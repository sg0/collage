#!/bin/bash
# collage_sweep.sh -- shared parameter-sweep engine for the Perlmutter study
# scripts (collage_{algos,split,tunables}_{cpu,gpu}.sbatch, via profile_*.sh).
# Every knob below can be overridden from the environment, e.g.
#   COLLS="allreduce bcast" PPN_LIST="4 64" sbatch -N 32 collage_algos_cpu.sbatch
# ALGOS_<collective>="a,b" restricts one collective to some algorithms.
#
# One "step" = one srun of collage = one (collective, PPN, layout, type, op,
# buffers, reduce mode) case, all algorithms + native, all message sizes.
#
#  - resumable: a step whose CSV already exists is skipped (FORCE=1 re-runs)
#  - time-aware: no new step is started when the job's remaining time is less
#    than STEP_MIN + SAFETY_MIN minutes
#  - every step is time-boxed (srun -t STEP_MIN) and recorded in manifest.tsv
#  - DRYRUN=1 prints the plan instead of running it

# ---------------------------------------------------------------- defaults
: "${NODE_TYPE:?set NODE_TYPE=cpu or gpu}"
: "${BIN:?set BIN to the collage binary}"
NODES="${NODES:-${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}}"
JOBID="${SLURM_JOB_ID:-local$$}"

COLLS="${COLLS:-allreduce reduce bcast allgather allgatherv alltoall alltoallv reduce_scatter reduce_scatter_block gather scatter barrier}"
REDUCTIONS="allreduce reduce reduce_scatter reduce_scatter_block"
TYPES="${TYPES:-double}"
OPS="${OPS:-sum}"                       # e.g. "sum max user user_nc"
LAYOUTS="${LAYOUTS:-world}"           # world node cross random random:P stride:K group:M
RANDOM_P="${RANDOM_P:-0.5}"
SIZES_ARGS="${SIZES_ARGS:---min 0 --max 4M --factor 2}"
ITERS="${ITERS:-50}"; WARMUP="${WARMUP:-5}"
ITERS_LARGE="${ITERS_LARGE:-10}"; WARMUP_LARGE="${WARMUP_LARGE:-2}"; LARGE="${LARGE:-64K}"
ROOT="${ROOT:-0}"
VDIST="${VDIST:-uniform}"
CHECK="${CHECK:-0}"                     # 1: also verify every case against native MPI
STEP_MIN="${STEP_MIN:-20}"              # time limit of one srun step (minutes)
SAFETY_MIN="${SAFETY_MIN:-2}"
EXTRA_ARGS="${EXTRA_ARGS:-}"            # passed to collage verbatim (e.g. "-s 0,64K -k 2,4")
DRYRUN="${DRYRUN:-0}"
FORCE="${FORCE:-0}"
STUDY="${STUDY:-algos}"
TRANSPORTS="${TRANSPORTS:-mpi}"         # GPU buffers: mpi | cray | mpich | acx (stream-triggered ones
                                        # need a binary built with ST=..., see build_perlmutter.sh)
OUTDIR="${OUTDIR:-${SCRATCH:-$PWD}/collage_study/${STUDY}/${NODE_TYPE}_${NODES}N}"
NODE_MEM_GB="${NODE_MEM_GB:?set NODE_MEM_GB (host memory per node)}"
MEM_FRACTION_PCT="${MEM_FRACTION_PCT:-50}"   # share of node memory for benchmark buffers

# --------------------------------------------------------------- helpers
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

# remaining wall time of this job in seconds (large if unknown / outside Slurm)
remaining_seconds() {
    local left
    [ -n "${SLURM_JOB_ID:-}" ] || { echo 999999999; return; }
    left=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null | head -1)
    case "$left" in
        ""|UNLIMITED|NOT_SET|INVALID) echo 999999999; return ;;
    esac
    local d=0 h=0 m=0 s=0 rest="$left"
    if [[ "$rest" == *-* ]]; then d="${rest%%-*}"; rest="${rest#*-}"; fi
    IFS=: read -r -a p <<< "$rest"
    case ${#p[@]} in
        3) h=${p[0]}; m=${p[1]}; s=${p[2]} ;;
        2) m=${p[0]}; s=${p[1]} ;;
        1) s=${p[0]} ;;
    esac
    echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}

# --split-mask spec for a layout name, or "-" when the layout does not apply
#   world  : one communicator of all ranks (no --split-mask)
#   node   : one group per node            (M = PPN, consecutive ranks)
#   cross  : groups of one rank per node   (M = NODES, stride K = PPN)
#   random : groups of PPN ranks, randomised (M = PPN, p = RANDOM_P)
#   random:P : as random, with p = P            (e.g. random:0.25)
#   stride:K : groups of PPN ranks spaced K apart (M = PPN, stride K)
#   group:M  : groups of M consecutive ranks     (e.g. group:256 = 2 CPU nodes at PPN 128)
layout_spec() {
    local name=$1 ppn=$2 ranks=$(( NODES * $2 ))
    local arg="${name#*:}"
    case "$name" in
        world)    echo "" ;;
        node)     [ "$ppn" -ge 2 ] && echo "$ppn" || echo "-" ;;
        cross)    [ "$ppn" -ge 2 ] && [ "$NODES" -ge 2 ] && echo "${NODES}:${ppn}" || echo "-" ;;
        random)   [ "$ppn" -ge 2 ] && echo "${ppn}:${RANDOM_P}" || echo "-" ;;
        random:*) [ "$ppn" -ge 2 ] && echo "${ppn}:${arg}" || echo "-" ;;
        stride:*) [ "$ppn" -ge 2 ] && [ $(( ranks % (ppn * arg) )) -eq 0 ] && echo "${ppn}:${arg}" || echo "-" ;;
        group:*)  [ "$arg" -ge 2 ] && [ "$arg" -lt "$ranks" ] && [ $(( ranks % arg )) -eq 0 ] && echo "$arg" || echo "-" ;;
        *) echo "unknown layout '$name'" >&2; echo "-" ;;
    esac
}

# -c and --cpu-bind for PPN tasks on a node with CORES physical cores (NERSC: 2*floor(cores/ppn))
cpus_per_task() { local cores=$1 ppn=$2; local c=$(( 2 * (cores / ppn) )); [ "$c" -ge 1 ] && echo "$c" || echo 1; }
cpu_bind_for()  { local cores=$1 ppn=$2; [ "$ppn" -le "$cores" ] && echo cores || echo threads; }

# ----------------------------------------------------------- the sweep
# arguments: PPN list, cores per node, srun extra args, buffers list, reduce-mode list
collage_sweep() {
    local ppn_list=$1 cores=$2 srun_extra=$3 buffers_list=$4 reduce_list=$5
    mkdir -p "$OUTDIR/raw" "$OUTDIR/logs"
    local manifest="$OUTDIR/manifest.tsv"
    [ -f "$manifest" ] || printf 'job\tstatus\trc\tseconds\tcsv\tcommand\n' > "$manifest"
    local planned=0 ran=0 skipped=0 failed=0 stopped=0

    log "collage sweep '$STUDY': $NODE_TYPE, $NODES nodes, job $JOBID"
    log "  binary : $BIN"
    log "  output : $OUTDIR"
    log "  PPN    : $ppn_list   layouts: $LAYOUTS   colls: $COLLS"

    for ppn in $ppn_list; do
        local c bind ranks
        c=$(cpus_per_task "$cores" "$ppn"); bind=$(cpu_bind_for "$cores" "$ppn")
        ranks=$(( NODES * ppn ))
        if [ "$ppn" -gt $(( 2 * cores )) ]; then log "skip PPN=$ppn (> hardware threads)"; continue; fi
        # per-rank buffer budget: a fraction of the node's memory divided by PPN
        local max_mem
        if [ -n "${MAX_MEM:-}" ]; then max_mem=$MAX_MEM
        else max_mem="$(( NODE_MEM_GB * MEM_FRACTION_PCT / 100 * 1024 / ppn ))M"; fi

        for layout in $LAYOUTS; do
            local spec; spec=$(layout_spec "$layout" "$ppn")
            [ "$spec" = "-" ] && continue
            local split_arg=""; [ -n "$spec" ] && split_arg="--split-mask=$spec"
            for coll in $COLLS; do
                local ops="-"; [[ " $REDUCTIONS " == *" $coll "* ]] && ops=$OPS
                local types=$TYPES; [ "$coll" = barrier ] && types=double
                for type in $types; do
                for op in $ops; do
                for buffers in $buffers_list; do
                    local reds="-"; [ "$buffers" = gpu ] && [ "$op" != "-" ] && reds=$reduce_list
                    local trs=mpi; [ "$buffers" = gpu ] && trs=$TRANSPORTS
                    local combos="" r t
                    for r in $reds; do for t in $trs; do combos="$combos $r/$t"; done; done
                    for combo in $combos; do
                        local red=${combo%/*} tr=${combo#*/}
                        planned=$((planned + 1))
                        local name="${coll}__${buffers}__ppn${ppn}__${layout//:/-}__${type}__${op}"
                        [ "$red" != "-" ] && name="${name}__${red}"
                        [ "$tr" != mpi ] && name="${name}__st-${tr}"
                        local csv="$OUTDIR/raw/${name}.csv" lg="$OUTDIR/logs/${name}.log"
                        if [ "$FORCE" != 1 ] && [ -s "$csv" ]; then skipped=$((skipped + 1)); continue; fi

                        local args=(-c "$coll" $SIZES_ARGS -t "$type" -r "$ROOT" --vdist "$VDIST"
                                    -i "$ITERS" -w "$WARMUP" --iters-large "$ITERS_LARGE"
                                    --warmup-large "$WARMUP_LARGE" --large "$LARGE" --max-mem "$max_mem"
                                    --csv "$csv" --tag "job=$JOBID;study=$STUDY;nodetype=$NODE_TYPE;layoutname=$layout")
                        [ "$op" != "-" ] && args+=(-o "$op")
                        # per-collective algorithm subset, e.g. ALGOS_bcast="chain,pipeline"
                        local algos_var="ALGOS_${coll}"
                        [ -n "${!algos_var:-}" ] && args+=(-a "${!algos_var}")
                        [ -n "$split_arg" ] && args+=("$split_arg")
                        [ "$CHECK" = 1 ] && args+=(--check)
                        if [ "$buffers" = gpu ]; then
                            args+=(--gpu ${GPU_MAP:+--gpu-map "$GPU_MAP"})
                            [ "$red" != "-" ] && args+=(--gpu-reduce "$red")
                            [ "$tr" != mpi ] && args+=(--transport "$tr")
                        fi
                        # shellcheck disable=SC2206
                        args+=($EXTRA_ARGS)
                        local cmd=(srun -N "$NODES" -n "$ranks" --ntasks-per-node="$ppn" -c "$c"
                                   --cpu-bind="$bind" $srun_extra -t "$STEP_MIN" "$BIN" "${args[@]}")

                        if [ "$DRYRUN" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; continue; fi

                        local left; left=$(remaining_seconds)
                        if [ "$left" -lt $(( (STEP_MIN + SAFETY_MIN) * 60 )) ]; then
                            log "stopping: $((left / 60)) min left, a step may take $STEP_MIN min (resubmit to resume)"
                            stopped=1; break 7   # out of all seven loops
                        fi
                        log "run [$ranks ranks] $name"
                        local t0=$SECONDS rc=0
                        "${cmd[@]}" > "$lg" 2>&1 || rc=$?
                        local dt=$(( SECONDS - t0 )) status=ok
                        if [ "$rc" -ne 0 ]; then
                            status=fail; failed=$((failed + 1))
                            grep -qiE "time limit|due to time" "$lg" && status=timeout
                            rm -f "$csv"   # not complete: retried on the next submission
                            log "  -> $status (rc=$rc), see $lg"
                        fi
                        ran=$((ran + 1))
                        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$JOBID" "$status" "$rc" "$dt" "${csv##*/}" \
                            "$(printf '%q ' "${cmd[@]}")" >> "$manifest"
                    done
                done; done; done; done
        done
    done
    if [ "$DRYRUN" = 1 ]; then log "dry run: $planned steps planned ($skipped already done)"; return 0; fi
    log "done: $planned planned, $ran run ($failed failed), $skipped already complete$([ $stopped = 1 ] && echo ', stopped early: resubmit to continue')"
}
