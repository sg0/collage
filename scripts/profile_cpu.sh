#!/bin/bash
# profile_cpu.sh -- Perlmutter CPU-node settings shared by every *_cpu.sbatch study.
# Node: 2x AMD EPYC 7763, 128 cores / 256 hardware threads, 512 GB, 1 Slingshot NIC.
# Each step uses -c = 2*floor(128/PPN) and --cpu-bind=cores (NERSC affinity
# guidance), so ranks are spread over both sockets and all 8 NUMA domains.

export COLLAGE_HOME="${COLLAGE_HOME:-$HOME/proj/collage}"
export BIN="${BIN:-$COLLAGE_HOME/bin/collage-cpu}"
export NODE_TYPE=cpu
export NODE_MEM_GB=512
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
CORES_PER_NODE=128
SRUN_EXTRA=""
BUFFERS="host"
GPU_REDUCE="-"

# shellcheck source=collage_sweep.sh
source "${SWEEP_LIB:-$COLLAGE_HOME/perlmutter/collage_sweep.sh}"

collage_run() {
    [ -x "$BIN" ] || { echo "collage binary not found: $BIN (run scripts/build_perlmutter.sh)"; exit 1; }
    mkdir -p "$OUTDIR"
    {
        echo "study $STUDY, job $JOBID: $NODES CPU nodes, $(date)"
        echo "nodes: ${SLURM_JOB_NODELIST:-?}"
        module list 2>&1 | sed 's/^/  /'
    } > "$OUTDIR/job_${JOBID}.info" 2>&1
    collage_sweep "$PPN_LIST" "$CORES_PER_NODE" "$SRUN_EXTRA" "$BUFFERS" "$GPU_REDUCE"
}
