#!/bin/bash
# Study "tunables" (CPU nodes): segment size, chain fan-out, k-nomial radix and
# request throttling of the algorithms that have them, on one communicator.
#
#   sbatch -N 32 collage_tunables_cpu.sbatch
#   ./submit_study.sh tunables cpu
#   EXTRA_ARGS="-s 0,16K,128K" COLLS=bcast sbatch -N 64 collage_tunables_cpu.sbatch
#
# Only the algorithms that take tunables run (plus native, listed in each
# ALGOS_<collective>, as the baseline);
# each combination becomes its own variant column, e.g. chain[seg=8K,fo=4]:
#   -s / --segsize   0 (unsegmented), 8K, 64K, 512K
#   --fanout         2, 4, 8          (chain algorithms)
#   -k / --radix     2, 4, 8          (k-nomial, k-Bruck)
#   --max-reqs       0 (unlimited), 4, 16
# Sizes step by 4x (0, 1, 4, ... 4M) to keep the number of cases manageable;
# set SIZES_ARGS="--min 0 --max 4M --factor 2" for the full grid.
# Results: $SCRATCH/collage_study/tunables/cpu_<N>N/raw/*.csv
#SBATCH -N 16
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH -J collage-tunables-cpu
#SBATCH --exclusive
#SBATCH --ntasks-per-node=128
#SBATCH --output=collage_tunables_cpu_%j.out

set -uo pipefail
export COLLAGE_HOME="${COLLAGE_HOME:-$HOME/proj/collage}"
PPN_LIST="${PPN_LIST:-16 128}"
STUDY=tunables
LAYOUTS="${LAYOUTS:-world}"
COLLS="${COLLS:-allreduce allgather alltoall bcast reduce gather scatter}"
SIZES_ARGS="${SIZES_ARGS:---min 0 --max 4M --factor 4}"
EXTRA_ARGS="${EXTRA_ARGS:--s 0,8K,64K,512K --fanout 2,4,8 -k 2,4,8 --max-reqs 0,4,16}"
STEP_MIN="${STEP_MIN:-30}"
ALGOS_allreduce="${ALGOS_allreduce:-nonoverlapping,ring_segmented,native}"
ALGOS_allgather="${ALGOS_allgather:-k_bruck,native}"
ALGOS_alltoall="${ALGOS_alltoall:-linear_sync,native}"
ALGOS_bcast="${ALGOS_bcast:-bintree,pipeline,chain,binomial,split_bintree,knomial,native}"
ALGOS_reduce="${ALGOS_reduce:-chain,pipeline,binary,binomial,in_order_binary,knomial,native}"
ALGOS_gather="${ALGOS_gather:-linear_sync,native}"
ALGOS_scatter="${ALGOS_scatter:-linear_nb,native}"
source "$COLLAGE_HOME/scripts/profile_cpu.sh"
collage_run
