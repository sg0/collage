#!/bin/bash
# Study "tunables" (GPU nodes): as collage_tunables_cpu.sbatch, one process per
# GPU with GPU buffers (segmentation matters differently when every segment
# is a device-memory message).
#
#   sbatch -N 32 collage_tunables_gpu.sbatch
#   ./submit_study.sh tunables gpu
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
# Results: $SCRATCH/collage_study/tunables/gpu_<N>N/raw/*.csv
#SBATCH -N 16
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH -J collage-tunables-gpu
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --output=collage_tunables_gpu_%j.out
# for the 80 GB A100 nodes (at most 256 nodes): #SBATCH -C gpu&hbm80g

set -uo pipefail
export COLLAGE_HOME="${COLLAGE_HOME:-$HOME/proj/collage}"
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
source "$COLLAGE_HOME/scripts/profile_gpu.sh"
collage_run
