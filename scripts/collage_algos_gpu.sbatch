#!/bin/bash
# Study "algos" (GPU nodes): every algorithm of every collective, plus native
# MPI, one process per GPU (PPN 4), buffers in GPU memory through CUDA-aware
# Cray MPICH. By default each case is also run with host buffers on the same
# nodes (BUFFERS="gpu host"), for a GPU-vs-host comparison.
#
#   sbatch -N 32 collage_algos_gpu.sbatch
#   ./submit_study.sh algos gpu
#   BUFFERS=gpu GPU_REDUCE="kernel host" sbatch -N 64 collage_algos_gpu.sbatch
#   DRYRUN=1 bash collage_algos_gpu.sbatch
#
# GPU binding, NIC policy and GPU_MAP: see profile_gpu.sh.
# Results: $SCRATCH/collage_study/algos/gpu_<N>N/raw/*.csv
#SBATCH -N 16
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -J collage-algos-gpu
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --output=collage_algos_gpu_%j.out
# for the 80 GB A100 nodes (at most 256 nodes): #SBATCH -C gpu&hbm80g

set -uo pipefail
export COLLAGE_HOME="${COLLAGE_HOME:-$HOME/proj/collage}"
STUDY=algos
LAYOUTS="${LAYOUTS:-world}"
BUFFERS="${BUFFERS:-gpu host}"
source "$COLLAGE_HOME/scripts/profile_gpu.sh"
collage_run
