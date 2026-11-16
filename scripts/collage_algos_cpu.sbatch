#!/bin/bash
# Study "algos" (CPU nodes): every algorithm of every collective, plus native
# MPI, on one communicator of all ranks, for 0..4 MB and a range of
# processes per node (PPN).
#
#   sbatch -N 32 collage_algos_cpu.sbatch           # one node count per job
#   ./submit_study.sh algos cpu                     # 16, 32, 64 and 128 nodes
#   PPN_LIST="16 128" COLLS="allreduce" OPS="sum max user_nc" sbatch -N 64 collage_algos_cpu.sbatch
#   DRYRUN=1 bash collage_algos_cpu.sbatch          # print the plan (login node)
#
# Resumable: resubmitting the same node count continues where the previous
# job stopped. Results: $SCRATCH/collage_study/algos/cpu_<N>N/raw/*.csv
#SBATCH -N 16
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -J collage-algos-cpu
#SBATCH --exclusive
#SBATCH --ntasks-per-node=128
#SBATCH --output=collage_algos_cpu_%j.out

set -uo pipefail
export COLLAGE_HOME="${COLLAGE_HOME:-$HOME/proj/collage}"
STUDY=algos
PPN_LIST="${PPN_LIST:-1 2 4 8 16 32 64 128}"
LAYOUTS="${LAYOUTS:-world}"
# COLLS, OPS, TYPES, SIZES_ARGS, ITERS, CHECK, ...: see collage_sweep.sh
source "$COLLAGE_HOME/scripts/profile_cpu.sh"
collage_run
