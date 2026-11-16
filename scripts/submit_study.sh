#!/bin/bash
# Submit collage studies for several node counts (one job per study, node
# type and node count).
#
#   ./submit_study.sh algos cpu                 # 16 32 64 128 CPU nodes
#   ./submit_study.sh split gpu
#   ./submit_study.sh all both                  # every study on both node types
#   NODE_COUNTS="16 128" TIME=08:00:00 ./submit_study.sh tunables gpu
#   ./submit_study.sh algos cpu --dependency=singleton   # extra sbatch options pass through
#
# studies:     algos | split | tunables | all   (see collage_<study>_<type>.sbatch)
# node types:  cpu | gpu | both
# TIME and QOS override the scripts' own limits; any sweep variable (COLLS,
# PPN_LIST, LAYOUTS, OPS, ...) set in your environment is passed to the jobs
# (sbatch exports the environment by default). Jobs are resumable: submit
# again with the same node counts to continue.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
usage() { echo "usage: $0 algos|split|tunables|all cpu|gpu|both [sbatch options]"; exit 1; }
[ $# -ge 2 ] || usage
study=$1 kind=$2; shift 2
case "$study" in algos|split|tunables) studies=$study ;; all) studies="algos split tunables" ;; *) usage ;; esac
case "$kind" in cpu|gpu) kinds=$kind ;; both) kinds="cpu gpu" ;; *) usage ;; esac
NODE_COUNTS="${NODE_COUNTS:-16 32 64 128}"
export COLLAGE_HOME="${COLLAGE_HOME:-$(cd "$here/.." && pwd)}"

for s in $studies; do
    for k in $kinds; do
        script="$here/collage_${s}_${k}.sbatch"
        for n in $NODE_COUNTS; do
            opts=(-N "$n" -J "collage-${s}-${k}-${n}N")
            [ -n "${QOS:-}" ] && opts+=(-q "$QOS")
            [ -n "${TIME:-}" ] && opts+=(-t "$TIME")
            sbatch "${opts[@]}" "$@" "$script"
        done
    done
done
