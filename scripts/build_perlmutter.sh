#!/bin/bash
# Build both collage binaries on a Perlmutter login node:
#   bin/collage-cpu : CPU nodes   (module load cpu)
#   bin/collage-gpu : GPU nodes   (module load gpu: cudatoolkit + craype-accel-nvidia80,
#                                  CUDA-aware Cray MPICH via the GTL library)
# Usage:  perlmutter/build_perlmutter.sh [cpu|gpu|both]   (default both)
#
# Cray wrappers: CC is used instead of mpicxx. CRAY_ACCEL_TARGET=nvidia80 links
# the GPU Transport Layer (without it, MPICH_GPU_SUPPORT_ENABLED=1 aborts with
# "GTL library is not linked"). nvcc compiles the reduction kernels for A100 (sm_80).
set -euo pipefail
cd "$(dirname "$0")/.."
# `module` is a shell function; load Lmod's init in non-interactive shells
if ! type module >/dev/null 2>&1; then
    for f in /usr/share/lmod/lmod/init/bash /opt/cray/pe/lmod/lmod/init/bash; do
        [ -f "$f" ] && { source "$f"; break; }
    done
fi
type module >/dev/null 2>&1 || { echo "the 'module' command is not available"; exit 1; }
which="${1:-both}"
mkdir -p bin

build_cpu() (
    module load cpu
    make clean >/dev/null
    make MPICXX=CC CXXFLAGS="-O3 -std=c++17 -Wall -Wextra -march=znver3"
    cp collage bin/collage-cpu
    echo "built bin/collage-cpu"
)

build_gpu() (
    module load gpu
    module load cudatoolkit craype-accel-nvidia80
    export CRAY_ACCEL_TARGET=nvidia80
    : "${CUDA_HOME:?cudatoolkit did not set CUDA_HOME}"
    make clean >/dev/null
    make CUDA=1 MPICXX=CC NVCC="$(command -v nvcc)" CUDA_HOME="$CUDA_HOME" \
         CUDA_ARCH="-arch=sm_80" NVCCFLAGS="-O3 -std=c++17 -arch=sm_80"
    cp collage bin/collage-gpu
    echo "built bin/collage-gpu"
)

case "$which" in
    cpu) build_cpu ;;
    gpu) build_gpu ;;
    both) build_cpu; build_gpu ;;
    *) echo "usage: $0 [cpu|gpu|both]"; exit 1 ;;
esac
make clean >/dev/null
