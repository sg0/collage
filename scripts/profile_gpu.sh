#!/bin/bash
# profile_gpu.sh -- Perlmutter GPU-node settings shared by every *_gpu.sbatch study.
# Node: 1x AMD EPYC 7763 (64 cores / 128 threads), 4x NVIDIA A100, 256 GB, 4 NICs.
#  - 4 tasks per node (one per GPU), -c 32 (= 2*floor(64/4)), --cpu-bind=cores:
#    local rank i runs on cores 16i..16i+15.
#  - --gpu-bind=none keeps all four GPUs visible to every task, so Cray MPICH
#    can use CUDA IPC between GPUs on a node; collage picks the device itself
#    with --gpu-map (GPU_MAP). 3,2,1,0 gives local rank i the GPU closest to its
#    cores (the mapping used by e.g. WarpX's Perlmutter profile); check
#    gpu_topology.txt, written by each job, to confirm on your nodes.
#  - MPICH_OFI_NIC_POLICY=GPU: each rank uses the NIC closest to its GPU.

export COLLAGE_HOME="${COLLAGE_HOME:-$HOME/proj/collage}"
export BIN="${BIN:-$COLLAGE_HOME/bin/collage-gpu}"
export NODE_TYPE=gpu
export NODE_MEM_GB=256
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export MPICH_GPU_SUPPORT_ENABLED=1      # required for CUDA-aware Cray MPICH
export MPICH_OFI_NIC_POLICY="${MPICH_OFI_NIC_POLICY:-GPU}"
export GPU_MAP="${GPU_MAP:-3,2,1,0}"
export MAX_MEM="${MAX_MEM:-16G}"        # per rank: well inside 40 GB HBM and the 64 GB host share
CORES_PER_NODE=64
SRUN_EXTRA="--gpus-per-node=4 --gpu-bind=none"
PPN_LIST=4                              # one process per GPU
BUFFERS="${BUFFERS:-gpu}"               # "gpu host": also run every case with host buffers
GPU_REDUCE="${GPU_REDUCE:-kernel}"      # kernel | mpi | host (reductions on GPU data)

# shellcheck source=collage_sweep.sh
source "${SWEEP_LIB:-$COLLAGE_HOME/scripts/collage_sweep.sh}"

collage_run() {
    [ -x "$BIN" ] || { echo "collage binary not found: $BIN (run scripts/build_perlmutter.sh)"; exit 1; }
    mkdir -p "$OUTDIR"
    {
        echo "study $STUDY, job $JOBID: $NODES GPU nodes, $(date)"
        echo "nodes: ${SLURM_JOB_NODELIST:-?}"
        echo "MPICH_GPU_SUPPORT_ENABLED=$MPICH_GPU_SUPPORT_ENABLED MPICH_OFI_NIC_POLICY=$MPICH_OFI_NIC_POLICY GPU_MAP=$GPU_MAP"
        module list 2>&1 | sed 's/^/  /'
    } > "$OUTDIR/job_${JOBID}.info" 2>&1
    if [ "$DRYRUN" != 1 ] && [ ! -s "$OUTDIR/gpu_topology.txt" ]; then
        # GPU <-> CPU/NUMA/NIC affinity of one node, to check GPU_MAP
        srun -N 1 -n 1 --gpus-per-node=4 --gpu-bind=none nvidia-smi topo -m > "$OUTDIR/gpu_topology.txt" 2>&1 || true
    fi
    collage_sweep "$PPN_LIST" "$CORES_PER_NODE" "$SRUN_EXTRA" "$BUFFERS" "$GPU_REDUCE"
}
