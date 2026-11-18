# collage studies on NERSC Perlmutter

These job scripts run `collage` parameter studies on 16–128 Perlmutter nodes,
and `../plots/plot_collage.py` turns the results into figures. There are three
studies, and each one has a CPU-node script and a GPU-node script.

| study | question | CPU script | GPU script |
|---|---|---|---|
| `algos` | How fast is every algorithm of every collective, compared with native Cray MPICH, from 0 B to 4 MB? | `collage_algos_cpu.sbatch`: PPN 1 2 4 … 128 | `collage_algos_gpu.sbatch`: PPN 4, GPU and host buffers |
| `split` | How do the algorithms behave when many groups run at once (`--split-mask`), and how much does group placement matter? | `collage_split_cpu.sbatch`: PPN 16 64 128 | `collage_split_gpu.sbatch`: PPN 4 |
| `tunables` | Which segment size, chain fan-out, k-nomial radix or request limit is best? | `collage_tunables_cpu.sbatch`: PPN 16 128 | `collage_tunables_gpu.sbatch`: PPN 4 |

## Files

| file | role |
|---|---|
| `build_perlmutter.sh` | builds `bin/collage-cpu` (CPU nodes) and `bin/collage-gpu` (GPU nodes, CUDA-aware) |
| `collage_<study>_<cpu\|gpu>.sbatch` | the six job scripts: the `#SBATCH` header plus the study's defaults |
| `profile_cpu.sh`, `profile_gpu.sh` | node-type settings shared by every study: binary, memory, affinity, GPU binding, MPICH environment |
| `collage_sweep.sh` | the sweep engine: loops, time guard, resume, manifest |
| `submit_study.sh` | submits a study for several node counts |

## 1. Build (login node)

```bash
cd $HOME/proj/collage          # or set COLLAGE_HOME to wherever the sources are
scripts/build_perlmutter.sh # both binaries; or: cpu | gpu
```

To also compile in Cray MPICH's stream-triggered transport (`--transport
cray`, see the main README), build the GPU binary with
`ST=cray scripts/build_perlmutter.sh gpu`. The plain transport stays
available in that binary, so `TRANSPORTS="mpi cray"` compares both in one job.
Whether the installed Cray MPICH supports these queues with CUDA streams is
not confirmed: HPE documents them with HIP streams. Run a short `--check` job
first.

The CPU binary is built with `module load cpu` and `CC -march=znver3`. The GPU
binary is built with `module load gpu cudatoolkit craype-accel-nvidia80` and
`CRAY_ACCEL_TARGET=nvidia80`, which links Cray MPICH's GPU transport layer. The
reduction kernels are built for `sm_80`.

## 2. Submit

```bash
scripts/submit_study.sh algos cpu       # 16, 32, 64, 128 CPU nodes: 4 jobs
scripts/submit_study.sh split gpu
scripts/submit_study.sh all both        # every study on both node types: 24 jobs
NODE_COUNTS="16 128" TIME=08:00:00 perlmutter/submit_study.sh tunables cpu
sbatch -N 64 scripts/collage_split_cpu.sbatch   # a single job
```

All scripts use account `m4626` and `-q regular`. Override these with
`sbatch -A`/`-q`, or with `QOS=` for `submit_study.sh`. The `debug` QOS allows
at most 8 nodes and 30 minutes, so it suits a smoke test:

```bash
NODE_COUNTS=2 QOS=debug TIME=00:20:00 COLLS=allreduce PPN_LIST=128 \
    scripts/submit_study.sh algos cpu
```

To print the plan without running anything (on a login node):

```bash
DRYRUN=1 SLURM_JOB_NUM_NODES=128 bash scripts/collage_split_cpu.sbatch
```

### Resubmit until done

One *step* is one `srun` of `collage`: a single (collective, PPN, layout, type,
op, buffers, reduce mode) case, covering all its algorithms and all sizes. Each
step is time-boxed with `srun -t STEP_MIN`. A job starts no new step when less
than `STEP_MIN + SAFETY_MIN` minutes remain, and finished steps are skipped on
the next submission. So you can submit the same study and node count again
until the log says `done: … 0 run`. A step that fails is recorded in
`manifest.tsv`, its CSV is removed, and it is retried on the next submission.
To give up on a case, take it out of `COLLS`/`LAYOUTS`/`PPN_LIST`.

### Knobs

Any of these can be set in the environment of `sbatch` or `submit_study.sh`.
Each study script only sets defaults.

| variable | meaning (default) |
|---|---|
| `COLLS` | collectives (all 12 in `algos`; 5 in `split`; 7 in `tunables`) |
| `PPN_LIST` | processes per node on CPU nodes (GPU nodes: always 4) |
| `LAYOUTS` | group layouts (see below) |
| `OPS`, `TYPES` | reduction ops and datatypes (`sum`, `double`); e.g. `OPS="sum max user_nc"` |
| `SIZES_ARGS` | sizes passed to collage (`--min 0 --max 4M --factor 2`; `tunables`: factor 4) |
| `ITERS`/`WARMUP`, `ITERS_LARGE`/`WARMUP_LARGE`, `LARGE` | iterations below/above `LARGE` (50/5, 10/2, 64K) |
| `EXTRA_ARGS` | passed verbatim to collage (`tunables`: `-s 0,8K,64K,512K --fanout 2,4,8 -k 2,4,8 --max-reqs 0,4,16`) |
| `ALGOS_<collective>` | algorithm subset for one collective, e.g. `ALGOS_bcast=chain,knomial,native` |
| `CHECK=1` | also verify every result against native MPI |
| `BUFFERS` | GPU scripts: `gpu`, `host` or `"gpu host"` (`algos` default: both) |
| `GPU_REDUCE` | local reductions on GPU data: `kernel` `mpi` `host` (`kernel`) |
| `GPU_MAP` | device of local rank i (`3,2,1,0`) |
| `TRANSPORTS` | GPU scripts: transports of the algorithms' messages, e.g. `"mpi cray"` (`mpi`; `cray` needs `ST=cray perlmutter/build_perlmutter.sh gpu`; `shmem` needs `ST=nvshmem perlmutter/build_perlmutter.sh gpu` and NERSC's `nvshmem` module, and runs over libfabric with NVSHMEM's host proxy here; `ucp` is for InfiniBand/RoCE clusters, since UCX has no device lanes on Slingshot) |
| `MAX_MEM` | per-rank buffer cap (CPU: half the node's memory / PPN; GPU: 16G) |
| `STEP_MIN`, `SAFETY_MIN` | step time limit and margin, in minutes (20 or 30, and 2) |
| `STUDY`, `OUTDIR` | output location (`$SCRATCH/collage_study/<study>/<cpu\|gpu>_<N>N`) |
| `FORCE=1`, `DRYRUN=1` | re-run finished steps; print the plan only |

### Layouts

| name | `--split-mask` | groups |
|---|---|---|
| `world` | (none) | one communicator of all ranks |
| `node` | `PPN` | one group per node: on-node traffic only |
| `cross` | `NODES:PPN` | one rank per node in each group: all traffic goes over the network |
| `stride:K` | `PPN:K` | PPN ranks spaced K apart, spanning K nodes |
| `random:P` | `PPN:P` | PPN ranks, randomised: small P is nearly consecutive, P = 1.0 is fully scattered |
| `random` | `PPN:$RANDOM_P` | as `random:P` with `RANDOM_P` (0.5) |
| `group:M` | `M` | M consecutive ranks, spanning several nodes |

A layout that does not fit the rank count is skipped. Examples are `stride:K`
when PPN·K does not divide the number of ranks, and `group:M` when M does not
divide it or is not smaller than it.

## Node settings

**CPU nodes** have 2 × EPYC 7763 (128 cores, 256 threads), 512 GB and one
NIC. Each step uses `-c 2·⌊128/PPN⌋` with `--cpu-bind=cores`, following NERSC's
affinity guidance, so ranks are spread over both sockets and all NUMA domains.

**GPU nodes** have one EPYC 7763 (64 cores), 4 × A100 (40 GB), 256 GB and four
NICs. Each step uses 4 tasks per node, `-c 32 --cpu-bind=cores`,
`--gpus-per-node=4 --gpu-bind=none`, and `MPICH_GPU_SUPPORT_ENABLED=1`.
`--gpu-bind=none` keeps all four GPUs visible to every task, so Cray MPICH can
use CUDA IPC between GPUs on the same node. collage picks the device itself
with `--gpu-map 3,2,1,0`, which gives local rank i the GPU nearest its cores.
`MPICH_OFI_NIC_POLICY=GPU` sends each rank's traffic through the NIC nearest
its GPU. Each GPU job writes `gpu_topology.txt` (`nvidia-smi topo -m`) so you
can confirm the mapping. For the 80 GB A100 nodes, use `-C "gpu&hbm80g"`
(there are 256 of them).

## Output

```
$SCRATCH/collage_study/<study>/<cpu|gpu>_<N>N/
    raw/<coll>__<buffers>__ppn<P>__<layout>__<type>__<op>[__<reduce>][__st-<transport>].csv
    logs/<same name>.log          collage's stdout/stderr (layout report, table)
    manifest.tsv                  every step: job, status, rc, seconds, csv, command
    job_<id>.info                 nodes, modules, environment
    gpu_topology.txt              GPU jobs
```

Every CSV row carries `tag=job=…;study=…;nodetype=…;layoutname=…`.

## 3. Plot

Requirements: `python3` with pandas, matplotlib and seaborn. On Perlmutter,
`module load python` provides all three.

```bash
python3 plots/plot_collage.py $SCRATCH/collage_study -o figs            # every figure
python3 plots/plot_collage.py $SCRATCH/collage_study/algos -o figs/algos --kind latency,best,scaling,ppn
python3 plots/plot_collage.py $SCRATCH/collage_study --study split -o figs/split --kind layout,random
python3 plots/plot_collage.py $SCRATCH/collage_study --study tunables -o figs/tunables --kind tunables
python3 plots/plot_collage.py $SCRATCH/collage_study --study algos --kind buffers --nodes 128 -o figs/gpu
```

| kind | figure |
|---|---|
| `latency` | latency vs size, one panel per node count, one line per algorithm (native: black dashed) |
| `best` | heatmap nodes × size: the fastest algorithm in each cell, coloured by its speedup over native (+ `best_algorithms.csv`) |
| `scaling` | latency vs node count at a few sizes (`--sizes 1K,64K,4M`) |
| `ppn` | latency vs processes per node, per node count (CPU) |
| `buffers` | host-buffer latency / GPU-buffer latency vs size (GPU `algos` study) |
| `transport` | CUDA-aware MPI latency / stream-triggered latency vs size, one figure set per transport (GPU runs with `TRANSPORTS="mpi cray"`) |
| `layout` | aggregate bandwidth vs size, one panel per algorithm, one line per layout (`split`) |
| `random` | aggregate bandwidth vs p of `PPN:p`, with p = 0 for the `node` layout, one line per algorithm (`split`) |
| `tunables` | per algorithm: rows of tunable settings × sizes, latency relative to the fastest row (native included), ● marks the fastest (+ `tunables_best.csv`) |

Filters: `--study --coll --algos --nodes --ppn --layout --buffers --transport --gpu-reduce
--op --type`. One figure is written for each combination of the configuration
columns that remain (study, buffers, reduce mode, PPN, layout, type, op, root).
Output formats are set with `--formats` (default `png,pdf`).
