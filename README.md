# collage — point-to-point collective algorithm benchmark

`collage` benchmarks the collective algorithms from Open MPI's `coll/base`
component (commit `ec987ee891f9d9b91f21edde9eab3f991634f57d`). Each algorithm
is re-implemented with plain MPI point-to-point calls. You pick one collective,
and `collage` times every algorithm for it (plus the MPI library's own
collective as a baseline) over a range of message sizes. By default the range
is 0 B to 4 MiB in powers of two. It can optionally check each result against
the native MPI collective.

The algorithm set is the one in the original headers. No algorithms were added.

I would like to acknowledge the implicit generosity and dedication of the MPI implementers, 
researchers, and, broadly the MPI Forum for continuing to steer HPC ahead.

## Layout

| file | contents |
|---|---|
| `utils.hpp` | type map, reduction ops, `CollBase` (dup'ed communicator, large-count-aware p2p wrappers, local reductions), port of `coll_base_topo.c` (binary/k-ary, binomial, in-order binomial, in-order binary, k-nomial and chain trees) |
| `allgather.hpp` … `scatter.hpp` | one class template per collective, `Coll<T>`, one member function per algorithm |
| `memory.hpp` | memory layer: host/GPU space, `Buf<T>` temporaries with a caching device allocator, overlap-safe copies, reduction dispatch |
| `gpu_ops.hpp`, `gpu_reduce.cu` | element-wise reduction operators and their CUDA kernels (built with `make CUDA=1`) |
| `bench.hpp` | one runner per collective (buffers, init, native reference, verification), the sweep/timing harness, CSV output |
| `bench.cpp` | command-line driver, runtime datatype dispatch |

The headers can also be used on their own:

```c++
#include "allreduce.hpp"
Allreduce<float> ar(MPI_COMM_WORLD);          // dups the communicator
ar.ring(sbuf, rbuf, count, MPI_SUM);           // or any MPI_Op, including user ops
```

## Build

```bash
make                    # uses mpicxx, C++17
make MPICXX=CC          # e.g. Cray wrappers
make NO_LARGE_COUNT=1   # MPI-3 libraries without MPI_Send_c & co.
make CUDA=1             # GPU buffers (--gpu); needs nvcc and a CUDA-aware MPI
make CUDA=1 CUDA_HOME=/opt/cuda CUDA_ARCH=-arch=sm_90
```

With an MPI-4 library, all counts go through the `MPI_*_c` large-count calls.
This means `size × 4 MiB` buffers work at any scale. Without MPI-4, any count
above `INT_MAX` raises an error.

## Run

```bash
# all allreduce algorithms, 0..4 MiB, float sum, verified against MPI_Allreduce
mpirun -np 128 ./collage -c allreduce -t float -o sum --check

# a non-builtin and a non-commutative reduction
mpirun -np 128 ./collage -c reduce -o user
mpirun -np 128 ./collage -c reduce -o user_nc

# a few bcast algorithms with a segment-size sweep (one column per value)
mpirun -np 256 ./collage -c bcast -a binomial,pipeline,knomial,native -s 0,8K,64K -k 2,4

# irregular alltoallv counts, explicit sizes, statistics to CSV
mpirun -np 64 ./collage -c alltoallv --vdist random --sizes 0,64,1K,64K,1M,4M --csv a2av.csv

# correctness only (fast, good for CI); exit status 2 on any mismatch
mpirun -np 7 ./collage -c reduce_scatter -o user_nc --check-only

./collage --list        # collectives, algorithms and the tunables each one uses
./collage --help
```

### Options

| option | meaning | default |
|---|---|---|
| `-c NAME` | collective: `allgather allgatherv allreduce alltoall alltoallv barrier bcast gather reduce reduce_scatter reduce_scatter_block scatter` | – |
| `-a LIST` | algorithms to run (`native` = the MPI library's collective) | all + native |
| `-t TYPE` | `int8 uint8 int16 uint16 int32 uint32 int64 uint64 float double` | `double` |
| `-o OP` | `sum prod max min land lor lxor band bor bxor user user_nc` | `sum` |
| `-m / -M / -f` | min bytes / max bytes / geometric factor | `0 / 4M / 2` |
| `--sizes LIST` | explicit sizes (overrides `-m/-M/-f`), e.g. `0,8,1K,4M` | – |
| `-r R` | root of `bcast reduce gather scatter` | `0` |
| `-s LIST` | segment size in bytes (`0` = no segmentation) | `0` |
| `--fanout LIST` | chains for `chain` algorithms | `4` |
| `-k LIST` | radix for `knomial` / `k_bruck` | `4` |
| `--max-reqs LIST` | outstanding-request limit (`0` = unlimited) | `0` |
| `--first-seg LIST` | first segment for gather `linear_sync` | `1K` |
| `--vdist D` | counts of the v-collectives: `uniform`, `random` (mean = size), `linear` | `uniform` |
| `-i / -w` | timed / warm-up iterations | `100 / 10` |
| `--iters-large / --warmup-large / --large` | same, for messages above `--large` | `20 / 5 / 64K` |
| `--max-mem B` | skip cases whose buffers exceed `B` per rank | `2G` |
| `--check`, `--check-only` | verify against native MPI (and time / don't time) | off |
| `--csv FILE` | `avg/min/max` µs and algorithmic bandwidth per (variant, size) | – |
| `--split-mask M[:K\|:p]` | run on N/M concurrent communicators of M ranks; members consecutive, K apart, or randomised by p (see below) | off |
| `--gpu` | user buffers and all algorithm temporaries in GPU memory (needs `make CUDA=1`, CUDA-aware MPI) | off |
| `--gpu-reduce MODE` | local reductions on GPU data: `kernel`, `mpi` or `host` (see below) | `kernel` |
| `--assume-cuda-aware` | use device buffers even if the library does not report CUDA awareness | off |

Tunable options take comma-separated lists. An algorithm that uses a tunable
gets one column per value, labelled for example `knomial[seg=8K,k=2]`.

### What "size" means

| collective | message size is … |
|---|---|
| bcast | bytes broadcast |
| reduce, allreduce | bytes reduced on each rank |
| allgather, gather | bytes contributed by each rank |
| scatter, reduce_scatter_block | bytes received by each rank |
| alltoall | bytes sent to each peer |
| alltoallv, allgatherv, reduce_scatter | *average* bytes per block (`--vdist`) |
| barrier | no data (single row) |

Sizes are rounded down to whole elements of `-t`.

### Output

```
# bytes      nonoverlapping recursivedoubling         ring ring_segmented ...  native
0                     81.18             60.15        44.66          44.99 ...   22.31
8                     83.69             45.86        42.88          54.85 ...   46.81
```

Each cell is the mean latency in µs, averaged over ranks (OSU convention). The
CSV file also has the min and max over ranks. Other cell values:

* `n/a`: the algorithm can't run in this configuration. Examples are
  `two_procs` on more than 2 ranks, or a commutative-only algorithm with
  `-o user_nc`. The reason is printed below the table.
* `skip`: the case would exceed `--max-mem`.
* `FAIL:...`: `--check` found wrong results.

The harness uses MPI collectives only for synchronisation (a barrier before
each timed iteration), statistics, and the `--check` reference. The algorithms
themselves only use point-to-point calls, derived datatypes and
`MPI_Reduce_local`.

### Concurrent groups (`--split-mask`)

```bash
# 32 ranks -> 4 communicators of 8 ranks, 4 rings running at the same time
mpiexec -n 32 ./collage -c allgather -a ring --split-mask=8       # consecutive
mpiexec -n 32 ./collage -c allgather -a ring --split-mask=8:4     # members 4 apart
mpiexec -n 32 ./collage -c allgather -a ring --split-mask=8:0.45  # randomised, p = 0.45
```

`--split-mask M[:K|:p]` splits the N processes into N/M communicators of M
ranks each; M must divide N. Every algorithm then runs on all groups at once.
Each group has its own instance of the collective, so `ring` gives N/M
independent rings and a rooted collective has one root per group. Think of the
ranks as laid out in a straight line. Which ranks form a group is set by the
optional suffix:

* **`M:K`: fixed stride.** The members of a group are exactly K ranks apart.
  The line is cut into blocks of M·K consecutive ranks, and each block holds K
  interleaved groups: `{b·MK + j + i·K : i = 0..M−1}` for `j = 0..K−1`. N must
  be divisible by M·K. `K = 1` (also what plain `M` means) gives consecutive
  ranks, and `K = N/M` gives `rank % (N/M)`. Example with N = 32 and
  `--split-mask=8:2`: `{0,2,…,14}`, `{1,3,…,15}`, `{16,18,…,30}`,
  `{17,19,…,31}`.
* **`M:p`: randomised members, p ∈ [0, 1].** The group size stays M, but the
  members are reshuffled from the consecutive layout. The tool performs about
  p·N/2 swaps. Each swap exchanges two not-yet-moved ranks from different
  groups that are at most `R = M·(N/M)^p` apart on the line. As a result,
  about a fraction p of all members end up outside their consecutive group:
  `8:0.5` is a 50-50 chance per member. Low p keeps groups nearly
  consecutive, and a foreign member typically comes from the adjoining group
  (at most about two groups away). High p moves almost every rank, over
  distances up to the whole line. This differs from `M:K`, where the distance
  between members is fixed. p is told apart from K by its decimal point:
  `8:1` is stride 1, `8:1.0` is p = 1, and `8:0.0` is the consecutive layout.
  The layout is drawn with `--seed` (default 42) on rank 0 and broadcast, so a
  given seed always reproduces the same groups.

Every run, with or without `--split-mask`, starts with a group-layout report.
Without the option, the report describes a single group of all N ranks on
`MPI_COMM_WORLD`:

```
# groups       : 4 concurrent communicators of 8 ranks, members 4 ranks apart (--split-mask 8:4)
# layout       : 4/4 groups non-consecutive, 75% of ranks outside their consecutive group,
#                mean member gap 4.00, max group span 28
# nodes        : 4 nodes, 8 ranks per node, block placement (consecutive ranks share a node)
#                each group spans 4..4 nodes (mean 4.00); 0/4 groups are intra-node
#   group 0    : {0,4,8,12,16,20,24,28}  (4 nodes)
#   ...
```

The report contains:

* how the groups were formed;
* how far the layout is from consecutive: the number of non-consecutive
  groups, the fraction of ranks outside their consecutive group, the mean gap
  between neighbouring members and the largest group span;
* how the ranks sit on nodes: the node count, ranks per node, and whether
  the launcher placed ranks in blocks or round-robin;
* how many nodes each group spans, and how many groups stay inside one node;
* the first four groups' members, each with its node count.

Nodes are identified by host name (`MPI_Get_processor_name`). The report shows
how a given layout interacts with the launcher's placement. For example,
`8:4` spreads every group over 4 nodes with block placement, but keeps each
group on a single node with round-robin placement.

Inside a group, ranks are ordered by world rank, and `--root` is the rank
*within* a group.

* All groups start each timed iteration together: the barrier before every
  iteration is on `MPI_COMM_WORLD`. The groups therefore contend for the
  network and shared links, which is the point of the mode.
* **Latency table:** a group's latency is the mean over its ranks. Each cell
  is the mean of that over the groups.
* **Aggregate bandwidth table** (printed after the latency table): the sum over
  groups of `bytes / group latency` in MB/s, where `bytes` follows the
  per-collective meaning of "size" above.
* **CSV:** adds `groups`, `group_ranks`, `layout` (`consecutive`, `stride=K`
  or `random:p=P:seed=S`), `group_min_us` / `group_max_us` (the fastest and
  slowest group) and `aggbw_MBps`.
* `--check` verifies every group against the native collective on that
  group's communicator.

How groups map onto nodes depends on how the launcher places ranks. For
example, with 8 ranks per node packed by core, `--split-mask 8` puts each
group inside one node, `--split-mask 8:K` with K ≥ 8 spreads each group over
8 nodes, and `M:p` gives everything in between.

### GPU buffers (`--gpu`)

```bash
make CUDA=1
mpirun -np 16 ./collage -c allreduce --gpu --check            # device buffers, CUDA reduction kernels
mpirun -np 16 ./collage -c allreduce --gpu --gpu-reduce mpi    # let MPI_Reduce_local reduce device data
mpirun -np 64 ./collage -c allgather -a ring --gpu --split-mask=8
```

With `--gpu`, every buffer an algorithm touches is in GPU memory, not only
the user buffers. MPI calls receive device pointers, which is why the library
must be CUDA-aware. The algorithms are unchanged, but the three things they do
outside MPI are routed through `memory.hpp`:

| what | host build / no `--gpu` | `--gpu` |
|---|---|---|
| temporary buffers (`Buf<T>`) | `new T[n]()` | `cudaMalloc`, served from a caching pool so that per-call temporaries cost no `cudaMalloc` after warm-up |
| local copies (rotations, self blocks, packing) | `memmove` | `cudaMemcpyAsync` + stream sync; overlapping ranges go through a temporary, since `cudaMemcpy` is undefined for overlap |
| local reductions | `MPI_Reduce_local` | `--gpu-reduce`: `kernel` = CUDA kernels for every predefined op × type (default); `mpi` = `MPI_Reduce_local` on device pointers (only if your library supports it); `host` = stage through pinned host memory |

User-defined ops (`-o user`, `-o user_nc`) are always reduced on the host,
because their function is CPU code. The run reports how many local reductions
on rank 0 took each path. Indexed derived datatypes are still passed to MPI
(alltoall `bruck`, allgatherv `bruck`/`neighborexchange`/`basic_default`,
reduce_scatter_block `recursivedoubling`). This relies on the library
supporting non-contiguous datatypes in device memory, which Open MPI and MPICH
do.

* **Which GPU.** Each process uses device `local rank % visible devices`. The
  local rank is read from the launcher's environment before `MPI_Init`
  (`OMPI_COMM_WORLD_LOCAL_RANK`, `MV2_COMM_WORLD_LOCAL_RANK`,
  `MPI_LOCALRANKID`, `PMI_LOCAL_RANK`, `PALS_LOCAL_RANKID`, `SLURM_LOCALID`,
  `FLUX_TASK_LOCAL_ID`), because several CUDA-aware MPIs set up their CUDA
  context inside `MPI_Init`. Otherwise it comes from an `MPI_COMM_TYPE_SHARED`
  split after init. If the launcher binds one GPU per process
  (`CUDA_VISIBLE_DEVICES`, `--gpus-per-task=1`), every process simply uses its
  only visible device. The header names rank 0's device and the source of the
  local rank.
* **CUDA awareness.** The library is asked: Open MPI via
  `MPIX_Query_cuda_support()`, MPICH ≥ 4 via `MPIX_GPU_query_support()`. If
  any rank answers "no", the run stops, because device pointers would crash a
  non-CUDA-aware library. `--assume-cuda-aware` overrides that.
  "Unknown" (no query available) gives a warning and continues.
* **Timing.** Every local copy and kernel synchronises the stream before
  returning, because the next step is usually an MPI call on the same buffer.
  This is the same constraint an MPI library's own GPU path has, so the
  timings include it. Inputs are generated on the host and uploaded, and
  `--check` downloads results. Neither transfer is inside the timed region.
* **Baseline.** `native` passes the same device buffers to the library's own
  collective, i.e. its CUDA-aware path.
* **Memory.** `--max-mem` then limits GPU memory per process. Cached device
  blocks are released before `MPI_Finalize`.
* **Caveats.** Whether the native collective (and therefore `--check`)
  supports a user-defined op on device buffers depends on the library; prefer
  predefined ops on GPUs. The `mpi` reduce mode likewise depends on the
  library accepting device pointers in `MPI_Reduce_local`.

## Reduction operations

`-o` accepts all predefined operations that make sense for the element type.
Bitwise operations are rejected for `float`/`double`. Two user-defined
operations are also available:

* `user`: element-wise sum registered with `MPI_Op_create` (commutative). It
  measures the cost of a non-builtin reduction kernel.
* `user_nc`: `f(a, b) = a`, which is associative but **non-commutative**. It
  exercises the order-preserving paths: in-order binary trees, the
  non-commutative branches of recursive doubling and butterfly, and the
  upstream fallbacks.

Algorithms that upstream documents or implements as commutative-only report
`n/a` for non-commutative ops instead of returning wrong results. These are:
reduce `chain/pipeline/binary/binomial/knomial`, allreduce
`ring/ring_segmented/allgather_reduce`, and reduce_scatter
`recursivehalving/ring`. Where upstream itself falls back for non-commutative
ops (e.g. `redscat_gather` → `linear`), the fallback is kept.

Any `MPI_Op` can be passed to the headers directly.

## Algorithms

| collective | algorithms (tunables) |
|---|---|
| allgather | recursivedoubling, sparbit, ring, neighborexchange, linear, k_bruck (k), direct_messaging |
| allgatherv | bruck, sparbit, ring, neighborexchange, two_procs, basic_default |
| allreduce | nonoverlapping (s, m), recursivedoubling, ring, ring_segmented (s), linear, redscat_allgather, allgather_reduce |
| alltoall | inplace, pairwise, bruck, linear_sync (m), two_procs, linear |
| alltoallv | inplace, pairwise, linear |
| barrier | doublering, recursivedoubling, bruck, two_procs, linear, tree |
| bcast | bintree (s), pipeline (s), chain (s, f), binomial (s), split_bintree (s), linear, knomial (s, k), scatter_allgather, scatter_allgather_ring |
| gather | binomial, linear_sync (first-seg), linear |
| reduce | chain (s, f, m), pipeline (s, m), binary (s, m), binomial (s, m), in_order_binary (s, m), linear, redscat_gather, knomial (k) |
| reduce_scatter | nonoverlapping (s, m), recursivehalving, ring, butterfly |
| reduce_scatter_block | basic_linear, recursivedoubling, recursivehalving, butterfly (→ butterfly_pof2 for 2^n ranks) |
| scatter | binomial, linear, linear_nb (m) |

Built-in fallbacks are kept as upstream wrote them. For example,
recursive-doubling allgather becomes `k_bruck(2)` on non-power-of-two sizes,
neighbor exchange becomes `ring` on odd sizes, and `scatter_allgather` becomes
`linear` when `count < size`.

**Composite algorithms.** Upstream, these call "whatever collective is
selected on the communicator". Here they are wired to fixed point-to-point
sub-algorithms, so nothing reaches the MPI library's collectives:

| algorithm | sub-steps used here |
|---|---|
| allreduce `nonoverlapping` | reduce `binomial` (`in_order_binary` if non-commutative) + bcast `binomial`, both with `-s` |
| allreduce `allgather_reduce` | allgather `recursivedoubling` + local reductions |
| allgather `linear` | gather `linear` + bcast `linear` (coll/basic behaviour) |
| allgatherv `basic_default` | linear gatherv + linear bcast of an indexed datatype |
| reduce_scatter `nonoverlapping` | reduce `binomial`/`in_order_binary` + linear scatterv |
| reduce_scatter_block `basic_linear` | reduce `linear` + scatter `linear` |

To use different sub-steps, change the corresponding line in the header.

## Verification

`--check` runs the native MPI collective once per size to build the
reference. It then re-initializes the inputs, runs each algorithm once on
poisoned output buffers, and compares the result element by element. 

Verification study is currently ongoing, please file bugs.

## License

collage is distributed under the 3-clause BSD license. The algorithm headers
are derived from Open MPI and also carry Open MPI's BSD license and copyright
notices. See [LICENSE](LICENSE) for both.
