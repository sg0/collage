# collage — point-to-point collective algorithm benchmark

`collage` benchmarks the collective algorithms from OpenMPI's `coll/base`
component (commit `ec987ee891f9d9b91f21edde9eab3f991634f57d`). Each algorithm
is re-implemented with plain MPI point-to-point calls. You pick one collective,
and `collage` times every algorithm for it (plus the MPI library's own
collective as a baseline) over a range of message sizes. By default the range
is 0 B to 4 MiB in powers of two. It can optionally check each result against
the native MPI collective.

The algorithm set is the one in the original headers. No algorithms were added.

## Layout

| file | contents |
|---|---|
| `utils.hpp` | type map, reduction ops, `CollBase` (dup'ed communicator, large-count-aware p2p wrappers, local reductions), port of `coll_base_topo.c` (binary/k-ary, binomial, in-order binomial, in-order binary, k-nomial and chain trees) |
| `allgather.hpp` … `scatter.hpp` | one class template per collective, `Coll<T>`, one member function per algorithm |
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
| `--split-mask M[:K\|:p]` | run on N/M concurrent communicators of M ranks; members consecutive, K apart, or randomized by p (see below) | off |

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
mpiexec -n 32 ./collage -c allgather -a ring --split-mask=8:0.45  # randomized, p = 0.45
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
* **`M:p`: randomized members, p ∈ [0, 1].** The group size stays M, but the
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

**Composite algorithms.** 

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
poisoned output buffers, and compares the result element by element. The
comparison covers the root for rooted collectives and every rank otherwise.
Inputs are small integers chosen per operation, so floating-point sums and
products are exact whatever the reduction order. Verification study is 
currently ongoing, please file bugs.

## License

collage is distributed under the 3-clause BSD license. The algorithm headers
are derived from OpenMPI and also carry OpenMPI's BSD license and copyright
notices. See [LICENSE](LICENSE) for both.
