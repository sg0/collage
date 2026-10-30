# collage — point-to-point collective algorithm benchmark

`collage` benchmarks the collective algorithms from OpenMPI's `coll/base`
component (commit `ec987ee891f9d9b91f21edde9eab3f991634f57d`). Each algorithm
is re-implemented with plain MPI point-to-point calls. You pick one collective,
and `collage` times every algorithm for it (plus the MPI library's own
collective as a baseline) over a range of message sizes. By default the range
is 0 B to 4 MiB in powers of two. It can optionally check each result against
the native MPI collective.

The algorithm set is taken from the specific commit; no new algorithms or options were added. 
This work was solely possible due to the implicit generosity and dedication of the MPI implementers,
researchers, and, broadly the MPI Forum for steering HPC ahead through the past decades.    

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
mpiexec -n 128 ./collage -c allreduce -t float -o sum --check

# a non-builtin and a non-commutative reduction
mpiexec -n 128 ./collage -c reduce -o user
mpiexec -n 128 ./collage -c reduce -o user_nc

# a few bcast algorithms with a segment-size sweep (one column per value)
mpiexec -n 256 ./collage -c bcast -a binomial,pipeline,knomial,native -s 0,8K,64K -k 2,4

# irregular alltoallv counts, explicit sizes, statistics to CSV
mpiexec -n 64 ./collage -c alltoallv --vdist random --sizes 0,64,1K,64K,1M,4M --csv a2av.csv

# correctness only (fast, good for CI); exit status 2 on any mismatch
mpiexec -n 7 ./collage -c reduce_scatter -o user_nc --check-only

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

## Changes relative to the draft and to upstream

The draft headers were partly copied Open MPI source (`MCA_PML_CALL`,
`ompi_datatype_*`, `goto err_hndl`) and partly unfinished ports. `utils.hpp`
was only present as a vim swap file. Everything was rewritten against the
upstream sources at the commit above, keeping control flow and message
schedules identical. These are the changes:

* **Buffers are typed, counts are elements.** Every algorithm takes `T*`
  buffers and `size_t` counts/displacements. The draft computed offsets as
  `(T*)buf + i * count * sizeof(T)`, which scales by `sizeof(T)` twice. It also
  had separate send/recv types `U`/`T`; one element type is used now.
* **Packing and datatype copies became `memcpy`.** `opal_convertor_pack` (the
  in-place alltoall(v)), `ompi_datatype_sndrcv` and
  `ompi_datatype_copy_content_same_ddt` are equivalent to `memcpy` for
  contiguous predefined types. Indexed datatypes are still built with
  `MPI_Type_indexed(_c)` where upstream uses them (alltoall bruck, allgatherv
  bruck / neighborexchange / basic_default, reduce_scatter_block
  recursivedoubling).
* **Persistent requests are freed.** Linear alltoall(v) now calls
  `MPI_Request_free` on its persistent requests after `MPI_Waitall`. The draft
  leaked them.
* **Sparbit tags are non-negative.** Sparbit uses `TAG_BASE + block` instead of
  upstream's negative internal tag space. At very large scale,
  `1000 + size` must not exceed `MPI_TAG_UB`. MPI guarantees at least 32767,
  and real implementations allow far more.
* **Signed arithmetic in bcast scatter_allgather(_ring).** At this commit,
  upstream uses `size_t` for `recv_count`/`send_count` and then tests
  `<= 0` / `< 0`. That can underflow when blocks run past the end of the
  buffer. The port uses signed 64-bit counts, which is what the code intends.
* **Zero-size messages run the algorithm.** MPI libraries return early for
  most collectives at count 0. Here, the 0-byte row runs each algorithm with
  empty messages, which measures its latency term. Segmented algorithms use one
  empty segment, and leaf processes in segmented reduce always send at least
  one message, so the tree never deadlocks. The exceptions are algorithms
  where upstream itself returns early for zero counts (`reduce_scatter_block`
  and reduce_scatter `recursivehalving`).
* **`TypeMap<long>` fixed.** The draft mapped `long` to `MPI_LONG_INT`, which
  is a pair type. The type map now uses fixed-width `MPI_INT*_T`/`MPI_UINT*_T`.
* **Errors abort, unsupported cases throw.** Errors go through `MPI_CHECK`,
  which reports file and line and aborts. Algorithms that can't run in a given
  configuration throw `NotApplicable`, and the harness reports `n/a`.

## Verification

`--check` runs the native MPI collective once per size to build the
reference. It then re-initialises the inputs, runs each algorithm once on
poisoned output buffers, and compares the result element by element. The
comparison covers the root for rooted collectives and every rank otherwise.
Inputs are small integers chosen per operation, so floating-point sums and
products are exact whatever the reduction order. Exhaustive verification is 
ongoing.
