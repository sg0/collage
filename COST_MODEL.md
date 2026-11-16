# Cost model of the MPI algorithms

The two tables give the **latency** (α) and **bandwidth** (β, plus γ for
reductions) terms of every algorithm in collage, as implemented in the headers.
They include the tunables collage can sweep (`-s`, `--fanout`, `-k`,
`--max-reqs`, `--first-seg`). They are leading-order Hockney estimates for
choosing and explaining algorithms. They do not predict timings exactly.

## Model and notation

A message of b bytes costs **α + bβ**. Reducing b bytes locally costs **bγ**.
Links are full duplex: a rank can send one message and receive one at the
same time (a `sendrecv` with one partner costs α + bβ). The time of an
algorithm is the cost along its critical path, usually at the busiest rank.

The latency table has two α columns because nonblocking messages can be counted
in two ways:

* **α, one-port.** A rank's messages are serialised, one start-up each. This is
  the classic Hockney count, and matches eager messages on one NIC.
* **rounds, overlapped.** Messages posted together and completed by one
  `waitall` share a single α, because their start-ups overlap. This is where
  `-k`, `--max-reqs` and the nonblocking algorithms differ from the blocking
  ones. The truth usually lies between the two columns. On Perlmutter's
  4-NIC GPU nodes it is closer to the right-hand one.

The bandwidth column counts the bytes on the busiest rank's port, which are
serialised under both views.

| symbol | meaning |
|---|---|
| p | ranks in the communicator (or in one group with `--split-mask`) |
| n | message size as collage reports it: bytes per rank for allgather/gather, per peer for alltoall, per rank's result for scatter/reduce_scatter_block, the whole vector for bcast/reduce/allreduce (see *What "size" means* in the README) |
| n̄, N | v-variants: mean block, and total N = p·n̄. reduce_scatter_block: N = p·n |
| λ, λ′, p′ | λ = ⌈log₂p⌉; λ′ = ⌊log₂p⌋; p′ = 2^λ′ (the largest power of two ≤ p) |
| λ_k | ⌈log_k p⌉ |
| [p≠p′] | 1 when p is not a power of two, else 0 |
| **s** (`-s`) | segment size in bytes; 0 means no segmentation |
| S | number of segments, ⌈n/s⌉ (S = 1 when s = 0) |
| **f** (`--fanout`) | number of chains hanging off the root (default 4) |
| **k** (`-k`) | radix of k-nomial trees and k-Bruck (default 4) |
| **m** (`--max-reqs`) | cap on outstanding requests (0 = unlimited) |
| **F** (`--first-seg`) | gather `linear_sync` first-segment size (default 1 KiB) |

### Segmented trees (the `generic` bcast and reduce)

The tree algorithms send S segments down (bcast) or up (reduce) a tree. The
cost depends on two numbers of the tree:

* **L**: the critical-path cost of one unsegmented message, in messages.
* **B**: the largest number of children of any rank (the pipeline bottleneck).

Then:

```
T ≈ (L + B·(S−1)) · (α + n/S · β  [+ n/S · γ for reduce])
rounds (overlapped) ≈ h + S − 1        (h = tree depth in levels)
```

| tree (algorithms) | L | B | h |
|---|---|---|---|
| pipeline, i.e. chain with f = 1 (`pipeline`) | p−1 | 1 | p−1 |
| f chains (`chain`) | f + ⌈(p−1)/f⌉ − 1 | f | ⌈(p−1)/f⌉ |
| binary (`bintree`, `binary`, `in_order_binary`) | 2λ′ | 2 | λ′ |
| binomial (`binomial`) | λ | λ | λ |
| k-nomial (`knomial`) | (k−1)·λ_k | (k−1)·λ_k | λ_k |

For large S the β term tends to **B·n**. The best segment count balances the
added α against the shortened pipeline fill:

```
S* = sqrt( (L − B) · nβ / (B·α) ),   i.e.   s* = n / S* = sqrt( B·α·n / ((L − B)·β) )
```

So segmentation pays off most for the pipeline and chains, where L ≫ B. It
gives less for binary trees. For binomial and k-nomial trees, L = B, so the
one-port model says segmentation never helps there. In practice, any gain
comes from overlap that this model ignores.

## Table 1: latency (α term)

| collective | algorithm | params | α coefficient, one-port | rounds, overlapped | condition / fallback |
|---|---|---|---|---|---|
| allgather | recursivedoubling | — | λ | λ | p not a power of two → `k_bruck` with k = 2 |
| | sparbit | — | p−1 | λ | one message per block, λ steps |
| | ring | — | p−1 | p−1 | |
| | neighborexchange | — | p/2 | p/2 | odd p → `ring` |
| | linear | — | 2(p−1) | p | linear gather to 0, then linear bcast |
| | k_bruck | k | (k−1)·λ_k | λ_k | |
| | direct_messaging | — | p−1 | 1 | |
| allgatherv | bruck | — | λ | λ | |
| | sparbit | — | p−1 | λ | |
| | ring | — | p−1 | p−1 | |
| | neighborexchange | — | p/2 | p/2 | odd p → `ring` |
| | two_procs | — | 1 | 1 | p = 2 only |
| | basic_default | — | 2(p−1) | 2 | linear gatherv, then linear bcast |
| allreduce | nonoverlapping | s, m | 2λ·S | 2(λ + S − 1) | binomial reduce + binomial bcast; non-commutative: in_order_binary reduce, (2λ′ + 2(S−1) + 1) + λ·S |
| | recursivedoubling | — | λ′ + 2[p≠p′] | λ′ + 2[p≠p′] | |
| | ring | — | 2(p−1) | 2(p−1) | count < p elements → `recursivedoubling`; commutative only |
| | ring_segmented | s | (P+1)(p−1), with P ≈ ⌊n/(p·s)⌋ phases | (P+1)(p−1) | s = 0 or n < p·s → `ring` |
| | linear | — | 2(p−1) | p | linear reduce, then linear bcast |
| | redscat_allgather | — | 2λ′ + 3[p≠p′] | 2λ′ + 3[p≠p′] | non-commutative → `linear` |
| | allgather_reduce | — | λ | λ | non-power-of-two allgather is k_bruck(2); commutative only |
| alltoall | inplace | — | ≈ p (2 messages per pair step) | ≈ p | |
| | pairwise | — | p−1 | p−1 | |
| | bruck | — | λ | λ | |
| | linear_sync | m | p−1 | ⌈(p−1)/m⌉ (1 if m = 0 or m ≥ p−1) | |
| | two_procs | — | 1 | 1 | p = 2 only |
| | linear | — | p−1 | 1 | |
| alltoallv | inplace | — | ≈ p | ≈ p | |
| | pairwise | — | p−1 | p−1 | |
| | linear | — | p−1 | 1 | |
| barrier | doublering | — | 2p | 2p | |
| | recursivedoubling | — | λ′ + 2[p≠p′] | λ′ + 2[p≠p′] | |
| | bruck | — | λ | λ | |
| | two_procs | — | 1 | 1 | p = 2 only |
| | linear | — | 2(p−1) | 2 | |
| | tree | — | 2λ | 2λ | |
| bcast | bintree | s | 2λ′ + 2(S−1) | λ′ + S − 1 | |
| | pipeline | s | p − 2 + S | p − 2 + S | |
| | chain | s, f | f + ⌈(p−1)/f⌉ − 1 + f(S−1) | ⌈(p−1)/f⌉ + S − 1 | |
| | binomial | s | λ·S | λ + S − 1 | |
| | split_bintree | s | 2λ′ + 2(S₂−1) + 1, with S₂ = ⌈n/(2s)⌉ | λ′ + S₂ | n < 2s → `chain` with f = 1 |
| | linear | — | p−1 | 1 | |
| | knomial | s, k | (k−1)·λ_k·S | λ_k + S − 1 | |
| | scatter_allgather | — | 2λ (+ ≤ λ′ for non-power-of-two p) | 2λ | count < p elements → `linear` |
| | scatter_allgather_ring | — | λ + p − 1 | λ + p − 1 | count < p elements → `linear` |
| gather | binomial | — | λ | λ | |
| | linear_sync | F | 2(p−1) + 1 | 2(p−1) + 1 | the root handshakes with each rank in turn; the second segment (n > F) overlaps |
| | linear | — | p−1 | p−1 | |
| reduce | chain | s, f, m | f + ⌈(p−1)/f⌉ − 1 + f(S−1) | ⌈(p−1)/f⌉ + S − 1 | commutative only |
| | pipeline | s, m | p − 2 + S | p − 2 + S | commutative only |
| | binary | s, m | 2λ′ + 2(S−1) | λ′ + S − 1 | commutative only |
| | binomial | s, m | λ·S | λ + S − 1 | commutative only |
| | in_order_binary | s, m | 2λ′ + 2(S−1) + [root ≠ p−1] | λ′ + S − 1 + [root ≠ p−1] | any op |
| | linear | — | p−1 | p−1 | any op |
| | redscat_gather | — | 2λ′ + 2[p≠p′] (+1 if the root is an excluded odd rank) | same | count < p′ elements or non-commutative → `linear` |
| | knomial | k | (k−1)·λ_k | λ_k | ignores s and m; commutative only |
| reduce_scatter | nonoverlapping | s, m | λ·S + (p−1) | λ + S | binomial reduce of N + linear scatterv; non-commutative: in_order_binary reduce |
| | recursivehalving | — | λ′ + 2[p≠p′] | λ′ + 2[p≠p′] | commutative only |
| | ring | — | p−1 | p−1 | commutative only |
| | butterfly | — | λ′ + 2[p≠p′] | λ′ + 2[p≠p′] | |
| reduce_scatter_block | basic_linear | — | 2(p−1) | p | linear reduce of N, then linear scatter |
| | recursivedoubling | — | λ (+ ≤ λ′ for non-power-of-two p) | λ | any op |
| | recursivehalving | — | λ′ + 2[p≠p′] | λ′ + 2[p≠p′] | non-commutative → `basic_linear` |
| | butterfly | — | λ′ + 2[p≠p′] | λ′ + 2[p≠p′] | p a power of two → `butterfly_pof2`: λ |
| scatter | binomial | — | λ | λ | |
| | linear | — | p−1 | p−1 | |
| | linear_nb | m | p−1 | ⌈(p−1)/m⌉ (1 if m ≤ 1) | every m-th send is blocking |

## Table 2: bandwidth (β term) and reduction (γ term)

These are bytes through the busiest rank's port, and bytes reduced on the
critical path. For the segmented trees, the closed forms below are for
S = 1; with segments, use the generic formula (β ≈ B·n as S grows).

| collective | algorithm | params | β coefficient (bytes) | γ coefficient (bytes) | notes |
|---|---|---|---|---|---|
| allgather | recursivedoubling | — | (p−1)·n | — | optimal |
| | sparbit | — | (p−1)·n | — | |
| | ring | — | (p−1)·n | — | |
| | neighborexchange | — | (p−1)·n | — | n, then 2n per step |
| | linear | — | (p²−1)·n | — | the root sends p·n to every rank |
| | k_bruck | k | (p−1)·n | — | plus a local rotation of up to p·n per rank |
| | direct_messaging | — | (p−1)·n | — | |
| allgatherv | bruck | — | N − n_rank ≈ (p−1)·n̄ | — | indexed datatypes, no packing copies |
| | sparbit | — | (p−1)·n̄ | — | |
| | ring | — | (p−1)·n̄ | — | skewed blocks: Σ of the largest block per step |
| | neighborexchange | — | (p−1)·n̄ | — | |
| | two_procs | — | n̄ | — | |
| | basic_default | — | (p−1)·n̄ + (p−1)·N = (p²−1)·n̄ | — | |
| allreduce | nonoverlapping | s, m | 2λ·n | λ·n | |
| | recursivedoubling | — | (λ′ + 2[p≠p′])·n | (λ′ + [p≠p′])·n | |
| | ring | — | 2(p−1)/p · n | (p−1)/p · n | bandwidth-optimal |
| | ring_segmented | s | 2(p−1)/p · n | (p−1)/p · n | segmenting adds only α |
| | linear | — | 2(p−1)·n | (p−1)·n | |
| | redscat_allgather | — | 2(p′−1)/p′ · n + 2n·[p≠p′] | (p′−1)/p′ · n + n/2·[p≠p′] | Rabenseifner |
| | allgather_reduce | — | (p−1)·n | (p−1)·n | trades bandwidth for latency |
| alltoall | inplace | — | (p−1)·n | — | plus ~p/2 local copies of n |
| | pairwise | — | (p−1)·n | — | |
| | bruck | — | ≈ (p/2)·λ·n | — | plus rotations (≈ 2p·n) and a per-step copy-back |
| | linear_sync | m | (p−1)·n | — | |
| | two_procs | — | n | — | |
| | linear | — | (p−1)·n | — | |
| alltoallv | inplace / pairwise / linear | — | (p−1)·n̄ | — | skewed counts: the busiest rank's total |
| barrier | all | — | 0 | — | zero-byte messages |
| bcast | bintree | s | 2λ′·n | — | |
| | pipeline | s | (p−1)·n (→ n as S grows) | — | |
| | chain | s, f | (f + ⌈(p−1)/f⌉ − 1)·n (→ f·n as S grows) | — | |
| | binomial | s | λ·n | — | segmentation does not reduce it |
| | split_bintree | s | λ′·n + n/2 | — | each half goes down one subtree, then pairs swap halves |
| | linear | — | (p−1)·n | — | |
| | knomial | s, k | (k−1)·λ_k·n | — | |
| | scatter_allgather | — | 2(p−1)/p · n | — | |
| | scatter_allgather_ring | — | 2(p−1)/p · n | — | |
| gather | binomial | — | (p−1)·n | — | |
| | linear_sync | F | (p−1)·n | — | the first min(n, F) bytes of each rank sit on the handshake path |
| | linear | — | (p−1)·n | — | |
| reduce | chain | s, f, m | (f + ⌈(p−1)/f⌉ − 1)·n (→ f·n) | same as β | |
| | pipeline | s, m | (p−1)·n (→ n) | same as β | |
| | binary | s, m | 2λ′·n (→ 2n) | same as β | |
| | in_order_binary | s, m | 2λ′·n (→ 2n), + n if root ≠ p−1 | 2λ′·n (→ 2n) | |
| | binomial | s, m | λ·n | λ·n | |
| | linear | — | (p−1)·n | (p−1)·n | |
| | redscat_gather | — | 2(p′−1)/p′ · n + n·[p≠p′] | (p′−1)/p′ · n + n/2·[p≠p′] | Rabenseifner |
| | knomial | k | (k−1)·λ_k·n | (k−1)·λ_k·n | |
| reduce_scatter | nonoverlapping | s, m | λ·N + (p−1)·n̄ | λ·N | |
| | recursivehalving | — | (p′−1)/p′ · N + (N + n̄)·[p≠p′] | (p′−1)/p′ · N + N·[p≠p′] | |
| | ring | — | (p−1)·n̄ | (p−1)·n̄ | |
| | butterfly | — | (p′−1)/p′ · N + (N + n̄)·[p≠p′] | (p′−1)/p′ · N + N·[p≠p′] | |
| reduce_scatter_block | basic_linear | — | (p−1)·N + (p−1)·n | (p−1)·N | |
| | recursivedoubling | — | (λ − (p−1)/p)·N | (λ − (p−1)/p)·N | |
| | recursivehalving | — | (p′−1)/p′ · N + (N + n)·[p≠p′] | (p′−1)/p′ · N + N·[p≠p′] | |
| | butterfly | — | (p′−1)/p′ · N + (N + n)·[p≠p′] | (p′−1)/p′ · N + N·[p≠p′] | power of two: (p−1)·n |
| scatter | binomial | — | (p−1)·n | — | |
| | linear | — | (p−1)·n | — | |
| | linear_nb | m | (p−1)·n | — | |

## How the tunables enter

* **s (segment size).** Pipelined trees take the generic formula: more segments
  add α·B per segment and shrink the β term towards B·n; the optimum is s*
  above. For `ring_segmented`, s sets the number of phases P ≈ n/(p·s). Each
  phase adds (p−1)α, and the β term stays the same. For `split_bintree`, s
  splits each half of the message.
* **f (fan-out of `chain`).** It trades depth ⌈(p−1)/f⌉ against the root's
  per-segment load f. With few segments a large f is better (shorter chains);
  with many segments a small f is better (β → f·n).
* **k (radix).** For `k_bruck`, larger k cuts rounds to λ_k but keeps
  (k−1)·λ_k one-port messages. It pays off when concurrent messages overlap.
  For `knomial` bcast/reduce, the root's load (k−1)·λ_k grows with k, and the
  depth λ_k shrinks.
* **m (`--max-reqs`).** For `alltoall linear_sync` and `scatter linear_nb`, it
  sets the rounds ⌈(p−1)/m⌉ directly. For the reduce trees, m only limits a
  leaf's outstanding synchronous sends when S > m. It does not change the ideal
  α–β cost; it limits unexpected-message buffering at the parent and can stall
  the pipeline when m is very small.
* **F (`--first-seg`).** It adds min(n, F)·β to each of the p−1 serialised
  handshakes in `gather linear_sync`.

## Fitting α, β and γ from collage runs

* **α and β.** Run `collage -c alltoall -a two_procs` on 2 ranks (on the same
  node or on different nodes). Its latency is α + nβ, so a straight line
  through the 8 B … 4 MB points gives α (intercept) and β (slope). Do this for
  both host and GPU buffers.
* **γ.** Run `collage -c allreduce -a recursivedoubling` on the same 2 ranks.
  It costs α + n(β + γ), so subtracting the line above gives γ. Fit γ per op,
  type and `--gpu-reduce` mode.
* **Prediction.** Insert the fitted values into the tables. Measured times well
  above the one-port prediction usually point to contention (try
  `--split-mask` layouts) or rendezvous protocol switches. Times below it show
  overlap, the right-hand α column.
