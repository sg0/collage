#pragma once
#ifndef COLLAGE_REDUCE_HPP
#define COLLAGE_REDUCE_HPP

/// Algorithms adapted from Open MPI coll/base (coll_base_reduce.c @ ec987ee)
/// The reduction operation `op` is any (predefined or user-defined) MPI_Op.
/// `segsize` is in bytes (0 = no segmentation); `max_outstanding_reqs` = 0
/// means no flow control on the leaves.

#include "utils.hpp"

template <typename T>
class Reduce : public CollBase
{
public:
    explicit Reduce(MPI_Comm comm = MPI_COMM_WORLD) : CollBase(comm), dt_(TypeMap<T>()) {}

    /// generic segmented tree reduce (ompi_coll_base_reduce_generic).
    /// Correct for non-commutative ops only on trees whose children hold
    /// lower ranks than their parent (the in-order binary tree).
    void generic(const T* sendbuf, T* recvbuf, size_t original_count, MPI_Op op, int root,
                 Tree const& tree, size_t count_by_segment, int max_outstanding_reqs)
    {
        T* inbuf[2] = {nullptr, nullptr};
        std::vector<T> inbuf_free[2], accumbuf_free;
        T* accumbuf = nullptr;
        const T* local_op_buffer = nullptr;
        const T* sendtmpbuf = sendbuf;
        MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        int num_segments, segindex, i, inbi;
        size_t recvcount, prevcount;
        const int rank = rank_;
        const bool commute = op_is_commute(op);

        /* single process: Open MPI never reaches this code (coll/self handles
         * such communicators); the tree would make the root send to itself */
        if (size_ == 1) {
            if (recvbuf) copy_elems(recvbuf, sendbuf, original_count);
            return;
        }

        /* zero-size reduce: run the algorithm with one empty segment */
        if (count_by_segment == 0) count_by_segment = 1;
        num_segments = (int)((original_count + count_by_segment - 1) / count_by_segment);
        if (num_segments < 1) num_segments = 1;
        const size_t segment_increment = count_by_segment;

        if (tree.tree_nextsize > 0) {
            /* non-leaf nodes - wait for children to send me data & forward up */
            accumbuf = recvbuf;
            if ((nullptr == accumbuf) || (root != rank)) {
                accumbuf_free.resize(std::max<size_t>(original_count, 1));
                accumbuf = accumbuf_free.data();
            }
            /* If this is a non-commutative operation we must copy
               sendbuf to the accumbuf, in order to simplify the loops */
            if (!commute) copy_elems(accumbuf, sendtmpbuf, original_count);

            /* Allocate two buffers for incoming segments */
            inbuf_free[0].resize(count_by_segment);
            inbuf[0] = inbuf_free[0].data();
            if ((num_segments > 1) || (tree.tree_nextsize > 1)) {
                inbuf_free[1].resize(count_by_segment);
                inbuf[1] = inbuf_free[1].data();
            }

            inbi = 0;
            recvcount = 0;
            for (segindex = 0; segindex <= num_segments; segindex++) {
                prevcount = recvcount;
                recvcount = count_by_segment;
                if (segindex == (num_segments - 1))
                    recvcount = original_count - count_by_segment * (size_t)segindex;

                for (i = 0; i < tree.tree_nextsize; i++) {
                    /* post irecv for current segindex on current child */
                    if (segindex < num_segments) {
                        T* local_recvbuf = inbuf[inbi];
                        if (0 == i) {
                            /* commutative: irecv directly into the accumulate buffer */
                            if (commute) local_recvbuf = accumbuf + (size_t)segindex * segment_increment;
                        }
                        irecv(local_recvbuf, recvcount, dt_, tree.tree_next[i], TAG_REDUCE, &reqs[inbi]);
                    }
                    /* wait on data from last child for previous segment */
                    wait(&reqs[inbi ^ 1]);
                    local_op_buffer = inbuf[inbi ^ 1];
                    if (i > 0) {
                        /* first op: combine own data with the data from down stream */
                        if (1 == i) {
                            if (commute)
                                local_op_buffer = sendtmpbuf + (size_t)segindex * segment_increment;
                        }
                        reduce_local(local_op_buffer, accumbuf + (size_t)segindex * segment_increment,
                                     recvcount, dt_, op);
                    } else if (segindex > 0) {
                        T* accumulator = accumbuf + (size_t)(segindex - 1) * segment_increment;
                        if (tree.tree_nextsize <= 1) {
                            if (commute)
                                local_op_buffer = sendtmpbuf + (size_t)(segindex - 1) * segment_increment;
                        }
                        reduce_local(local_op_buffer, accumulator, prevcount, dt_, op);

                        /* pass to the next process unless you are the root */
                        if (rank != tree.tree_root)
                            send(accumulator, prevcount, dt_, tree.tree_prev, TAG_REDUCE);

                        /* we do num_segment+1 steps for pipelining */
                        if (segindex == num_segments) break;
                    }
                    inbi = inbi ^ 1;
                }
            }
        } else {
            /* leaf nodes */
            if ((0 == max_outstanding_reqs) || (num_segments <= max_outstanding_reqs)) {
                /* blocking sends to the parent (at least one, possibly empty) */
                segindex = 0;
                do {
                    if (original_count < count_by_segment) count_by_segment = original_count;
                    send(sendbuf + (size_t)segindex * segment_increment, count_by_segment, dt_,
                         tree.tree_prev, TAG_REDUCE);
                    segindex++;
                    original_count -= count_by_segment;
                } while (original_count > 0);
            } else {
                /* flow control: max_outstanding_reqs non-blocking synchronous sends */
                int creq = 0;
                std::vector<MPI_Request> sreq(max_outstanding_reqs, MPI_REQUEST_NULL);
                for (segindex = 0; segindex < max_outstanding_reqs; segindex++) {
                    issend(sendbuf + (size_t)segindex * segment_increment, count_by_segment,
                           tree.tree_prev, &sreq[segindex]);
                    original_count -= count_by_segment;
                }
                creq = 0;
                while (original_count > 0) {
                    wait(&sreq[creq]);
                    if (original_count < count_by_segment) count_by_segment = original_count;
                    issend(sendbuf + (size_t)segindex * segment_increment, count_by_segment,
                           tree.tree_prev, &sreq[creq]);
                    creq = (creq + 1) % max_outstanding_reqs;
                    segindex++;
                    original_count -= count_by_segment;
                }
                waitall(max_outstanding_reqs, sreq.data());
            }
        }
    }

    /// 1. chain (fanout chains, segmented) -- commutative ops only
    void chain(const T* sbuf, T* rbuf, size_t count, MPI_Op op, int root, size_t segsize,
               int fanout, int max_outstanding_reqs)
    {
        require_commutative(op, "chain");
        Tree tree = topo::build_chain(fanout, rank_, size_, root);
        size_t segcount = count;
        computed_segcount(segsize, sizeof(T), segcount);
        generic(sbuf, rbuf, count, op, root, tree, segcount, max_outstanding_reqs);
    }

    /// 2. pipeline (segmented) -- commutative ops only
    void pipeline(const T* sbuf, T* rbuf, size_t count, MPI_Op op, int root, size_t segsize,
                  int max_outstanding_reqs)
    {
        require_commutative(op, "pipeline");
        Tree tree = topo::build_chain(1, rank_, size_, root);
        size_t segcount = count;
        computed_segcount(segsize, sizeof(T), segcount);
        generic(sbuf, rbuf, count, op, root, tree, segcount, max_outstanding_reqs);
    }

    /// 3. binary tree (segmented) -- commutative ops only
    void binary(const T* sbuf, T* rbuf, size_t count, MPI_Op op, int root, size_t segsize,
                int max_outstanding_reqs)
    {
        require_commutative(op, "binary");
        Tree tree = topo::build_tree(2, rank_, size_, root);
        size_t segcount = count;
        computed_segcount(segsize, sizeof(T), segcount);
        generic(sbuf, rbuf, count, op, root, tree, segcount, max_outstanding_reqs);
    }

    /// 4. (in-order) binomial tree (segmented) -- commutative ops only
    void binomial(const T* sbuf, T* rbuf, size_t count, MPI_Op op, int root, size_t segsize,
                  int max_outstanding_reqs)
    {
        require_commutative(op, "binomial");
        Tree tree = topo::build_in_order_bmtree(rank_, size_, root);
        size_t segcount = count;
        computed_segcount(segsize, sizeof(T), segcount);
        generic(sbuf, rbuf, count, op, root, tree, segcount, max_outstanding_reqs);
    }

    /// 5. in-order binary tree (also valid for non-commutative ops).
    ///    The tree is rooted at size-1; the result is forwarded to `root`.
    void in_order_binary(const T* sendbuf, T* recvbuf, size_t count, MPI_Op op, int root,
                         size_t segsize, int max_outstanding_reqs)
    {
        const int rank = rank_, size = size_;
        Tree tree = topo::build_in_order_bintree(rank, size);
        size_t segcount = count;
        computed_segcount(segsize, sizeof(T), segcount);

        int io_root = size - 1;
        const T* use_this_sendbuf = sendbuf;
        T* use_this_recvbuf = recvbuf;
        std::vector<T> tmpbuf;
        if (io_root != root && io_root == rank) {
            tmpbuf.resize(std::max<size_t>(count, 1));
            use_this_recvbuf = tmpbuf.data();
        }

        generic(use_this_sendbuf, use_this_recvbuf, count, op, io_root, tree, segcount,
                max_outstanding_reqs);

        if (io_root != root) {
            if (root == rank) recv(recvbuf, count, dt_, io_root, TAG_REDUCE);
            else if (io_root == rank) send(use_this_recvbuf, count, dt_, root, TAG_REDUCE);
        }
    }

    /// 6. basic linear: root receives from everybody in rank order (size-1 .. 0)
    void linear(const T* sbuf, T* rbuf, size_t count, MPI_Op op, int root)
    {
        int i;
        const int rank = rank_, size = size_;

        if (rank != root) {
            send(sbuf, count, dt_, root, TAG_REDUCE);
            return;
        }
        std::vector<T> pml_buffer;
        if (size > 1) pml_buffer.resize(count);

        /* Initialize the receive buffer. */
        if (rank == (size - 1)) copy_elems(rbuf, sbuf, count);
        else recv(rbuf, count, dt_, size - 1, TAG_REDUCE);

        /* Loop receiving and calling reduction function */
        for (i = size - 2; i >= 0; --i) {
            const T* inbuf;
            if (rank == i) {
                inbuf = sbuf;
            } else {
                recv(pml_buffer.data(), count, dt_, i, TAG_REDUCE);
                inbuf = pml_buffer.data();
            }
            reduce_local(inbuf, rbuf, count, dt_, op);
        }
    }

    /// 7. Rabenseifner: reduce-scatter (recursive halving) + binomial gather.
    ///    Falls back to linear for count < 2^floor(log2 p) or non-commutative ops.
    void redscat_gather(const T* sbuf, T* rbuf, size_t count, MPI_Op op, int root)
    {
        const int comm_size = size_, rank = rank_;
        int nsteps = hibit(comm_size); /* ilog2(comm_size) */
        if (-1 == nsteps) throw std::runtime_error("redscat_gather: bad communicator size");
        int nprocs_pof2 = 1 << nsteps;
        if (nprocs_pof2 < 2 || count < (size_t)nprocs_pof2 || !op_is_commute(op)) {
            linear(sbuf, rbuf, count, op, root);
            return;
        }

        std::vector<T> tmp_buf(count), rbuf_raw;
        if (rank != root) {
            rbuf_raw.resize(count);
            rbuf = rbuf_raw.data();
        }
        copy_elems(rbuf, sbuf, count);

        /* Step 1. Reduce the number of processes to the nearest lower power of two */
        int vrank, step;
        size_t wsize;
        int nprocs_rem = comm_size - nprocs_pof2;
        if (rank < 2 * nprocs_rem) {
            size_t count_lhalf = count / 2;
            size_t count_rhalf = count - count_lhalf;
            if (rank % 2 != 0) {
                /* Odd process -- exchange with rank - 1 */
                sendrecv(rbuf, count_lhalf, dt_, rank - 1, TAG_REDUCE,
                         tmp_buf.data() + count_lhalf, count_rhalf, dt_, rank - 1, TAG_REDUCE);
                reduce_local(tmp_buf.data() + count_lhalf, rbuf + count_lhalf, count_rhalf, dt_, op);
                send(rbuf + count_lhalf, count_rhalf, dt_, rank - 1, TAG_REDUCE);
                vrank = -1;
            } else {
                /* Even process -- exchange with rank + 1 */
                sendrecv(rbuf + count_lhalf, count_rhalf, dt_, rank + 1, TAG_REDUCE,
                         tmp_buf.data(), count_lhalf, dt_, rank + 1, TAG_REDUCE);
                reduce_local(tmp_buf.data(), rbuf, count_lhalf, dt_, op);
                recv(rbuf + count_lhalf, count_rhalf, dt_, rank + 1, TAG_REDUCE);
                vrank = rank / 2;
            }
        } else {
            vrank = rank - nprocs_rem;
        }

        /* Step 2. Reduce-scatter with recursive vector halving and distance doubling */
        std::vector<size_t> rindex(nsteps), sindex(nsteps), rcount(nsteps), scount(nsteps);
        if (vrank != -1) {
            step = 0;
            wsize = count;
            sindex[0] = rindex[0] = 0;
            for (int mask = 1; mask < nprocs_pof2; mask <<= 1) {
                int vdest = vrank ^ mask;
                int dest = (vdest < nprocs_rem) ? vdest * 2 : vdest + nprocs_rem;
                if (rank < dest) {
                    rcount[step] = wsize / 2;
                    scount[step] = wsize - rcount[step];
                    sindex[step] = rindex[step] + rcount[step];
                } else {
                    scount[step] = wsize / 2;
                    rcount[step] = wsize - scount[step];
                    rindex[step] = sindex[step] + scount[step];
                }
                sendrecv(rbuf + sindex[step], scount[step], dt_, dest, TAG_REDUCE,
                         tmp_buf.data() + rindex[step], rcount[step], dt_, dest, TAG_REDUCE);
                /* Local reduce: rbuf[] = tmp_buf[] <op> rbuf[] */
                reduce_local(tmp_buf.data() + rindex[step], rbuf + rindex[step], rcount[step], dt_, op);
                if (step + 1 < nsteps) {
                    rindex[step + 1] = rindex[step];
                    sindex[step + 1] = rindex[step];
                    wsize = rcount[step];
                    step++;
                }
            }
        }

        /* Setup the root process for gather operation. */
        int vroot = 0;
        if (root < 2 * nprocs_rem) {
            if (root % 2 != 0) {
                vroot = 0;
                if (rank == root) {
                    /* root was excluded on step 1: receive data from process 0 */
                    rindex[0] = 0;
                    step = 0, wsize = count;
                    for (int mask = 1; mask < nprocs_pof2; mask *= 2) {
                        rcount[step] = wsize / 2;
                        scount[step] = wsize - rcount[step];
                        rindex[step] = 0;
                        sindex[step] = rcount[step];
                        step++;
                        wsize /= 2;
                    }
                    recv(rbuf, rcount[nsteps - 1], dt_, 0, TAG_REDUCE);
                    vrank = 0;
                } else if (vrank == 0) {
                    send(rbuf, rcount[nsteps - 1], dt_, root, TAG_REDUCE);
                    vrank = -1;
                }
            } else {
                vroot = root / 2;
            }
        } else {
            vroot = root - nprocs_rem;
        }

        /* Step 3. Gather result at the vroot by the binomial tree algorithm. */
        if (vrank != -1) {
            int vdest_tree, vroot_tree;
            step = nsteps - 1;
            for (int mask = nprocs_pof2 >> 1; mask > 0; mask >>= 1) {
                int vdest = vrank ^ mask;
                int dest = (vdest < nprocs_rem) ? vdest * 2 : vdest + nprocs_rem;
                if ((vdest == 0) && (root < 2 * nprocs_rem) && (root % 2 != 0)) dest = root;

                vdest_tree = vdest >> step;
                vdest_tree <<= step;
                vroot_tree = vroot >> step;
                vroot_tree <<= step;
                if (vdest_tree == vroot_tree) {
                    send(rbuf + rindex[step], rcount[step], dt_, dest, TAG_REDUCE);
                    break;
                } else {
                    recv(rbuf + sindex[step], scount[step], dt_, dest, TAG_REDUCE);
                }
                step--;
            }
        }
    }

    /// 8. k-nomial tree (single transfer per child) -- commutative ops only
    void knomial(const T* sendbuf, T* recvbuf, size_t count, MPI_Op op, int root,
                 size_t /*segsize*/, int /*max_outstanding_reqs*/, int radix)
    {
        require_commutative(op, "knomial");
        const int rank = rank_;
        Tree tree = topo::build_kmtree(rank, size_, root, radix);
        int num_children = tree.tree_nextsize;
        bool is_leaf = (tree.tree_nextsize == 0);

        std::vector<T> reduce_buf(count), child_buf;
        copy_elems(reduce_buf.data(), sendbuf, count);

        int num_reqs = 0;
        std::vector<MPI_Request> reqs;
        if (!is_leaf) {
            child_buf.resize(count * (size_t)num_children);
            reqs.assign(num_children, MPI_REQUEST_NULL);
        }
        for (int i = 0; i < num_children; i++) {
            int child = tree.tree_next[i];
            irecv(child_buf.data() + (size_t)i * count, count, dt_, child, TAG_REDUCE,
                  &reqs[num_reqs++]);
        }
        if (num_reqs > 0) waitall(num_reqs, reqs.data());

        for (int i = 0; i < num_children; i++)
            reduce_local(child_buf.data() + (size_t)i * count, reduce_buf.data(), count, dt_, op);

        if (rank != root) send(reduce_buf.data(), count, dt_, tree.tree_prev, TAG_REDUCE);
        if (rank == root) copy_elems(recvbuf, reduce_buf.data(), count);
    }

private:
    MPI_Datatype dt_;

    void issend(const T* buf, size_t count, int dest, MPI_Request* req) const
    {
#if COLLAGE_LARGE_COUNT
        MPI_CHECK(MPI_Issend_c(buf, cnt(count), dt_, dest, TAG_REDUCE, comm_, req));
#else
        MPI_CHECK(MPI_Issend(buf, cnt(count), dt_, dest, TAG_REDUCE, comm_, req));
#endif
    }
    static void require_commutative(MPI_Op op, const char* algo)
    {
        if (!op_is_commute(op))
            throw NotApplicable(std::string(algo) + " requires a commutative operation");
    }
};

#endif
