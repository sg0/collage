#pragma once
#ifndef COLLAGE_REDUCE_SCATTER_HPP
#define COLLAGE_REDUCE_SCATTER_HPP

/// Algorithms adapted from Open MPI coll/base (coll_base_reduce_scatter.c @ ec987ee)
/// sbuf holds sum(rcounts) elements; rank i receives rcounts[i] reduced elements.

#include "reduce.hpp"
#include "scatter.hpp"
#include "utils.hpp"

template <typename T>
class ReduceScatter : public CollBase
{
public:
    using counts_t = std::vector<size_t>;

    explicit ReduceScatter(MPI_Comm comm = MPI_COMM_WORLD)
        : CollBase(comm), dt_(TypeMap<T>()), reduce_(comm), scatter_(comm) {}

    /// 1. non-overlapping: reduce to rank 0, then scatterv.
    ///    Upstream uses the communicator's reduce/scatterv; here the
    ///    point-to-point binomial reduce (in-order binary for non-commutative
    ///    ops) and a linear scatterv are used.
    void nonoverlapping(const T* sbuf, T* rbuf, counts_t const& rcounts, MPI_Op op,
                        size_t segsize, int max_outstanding_reqs)
    {
        const int size = size_, rank = rank_;
        const int root = 0;
        size_t total_count = 0;
        for (int i = 0; i < size; i++) total_count += rcounts[i];

        std::vector<T> tmprbuf_free;
        T* tmprbuf = rbuf;
        if (root == rank) {
            /* temporary receive buffer on root: rbuf only holds rcounts[root] */
            tmprbuf_free.resize(std::max<size_t>(total_count, 1));
            tmprbuf = tmprbuf_free.data();
        }
        if (op_is_commute(op))
            reduce_.binomial(sbuf, tmprbuf, total_count, op, root, segsize, max_outstanding_reqs);
        else
            reduce_.in_order_binary(sbuf, tmprbuf, total_count, op, root, segsize, max_outstanding_reqs);

        std::vector<size_t> displs(size);
        displs[0] = 0;
        for (int i = 1; i < size; i++) displs[i] = displs[i - 1] + rcounts[i - 1];
        scatter_.linear_v(tmprbuf, rcounts, displs, rbuf, rcounts[rank], root);
    }

    /// 2. recursive halving (commutative operations only)
    void recursivehalving(const T* sbuf, T* rbuf, counts_t const& rcounts, MPI_Op op)
    {
        int i, tmp_size, remain = 0, tmp_rank;
        const int size = size_, rank = rank_;
        size_t count;

        require_commutative(op, "recursivehalving");

        std::vector<size_t> disps(size);
        disps[0] = 0;
        for (i = 0; i < (size - 1); ++i) disps[i + 1] = disps[i] + rcounts[i];
        count = disps[size - 1] + rcounts[size - 1];

        /* short cut the trivial case */
        if (0 == count) return;

        std::vector<T> recv_buf(count), result_buf(count);
        copy_elems(result_buf.data(), sbuf, count);

        tmp_size = next_poweroftwo(size);
        tmp_size >>= 1;
        remain = size - tmp_size;

        if (rank < 2 * remain) {
            if ((rank & 1) == 0) {
                send(result_buf.data(), count, dt_, rank + 1, TAG_REDUCE_SCATTER);
                tmp_rank = -1;
            } else {
                recv(recv_buf.data(), count, dt_, rank - 1, TAG_REDUCE_SCATTER);
                reduce_local(recv_buf.data(), result_buf.data(), count, dt_, op);
                tmp_rank = rank / 2;
            }
        } else {
            tmp_rank = rank - remain;
        }

        if (tmp_rank >= 0) {
            int mask, send_index, recv_index, last_index;
            std::vector<size_t> tmp_rcounts(tmp_size), tmp_disps(tmp_size);
            for (i = 0; i < tmp_size; ++i) {
                if (i < remain) tmp_rcounts[i] = rcounts[i * 2 + 1] + rcounts[i * 2];
                else tmp_rcounts[i] = rcounts[i + remain];
            }
            tmp_disps[0] = 0;
            for (i = 0; i < tmp_size - 1; ++i) tmp_disps[i + 1] = tmp_disps[i] + tmp_rcounts[i];

            mask = tmp_size >> 1;
            send_index = recv_index = 0;
            last_index = tmp_size;
            while (mask > 0) {
                int tmp_peer, peer;
                size_t send_count, recv_count;
                MPI_Request request = MPI_REQUEST_NULL;

                tmp_peer = tmp_rank ^ mask;
                peer = (tmp_peer < remain) ? tmp_peer * 2 + 1 : tmp_peer + remain;

                send_count = recv_count = 0;
                if (tmp_rank < tmp_peer) {
                    send_index = recv_index + mask;
                    for (i = send_index; i < last_index; ++i) send_count += tmp_rcounts[i];
                    for (i = recv_index; i < send_index; ++i) recv_count += tmp_rcounts[i];
                } else {
                    recv_index = send_index + mask;
                    for (i = send_index; i < recv_index; ++i) send_count += tmp_rcounts[i];
                    for (i = recv_index; i < last_index; ++i) recv_count += tmp_rcounts[i];
                }

                if (recv_count > 0)
                    irecv(recv_buf.data() + tmp_disps[recv_index], recv_count, dt_, peer,
                          TAG_REDUCE_SCATTER, &request);
                if (send_count > 0)
                    send(result_buf.data() + tmp_disps[send_index], send_count, dt_, peer,
                         TAG_REDUCE_SCATTER);
                if (recv_count > 0) {
                    wait(&request);
                    reduce_local(recv_buf.data() + tmp_disps[recv_index],
                                 result_buf.data() + tmp_disps[recv_index], recv_count, dt_, op);
                }
                send_index = recv_index;
                last_index = recv_index + mask;
                mask >>= 1;
            }

            if (0 != rcounts[rank]) copy_elems(rbuf, result_buf.data() + disps[rank], rcounts[rank]);
        }

        /* non-power of two case: odd procs send the even procs their results */
        if (rank < (2 * remain)) {
            if ((rank & 1) == 0) {
                if (rcounts[rank]) recv(rbuf, rcounts[rank], dt_, rank + 1, TAG_REDUCE_SCATTER);
            } else {
                if (rcounts[rank - 1])
                    send(result_buf.data() + disps[rank - 1], rcounts[rank - 1], dt_, rank - 1,
                         TAG_REDUCE_SCATTER);
            }
        }
    }

    /// 3. ring (commutative operations only)
    void ring(const T* sbuf, T* rbuf, counts_t const& rcounts, MPI_Op op)
    {
        int i, k, recv_from, send_to, inbi;
        const int size = size_, rank = rank_;
        size_t total_count, max_block_count;
        MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

        std::vector<size_t> displs(size);
        displs[0] = 0;
        total_count = rcounts[0];
        max_block_count = rcounts[0];
        for (i = 1; i < size; i++) {
            displs[i] = total_count;
            total_count += rcounts[i];
            if (max_block_count < rcounts[i]) max_block_count = rcounts[i];
        }

        if (1 == size) {
            copy_elems(rbuf, sbuf, total_count);
            return;
        }
        require_commutative(op, "ring");

        std::vector<T> accumbuf(total_count), inbuf_v[2];
        T* inbuf[2] = {nullptr, nullptr};
        inbuf_v[0].resize(max_block_count);
        inbuf[0] = inbuf_v[0].data();
        if (size > 2) {
            inbuf_v[1].resize(max_block_count);
            inbuf[1] = inbuf_v[1].data();
        }
        copy_elems(accumbuf.data(), sbuf, total_count);

        send_to = (rank + 1) % size;
        recv_from = (rank + size - 1) % size;

        inbi = 0;
        irecv(inbuf[inbi], max_block_count, dt_, recv_from, TAG_REDUCE_SCATTER, &reqs[inbi]);
        send(accumbuf.data() + displs[recv_from], rcounts[recv_from], dt_, send_to, TAG_REDUCE_SCATTER);

        for (k = 2; k < size; k++) {
            const int prevblock = (rank + size - k) % size;
            inbi = inbi ^ 0x1;
            irecv(inbuf[inbi], max_block_count, dt_, recv_from, TAG_REDUCE_SCATTER, &reqs[inbi]);
            wait(&reqs[inbi ^ 0x1]);
            T* tmprecv = accumbuf.data() + displs[prevblock];
            reduce_local(inbuf[inbi ^ 0x1], tmprecv, rcounts[prevblock], dt_, op);
            send(tmprecv, rcounts[prevblock], dt_, send_to, TAG_REDUCE_SCATTER);
        }

        wait(&reqs[inbi]);
        T* tmprecv = accumbuf.data() + displs[rank];
        reduce_local(inbuf[inbi], tmprecv, rcounts[rank], dt_, op);
        copy_elems(rbuf, tmprecv, rcounts[rank]);
    }

    /// 4. butterfly (J.L. Traff, "An improved algorithm for (non-commutative)
    ///    reduce-scatter")
    void butterfly(const T* sbuf, T* rbuf, counts_t const& rcounts, MPI_Op op)
    {
        const int comm_size = size_, rank = rank_;
        if (comm_size < 2) {
            copy_elems(rbuf, sbuf, rcounts[0]);
            return;
        }

        std::vector<size_t> displs(comm_size);
        displs[0] = 0;
        for (int i = 1; i < comm_size; i++) displs[i] = displs[i - 1] + rcounts[i - 1];
        size_t totalcount = displs[comm_size - 1] + rcounts[comm_size - 1];

        std::vector<T> tmpbuf0(std::max<size_t>(totalcount, 1)), tmpbuf1(std::max<size_t>(totalcount, 1));
        T* psend = tmpbuf0.data();
        T* precv = tmpbuf1.data();
        copy_elems(psend, sbuf, totalcount);

        int nprocs_pof2 = next_poweroftwo(comm_size);
        nprocs_pof2 >>= 1;
        int nprocs_rem = comm_size - nprocs_pof2;
        int log2_size = hibit(nprocs_pof2); /* opal_cube_dim of a power of two */
        int vrank = -1;

        auto sum_counts = [&](int lo, int hi) -> size_t {
            lo = (lo < nprocs_rem) ? lo * 2 : lo + nprocs_rem;
            hi = (hi < nprocs_rem) ? hi * 2 + 1 : hi + nprocs_rem;
            return displs[hi] + rcounts[hi] - displs[lo];
        };

        if (rank < 2 * nprocs_rem) {
            if ((rank % 2) == 0) {
                send(psend, totalcount, dt_, rank + 1, TAG_REDUCE_SCATTER);
                vrank = -1;
            } else {
                recv(precv, totalcount, dt_, rank - 1, TAG_REDUCE_SCATTER);
                reduce_local(precv, psend, totalcount, dt_, op);
                vrank = rank / 2;
            }
        } else {
            vrank = rank - nprocs_rem;
        }

        if (vrank != -1) {
            int nblocks = nprocs_pof2, send_index = 0, recv_index = 0;
            for (int mask = 1; mask < nprocs_pof2; mask <<= 1) {
                int vpeer = vrank ^ mask;
                int peer = (vpeer < nprocs_rem) ? vpeer * 2 + 1 : vpeer + nprocs_rem;
                nblocks /= 2;
                if ((vrank & mask) == 0) send_index += nblocks;
                else recv_index += nblocks;

                size_t send_count = sum_counts(send_index, send_index + nblocks - 1);
                int index = (send_index < nprocs_rem) ? 2 * send_index : nprocs_rem + send_index;
                size_t sdispl = displs[index];

                size_t recv_count = sum_counts(recv_index, recv_index + nblocks - 1);
                index = (recv_index < nprocs_rem) ? 2 * recv_index : nprocs_rem + recv_index;
                size_t rdispl = displs[index];

                sendrecv(psend + sdispl, send_count, dt_, peer, TAG_REDUCE_SCATTER,
                         precv + rdispl, recv_count, dt_, peer, TAG_REDUCE_SCATTER);

                if (vrank < vpeer) {
                    /* precv = psend <op> precv */
                    reduce_local(psend + rdispl, precv + rdispl, recv_count, dt_, op);
                    std::swap(psend, precv);
                } else {
                    /* psend = precv <op> psend */
                    reduce_local(precv + rdispl, psend + rdispl, recv_count, dt_, op);
                }
                send_index = recv_index;
            }

            /* Exchange results with remote process according to a mirror permutation. */
            int vpeer = (int)mirror_perm(vrank, log2_size);
            int peer = (vpeer < nprocs_rem) ? vpeer * 2 + 1 : vpeer + nprocs_rem;
            int index = (send_index < nprocs_rem) ? 2 * send_index : nprocs_rem + send_index;

            if (vpeer < nprocs_rem) {
                /* two blocks: send the first one to the excluded process */
                send(psend + displs[index], rcounts[index], dt_, peer - 1, TAG_REDUCE_SCATTER);
            }
            if (vpeer < nprocs_rem) index++;
            if (vpeer != vrank) {
                sendrecv(psend + displs[index], rcounts[index], dt_, peer, TAG_REDUCE_SCATTER,
                         rbuf, rcounts[rank], dt_, peer, TAG_REDUCE_SCATTER);
            } else {
                copy_elems(rbuf, psend + displs[rank], rcounts[rank]);
            }
        } else {
            /* Excluded process: receive result */
            int vpeer = (int)mirror_perm((rank + 1) / 2, log2_size);
            int peer = (vpeer < nprocs_rem) ? vpeer * 2 + 1 : vpeer + nprocs_rem;
            recv(rbuf, rcounts[rank], dt_, peer, TAG_REDUCE_SCATTER);
        }
    }

private:
    MPI_Datatype dt_;
    Reduce<T> reduce_;
    Scatter<T> scatter_;

    static void require_commutative(MPI_Op op, const char* algo)
    {
        if (!op_is_commute(op))
            throw NotApplicable(std::string(algo) + " requires a commutative operation");
    }
};

#endif
