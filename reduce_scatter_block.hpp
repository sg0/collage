#pragma once
#ifndef COLLAGE_REDUCE_SCATTER_BLOCK_HPP
#define COLLAGE_REDUCE_SCATTER_BLOCK_HPP

/// Algorithms adapted from Open MPI coll/base (coll_base_reduce_scatter_block.c @ ec987ee)
/// sbuf holds size*rcount elements; every rank receives rcount reduced elements.

#include "reduce.hpp"
#include "scatter.hpp"
#include "utils.hpp"

template <typename T>
class ReduceScatterBlock : public CollBase
{
public:
    explicit ReduceScatterBlock(MPI_Comm comm = MPI_COMM_WORLD)
        : CollBase(comm), dt_(TypeMap<T>()), reduce_(comm), scatter_(comm) {}

    /// 1. basic linear: reduce to 0, then scatter. Upstream uses the
    ///    communicator's reduce/scatter; here the linear point-to-point ones
    ///    (as in Open MPI's coll/basic component).
    void basic_linear(const T* sbuf, T* rbuf, size_t rcount, MPI_Op op)
    {
        const int rank = rank_, size = size_;
        size_t count = rcount * (size_t)size;
        if (0 == count) return;

        std::vector<T> recv_buf;
        if (count > (size_t)INT_MAX) {
            /* large payload: reduce per block, then send to each rank */
            if (0 == rank) recv_buf.resize(rcount);
            for (int i = 0; i < size; ++i) {
                const T* sbuf_ptr = sbuf + rcount * (size_t)i;
                reduce_.linear(sbuf_ptr, recv_buf.data(), rcount, op, 0);
                if (0 == rank) {
                    if (i == rank) copy_elems(rbuf, recv_buf.data(), rcount);
                    else send(recv_buf.data(), rcount, dt_, i, TAG_REDUCE_SCATTER_BLOCK);
                } else if (i == rank) {
                    recv(rbuf, rcount, dt_, 0, TAG_REDUCE_SCATTER_BLOCK);
                }
            }
        } else {
            if (0 == rank) recv_buf.resize(count);
            reduce_.linear(sbuf, recv_buf.data(), count, op, 0);
            scatter_.linear(recv_buf.data(), rcount, rbuf, rcount, 0);
        }
    }

    /// 2. recursive doubling (supports non-commutative ops)
    void recursivedoubling(const T* sbuf, T* rbuf, size_t rcount, MPI_Op op)
    {
        const int comm_size = size_, rank = rank_;
        if (rcount == 0) return;
        if (comm_size < 2) {
            copy_elems(rbuf, sbuf, rcount);
            return;
        }
        size_t totalcount = (size_t)comm_size * rcount;
#if !COLLAGE_LARGE_COUNT
        if (totalcount > (size_t)INT_MAX) {
            basic_linear(sbuf, rbuf, rcount, op);
            return;
        }
#endif
        std::vector<T> tmpbuf_v(totalcount), tmprecv_v(totalcount);
        T* tmpbuf = tmpbuf_v.data();
        T* tmprecv = tmprecv_v.data();
        copy_elems(tmpbuf, sbuf, totalcount);

        const bool is_commutative = op_is_commute(op);
        size_t blocklens[2], displs[2];

        for (int mask = 1; mask < comm_size; mask <<= 1) {
            int remote = rank ^ mask;
            int cur_tree_root = rounddown(rank, mask);
            int remote_tree_root = rounddown(remote, mask);

            /* Send type: [0..cur_tree_root - 1], [cur_tree_root + mask, p - 1] */
            blocklens[0] = rcount * (size_t)cur_tree_root;
            blocklens[1] = (comm_size >= cur_tree_root + mask)
                               ? rcount * (size_t)(comm_size - cur_tree_root - mask)
                               : 0;
            displs[0] = 0;
            displs[1] = (size_t)comm_size * rcount - blocklens[1];
            MPI_Datatype dtypesend = make_indexed2(blocklens, displs);

            /* Recv type: [0..remote_tree_root - 1], [remote_tree_root + mask, p - 1] */
            blocklens[0] = rcount * (size_t)remote_tree_root;
            blocklens[1] = (comm_size >= remote_tree_root + mask)
                               ? rcount * (size_t)(comm_size - remote_tree_root - mask)
                               : 0;
            displs[0] = 0;
            displs[1] = (size_t)comm_size * rcount - blocklens[1];
            MPI_Datatype dtyperecv = make_indexed2(blocklens, displs);

            int is_block_received = 0;
            if (remote < comm_size) {
                sendrecv(tmpbuf, 1, dtypesend, remote, TAG_REDUCE_SCATTER_BLOCK,
                         tmprecv, 1, dtyperecv, remote, TAG_REDUCE_SCATTER_BLOCK);
                is_block_received = 1;
            }

            /* Non-power-of-two case */
            if (remote_tree_root + mask > comm_size) {
                int nprocs_alldata = comm_size - cur_tree_root - mask;
                for (int rhalving_mask = mask >> 1; rhalving_mask > 0; rhalving_mask >>= 1) {
                    remote = rank ^ rhalving_mask;
                    int tree_root = rounddown(rank, rhalving_mask << 1);
                    if ((remote > rank) && (rank < tree_root + nprocs_alldata) &&
                        (remote >= tree_root + nprocs_alldata)) {
                        send(tmprecv, 1, dtyperecv, remote, TAG_REDUCE_SCATTER_BLOCK);
                    } else if ((remote < rank) && (remote < tree_root + nprocs_alldata) &&
                               (rank >= tree_root + nprocs_alldata)) {
                        recv(tmprecv, 1, dtyperecv, remote, TAG_REDUCE_SCATTER_BLOCK);
                        is_block_received = 1;
                    }
                }
            }

            if (is_block_received) {
                /* After reduction the result must be in tmpbuf */
                if (is_commutative || (remote_tree_root < cur_tree_root)) {
                    reduce_local(tmprecv, tmpbuf, blocklens[0], dt_, op);
                    reduce_local(tmprecv + displs[1], tmpbuf + displs[1], blocklens[1], dt_, op);
                } else {
                    reduce_local(tmpbuf, tmprecv, blocklens[0], dt_, op);
                    reduce_local(tmpbuf + displs[1], tmprecv + displs[1], blocklens[1], dt_, op);
                    /* copy the blocks described by dtyperecv from tmprecv to tmpbuf */
                    copy_elems(tmpbuf, tmprecv, blocklens[0]);
                    copy_elems(tmpbuf + displs[1], tmprecv + displs[1], blocklens[1]);
                }
            }
            MPI_Type_free(&dtypesend);
            MPI_Type_free(&dtyperecv);
        }
        copy_elems(rbuf, tmpbuf + (size_t)rank * rcount, rcount);
    }

    /// 3. recursive halving (non-commutative ops fall back to basic_linear)
    void recursivehalving(const T* sbuf, T* rbuf, size_t rcount, MPI_Op op)
    {
        const int comm_size = size_, rank = rank_;
        if (rcount == 0) return;
        if (comm_size < 2) {
            copy_elems(rbuf, sbuf, rcount);
            return;
        }
        if (!op_is_commute(op)) {
            basic_linear(sbuf, rbuf, rcount, op);
            return;
        }
        size_t totalcount = (size_t)comm_size * rcount;
        std::vector<T> tmpbuf_v(totalcount), tmprecv_v(totalcount);
        T* tmpbuf = tmpbuf_v.data();
        T* tmprecv = tmprecv_v.data();
        copy_elems(tmpbuf, sbuf, totalcount);

        int nprocs_pof2 = next_poweroftwo(comm_size);
        nprocs_pof2 >>= 1;
        int nprocs_rem = comm_size - nprocs_pof2;
        int vrank = -1;
        if (rank < 2 * nprocs_rem) {
            if ((rank % 2) == 0) {
                send(tmpbuf, totalcount, dt_, rank + 1, TAG_REDUCE_SCATTER_BLOCK);
                vrank = -1;
            } else {
                recv(tmprecv, totalcount, dt_, rank - 1, TAG_REDUCE_SCATTER_BLOCK);
                reduce_local(tmprecv, tmpbuf, totalcount, dt_, op);
                vrank = rank / 2;
            }
        } else {
            vrank = rank - nprocs_rem;
        }

        if (vrank != -1) {
            int send_index = 0, recv_index = 0, last_index = nprocs_pof2;
            for (int mask = nprocs_pof2 >> 1; mask > 0; mask >>= 1) {
                int vpeer = vrank ^ mask;
                int peer = (vpeer < nprocs_rem) ? vpeer * 2 + 1 : vpeer + nprocs_rem;
                size_t send_count = 0, recv_count = 0;
                if (vrank < vpeer) {
                    send_index = recv_index + mask;
                    send_count = rcount * (size_t)range_sum(send_index, last_index - 1, nprocs_rem - 1);
                    recv_count = rcount * (size_t)range_sum(recv_index, send_index - 1, nprocs_rem - 1);
                } else {
                    recv_index = send_index + mask;
                    send_count = rcount * (size_t)range_sum(send_index, recv_index - 1, nprocs_rem - 1);
                    recv_count = rcount * (size_t)range_sum(recv_index, last_index - 1, nprocs_rem - 1);
                }
                size_t rdispl = rcount * (size_t)((recv_index <= nprocs_rem - 1) ? 2 * recv_index
                                                                                  : nprocs_rem + recv_index);
                size_t sdispl = rcount * (size_t)((send_index <= nprocs_rem - 1) ? 2 * send_index
                                                                                  : nprocs_rem + send_index);
                MPI_Request request = MPI_REQUEST_NULL;
                if (recv_count > 0)
                    irecv(tmprecv + rdispl, recv_count, dt_, peer, TAG_REDUCE_SCATTER_BLOCK, &request);
                if (send_count > 0)
                    send(tmpbuf + sdispl, send_count, dt_, peer, TAG_REDUCE_SCATTER_BLOCK);
                if (recv_count > 0) {
                    wait(&request);
                    reduce_local(tmprecv + rdispl, tmpbuf + rdispl, recv_count, dt_, op);
                }
                send_index = recv_index;
                last_index = recv_index + mask;
            }
            copy_elems(rbuf, tmpbuf + (size_t)rank * rcount, rcount);
        }

        /* Step 3. Send the result to excluded even ranks */
        if (rank < 2 * nprocs_rem) {
            if ((rank % 2) == 0) recv(rbuf, rcount, dt_, rank + 1, TAG_REDUCE_SCATTER_BLOCK);
            else send(tmpbuf + (size_t)(rank - 1) * rcount, rcount, dt_, rank - 1, TAG_REDUCE_SCATTER_BLOCK);
        }
    }

    /// 4. butterfly (J.L. Traff); power-of-two sizes use butterfly_pof2
    void butterfly(const T* sbuf, T* rbuf, size_t rcount, MPI_Op op)
    {
        const int comm_size = size_, rank = rank_;
        if (rcount == 0) return;
        if (comm_size < 2) {
            copy_elems(rbuf, sbuf, rcount);
            return;
        }
        if (!(comm_size & (comm_size - 1))) {
            butterfly_pof2(sbuf, rbuf, rcount, op);
            return;
        }
        size_t totalcount = (size_t)comm_size * rcount;
        std::vector<T> tmpbuf0(totalcount), tmpbuf1(totalcount);
        T* psend = tmpbuf0.data();
        T* precv = tmpbuf1.data();
        copy_elems(psend, sbuf, totalcount);

        int nprocs_pof2 = next_poweroftwo(comm_size);
        nprocs_pof2 >>= 1;
        int nprocs_rem = comm_size - nprocs_pof2;
        int log2_size = hibit(nprocs_pof2);
        int vrank = -1;
        if (rank < 2 * nprocs_rem) {
            if ((rank % 2) == 0) {
                send(psend, totalcount, dt_, rank + 1, TAG_REDUCE_SCATTER_BLOCK);
                vrank = -1;
            } else {
                recv(precv, totalcount, dt_, rank - 1, TAG_REDUCE_SCATTER_BLOCK);
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

                size_t send_count = rcount * (size_t)range_sum(send_index, send_index + nblocks - 1, nprocs_rem - 1);
                size_t recv_count = rcount * (size_t)range_sum(recv_index, recv_index + nblocks - 1, nprocs_rem - 1);
                size_t sdispl = rcount * (size_t)((send_index <= nprocs_rem - 1) ? 2 * send_index
                                                                                  : nprocs_rem + send_index);
                size_t rdispl = rcount * (size_t)((recv_index <= nprocs_rem - 1) ? 2 * recv_index
                                                                                  : nprocs_rem + recv_index);
                sendrecv(psend + sdispl, send_count, dt_, peer, TAG_REDUCE_SCATTER_BLOCK,
                         precv + rdispl, recv_count, dt_, peer, TAG_REDUCE_SCATTER_BLOCK);
                if (vrank < vpeer) {
                    reduce_local(psend + rdispl, precv + rdispl, recv_count, dt_, op);
                    std::swap(psend, precv);
                } else {
                    reduce_local(precv + rdispl, psend + rdispl, recv_count, dt_, op);
                }
                send_index = recv_index;
            }

            int vpeer = (int)mirror_perm(vrank, log2_size);
            int peer = (vpeer < nprocs_rem) ? vpeer * 2 + 1 : vpeer + nprocs_rem;
            if (vpeer < nprocs_rem) {
                size_t sdispl = rcount * (size_t)((send_index <= nprocs_rem - 1) ? 2 * send_index
                                                                                  : nprocs_rem + send_index);
                send(psend + sdispl, rcount, dt_, peer - 1, TAG_REDUCE_SCATTER_BLOCK);
            }
            size_t sdispl = rcount * (size_t)((send_index <= nprocs_rem - 1) ? 2 * send_index
                                                                              : nprocs_rem + send_index);
            if (vpeer < nprocs_rem) sdispl += rcount;
            if (vpeer != vrank) {
                sendrecv(psend + sdispl, rcount, dt_, peer, TAG_REDUCE_SCATTER_BLOCK,
                         rbuf, rcount, dt_, peer, TAG_REDUCE_SCATTER_BLOCK);
            } else {
                copy_elems(rbuf, psend + sdispl, rcount);
            }
        } else {
            int vpeer = (int)mirror_perm((rank + 1) / 2, log2_size);
            int peer = (vpeer < nprocs_rem) ? vpeer * 2 + 1 : vpeer + nprocs_rem;
            recv(rbuf, rcount, dt_, peer, TAG_REDUCE_SCATTER_BLOCK);
        }
    }

    /// 4b. butterfly for power-of-two sizes (called by butterfly)
    void butterfly_pof2(const T* sbuf, T* rbuf, size_t rcount, MPI_Op op)
    {
        const int comm_size = size_, rank = rank_;
        if (rcount == 0) return;
        if (comm_size < 2) {
            copy_elems(rbuf, sbuf, rcount);
            return;
        }
        size_t totalcount = (size_t)comm_size * rcount;
        std::vector<T> tmpbuf0(totalcount), tmpbuf1(totalcount);
        T* psend = tmpbuf0.data();
        T* precv = tmpbuf1.data();

        /* Permute the blocks according to a mirror permutation */
        int log2_comm_size = hibit(next_poweroftwo_inclusive(comm_size)); /* opal_cube_dim */
        for (int i = 0; i < comm_size; i++)
            copy_elems(psend + (size_t)mirror_perm(i, log2_comm_size) * rcount,
                       sbuf + (size_t)i * rcount, rcount);

        size_t nblocks = totalcount, send_index = 0, recv_index = 0;
        for (int mask = 1; mask < comm_size; mask <<= 1) {
            int peer = rank ^ mask;
            nblocks /= 2;
            if ((rank & mask) == 0) send_index += nblocks;
            else recv_index += nblocks;
            sendrecv(psend + send_index, nblocks, dt_, peer, TAG_REDUCE_SCATTER_BLOCK,
                     precv + recv_index, nblocks, dt_, peer, TAG_REDUCE_SCATTER_BLOCK);
            if (rank < peer) {
                reduce_local(psend + recv_index, precv + recv_index, nblocks, dt_, op);
                std::swap(psend, precv);
            } else {
                reduce_local(precv + recv_index, psend + recv_index, nblocks, dt_, op);
            }
            send_index = recv_index;
        }
        copy_elems(rbuf, psend + recv_index, rcount);
    }

private:
    MPI_Datatype dt_;
    Reduce<T> reduce_;
    Scatter<T> scatter_;

    /* sum of elems in intersection of [a, b] and [0, r] (blocks of excluded procs count twice) */
    static int range_sum(int a, int b, int r)
    {
        if (r < a) return b - a + 1;
        else if (r > b) return 2 * (b - a + 1);
        return 2 * (r - a + 1) + b - r;
    }

    MPI_Datatype make_indexed2(const size_t blen[2], const size_t disp[2])
    {
        MPI_Datatype t;
#if COLLAGE_LARGE_COUNT
        MPI_Count bl[2] = {(MPI_Count)blen[0], (MPI_Count)blen[1]};
        MPI_Count dp[2] = {(MPI_Count)disp[0], (MPI_Count)disp[1]};
        MPI_CHECK(MPI_Type_indexed_c(2, bl, dp, dt_, &t));
#else
        int bl[2] = {cnt(blen[0]), cnt(blen[1])};
        int dp[2] = {cnt(disp[0]), cnt(disp[1])};
        MPI_CHECK(MPI_Type_indexed(2, bl, dp, dt_, &t));
#endif
        MPI_CHECK(MPI_Type_commit(&t));
        return t;
    }
};

#endif
