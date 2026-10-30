#pragma once
#ifndef COLLAGE_ALLGATHERV_HPP
#define COLLAGE_ALLGATHERV_HPP

/// Algorithms adapted from Open MPI coll/base (coll_base_allgatherv.c @ ec987ee)
/// rcounts / rdispls are in elements of T (as in MPI_Allgatherv).

#include <cmath>

#include "gather.hpp"
#include "utils.hpp"

template <typename T>
class Allgatherv : public CollBase
{
public:
    using counts_t = std::vector<size_t>;
    using displs_t = std::vector<size_t>;

    explicit Allgatherv(MPI_Comm comm = MPI_COMM_WORLD)
        : CollBase(comm), dt_(TypeMap<T>()), gather_(comm) {}

    /// 1. Bruck (log(p) steps, indexed datatypes select the blocks)
    void bruck(const T* sbuf, size_t scount, T* rbuf, counts_t const& rcounts,
               displs_t const& rdispls)
    {
        int sendto, recvfrom, distance, blockcount, i;
        const int size = size_, rank = rank_;

        copy_elems(rbuf + rdispls[rank], sbuf, std::min(scount, rcounts[rank]));

        std::vector<size_t> new_rcounts(size), new_rdispls(size), new_scounts(size), new_sdispls(size);

        for (distance = 1; distance < size; distance <<= 1) {
            recvfrom = (rank + distance) % size;
            sendto = (rank - distance + size) % size;

            if (distance <= (size >> 1)) blockcount = distance;
            else blockcount = size - distance;

            /* create send and receive datatypes */
            for (i = 0; i < blockcount; i++) {
                const int tmp_srank = (rank + i) % size;
                const int tmp_rrank = (recvfrom + i) % size;
                new_scounts[i] = rcounts[tmp_srank];
                new_sdispls[i] = rdispls[tmp_srank];
                new_rcounts[i] = rcounts[tmp_rrank];
                new_rdispls[i] = rdispls[tmp_rrank];
            }
            MPI_Datatype new_sdtype = make_indexed(blockcount, new_scounts, new_sdispls);
            MPI_Datatype new_rdtype = make_indexed(blockcount, new_rcounts, new_rdispls);

            sendrecv(rbuf, 1, new_sdtype, sendto, TAG_ALLGATHERV,
                     rbuf, 1, new_rdtype, recvfrom, TAG_ALLGATHERV);

            MPI_Type_free(&new_sdtype);
            MPI_Type_free(&new_rdtype);
        }
    }

    /// 2. Sparbit (see allgather.hpp); tags are TAG_BASE + block
    void sparbit(const T* sbuf, size_t scount, T* rbuf, counts_t const& rcounts,
                 displs_t const& rdispls)
    {
        const int comm_size = size_, rank = rank_;
        int comm_log = 0, exclusion = 0, data_expected = 1, transfer_count = 0, step_requests = 0;
        int sendto, recvfrom, send_disp, recv_disp;
        uint32_t last_ignore, ignore_steps, distance = 1;

        T* tmprecv = rbuf;
        copy_elems(tmprecv + rdispls[rank], sbuf, scount);
        T* tmpsend = tmprecv;
        if (comm_size == 1) return;

        std::vector<MPI_Request> requests(2 * (size_t)comm_size, MPI_REQUEST_NULL);

        comm_log = (int)std::ceil(std::log((double)comm_size) / std::log(2.0));
        distance <<= comm_log - 1;

        last_ignore = __builtin_ctz(comm_size);
        ignore_steps = (~((uint32_t)comm_size >> last_ignore) | 1) << last_ignore;

        for (int i = 0; i < comm_log; ++i) {
            sendto = (rank + distance) % comm_size;
            recvfrom = (rank - distance + comm_size) % comm_size;
            exclusion = (distance & ignore_steps) == distance;

            for (transfer_count = 0; transfer_count < data_expected - exclusion; transfer_count++) {
                send_disp = (rank - 2 * transfer_count * distance + comm_size) % comm_size;
                recv_disp = (rank - (2 * transfer_count + 1) * distance + comm_size) % comm_size;
                if (rcounts[send_disp] > 0)
                    isend(tmpsend + rdispls[send_disp], rcounts[send_disp], dt_, sendto,
                          TAG_BASE + send_disp, &requests[step_requests++]);
                if (rcounts[recv_disp] > 0)
                    irecv(tmprecv + rdispls[recv_disp], rcounts[recv_disp], dt_, recvfrom,
                          TAG_BASE + recv_disp, &requests[step_requests++]);
            }
            waitall(step_requests, requests.data());

            distance >>= 1;
            data_expected = (data_expected << 1) - exclusion;
            exclusion = step_requests = 0;
        }
    }

    /// 3. ring
    void ring(const T* sbuf, size_t scount, T* rbuf, counts_t const& rcounts,
              displs_t const& rdisps)
    {
        int sendto, recvfrom, i, recvdatafrom, senddatafrom;
        const int size = size_, rank = rank_;

        copy_elems(rbuf + rdisps[rank], sbuf, std::min(scount, rcounts[rank]));

        sendto = (rank + 1) % size;
        recvfrom = (rank - 1 + size) % size;

        for (i = 0; i < size - 1; i++) {
            recvdatafrom = (rank - i - 1 + size) % size;
            senddatafrom = (rank - i + size) % size;
            sendrecv(rbuf + rdisps[senddatafrom], rcounts[senddatafrom], dt_, sendto, TAG_ALLGATHERV,
                     rbuf + rdisps[recvdatafrom], rcounts[recvdatafrom], dt_, recvfrom, TAG_ALLGATHERV);
        }
    }

    /// 4. neighbor exchange (even sizes; odd sizes fall back to ring)
    void neighborexchange(const T* sbuf, size_t scount, T* rbuf, counts_t const& rcounts,
                          displs_t const& rdispls)
    {
        int i, even_rank;
        int neighbor[2], offset_at_step[2], recv_data_from[2], send_data_from;
        const int size = size_, rank = rank_;

        if (size % 2) {
            ring(sbuf, scount, rbuf, rcounts, rdispls);
            return;
        }

        copy_elems(rbuf + rdispls[rank], sbuf, std::min(scount, rcounts[rank]));

        even_rank = !(rank % 2);
        if (even_rank) {
            neighbor[0] = (rank + 1) % size;
            neighbor[1] = (rank - 1 + size) % size;
            recv_data_from[0] = rank;
            recv_data_from[1] = rank;
            offset_at_step[0] = (+2);
            offset_at_step[1] = (-2);
        } else {
            neighbor[0] = (rank - 1 + size) % size;
            neighbor[1] = (rank + 1) % size;
            recv_data_from[0] = neighbor[0];
            recv_data_from[1] = neighbor[0];
            offset_at_step[0] = (-2);
            offset_at_step[1] = (+2);
        }

        /* First step: exchange a single block with neighbor[0]. */
        sendrecv(rbuf + rdispls[rank], rcounts[rank], dt_, neighbor[0], TAG_ALLGATHERV,
                 rbuf + rdispls[neighbor[0]], rcounts[neighbor[0]], dt_, neighbor[0], TAG_ALLGATHERV);

        if (even_rank) send_data_from = rank;
        else send_data_from = recv_data_from[0];

        std::vector<size_t> sc(2), sd(2), rc(2), rd(2);
        for (i = 1; i < (size / 2); i++) {
            const int i_parity = i % 2;
            recv_data_from[i_parity] =
                (recv_data_from[i_parity] + offset_at_step[i_parity] + size) % size;

            /* We send data from ranks (send_data_from) and (send_data_from+1),
               and receive data from ranks (recv_data_from[i_parity]) and
               (recv_data_from[i_parity]+1). */
            sc[0] = rcounts[send_data_from];
            sc[1] = rcounts[send_data_from + 1];
            sd[0] = rdispls[send_data_from];
            sd[1] = rdispls[send_data_from + 1];
            MPI_Datatype new_sdtype = make_indexed(2, sc, sd);

            rc[0] = rcounts[recv_data_from[i_parity]];
            rc[1] = rcounts[recv_data_from[i_parity] + 1];
            rd[0] = rdispls[recv_data_from[i_parity]];
            rd[1] = rdispls[recv_data_from[i_parity] + 1];
            MPI_Datatype new_rdtype = make_indexed(2, rc, rd);

            sendrecv(rbuf, 1, new_sdtype, neighbor[i_parity], TAG_ALLGATHERV,
                     rbuf, 1, new_rdtype, neighbor[i_parity], TAG_ALLGATHERV);

            send_data_from = recv_data_from[i_parity];
            MPI_Type_free(&new_sdtype);
            MPI_Type_free(&new_rdtype);
        }
    }

    /// 5. two processes only
    void two_procs(const T* sbuf, size_t scount, T* rbuf, counts_t const& rcounts,
                   displs_t const& rdispls)
    {
        if (2 != size_) throw NotApplicable("two_procs requires exactly 2 processes");
        int remote = rank_ ^ 0x1;
        sendrecv(sbuf, scount, dt_, remote, TAG_ALLGATHERV,
                 rbuf + rdispls[remote], rcounts[remote], dt_, remote, TAG_ALLGATHERV);
        copy_elems(rbuf + rdispls[rank_], sbuf, std::min(scount, rcounts[rank_]));
    }

    /// 6. basic default: gatherv to rank 0, then broadcast of an indexed datatype.
    ///    The gatherv and the broadcast are linear point-to-point implementations.
    void basic_default(const T* sbuf, size_t scount, T* rbuf, counts_t const& rcounts,
                       displs_t const& disps)
    {
        const int size = size_, rank = rank_;
        const int root = 0;

        gather_.linear_v(sbuf, scount, rbuf, rcounts, disps, root);

        MPI_Datatype newtype = make_indexed(size, rcounts, disps);
        /* linear broadcast (as ompi_coll_base_bcast_intra_basic_linear) */
        if (size > 1) {
            if (rank != root) {
                recv(rbuf, 1, newtype, root, TAG_ALLGATHERV);
            } else {
                std::vector<MPI_Request> reqs(size - 1, MPI_REQUEST_NULL);
                MPI_Request* preq = reqs.data();
                for (int i = 0; i < size; ++i) {
                    if (i == rank) continue;
                    isend(rbuf, 1, newtype, i, TAG_ALLGATHERV, preq++);
                }
                waitall(size - 1, reqs.data());
            }
        }
        MPI_Type_free(&newtype);
    }

private:
    MPI_Datatype dt_;
    Gather<T> gather_;

    MPI_Datatype make_indexed(int n, std::vector<size_t> const& blen, std::vector<size_t> const& disp)
    {
        MPI_Datatype t;
#if COLLAGE_LARGE_COUNT
        std::vector<MPI_Count> bl(blen.begin(), blen.begin() + n), dp(disp.begin(), disp.begin() + n);
        MPI_CHECK(MPI_Type_indexed_c((MPI_Count)n, bl.data(), dp.data(), dt_, &t));
#else
        std::vector<int> bl(n), dp(n);
        for (int i = 0; i < n; ++i) { bl[i] = cnt(blen[i]); dp[i] = cnt(disp[i]); }
        MPI_CHECK(MPI_Type_indexed(n, bl.data(), dp.data(), dt_, &t));
#endif
        MPI_CHECK(MPI_Type_commit(&t));
        return t;
    }
};

#endif
