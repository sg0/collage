#pragma once
#ifndef COLLAGE_ALLGATHER_HPP
#define COLLAGE_ALLGATHER_HPP

/// Algorithms adapted from Open MPI coll/base (coll_base_allgather.c @ ec987ee)
/// Every rank contributes `count` elements; everybody receives size*count.

#include <cmath>

#include "bcast.hpp"
#include "gather.hpp"
#include "utils.hpp"

template <typename T>
class Allgather : public CollBase
{
public:
    explicit Allgather(MPI_Comm comm = MPI_COMM_WORLD)
        : CollBase(comm), dt_(TypeMap<T>()), gather_(comm), bcast_(comm) {}

    /// 1. recursive doubling (power-of-two sizes; otherwise falls back to
    ///    k_bruck with radix 2, as upstream does)
    void recursivedoubling(const T* sbuf, size_t scount, T* rbuf, size_t rcount)
    {
        int remote, distance, sendblocklocation;
        const int size = size_, rank = rank_;
        int pow2size = next_poweroftwo(size) >> 1;

        if (pow2size != size) {
            k_bruck(sbuf, scount, rbuf, rcount, 2);
            return;
        }

        copy_elems(rbuf + (size_t)rank * rcount, sbuf, scount);

        sendblocklocation = rank;
        for (distance = 0x1; distance < size; distance <<= 1) {
            remote = rank ^ distance;
            T *tmpsend, *tmprecv;
            if (rank < remote) {
                tmpsend = rbuf + (size_t)sendblocklocation * rcount;
                tmprecv = rbuf + (size_t)(sendblocklocation + distance) * rcount;
            } else {
                tmpsend = rbuf + (size_t)sendblocklocation * rcount;
                tmprecv = rbuf + (size_t)(sendblocklocation - distance) * rcount;
                sendblocklocation -= distance;
            }
            sendrecv(tmpsend, (size_t)distance * rcount, dt_, remote, TAG_ALLGATHER,
                     tmprecv, (size_t)distance * rcount, dt_, remote, TAG_ALLGATHER);
        }
    }

    /// 2. Sparbit: Bruck-like with inverted distances and non-decreasing
    ///    message sizes ("Sparbit: a new logarithmic-cost and data
    ///    locality-aware MPI Allgather algorithm").
    ///    Each block uses its own tag: TAG_BASE + block (upstream uses a
    ///    negative internal tag space that is not available to applications).
    void sparbit(const T* sbuf, size_t scount, T* rbuf, size_t rcount)
    {
        const int comm_size = size_, rank = rank_;
        int comm_log = 0, exclusion = 0, data_expected = 1, transfer_count = 0;
        int sendto, recvfrom, send_disp, recv_disp;
        uint32_t last_ignore, ignore_steps, distance = 1;

        T* tmprecv = rbuf;
        copy_elems(tmprecv + (size_t)rank * rcount, sbuf, scount);
        T* tmpsend = tmprecv;
        if (comm_size == 1) return;

        std::vector<MPI_Request> requests(comm_size, MPI_REQUEST_NULL);

        /* calculate log2 of the total process count */
        comm_log = (int)std::ceil(std::log((double)comm_size) / std::log(2.0));
        distance <<= comm_log - 1;

        last_ignore = __builtin_ctz(comm_size);
        ignore_steps = (~((uint32_t)comm_size >> last_ignore) | 1) << last_ignore;

        /* perform the parallel binomial tree distribution steps */
        for (int i = 0; i < comm_log; ++i) {
            sendto = (rank + distance) % comm_size;
            recvfrom = (rank - distance + comm_size) % comm_size;
            exclusion = (distance & ignore_steps) == distance;

            for (transfer_count = 0; transfer_count < data_expected - exclusion; transfer_count++) {
                send_disp = (rank - 2 * transfer_count * distance + comm_size) % comm_size;
                recv_disp = (rank - (2 * transfer_count + 1) * distance + comm_size) % comm_size;
                isend(tmpsend + (size_t)send_disp * scount, scount, dt_, sendto,
                      TAG_BASE + send_disp, &requests[transfer_count]);
                irecv(tmprecv + (size_t)recv_disp * rcount, rcount, dt_, recvfrom,
                      TAG_BASE + recv_disp, &requests[data_expected - exclusion + transfer_count]);
            }
            waitall(transfer_count * 2, requests.data());

            distance >>= 1;
            /* data expected for the next step */
            data_expected = (data_expected << 1) - exclusion;
            exclusion = 0;
        }
    }

    /// 3. ring
    void ring(const T* sbuf, size_t scount, T* rbuf, size_t rcount)
    {
        int sendto, recvfrom, i, recvdatafrom, senddatafrom;
        const int size = size_, rank = rank_;

        copy_elems(rbuf + (size_t)rank * rcount, sbuf, scount);

        sendto = (rank + 1) % size;
        recvfrom = (rank - 1 + size) % size;

        for (i = 0; i < size - 1; i++) {
            recvdatafrom = (rank - i - 1 + size) % size;
            senddatafrom = (rank - i + size) % size;
            T* tmprecv = rbuf + (size_t)recvdatafrom * rcount;
            T* tmpsend = rbuf + (size_t)senddatafrom * rcount;
            sendrecv(tmpsend, rcount, dt_, sendto, TAG_ALLGATHER,
                     tmprecv, rcount, dt_, recvfrom, TAG_ALLGATHER);
        }
    }

    /// 4. neighbor exchange (even sizes; odd sizes fall back to ring)
    void neighborexchange(const T* sbuf, size_t scount, T* rbuf, size_t rcount)
    {
        int i, even_rank;
        int neighbor[2], offset_at_step[2], recv_data_from[2], send_data_from;
        const int size = size_, rank = rank_;

        if (size % 2) {
            ring(sbuf, scount, rbuf, rcount);
            return;
        }

        copy_elems(rbuf + (size_t)rank * rcount, sbuf, scount);

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
        sendrecv(rbuf + (size_t)rank * rcount, rcount, dt_, neighbor[0], TAG_ALLGATHER,
                 rbuf + (size_t)neighbor[0] * rcount, rcount, dt_, neighbor[0], TAG_ALLGATHER);

        if (even_rank) send_data_from = rank;
        else send_data_from = recv_data_from[0];

        for (i = 1; i < (size / 2); i++) {
            const int i_parity = i % 2;
            recv_data_from[i_parity] =
                (recv_data_from[i_parity] + offset_at_step[i_parity] + size) % size;

            T* tmprecv = rbuf + (size_t)recv_data_from[i_parity] * rcount;
            T* tmpsend = rbuf + (size_t)send_data_from * rcount;
            sendrecv(tmpsend, 2 * rcount, dt_, neighbor[i_parity], TAG_ALLGATHER,
                     tmprecv, 2 * rcount, dt_, neighbor[i_parity], TAG_ALLGATHER);
            send_data_from = recv_data_from[i_parity];
        }
    }

    /// 5. basic linear: gather to rank 0, then broadcast.
    ///    The gather and broadcast are the basic linear point-to-point ones.
    void linear(const T* sbuf, size_t scount, T* rbuf, size_t rcount)
    {
        const int root = 0;
        gather_.linear(sbuf, scount, rbuf, rcount, root);
        bcast_.linear(rbuf, rcount * (size_t)size_, root);
    }

    /// 6. k-port Bruck with radix k
    void k_bruck(const T* sbuf, size_t scount, T* rbuf, size_t rcount, int radix)
    {
        int dst, src, recvcount, distance;
        const int size = size_, rank = rank_;
        int num_reqs, max_reqs = 0;
        std::vector<T> tmp_buf;

        if (radix < 2) throw std::invalid_argument("k_bruck: radix must be >= 2");

        if (0 != rank) tmp_buf.resize(rcount * (size_t)(size - rank));

        /* everyone has its own block at location 0 */
        copy_elems(rbuf, sbuf, scount);

        max_reqs = 2 * (radix - 1);
        std::vector<MPI_Request> reqs(max_reqs, MPI_REQUEST_NULL);
        recvcount = 1;
        T* tmpsend = rbuf;
        for (distance = 1; distance < size; distance *= radix) {
            num_reqs = 0;
            for (int j = 1; j < radix; j++) {
                if (distance * j >= size) break;
                src = (rank + distance * j) % size;
                dst = (rank - distance * j + size) % size;

                T* tmprecv = tmpsend + (size_t)distance * j * rcount;

                if (distance <= (size / radix)) recvcount = distance;
                else recvcount = (distance < (size - distance * j) ? distance : (size - distance * j));

                irecv(tmprecv, (size_t)recvcount * rcount, dt_, src, TAG_ALLGATHER, &reqs[num_reqs++]);
                isend(tmpsend, (size_t)recvcount * rcount, dt_, dst, TAG_ALLGATHER, &reqs[num_reqs++]);
            }
            waitall(num_reqs, reqs.data());
        }

        /* Finalization step: on all ranks except 0, data needs to be shifted locally */
        if (0 != rank) {
            copy_elems(tmp_buf.data(), rbuf, (size_t)(size - rank) * rcount);
            copy_elems(rbuf, rbuf + (size_t)(size - rank) * rcount, (size_t)rank * rcount);
            copy_elems(rbuf + (size_t)rank * rcount, tmp_buf.data(), (size_t)(size - rank) * rcount);
        }
    }

    /// 7. direct messaging: non-blocking exchange with every peer
    void direct_messaging(const T* sbuf, size_t scount, T* rbuf, size_t rcount)
    {
        const int comm_size = size_, rank = rank_;
        int max_reqs = 0;

        copy_elems(rbuf + (size_t)rank * rcount, sbuf, scount);
        const T* tmpsend = sbuf;

        std::vector<MPI_Request> reqs(2 * (size_t)(comm_size - 1) + 1, MPI_REQUEST_NULL);
        for (int i = 1; i < comm_size; ++i) {
            int peer_rank = (rank + i) % comm_size;
            T* tmprecv = rbuf + (size_t)peer_rank * rcount;
            irecv(tmprecv, rcount, dt_, peer_rank, TAG_ALLGATHER, &reqs[max_reqs++]);
            isend(tmpsend, scount, dt_, peer_rank, TAG_ALLGATHER, &reqs[max_reqs++]);
        }
        waitall(max_reqs, reqs.data());
    }

private:
    MPI_Datatype dt_;
    Gather<T> gather_;
    Bcast<T> bcast_;
};

#endif
