#pragma once
#ifndef COLLAGE_BARRIER_HPP
#define COLLAGE_BARRIER_HPP

/// Algorithms adapted from Open MPI coll/base (coll_base_barrier.c @ ec987ee)
/// No data is moved: zero-byte messages signal peer synchronisation.

#include "utils.hpp"

class Barrier : public CollBase
{
public:
    explicit Barrier(MPI_Comm comm = MPI_COMM_WORLD) : CollBase(comm) {}

    /// 1. double ring; last ring of sends is synchronous
    void doublering()
    {
        const int size = size_, rank = rank_;
        if (1 == size) return;
        int left = ((size + rank - 1) % size);
        int right = ((rank + 1) % size);

        if (rank > 0) recv(nullptr, 0, MPI_BYTE, left, TAG_BARRIER);   /* from the left */
        send(nullptr, 0, MPI_BYTE, right, TAG_BARRIER);                  /* to the right */
        if (rank == 0) recv(nullptr, 0, MPI_BYTE, left, TAG_BARRIER);  /* root from last */

        /* Allow nodes to exit */
        if (rank > 0) recv(nullptr, 0, MPI_BYTE, left, TAG_BARRIER);
        ssend(nullptr, 0, MPI_BYTE, right, TAG_BARRIER);
        if (rank == 0) recv(nullptr, 0, MPI_BYTE, left, TAG_BARRIER);
    }

    /// 2. recursive doubling (extra step for non power-of-two sizes)
    void recursivedoubling()
    {
        const int size = size_, rank = rank_;
        int adjsize, mask, remote;
        if (1 == size) return;

        /* do nearest power of 2 less than size calc */
        adjsize = next_poweroftwo(size);
        adjsize >>= 1;

        /* if size is not exact power of two, perform an extra step */
        if (adjsize != size) {
            if (rank >= adjsize) {
                /* send message to lower ranked node */
                remote = rank - adjsize;
                sendrecv_zero(remote, TAG_BARRIER, remote, TAG_BARRIER);
            } else if (rank < (size - adjsize)) {
                /* receive message from high level rank */
                recv(nullptr, 0, MPI_BYTE, rank + adjsize, TAG_BARRIER);
            }
        }

        /* exchange messages */
        if (rank < adjsize) {
            mask = 0x1;
            while (mask < adjsize) {
                remote = rank ^ mask;
                mask <<= 1;
                if (remote >= adjsize) continue;
                sendrecv_zero(remote, TAG_BARRIER, remote, TAG_BARRIER);
            }
        }

        /* non-power of 2 case */
        if (adjsize != size) {
            if (rank < (size - adjsize)) {
                /* send enter message to higher ranked node */
                remote = rank + adjsize;
                ssend(nullptr, 0, MPI_BYTE, remote, TAG_BARRIER);
            }
        }
    }

    /// 3. Bruck (dissemination): exchange with rank-2^k and rank+2^k
    void bruck()
    {
        const int size = size_, rank = rank_;
        if (1 == size) return;
        for (int distance = 1; distance < size; distance <<= 1) {
            int from = (rank + size - distance) % size;
            int to = (rank + distance) % size;
            sendrecv_zero(to, TAG_BARRIER, from, TAG_BARRIER);
        }
    }

    /// 4. two processes only
    void two_procs()
    {
        if (1 == size_) return;
        if (2 != size_) throw NotApplicable("two_procs requires exactly 2 processes");
        int remote = (rank_ + 1) & 0x1;
        sendrecv_zero(remote, TAG_BARRIER, remote, TAG_BARRIER);
    }

    /// 5. basic linear: everybody reports to rank 0, rank 0 releases everybody
    void linear()
    {
        const int size = size_, rank = rank_;
        if (1 == size) return;
        if (rank > 0) {
            send(nullptr, 0, MPI_BYTE, 0, TAG_BARRIER);
            recv(nullptr, 0, MPI_BYTE, 0, TAG_BARRIER);
        } else {
            std::vector<MPI_Request> requests(size, MPI_REQUEST_NULL);
            for (int i = 1; i < size; ++i)
                irecv(nullptr, 0, MPI_BYTE, MPI_ANY_SOURCE, TAG_BARRIER, &requests[i]);
            waitall(size - 1, requests.data() + 1);
            for (int i = 1; i < size; ++i) send(nullptr, 0, MPI_BYTE, i, TAG_BARRIER);
        }
    }

    /// 6. tree: up the (hypercube) tree and back down
    void tree()
    {
        const int size = size_, rank = rank_;
        int depth, jump, partner;
        if (1 == size) return;

        /* Find the nearest power of 2 of the communicator size. */
        depth = next_poweroftwo_inclusive(size);

        for (jump = 1; jump < depth; jump <<= 1) {
            partner = rank ^ jump;
            if (!(partner & (jump - 1)) && partner < size) {
                if (partner > rank) recv(nullptr, 0, MPI_BYTE, partner, TAG_BARRIER);
                else if (partner < rank) send(nullptr, 0, MPI_BYTE, partner, TAG_BARRIER);
            }
        }
        depth >>= 1;
        for (jump = depth; jump > 0; jump >>= 1) {
            partner = rank ^ jump;
            if (!(partner & (jump - 1)) && partner < size) {
                if (partner > rank) send(nullptr, 0, MPI_BYTE, partner, TAG_BARRIER);
                else if (partner < rank) recv(nullptr, 0, MPI_BYTE, partner, TAG_BARRIER);
            }
        }
    }
};

#endif
