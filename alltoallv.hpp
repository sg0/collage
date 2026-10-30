#pragma once
#ifndef COLLAGE_ALLTOALLV_HPP
#define COLLAGE_ALLTOALLV_HPP

/// Algorithms adapted from Open MPI coll/base (coll_base_alltoallv.c @ ec987ee)
/// counts / displacements are in elements of T (as in MPI_Alltoallv).

#include "utils.hpp"

template <typename T>
class Alltoallv : public CollBase
{
public:
    using counts_t = std::vector<size_t>;
    using displs_t = std::vector<size_t>;

    explicit Alltoallv(MPI_Comm comm = MPI_COMM_WORLD) : CollBase(comm), dt_(TypeMap<T>()) {}

    /// 1. in place (mca_coll_base_alltoallv_intra_basic_inplace)
    ///    rbuf holds the send data on input (laid out with rcounts/rdisps,
    ///    which must equal the send layout) and the result on output.
    void inplace(T* rbuf, counts_t const& rcounts, displs_t const& rdisps)
    {
        int i, left, right;
        const int size = size_, rank = rank_;
        MPI_Request req = MPI_REQUEST_NULL;
        size_t max_size = 0;

        for (i = 0; i < size; ++i) {
            if (i == rank) continue;
            max_size = std::max(rcounts[i], max_size);
        }
        if ((1 == size) || (0 == max_size)) return;

        std::vector<T> tmp_buffer(max_size);

        for (i = 1; i <= (size >> 1); ++i) {
            right = (rank + i) % size;
            left = (rank + size - i) % size;

            if (0 != rcounts[right]) { /* nothing to exchange with the peer on the right */
                /* "pack" the block destined to the right peer */
                copy_elems(tmp_buffer.data(), rbuf + rdisps[right], rcounts[right]);
                /* Receive data from the right */
                irecv(rbuf + rdisps[right], rcounts[right], dt_, right, TAG_ALLTOALLV, &req);
            }

            if ((left != right) && (0 != rcounts[left])) {
                /* Send data to the left */
                send(rbuf + rdisps[left], rcounts[left], dt_, left, TAG_ALLTOALLV);
                wait(&req);
                /* Receive data from the left */
                irecv(rbuf + rdisps[left], rcounts[left], dt_, left, TAG_ALLTOALLV, &req);
            }

            if (0 != rcounts[right]) {
                /* Send data to the right */
                send(tmp_buffer.data(), rcounts[right], dt_, right, TAG_ALLTOALLV);
            }

            wait(&req);
        }
    }

    /// 2. pairwise exchange
    void pairwise(const T* sbuf, counts_t const& scounts, displs_t const& sdisps,
                  T* rbuf, counts_t const& rcounts, displs_t const& rdisps)
    {
        int step, sendto, recvfrom;
        const int size = size_, rank = rank_;
        MPI_Request req;

        for (step = 0; step < size; step++) {
            req = MPI_REQUEST_NULL;
            sendto = (rank + step) % size;
            recvfrom = (rank + size - step) % size;

            const T* psnd = sbuf + sdisps[sendto];
            T* prcv = rbuf + rdisps[recvfrom];

            if (0 < rcounts[recvfrom])
                irecv(prcv, rcounts[recvfrom], dt_, recvfrom, TAG_ALLTOALLV, &req);
            if (0 < scounts[sendto])
                send(psnd, scounts[sendto], dt_, sendto, TAG_ALLTOALLV);
            if (MPI_REQUEST_NULL != req) wait(&req);
        }
    }

    /// 3. basic linear (persistent requests)
    void linear(const T* sbuf, counts_t const& scounts, displs_t const& sdisps,
                T* rbuf, counts_t const& rcounts, displs_t const& rdisps)
    {
        int i, nreqs;
        const int size = size_, rank = rank_;

        /* Simple optimization - handle send to self first */
        if (0 < scounts[rank])
            copy_elems(rbuf + rdisps[rank], sbuf + sdisps[rank], std::min(scounts[rank], rcounts[rank]));

        if (1 == size) return;

        nreqs = 0;
        std::vector<MPI_Request> reqs(2 * (size_t)size, MPI_REQUEST_NULL);
        MPI_Request* preq = reqs.data();

        /* Post all receives first */
        for (i = 0; i < size; ++i) {
            if (i == rank) continue;
            if (0 < rcounts[i]) {
                ++nreqs;
                recv_init(rbuf + rdisps[i], rcounts[i], dt_, i, TAG_ALLTOALLV, preq++);
            }
        }
        /* Now post all sends */
        for (i = 0; i < size; ++i) {
            if (i == rank) continue;
            if (0 < scounts[i]) {
                ++nreqs;
                send_init(sbuf + sdisps[i], scounts[i], dt_, i, TAG_ALLTOALLV, preq++);
            }
        }

        if (nreqs > 0) {
            MPI_CHECK(MPI_Startall(nreqs, reqs.data()));
            waitall(nreqs, reqs.data());
            /* Free the requests in all cases as they are persistent */
            for (i = 0; i < nreqs; ++i) MPI_Request_free(&reqs[i]);
        }
    }

private:
    MPI_Datatype dt_;
};

#endif
