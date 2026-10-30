#pragma once
#ifndef COLLAGE_ALLTOALL_HPP
#define COLLAGE_ALLTOALL_HPP

/// Algorithms adapted from Open MPI coll/base (coll_base_alltoall.c @ ec987ee)
/// Every rank sends `count` elements of type T to every rank (block j of sbuf
/// goes to rank j, block i of rbuf comes from rank i).

#include "utils.hpp"

template <typename T>
class Alltoall : public CollBase
{
public:
    explicit Alltoall(MPI_Comm comm = MPI_COMM_WORLD) : CollBase(comm), dt_(TypeMap<T>()) {}

    /// 1. in place (mca_coll_base_alltoall_intra_basic_inplace)
    ///    rbuf holds the send data on input and the result on output.
    void inplace(T* rbuf, size_t rcount)
    {
        int i, left, right;
        MPI_Request req = MPI_REQUEST_NULL;
        const int size = size_, rank = rank_;

        if ((1 == size) || (0 == rcount)) return;

        std::vector<T> tmp_buffer(rcount);

        for (i = 1; i <= (size >> 1); ++i) {
            right = (rank + i) % size;
            left = (rank + size - i) % size;

            /* "pack" the block destined to the right peer */
            copy_elems(tmp_buffer.data(), rbuf + (size_t)right * rcount, rcount);

            /* Receive data from the right */
            irecv(rbuf + (size_t)right * rcount, rcount, dt_, right, TAG_ALLTOALL, &req);

            if (left != right) {
                /* Send data to the left */
                send(rbuf + (size_t)left * rcount, rcount, dt_, left, TAG_ALLTOALL);
                wait(&req);
                /* Receive data from the left */
                irecv(rbuf + (size_t)left * rcount, rcount, dt_, left, TAG_ALLTOALL, &req);
            }

            /* Send data to the right */
            send(tmp_buffer.data(), rcount, dt_, right, TAG_ALLTOALL);
            wait(&req);
        }
    }

    /// 2. pairwise exchange
    void pairwise(const T* sbuf, size_t scount, T* rbuf, size_t rcount)
    {
        int step, sendto, recvfrom;
        const int size = size_, rank = rank_;

        /* Perform pairwise exchange - starting from 1 so the local copy is last */
        for (step = 1; step < size + 1; step++) {
            sendto = (rank + step) % size;
            recvfrom = (rank + size - step) % size;
            const T* tmpsend = sbuf + (size_t)sendto * scount;
            T* tmprecv = rbuf + (size_t)recvfrom * rcount;
            sendrecv(tmpsend, scount, dt_, sendto, TAG_ALLTOALL,
                     tmprecv, rcount, dt_, recvfrom, TAG_ALLTOALL);
        }
    }

    /// 3. Bruck (log(p) steps, uses an indexed datatype per step)
    void bruck(const T* sbuf, size_t scount, T* rbuf, size_t rcount)
    {
        int i, sendto, recvfrom, distance;
        const int size = size_, rank = rank_;

        std::vector<T> tmpbuf((size_t)size * rcount);

        /* Step 1 - local rotation - shift up by rank */
        copy_elems(tmpbuf.data(), sbuf + (size_t)rank * scount, (size_t)(size - rank) * scount);
        if (rank != 0)
            copy_elems(tmpbuf.data() + (size_t)(size - rank) * rcount, sbuf, (size_t)rank * scount);

        /* perform communication step */
        std::vector<size_t> blen, bdisp;
        for (distance = 1; distance < size; distance <<= 1) {
            sendto = (rank + distance) % size;
            recvfrom = (rank - distance + size) % size;

            /* Create datatype describing data sent/received */
            blen.clear();
            bdisp.clear();
            for (i = distance; i < size; i += 2 * distance) {
                int nblocks = distance;
                if (i + distance >= size) nblocks = size - i;
                blen.push_back(rcount * (size_t)nblocks);
                bdisp.push_back((size_t)i * rcount);
            }
            MPI_Datatype new_ddt = make_indexed(blen, bdisp);

            /* Sendreceive */
            sendrecv(tmpbuf.data(), 1, new_ddt, sendto, TAG_ALLTOALL,
                     rbuf, 1, new_ddt, recvfrom, TAG_ALLTOALL);

            /* Copy back new data from recvbuf to tmpbuf */
            for (size_t b = 0; b < blen.size(); ++b)
                copy_elems(tmpbuf.data() + bdisp[b], rbuf + bdisp[b], blen[b]);

            MPI_Type_free(&new_ddt);
        }

        /* Step 3 - local rotation */
        for (i = 0; i < size; i++)
            copy_elems(rbuf + (size_t)((rank - i + size) % size) * rcount,
                       tmpbuf.data() + (size_t)i * rcount, rcount);
    }

    /// 4. linear with a limited number of outstanding requests
    void linear_sync(const T* sbuf, size_t scount, T* rbuf, size_t rcount,
                     int max_outstanding_reqs)
    {
        int ri, si, nrreqs, nsreqs, total_reqs, nreqs = 0;
        const int size = size_, rank = rank_;

        /* simple optimization */
        copy_elems(rbuf + (size_t)rank * rcount, sbuf + (size_t)rank * scount, scount);

        if (1 == size) return;

        total_reqs = (((max_outstanding_reqs > (size - 1)) || (max_outstanding_reqs <= 0))
                          ? (size - 1)
                          : (max_outstanding_reqs));
        std::vector<MPI_Request> reqs(2 * (size_t)total_reqs, MPI_REQUEST_NULL);

        /* Post first batch of irecv and isend requests  */
        for (nreqs = 0, nrreqs = 0, ri = (rank + 1) % size; nreqs < total_reqs;
             ri = (ri + 1) % size, ++nrreqs) {
            irecv(rbuf + (size_t)ri * rcount, rcount, dt_, ri, TAG_ALLTOALL, &reqs[nreqs]);
            nreqs++;
        }
        for (nsreqs = 0, si = (rank + size - 1) % size; nreqs < 2 * total_reqs;
             si = (si + size - 1) % size, ++nsreqs) {
            isend(sbuf + (size_t)si * scount, scount, dt_, si, TAG_ALLTOALL, &reqs[nreqs]);
            nreqs++;
        }

        if (nreqs == 2 * (size - 1)) {
            waitall(nreqs, reqs.data());
        } else {
            int ncreqs = 0;
            while (ncreqs < 2 * (size - 1)) {
                int completed;
                waitany(2 * total_reqs, reqs.data(), &completed);
                reqs[completed] = MPI_REQUEST_NULL;
                ncreqs++;
                if (completed < total_reqs) {
                    if (nrreqs < (size - 1)) {
                        irecv(rbuf + (size_t)ri * rcount, rcount, dt_, ri, TAG_ALLTOALL,
                              &reqs[completed]);
                        ++nrreqs;
                        ri = (ri + 1) % size;
                    }
                } else {
                    if (nsreqs < (size - 1)) {
                        isend(sbuf + (size_t)si * scount, scount, dt_, si, TAG_ALLTOALL,
                              &reqs[completed]);
                        ++nsreqs;
                        si = (si + size - 1) % size;
                    }
                }
            }
        }
    }

    /// 5. two processes only
    void two_procs(const T* sbuf, size_t scount, T* rbuf, size_t rcount)
    {
        if (2 != size_) throw NotApplicable("two_procs requires exactly 2 processes");
        int remote = rank_ ^ 1;
        sendrecv(sbuf + (size_t)remote * scount, scount, dt_, remote, TAG_ALLTOALL,
                 rbuf + (size_t)remote * rcount, rcount, dt_, remote, TAG_ALLTOALL);
        copy_elems(rbuf + (size_t)rank_ * rcount, sbuf + (size_t)rank_ * scount, scount);
    }

    /// 6. basic linear (persistent requests, all posted at once)
    void linear(const T* sbuf, size_t scount, T* rbuf, size_t rcount)
    {
        int i, nreqs = 0;
        const int size = size_, rank = rank_;

        copy_elems(rbuf + (size_t)rank * rcount, sbuf + (size_t)rank * scount, scount);
        if (1 == size) return;

        std::vector<MPI_Request> req(2 * (size_t)(size - 1), MPI_REQUEST_NULL);
        MPI_Request* rreq = req.data();

        /* Post all receives first -- a simple optimization */
        for (nreqs = 0, i = (rank + 1) % size; i != rank; i = (i + 1) % size, ++rreq) {
            nreqs++;
            recv_init(rbuf + (size_t)i * rcount, rcount, dt_, i, TAG_ALLTOALL, rreq);
        }
        /* Now post all sends in reverse order */
        MPI_Request* sreq = rreq;
        for (i = (rank + size - 1) % size; i != rank; i = (i + size - 1) % size, ++sreq) {
            nreqs++;
            send_init(sbuf + (size_t)i * scount, scount, dt_, i, TAG_ALLTOALL, sreq);
        }

        MPI_CHECK(MPI_Startall(nreqs, req.data()));
        waitall(nreqs, req.data());
        for (i = 0; i < nreqs; ++i) MPI_Request_free(&req[i]);
    }

private:
    MPI_Datatype dt_;

    MPI_Datatype make_indexed(const std::vector<size_t>& blen, const std::vector<size_t>& disp)
    {
        MPI_Datatype t;
#if COLLAGE_LARGE_COUNT
        std::vector<MPI_Count> bl(blen.begin(), blen.end()), dp(disp.begin(), disp.end());
        MPI_CHECK(MPI_Type_indexed_c((MPI_Count)bl.size(), bl.data(), dp.data(), dt_, &t));
#else
        std::vector<int> bl, dp;
        for (size_t i = 0; i < blen.size(); ++i) { bl.push_back(cnt(blen[i])); dp.push_back(cnt(disp[i])); }
        MPI_CHECK(MPI_Type_indexed((int)bl.size(), bl.data(), dp.data(), dt_, &t));
#endif
        MPI_CHECK(MPI_Type_commit(&t));
        return t;
    }
};

#endif
