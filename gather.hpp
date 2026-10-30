#pragma once
#ifndef COLLAGE_GATHER_HPP
#define COLLAGE_GATHER_HPP

/// Algorithms adapted from Open MPI coll/base (coll_base_gather.c @ ec987ee)
/// Every rank contributes `count` elements; the root receives size*count.

#include "utils.hpp"

template <typename T>
class Gather : public CollBase
{
public:
    explicit Gather(MPI_Comm comm = MPI_COMM_WORLD) : CollBase(comm), dt_(TypeMap<T>()) {}

    /// 1. binomial tree (in-order binomial tree rooted at `root`)
    void binomial(const T* sbuf, size_t scount, T* rbuf, size_t rcount, int root)
    {
        int i, vrank;
        const int size = size_, rank = rank_;
        size_t total_recv = 0;
        T* ptmp = nullptr;
        std::vector<T> tempbuf;

        Tree bmtree = topo::build_in_order_bmtree(rank, size, root);
        vrank = (rank - root + size) % size;

        if (rank == root) {
            if (0 == root) {
                /* root on 0, just use the recv buffer */
                ptmp = rbuf;
                copy_elems(ptmp, sbuf, scount);
            } else {
                /* root is not on 0, allocate temp buffer for recv, rotate data at the end */
                tempbuf.resize(rcount * (size_t)size);
                ptmp = tempbuf.data();
                copy_elems(ptmp, sbuf, scount);
            }
            total_recv = rcount;
        } else if (!(vrank % 2)) {
            /* other non-leaf nodes, allocate temp buffer for data received from
             * children, the most we need is half of the total data elements due
             * to the property of binomial tree */
            tempbuf.resize(scount * (size_t)size);
            ptmp = tempbuf.data();
            copy_elems(ptmp, sbuf, scount);
            rcount = scount;
            total_recv = rcount;
        } else {
            /* leaf nodes, no temp buffer needed */
            ptmp = const_cast<T*>(sbuf);
            total_recv = scount;
        }

        if (!(vrank % 2)) {
            /* all non-leaf nodes recv from children */
            for (i = 0; i < bmtree.tree_nextsize; i++) {
                int mycount = 0, vkid;
                vkid = (bmtree.tree_next[i] - root + size) % size;
                mycount = vkid - vrank;
                if (mycount > (size - vkid)) mycount = size - vkid;
                size_t nrecv = (size_t)mycount * rcount;

                recv(ptmp + total_recv, rcount * (size_t)size - total_recv, dt_,
                     bmtree.tree_next[i], TAG_GATHER);
                total_recv += nrecv;
            }
        }

        if (rank != root) {
            /* all nodes except root send to parents */
            send(ptmp, total_recv, dt_, bmtree.tree_prev, TAG_GATHER);
        }

        if (rank == root && root != 0) {
            /* rotate received data on root if root != 0 */
            copy_elems(rbuf + (size_t)root * rcount, ptmp, rcount * (size_t)(size - root));
            copy_elems(rbuf, ptmp + rcount * (size_t)(size - root), rcount * (size_t)root);
        }
    }

    /// 2. linear with synchronisation: first segment (first_segment_size bytes)
    ///    is sent only after a zero-byte "go" message from the root.
    void linear_sync(const T* sbuf, size_t scount, T* rbuf, size_t rcount, int root,
                     size_t first_segment_size)
    {
        int i;
        const int size = size_, rank = rank_;
        size_t first_segment_count;

        if (rank != root) {
            first_segment_count = scount;
            computed_segcount(first_segment_size, sizeof(T), first_segment_count);

            recv(rbuf, 0, MPI_BYTE, root, TAG_GATHER);
            send(sbuf, first_segment_count, dt_, root, TAG_GATHER);
            send(sbuf + first_segment_count, scount - first_segment_count, dt_, root, TAG_GATHER);
        } else {
            std::vector<MPI_Request> reqs(size, MPI_REQUEST_NULL);
            MPI_Request first_segment_req;

            first_segment_count = rcount;
            computed_segcount(first_segment_size, sizeof(T), first_segment_count);

            for (i = 0; i < size; ++i) {
                if (i == rank) {
                    reqs[i] = MPI_REQUEST_NULL;
                    continue;
                }
                /* irecv for the first segment from i */
                T* ptmp = rbuf + (size_t)i * rcount;
                irecv(ptmp, first_segment_count, dt_, i, TAG_GATHER, &first_segment_req);
                /* send sync message */
                send(rbuf, 0, MPI_BYTE, i, TAG_GATHER);
                /* irecv for the second segment */
                ptmp = rbuf + (size_t)i * rcount + first_segment_count;
                irecv(ptmp, rcount - first_segment_count, dt_, i, TAG_GATHER, &reqs[i]);
                /* wait on the first segment to complete */
                wait(&first_segment_req);
            }
            /* copy local data */
            copy_elems(rbuf + (size_t)rank * rcount, sbuf, scount);
            /* wait all second segments to complete */
            waitall(size, reqs.data());
        }
    }

    /// 3. basic linear
    void linear(const T* sbuf, size_t scount, T* rbuf, size_t rcount, int root)
    {
        int i;
        const int size = size_, rank = rank_;
        if (rank != root) {
            send(sbuf, scount, dt_, root, TAG_GATHER);
            return;
        }
        T* ptmp = rbuf;
        for (i = 0; i < size; ++i, ptmp += rcount) {
            if (i == rank) copy_elems(ptmp, sbuf, scount);
            else recv(ptmp, rcount, dt_, i, TAG_GATHER);
        }
    }

    /// helper used by composite algorithms (allgatherv basic_default):
    /// linear gatherv as in Open MPI's coll/basic component.
    void linear_v(const T* sbuf, size_t scount, T* rbuf, std::vector<size_t> const& rcounts,
                  std::vector<size_t> const& displs, int root)
    {
        const int size = size_, rank = rank_;
        if (rank != root) {
            if (scount > 0) send(sbuf, scount, dt_, root, TAG_GATHERV);
            return;
        }
        std::vector<MPI_Request> reqs;
        reqs.reserve(size);
        for (int i = 0; i < size; ++i) {
            T* ptmp = rbuf + displs[i];
            if (i == rank) {
                copy_elems(ptmp, sbuf, std::min(scount, rcounts[i]));
            } else if (rcounts[i] > 0) {
                reqs.emplace_back(MPI_REQUEST_NULL);
                irecv(ptmp, rcounts[i], dt_, i, TAG_GATHERV, &reqs.back());
            }
        }
        waitall((int)reqs.size(), reqs.data());
    }

private:
    MPI_Datatype dt_;
};

#endif
