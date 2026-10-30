#pragma once
#ifndef COLLAGE_SCATTER_HPP
#define COLLAGE_SCATTER_HPP

/// Algorithms adapted from Open MPI coll/base (coll_base_scatter.c @ ec987ee)
/// The root sends `count` elements to every rank.

#include "utils.hpp"

template <typename T>
class Scatter : public CollBase
{
public:
    explicit Scatter(MPI_Comm comm = MPI_COMM_WORLD) : CollBase(comm), dt_(TypeMap<T>()) {}

    /// 1. binomial tree
    ///  Time complexity: alpha*log(p) + beta*m*(p-1)/p
    void binomial(const T* sbuf, size_t scount, T* rbuf, size_t rcount, int root)
    {
        int vrank;
        const int size = size_, rank = rank_;
        size_t curr_count;
        T* ptmp;
        std::vector<T> tempbuf;
        MPI_Status status;

        Tree bmtree = topo::build_in_order_bmtree(rank, size, root);

        vrank = (rank - root + size) % size;
        ptmp = rbuf; /* by default suppose leaf nodes, just use rbuf */

        if (vrank % 2) { /* leaves */
            recv(rbuf, rcount, dt_, bmtree.tree_prev, TAG_SCATTER, &status);
            return;
        }

        if (rank == root) { /* root and non-leafs */
            ptmp = const_cast<T*>(sbuf); /* if root == 0, just use the send buffer */
            if (0 != root) {
                tempbuf.resize(scount * (size_t)size);
                ptmp = tempbuf.data();
                /* rotate data so they will eventually be in the right place */
                copy_elems(ptmp + (size_t)(size - root) * scount, sbuf, scount * (size_t)root);
                copy_elems(ptmp, sbuf + (size_t)root * scount, scount * (size_t)(size - root));
            }
            curr_count = scount * (size_t)size;
        } else { /* (!(vrank % 2)) */
            scount = rcount;
            /* non-root, non-leaf nodes: allocate temp buffer for the subtree */
            int vparent = (bmtree.tree_prev - root + size) % size;
            int subtree_size = vrank - vparent;
            if (size - vrank < subtree_size) subtree_size = size - vrank;
            size_t packed_size = scount * (size_t)subtree_size;

            tempbuf.resize(packed_size);
            ptmp = tempbuf.data();

            /* recv from parent on non-root */
            recv(ptmp, packed_size, dt_, bmtree.tree_prev, TAG_SCATTER, &status);
            curr_count = get_count(&status, dt_);
        }

        /* local copy to rbuf */
        copy_elems(rbuf, ptmp, std::min(scount, rcount));

        /* send to children on all non-leaf */
        for (int i = bmtree.tree_nextsize - 1; i >= 0; i--) {
            int vchild = (bmtree.tree_next[i] - root + size) % size;
            int send_count = vchild - vrank;
            if (send_count > size - vchild) send_count = size - vchild;
            size_t nsend = (size_t)send_count * scount;
            send(ptmp + (curr_count - nsend), nsend, dt_, bmtree.tree_next[i], TAG_SCATTER);
            curr_count -= nsend;
        }
    }

    /// 2. basic linear
    void linear(const T* sbuf, size_t scount, T* rbuf, size_t rcount, int root)
    {
        int i;
        const int size = size_, rank = rank_;
        if (rank != root) {
            recv(rbuf, rcount, dt_, root, TAG_SCATTER);
            return;
        }
        const T* ptmp = sbuf;
        for (i = 0; i < size; ++i, ptmp += scount) {
            if (i == rank) copy_elems(rbuf, ptmp, std::min(scount, rcount));
            else send(ptmp, scount, dt_, i, TAG_SCATTER);
        }
    }

    /// 3. linear with non-blocking sends; every max_reqs-th send is blocking
    ///    (acts as a local resource flush).
    void linear_nb(const T* sbuf, size_t scount, T* rbuf, size_t rcount, int root, int max_reqs)
    {
        int i, nreqs;
        const int size = size_, rank = rank_;

        if (rank != root) {
            recv(rbuf, rcount, dt_, root, TAG_SCATTER);
            return;
        }
        if (max_reqs <= 1) {
            max_reqs = 0;
            nreqs = size - 1; /* no send for myself */
        } else {
            nreqs = size - (size / max_reqs);
        }
        std::vector<MPI_Request> reqs(std::max(nreqs, 1), MPI_REQUEST_NULL);
        MPI_Request* preq = reqs.data();

        const T* ptmp = sbuf;
        for (i = 0; i < size; ++i, ptmp += scount) {
            if (i == rank) {
                copy_elems(rbuf, ptmp, std::min(scount, rcount));
            } else {
                if (!max_reqs || (i % max_reqs)) isend(ptmp, scount, dt_, i, TAG_SCATTER, preq++);
                else send(ptmp, scount, dt_, i, TAG_SCATTER);
            }
        }
        waitall((int)(preq - reqs.data()), reqs.data());
    }

    /// helper used by composite algorithms (reduce_scatter nonoverlapping):
    /// linear scatterv as in Open MPI's coll/basic component.
    void linear_v(const T* sbuf, std::vector<size_t> const& scounts,
                  std::vector<size_t> const& displs, T* rbuf, size_t rcount, int root)
    {
        const int size = size_, rank = rank_;
        if (rank != root) {
            if (rcount > 0) recv(rbuf, rcount, dt_, root, TAG_SCATTERV);
            return;
        }
        std::vector<MPI_Request> reqs;
        reqs.reserve(size);
        for (int i = 0; i < size; ++i) {
            const T* ptmp = sbuf + displs[i];
            if (i == rank) {
                if (rbuf != ptmp) copy_elems(rbuf, ptmp, std::min(scounts[i], rcount));
            } else if (scounts[i] > 0) {
                reqs.emplace_back(MPI_REQUEST_NULL);
                isend(ptmp, scounts[i], dt_, i, TAG_SCATTERV, &reqs.back());
            }
        }
        waitall((int)reqs.size(), reqs.data());
    }

private:
    MPI_Datatype dt_;
};

#endif
