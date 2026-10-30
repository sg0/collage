#pragma once
#ifndef COLLAGE_BCAST_HPP
#define COLLAGE_BCAST_HPP

/// Algorithms adapted from Open MPI coll/base (coll_base_bcast.c @ ec987ee)
/// `segsize` is in bytes (0 = no segmentation), as in Open MPI.

#include "utils.hpp"

template <typename T>
class Bcast : public CollBase
{
public:
    explicit Bcast(MPI_Comm comm = MPI_COMM_WORLD) : CollBase(comm), dt_(TypeMap<T>()) {}

    /// generic segmented, pipelined tree broadcast (ompi_coll_base_bcast_intra_generic)
    void generic(T* buffer, size_t original_count, int root, size_t count_by_segment,
                 Tree const& tree)
    {
        int i, segindex, req_index;
        int num_segments;
        size_t sendcount;
        T* tmpbuf = buffer;
        MPI_Request recv_reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        std::vector<MPI_Request> send_reqs(std::max(tree.tree_nextsize, 1), MPI_REQUEST_NULL);

        if (size_ < 2) return;
        /* zero-size broadcast: run the algorithm with one empty segment */
        if (count_by_segment == 0) count_by_segment = 1;
        num_segments = (int)((original_count + count_by_segment - 1) / count_by_segment);
        if (num_segments < 1) num_segments = 1;
        const size_t realsegsize = count_by_segment;

        if (rank_ == root) {
            /* For each segment: send segment to all children. */
            sendcount = count_by_segment;
            for (segindex = 0; segindex < num_segments; segindex++) {
                if (segindex == (num_segments - 1))
                    sendcount = original_count - (size_t)segindex * count_by_segment;
                for (i = 0; i < tree.tree_nextsize; i++)
                    isend(tmpbuf, sendcount, dt_, tree.tree_next[i], TAG_BCAST, &send_reqs[i]);
                /* complete the sends before starting the next sends */
                waitall(tree.tree_nextsize, send_reqs.data());
                tmpbuf += realsegsize;
            }
        } else if (tree.tree_nextsize > 0) {
            /* Intermediate nodes: create the pipeline */
            req_index = 0;
            irecv(tmpbuf, count_by_segment, dt_, tree.tree_prev, TAG_BCAST, &recv_reqs[req_index]);

            for (segindex = 1; segindex < num_segments; segindex++) {
                req_index = req_index ^ 0x1;
                /* post new irecv */
                irecv(tmpbuf + realsegsize, count_by_segment, dt_, tree.tree_prev, TAG_BCAST,
                      &recv_reqs[req_index]);
                /* wait for and forward the previous segment to children */
                wait(&recv_reqs[req_index ^ 0x1]);
                for (i = 0; i < tree.tree_nextsize; i++)
                    isend(tmpbuf, count_by_segment, dt_, tree.tree_next[i], TAG_BCAST,
                          &send_reqs[i]);
                waitall(tree.tree_nextsize, send_reqs.data());
                tmpbuf += realsegsize;
            }
            /* Process the last segment */
            wait(&recv_reqs[req_index]);
            sendcount = original_count - (size_t)(num_segments - 1) * count_by_segment;
            for (i = 0; i < tree.tree_nextsize; i++)
                isend(tmpbuf, sendcount, dt_, tree.tree_next[i], TAG_BCAST, &send_reqs[i]);
            waitall(tree.tree_nextsize, send_reqs.data());
        } else {
            /* Leaf nodes: receive all segments from parent */
            req_index = 0;
            irecv(tmpbuf, count_by_segment, dt_, tree.tree_prev, TAG_BCAST, &recv_reqs[req_index]);
            for (segindex = 1; segindex < num_segments; segindex++) {
                req_index = req_index ^ 0x1;
                tmpbuf += realsegsize;
                irecv(tmpbuf, count_by_segment, dt_, tree.tree_prev, TAG_BCAST,
                      &recv_reqs[req_index]);
                wait(&recv_reqs[req_index ^ 0x1]);
            }
            wait(&recv_reqs[req_index]);
        }
    }

    /// 1. binary tree (segmented)
    void bintree(T* buffer, size_t count, int root, size_t segsize)
    {
        Tree tree = topo::build_tree(2, rank_, size_, root);
        size_t segcount = count;
        computed_segcount(segsize, sizeof(T), segcount);
        generic(buffer, count, root, segcount, tree);
    }

    /// 2. pipeline (chain with fanout 1, segmented)
    void pipeline(T* buffer, size_t count, int root, size_t segsize)
    {
        Tree tree = topo::build_chain(1, rank_, size_, root);
        size_t segcount = count;
        computed_segcount(segsize, sizeof(T), segcount);
        generic(buffer, count, root, segcount, tree);
    }

    /// 3. `chains` chains hanging off the root (segmented)
    void chain(T* buffer, size_t count, int root, size_t segsize, int chains)
    {
        Tree tree = topo::build_chain(chains, rank_, size_, root);
        size_t segcount = count;
        computed_segcount(segsize, sizeof(T), segcount);
        generic(buffer, count, root, segcount, tree);
    }

    /// 4. binomial tree (segmented)
    void binomial(T* buffer, size_t count, int root, size_t segsize)
    {
        Tree tree = topo::build_bmtree(rank_, size_, root);
        size_t segcount = count;
        computed_segcount(segsize, sizeof(T), segcount);
        generic(buffer, count, root, segcount, tree);
    }

    /// 5. split binary tree: each half of the message goes down one subtree,
    ///    then pairs exchange halves.
    void split_bintree(T* buffer, size_t count, int root, size_t segsize)
    {
        const int size = size_, rank = rank_;
        int segindex, i, lr, pair;
        size_t counts[2], segcount[2], sendcount[2], realsegsize[2];
        int num_segments[2];
        T* tmpbuf[2];
        MPI_Request base_req, new_req;
        const size_t type_size = sizeof(T);

        if (size == 1) return;

        Tree tree = topo::build_tree(2, rank, size, root);

        /* Determine number of segments and number of elements per segment */
        counts[0] = count / 2;
        if (count % 2 != 0) counts[0]++;
        counts[1] = count - counts[0];
        if (segsize > 0) {
            if (segsize < type_size) segsize = type_size; /* push segsize up to hold one type */
            segcount[0] = segcount[1] = segsize / type_size;
            num_segments[0] = (int)(counts[0] / segcount[0]);
            if ((counts[0] % segcount[0]) != 0) num_segments[0]++;
            num_segments[1] = (int)(counts[1] / segcount[1]);
            if ((counts[1] % segcount[1]) != 0) num_segments[1]++;
        } else {
            segcount[0] = counts[0];
            segcount[1] = counts[1];
            num_segments[0] = num_segments[1] = 1;
        }

        /* if the message is too small to be split into segments */
        if ((counts[0] == 0 || counts[1] == 0) || (segsize > counts[0] * type_size) ||
            (segsize > counts[1] * type_size)) {
            /* call linear version here ! */
            chain(buffer, count, root, segsize, 1);
            return;
        }

        realsegsize[0] = segcount[0];
        realsegsize[1] = segcount[1];
        tmpbuf[0] = buffer;
        tmpbuf[1] = buffer + counts[0];

        /* determine if I am left (0) or right (1), (root is right) */
        lr = ((rank + size - root) % size + 1) % 2;

        if (rank == root) {
            sendcount[0] = segcount[0];
            sendcount[1] = segcount[1];
            for (segindex = 0; segindex < num_segments[0]; segindex++) {
                for (i = 0; i < tree.tree_nextsize && i < 2; i++) {
                    if (segindex >= num_segments[i]) continue;
                    if (segindex == (num_segments[i] - 1))
                        sendcount[i] = counts[i] - (size_t)segindex * segcount[i];
                    send(tmpbuf[i], sendcount[i], dt_, tree.tree_next[i], TAG_BCAST);
                    tmpbuf[i] += realsegsize[i];
                }
            }
        } else if (tree.tree_nextsize > 0) {
            sendcount[lr] = segcount[lr];
            irecv(tmpbuf[lr], sendcount[lr], dt_, tree.tree_prev, TAG_BCAST, &base_req);
            for (segindex = 1; segindex < num_segments[lr]; segindex++) {
                if (segindex == (num_segments[lr] - 1))
                    sendcount[lr] = counts[lr] - (size_t)segindex * segcount[lr];
                irecv(tmpbuf[lr] + realsegsize[lr], sendcount[lr], dt_, tree.tree_prev,
                      TAG_BCAST, &new_req);
                wait(&base_req);
                for (i = 0; i < tree.tree_nextsize; i++)
                    send(tmpbuf[lr], segcount[lr], dt_, tree.tree_next[i], TAG_BCAST);
                base_req = new_req;
                tmpbuf[lr] += realsegsize[lr];
            }
            wait(&base_req);
            for (i = 0; i < tree.tree_nextsize; i++)
                send(tmpbuf[lr], sendcount[lr], dt_, tree.tree_next[i], TAG_BCAST);
        } else {
            /* leaf nodes: just consume segments as fast as possible */
            sendcount[lr] = segcount[lr];
            for (segindex = 0; segindex < num_segments[lr]; segindex++) {
                if (segindex == (num_segments[lr] - 1))
                    sendcount[lr] = counts[lr] - (size_t)segindex * segcount[lr];
                recv(tmpbuf[lr], sendcount[lr], dt_, tree.tree_prev, TAG_BCAST);
                tmpbuf[lr] += realsegsize[lr];
            }
        }

        /* reset the buffer pointers */
        tmpbuf[0] = buffer;
        tmpbuf[1] = buffer + counts[0];

        /* Step 2: exchange halves with the pair in the opposite subtree */
        if (lr == 0) pair = (rank + 1) % size;
        else pair = (rank + size - 1) % size;

        if ((size % 2) != 0 && rank != root) {
            sendrecv(tmpbuf[lr], counts[lr], dt_, pair, TAG_BCAST,
                     tmpbuf[(lr + 1) % 2], counts[(lr + 1) % 2], dt_, pair, TAG_BCAST);
        } else if ((size % 2) == 0) {
            if (rank == root) {
                /* root sends right buffer to the last node */
                send(tmpbuf[1], counts[1], dt_, (root + size - 1) % size, TAG_BCAST);
            } else if (rank == (root + size - 1) % size) {
                /* last node receives right buffer from the root */
                recv(tmpbuf[1], counts[1], dt_, root, TAG_BCAST);
            } else {
                sendrecv(tmpbuf[lr], counts[lr], dt_, pair, TAG_BCAST,
                         tmpbuf[(lr + 1) % 2], counts[(lr + 1) % 2], dt_, pair, TAG_BCAST);
            }
        }
    }

    /// 6. basic linear: root isends to everybody
    void linear(T* buff, size_t count, int root)
    {
        const int size = size_, rank = rank_;
        if (1 == size) return;
        if (rank != root) {
            recv(buff, count, dt_, root, TAG_BCAST);
            return;
        }
        std::vector<MPI_Request> reqs(size - 1, MPI_REQUEST_NULL);
        MPI_Request* preq = reqs.data();
        for (int i = 0; i < size; ++i) {
            if (i == rank) continue;
            isend(buff, count, dt_, i, TAG_BCAST, preq++);
        }
        waitall(size - 1, reqs.data());
    }

    /// 7. k-nomial tree (segmented)
    void knomial(T* buf, size_t count, int root, size_t segsize, int radix)
    {
        Tree tree = topo::build_kmtree(rank_, size_, root, radix);
        size_t segcount = count;
        computed_segcount(segsize, sizeof(T), segcount);
        generic(buf, count, root, segcount, tree);
    }

    /// 8. binomial scatter + recursive-doubling allgather (falls back to linear
    ///    when count < comm_size). Counts are handled as signed quantities
    ///    (the upstream size_t version can underflow for small counts).
    void scatter_allgather(T* buf, size_t count_, int root, size_t /*segsize*/)
    {
        MPI_Status status;
        const int comm_size = size_, rank = rank_;
        if (comm_size < 2) return;
        if (count_ < (size_t)comm_size) {
            linear(buf, count_, root);
            return;
        }
        const int64_t count = (int64_t)count_;
        int vrank = (rank - root + comm_size) % comm_size;
        int64_t recv_count = 0, send_count = 0;
        int64_t scatter_count = (count + comm_size - 1) / comm_size; /* ceil(count / comm_size) */
        int64_t curr_count = (rank == root) ? count : 0;

        /* Scatter by binomial tree: receive data from parent */
        int mask = 0x1;
        while (mask < comm_size) {
            if (vrank & mask) {
                int parent = (rank - mask + comm_size) % comm_size;
                recv_count = count - vrank * scatter_count;
                if (recv_count <= 0) {
                    curr_count = 0;
                } else {
                    recv(buf + vrank * scatter_count, (size_t)recv_count, dt_, parent, TAG_BCAST,
                         &status);
                    curr_count = (int64_t)get_count(&status, dt_);
                }
                break;
            }
            mask <<= 1;
        }

        /* Scatter by binomial tree: send data to child processes */
        mask >>= 1;
        while (mask > 0) {
            if (vrank + mask < comm_size) {
                send_count = curr_count - scatter_count * mask;
                if (send_count > 0) {
                    int child = (rank + mask) % comm_size;
                    send(buf + scatter_count * (vrank + mask), (size_t)send_count, dt_, child,
                         TAG_BCAST);
                    curr_count -= send_count;
                }
            }
            mask >>= 1;
        }

        /* Allgather by recursive doubling */
        int64_t rem_count = count - vrank * scatter_count;
        curr_count = (scatter_count < rem_count) ? scatter_count : rem_count;
        if (curr_count < 0) curr_count = 0;

        mask = 0x1;
        while (mask < comm_size) {
            int vremote = vrank ^ mask;
            int remote = (vremote + root) % comm_size;
            int vrank_tree_root = rounddown(vrank, mask);
            int vremote_tree_root = rounddown(vremote, mask);

            if (vremote < comm_size) {
                int64_t send_offset = vrank_tree_root * scatter_count;
                int64_t recv_offset = vremote_tree_root * scatter_count;
                recv_count = count - vremote_tree_root * scatter_count;
                if (recv_count < 0) recv_count = 0;
                sendrecv(buf + send_offset, (size_t)curr_count, dt_, remote, TAG_BCAST,
                         buf + recv_offset, (size_t)recv_count, dt_, remote, TAG_BCAST, &status);
                recv_count = (int64_t)get_count(&status, dt_);
                curr_count += recv_count;
            }

            /* Non-power-of-two case: if a process did not have a destination
             * process to communicate with, send it the current result
             * (recursive halving is used to find the process). */
            if (vremote_tree_root + mask > comm_size) {
                int nprocs_alldata = comm_size - vrank_tree_root - mask;
                int64_t offset = scatter_count * (vrank_tree_root + mask);
                for (int rhalving_mask = mask >> 1; rhalving_mask > 0; rhalving_mask >>= 1) {
                    vremote = vrank ^ rhalving_mask;
                    remote = (vremote + root) % comm_size;
                    int tree_root = rounddown(vrank, rhalving_mask << 1);
                    if ((vremote > vrank) && (vrank < tree_root + nprocs_alldata) &&
                        (vremote >= tree_root + nprocs_alldata)) {
                        send(buf + offset, (size_t)recv_count, dt_, remote, TAG_BCAST);
                    } else if ((vremote < vrank) && (vremote < tree_root + nprocs_alldata) &&
                               (vrank >= tree_root + nprocs_alldata)) {
                        /* upstream passes `count` as the bound; bound it by the
                         * space left in the buffer (may be 0 past the end) */
                        recv(buf + offset, (size_t)std::max<int64_t>(0, count - offset), dt_, remote, TAG_BCAST,
                             &status);
                        recv_count = (int64_t)get_count(&status, dt_);
                        curr_count += recv_count;
                    }
                }
            }
            mask <<= 1;
        }
    }

    /// 9. binomial scatter + ring allgather (falls back to linear when count < comm_size)
    void scatter_allgather_ring(T* buf, size_t count_, int root, size_t /*segsize*/)
    {
        MPI_Status status;
        const int comm_size = size_, rank = rank_;
        if (comm_size < 2) return;
        if (count_ < (size_t)comm_size) {
            linear(buf, count_, root);
            return;
        }
        const int64_t count = (int64_t)count_;
        int vrank = (rank - root + comm_size) % comm_size;
        int64_t recv_count = 0, send_count = 0;
        int64_t scatter_count = (count + comm_size - 1) / comm_size;
        int64_t curr_count = (rank == root) ? count : 0;

        int mask = 1;
        while (mask < comm_size) {
            if (vrank & mask) {
                int parent = (rank - mask + comm_size) % comm_size;
                recv_count = count - vrank * scatter_count;
                if (recv_count <= 0) {
                    curr_count = 0;
                } else {
                    recv(buf + vrank * scatter_count, (size_t)recv_count, dt_, parent, TAG_BCAST,
                         &status);
                    curr_count = (int64_t)get_count(&status, dt_);
                }
                break;
            }
            mask <<= 1;
        }

        mask >>= 1;
        while (mask > 0) {
            if (vrank + mask < comm_size) {
                send_count = curr_count - scatter_count * mask;
                if (send_count > 0) {
                    int child = (rank + mask) % comm_size;
                    send(buf + scatter_count * (vrank + mask), (size_t)send_count, dt_, child,
                         TAG_BCAST);
                    curr_count -= send_count;
                }
            }
            mask >>= 1;
        }

        /* Allgather by a ring algorithm */
        int left = (rank - 1 + comm_size) % comm_size;
        int right = (rank + 1) % comm_size;
        int send_block = vrank;
        int recv_block = (vrank - 1 + comm_size) % comm_size;

        for (int i = 1; i < comm_size; i++) {
            recv_count = (scatter_count < count - recv_block * scatter_count)
                             ? scatter_count
                             : count - recv_block * scatter_count;
            if (recv_count < 0) recv_count = 0;
            int64_t recv_offset = recv_block * scatter_count;

            send_count = (scatter_count < count - send_block * scatter_count)
                             ? scatter_count
                             : count - send_block * scatter_count;
            if (send_count < 0) send_count = 0;
            int64_t send_offset = send_block * scatter_count;

            sendrecv(buf + send_offset, (size_t)send_count, dt_, right, TAG_BCAST,
                     buf + recv_offset, (size_t)recv_count, dt_, left, TAG_BCAST);
            send_block = recv_block;
            recv_block = (recv_block - 1 + comm_size) % comm_size;
        }
    }

private:
    MPI_Datatype dt_;
};

#endif
