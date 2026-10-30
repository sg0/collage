#pragma once
#ifndef COLLAGE_ALLREDUCE_HPP
#define COLLAGE_ALLREDUCE_HPP

/// Algorithms adapted from Open MPI coll/base (coll_base_allreduce.c @ ec987ee)
/// The reduction operation `op` is any (predefined or user-defined) MPI_Op.

#include "allgather.hpp"
#include "bcast.hpp"
#include "reduce.hpp"
#include "utils.hpp"

template <typename T>
class Allreduce : public CollBase
{
public:
    explicit Allreduce(MPI_Comm comm = MPI_COMM_WORLD)
        : CollBase(comm), dt_(TypeMap<T>()), reduce_(comm), bcast_(comm), allgather_(comm) {}

    /// 1. non-overlapping: reduce to 0 followed by a broadcast from 0.
    ///    Upstream calls whatever reduce/bcast is selected for the communicator;
    ///    here the point-to-point binomial reduce (in-order binary for
    ///    non-commutative ops) and binomial bcast are used, with `segsize`.
    void nonoverlapping(const T* sbuf, T* rbuf, size_t count, MPI_Op op, size_t segsize,
                        int max_outstanding_reqs)
    {
        if (op_is_commute(op)) reduce_.binomial(sbuf, rbuf, count, op, 0, segsize, max_outstanding_reqs);
        else reduce_.in_order_binary(sbuf, rbuf, count, op, 0, segsize, max_outstanding_reqs);
        bcast_.binomial(rbuf, count, 0, segsize);
    }

    /// 2. recursive doubling (handles non power-of-two sizes and
    ///    non-commutative operations)
    void recursivedoubling(const T* sbuf, T* rbuf, size_t count, MPI_Op op)
    {
        int adjsize, remote, distance, newrank, newremote, extra_ranks;
        const int size = size_, rank = rank_;

        if (1 == size) {
            copy_elems(rbuf, sbuf, count);
            return;
        }

        std::vector<T> inplacebuf(count);
        copy_elems(inplacebuf.data(), sbuf, count);
        T* tmpsend = inplacebuf.data();
        T* tmprecv = rbuf;

        adjsize = next_poweroftwo(size);
        adjsize >>= 1;

        extra_ranks = size - adjsize;
        if (rank < (2 * extra_ranks)) {
            if (0 == (rank % 2)) {
                send(tmpsend, count, dt_, (rank + 1), TAG_ALLREDUCE);
                newrank = -1;
            } else {
                recv(tmprecv, count, dt_, (rank - 1), TAG_ALLREDUCE);
                /* tmpsend = tmprecv (op) tmpsend */
                reduce_local(tmprecv, tmpsend, count, dt_, op);
                newrank = rank >> 1;
            }
        } else {
            newrank = rank - extra_ranks;
        }

        for (distance = 0x1; distance < adjsize; distance <<= 1) {
            if (newrank < 0) break;
            newremote = newrank ^ distance;
            remote = (newremote < extra_ranks) ? (newremote * 2 + 1) : (newremote + extra_ranks);

            sendrecv_actual(tmpsend, count, dt_, remote, TAG_ALLREDUCE,
                            tmprecv, count, dt_, remote, TAG_ALLREDUCE);
            if (rank < remote) {
                /* tmprecv = tmpsend (op) tmprecv */
                reduce_local(tmpsend, tmprecv, count, dt_, op);
                std::swap(tmprecv, tmpsend);
            } else {
                /* tmpsend = tmprecv (op) tmpsend */
                reduce_local(tmprecv, tmpsend, count, dt_, op);
            }
        }

        if (rank < (2 * extra_ranks)) {
            if (0 == (rank % 2)) {
                recv(rbuf, count, dt_, (rank + 1), TAG_ALLREDUCE);
                tmpsend = rbuf;
            } else {
                send(tmpsend, count, dt_, (rank - 1), TAG_ALLREDUCE);
            }
        }

        /* Ensure that the final result is in rbuf */
        if (tmpsend != rbuf) copy_elems(rbuf, tmpsend, count);
    }

    /// 3. ring (reduce-scatter ring + allgather ring); count < size falls back
    ///    to recursive doubling. Commutative operations only.
    void ring(const T* sbuf, T* rbuf, size_t count, MPI_Op op)
    {
        int k, recv_from, send_to, inbi;
        size_t early_segcount, late_segcount, split_rank, max_segcount, block_count;
        size_t block_offset;
        const int size = size_, rank = rank_;
        MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

        if (1 == size) {
            copy_elems(rbuf, sbuf, count);
            return;
        }
        require_commutative(op, "ring");
        if (count < (size_t)size) {
            recursivedoubling(sbuf, rbuf, count, op);
            return;
        }

        compute_blockcount(count, size, split_rank, early_segcount, late_segcount);
        max_segcount = early_segcount;

        std::vector<T> inbuf_v[2];
        T* inbuf[2] = {nullptr, nullptr};
        inbuf_v[0].resize(max_segcount);
        inbuf[0] = inbuf_v[0].data();
        if (size > 2) {
            inbuf_v[1].resize(max_segcount);
            inbuf[1] = inbuf_v[1].data();
        }

        copy_elems(rbuf, sbuf, count);

        auto boff = [&](size_t b) {
            return (b < split_rank) ? (b * early_segcount) : (b * late_segcount + split_rank);
        };
        auto bcnt = [&](size_t b) { return (b < split_rank) ? early_segcount : late_segcount; };

        /* Computation loop */
        send_to = (rank + 1) % size;
        recv_from = (rank + size - 1) % size;

        inbi = 0;
        irecv(inbuf[inbi], max_segcount, dt_, recv_from, TAG_ALLREDUCE, &reqs[inbi]);
        block_offset = boff(rank);
        block_count = bcnt(rank);
        send(rbuf + block_offset, block_count, dt_, send_to, TAG_ALLREDUCE);

        for (k = 2; k < size; k++) {
            const int prevblock = (rank + size - k + 1) % size;
            inbi = inbi ^ 0x1;
            irecv(inbuf[inbi], max_segcount, dt_, recv_from, TAG_ALLREDUCE, &reqs[inbi]);
            wait(&reqs[inbi ^ 0x1]);
            /* rbuf[prevblock] = inbuf[inbi ^ 0x1] (op) rbuf[prevblock] */
            block_offset = boff(prevblock);
            block_count = bcnt(prevblock);
            T* tmprecv = rbuf + block_offset;
            reduce_local(inbuf[inbi ^ 0x1], tmprecv, block_count, dt_, op);
            send(tmprecv, block_count, dt_, send_to, TAG_ALLREDUCE);
        }

        /* Wait on the last block to arrive */
        wait(&reqs[inbi]);
        recv_from = (rank + 1) % size;
        block_offset = boff(recv_from);
        block_count = bcnt(recv_from);
        reduce_local(inbuf[inbi], rbuf + block_offset, block_count, dt_, op);

        /* Distribution loop - variation of ring allgather */
        send_to = (rank + 1) % size;
        recv_from = (rank + size - 1) % size;
        for (k = 0; k < size - 1; k++) {
            const int recv_data_from = (rank + size - k) % size;
            const int send_data_from = (rank + 1 + size - k) % size;
            sendrecv(rbuf + boff(send_data_from), bcnt(send_data_from), dt_, send_to, TAG_ALLREDUCE,
                     rbuf + boff(recv_data_from), max_segcount, dt_, recv_from, TAG_ALLREDUCE);
        }
    }

    /// 4. segmented ring (phases of segsize bytes); falls back to ring when
    ///    count < size * segcount. Commutative operations only.
    void ring_segmented(const T* sbuf, T* rbuf, size_t count, MPI_Op op, size_t segsize)
    {
        int k, recv_from, send_to, num_phases, phase;
        size_t early_blockcount, late_blockcount, split_rank, segcount, max_segcount, tmp1, tmp2;
        const int size = size_, rank = rank_;
        MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

        if (1 == size) {
            copy_elems(rbuf, sbuf, count);
            return;
        }
        require_commutative(op, "ring_segmented");

        segcount = count;
        computed_segcount(segsize, sizeof(T), segcount);
        if (segcount == 0 || count < (size_t)size * segcount) {
            ring(sbuf, rbuf, count, op);
            return;
        }

        num_phases = (int)(count / ((size_t)size * segcount));
        if ((count % ((size_t)size * segcount) >= (size_t)size) &&
            (count % ((size_t)size * segcount) > (((size_t)size * segcount) / 2))) {
            num_phases++;
        }

        compute_blockcount(count, size, split_rank, early_blockcount, late_blockcount);
        compute_blockcount(early_blockcount, num_phases, tmp1, max_segcount, tmp2);

        std::vector<T> inbuf_v[2];
        T* inbuf[2] = {nullptr, nullptr};
        inbuf_v[0].resize(max_segcount);
        inbuf[0] = inbuf_v[0].data();
        if (size > 2) {
            inbuf_v[1].resize(max_segcount);
            inbuf[1] = inbuf_v[1].data();
        }

        copy_elems(rbuf, sbuf, count);

        auto boff = [&](size_t b) {
            return (b < split_rank) ? (b * early_blockcount) : (b * late_blockcount + split_rank);
        };
        auto bcnt = [&](size_t b) { return (b < split_rank) ? early_blockcount : late_blockcount; };
        /* phase offset / count within a block of block_count elements */
        auto phase_part = [&](size_t block_count, int ph, size_t& phase_offset, size_t& phase_count) {
            size_t split_phase, early_phase_segcount, late_phase_segcount;
            compute_blockcount(block_count, num_phases, split_phase, early_phase_segcount,
                               late_phase_segcount);
            phase_count = (((size_t)ph < split_phase) ? early_phase_segcount : late_phase_segcount);
            phase_offset = (((size_t)ph < split_phase) ? ((size_t)ph * early_phase_segcount)
                                                       : ((size_t)ph * late_phase_segcount + split_phase));
        };

        for (phase = 0; phase < num_phases; phase++) {
            size_t phase_offset, phase_count, block_offset;
            int inbi;

            send_to = (rank + 1) % size;
            recv_from = (rank + size - 1) % size;

            inbi = 0;
            irecv(inbuf[inbi], max_segcount, dt_, recv_from, TAG_ALLREDUCE, &reqs[inbi]);

            block_offset = boff(rank);
            phase_part(bcnt(rank), phase, phase_offset, phase_count);
            send(rbuf + block_offset + phase_offset, phase_count, dt_, send_to, TAG_ALLREDUCE);

            for (k = 2; k < size; k++) {
                const int prevblock = (rank + size - k + 1) % size;
                inbi = inbi ^ 0x1;
                irecv(inbuf[inbi], max_segcount, dt_, recv_from, TAG_ALLREDUCE, &reqs[inbi]);
                wait(&reqs[inbi ^ 0x1]);

                block_offset = boff(prevblock);
                phase_part(bcnt(prevblock), phase, phase_offset, phase_count);
                T* tmprecv = rbuf + block_offset + phase_offset;
                reduce_local(inbuf[inbi ^ 0x1], tmprecv, phase_count, dt_, op);
                send(tmprecv, phase_count, dt_, send_to, TAG_ALLREDUCE);
            }

            wait(&reqs[inbi]);
            recv_from = (rank + 1) % size;
            block_offset = boff(recv_from);
            phase_part(bcnt(recv_from), phase, phase_offset, phase_count);
            reduce_local(inbuf[inbi], rbuf + block_offset + phase_offset, phase_count, dt_, op);
        }

        /* Distribution loop - variation of ring allgather */
        send_to = (rank + 1) % size;
        recv_from = (rank + size - 1) % size;
        for (k = 0; k < size - 1; k++) {
            const int recv_data_from = (rank + size - k) % size;
            const int send_data_from = (rank + 1 + size - k) % size;
            sendrecv(rbuf + boff(send_data_from), bcnt(send_data_from), dt_, send_to, TAG_ALLREDUCE,
                     rbuf + boff(recv_data_from), early_blockcount, dt_, recv_from, TAG_ALLREDUCE);
        }
    }

    /// 5. basic linear: linear reduce to 0 + linear bcast from 0
    void linear(const T* sbuf, T* rbuf, size_t count, MPI_Op op)
    {
        reduce_.linear(sbuf, rbuf, count, op, 0);
        bcast_.linear(rbuf, count, 0);
    }

    /// 6. Rabenseifner: reduce-scatter (recursive halving) + allgather
    ///    (recursive doubling). Non-commutative ops fall back to linear.
    void redscat_allgather(const T* sbuf, T* rbuf, size_t count, MPI_Op op)
    {
        const int comm_size = size_, rank = rank_;
        if (!op_is_commute(op)) {
            linear(sbuf, rbuf, count, op);
            return;
        }
        int nsteps = hibit(comm_size); /* ilog2(comm_size) */
        if (-1 == nsteps) throw std::runtime_error("redscat_allgather: bad communicator size");
        int nprocs_pof2 = 1 << nsteps;

        std::vector<T> tmp_buf(count);
        copy_elems(rbuf, sbuf, count);

        int vrank, step;
        size_t wsize;
        int nprocs_rem = comm_size - nprocs_pof2;
        if (rank < 2 * nprocs_rem) {
            size_t count_lhalf = count / 2;
            size_t count_rhalf = count - count_lhalf;
            if (rank % 2 != 0) {
                sendrecv(rbuf, count_lhalf, dt_, rank - 1, TAG_ALLREDUCE,
                         tmp_buf.data() + count_lhalf, count_rhalf, dt_, rank - 1, TAG_ALLREDUCE);
                reduce_local(tmp_buf.data() + count_lhalf, rbuf + count_lhalf, count_rhalf, dt_, op);
                send(rbuf + count_lhalf, count_rhalf, dt_, rank - 1, TAG_ALLREDUCE);
                vrank = -1;
            } else {
                sendrecv(rbuf + count_lhalf, count_rhalf, dt_, rank + 1, TAG_ALLREDUCE,
                         tmp_buf.data(), count_lhalf, dt_, rank + 1, TAG_ALLREDUCE);
                reduce_local(tmp_buf.data(), rbuf, count_lhalf, dt_, op);
                recv(rbuf + count_lhalf, count_rhalf, dt_, rank + 1, TAG_ALLREDUCE);
                vrank = rank / 2;
            }
        } else {
            vrank = rank - nprocs_rem;
        }

        std::vector<size_t> rindex(std::max(nsteps, 1)), sindex(std::max(nsteps, 1)),
            rcount(std::max(nsteps, 1)), scount(std::max(nsteps, 1));
        if (vrank != -1) {
            step = 0;
            wsize = count;
            sindex[0] = rindex[0] = 0;
            for (int mask = 1; mask < nprocs_pof2; mask <<= 1) {
                int vdest = vrank ^ mask;
                int dest = (vdest < nprocs_rem) ? vdest * 2 : vdest + nprocs_rem;
                if (rank < dest) {
                    rcount[step] = wsize / 2;
                    scount[step] = wsize - rcount[step];
                    sindex[step] = rindex[step] + rcount[step];
                } else {
                    scount[step] = wsize / 2;
                    rcount[step] = wsize - scount[step];
                    rindex[step] = sindex[step] + scount[step];
                }
                sendrecv(rbuf + sindex[step], scount[step], dt_, dest, TAG_ALLREDUCE,
                         tmp_buf.data() + rindex[step], rcount[step], dt_, dest, TAG_ALLREDUCE);
                reduce_local(tmp_buf.data() + rindex[step], rbuf + rindex[step], rcount[step], dt_, op);
                if (step + 1 < nsteps) {
                    rindex[step + 1] = rindex[step];
                    sindex[step + 1] = rindex[step];
                    wsize = rcount[step];
                    step++;
                }
            }

            /* Step 3. Allgather by the recursive doubling algorithm. */
            step = nsteps - 1;
            for (int mask = nprocs_pof2 >> 1; mask > 0; mask >>= 1) {
                int vdest = vrank ^ mask;
                int dest = (vdest < nprocs_rem) ? vdest * 2 : vdest + nprocs_rem;
                sendrecv(rbuf + rindex[step], rcount[step], dt_, dest, TAG_ALLREDUCE,
                         rbuf + sindex[step], scount[step], dt_, dest, TAG_ALLREDUCE);
                step--;
            }
        }

        /* Step 4. Send total result to excluded odd ranks. */
        if (rank < 2 * nprocs_rem) {
            if (rank % 2 != 0) recv(rbuf, count, dt_, rank - 1, TAG_ALLREDUCE);
            else send(rbuf, count, dt_, rank + 1, TAG_ALLREDUCE);
        }
    }

    /// 7. allgather + local reduce (trades bandwidth for latency).
    ///    Upstream uses the communicator's allgather; here the point-to-point
    ///    recursive-doubling allgather (k-Bruck radix 2 when not power of two).
    ///    Commutative operations only.
    void allgather_reduce(const T* sbuf, T* rbuf, size_t count, MPI_Op op)
    {
        require_commutative(op, "allgather_reduce");
        const int size = size_;
        std::vector<T> partial_buf(count * (size_t)size), tmpsend(count);
        copy_elems(tmpsend.data(), sbuf, count);

        allgather_.recursivedoubling(tmpsend.data(), count, partial_buf.data(), count);

        for (int target = 1; target < size; target++)
            reduce_local(partial_buf.data() + (size_t)target * count, partial_buf.data(), count, dt_, op);

        copy_elems(rbuf, partial_buf.data(), count);
    }

private:
    MPI_Datatype dt_;
    Reduce<T> reduce_;
    Bcast<T> bcast_;
    Allgather<T> allgather_;

    static void require_commutative(MPI_Op op, const char* algo)
    {
        if (!op_is_commute(op))
            throw NotApplicable(std::string(algo) + " requires a commutative operation");
    }
};

#endif
