#pragma once
#ifndef COLLAGE_WIRE_HPP
#define COLLAGE_WIRE_HPP

/*
 * Wire accounting: the bytes each rank sends to each other rank in one call
 * of an algorithm (wirebw, traffic factor; see README "Bandwidth").
 *
 * Counting is switched on only for one untimed call per (algorithm, size).
 * In the timed iterations the send wrappers test one flag (on) and nothing
 * else happens.
 *
 * Every send of every algorithm goes through CollBase::note_send, which
 * calls add() with the destination's MPI_COMM_WORLD rank, so algorithms that
 * use several communicators (composites, --split-mask groups) are counted
 * the same way. Messages to self are not on any wire and are not counted.
 * Receives need no counting: what rank r receives is what the others sent
 * to r, summed in collect().
 */

#include <mpi.h>

#include <algorithm>
#include <vector>

namespace wire {

struct State {
    bool on = false;
    std::vector<double> to;  /* bytes sent to each world rank during the counted call */
};

inline State& st()
{
    static State s;
    return s;
}

inline void add(int world_dest, double bytes)
{
    State& s = st();
    if (world_dest >= 0 && (size_t)world_dest < s.to.size()) s.to[(size_t)world_dest] += bytes;
}

inline void begin(MPI_Comm world)
{
    int n = 0;
    MPI_Comm_size(world, &n);
    State& s = st();
    s.to.assign((size_t)n, 0.0);
    s.on = true;
}

inline void end() { st().on = false; }

/* this rank's traffic of the counted call */
struct Traffic {
    double sent = 0, recv = 0;            /* bytes, to / from other ranks */
    double sent_inter = 0, recv_inter = 0; /* of those, to / from ranks on other nodes */
    double port() const { return std::max(sent, recv); }       /* full-duplex port: the busier direction */
    double port_inter() const { return std::max(sent_inter, recv_inter); }
};

/* collective over world; node_of: world rank -> node (empty: one node) */
inline Traffic collect(MPI_Comm world, const std::vector<int>& node_of)
{
    State& s = st();
    int me = 0, n = 0;
    MPI_Comm_rank(world, &me);
    MPI_Comm_size(world, &n);
    auto node = [&](int r) { return node_of.empty() ? 0 : node_of[(size_t)r]; };
    /* split each destination's bytes into intra- and inter-node, then let
     * every rank sum what was sent to it */
    std::vector<double> both(2 * (size_t)n, 0.0);
    Traffic t;
    for (int r = 0; r < n; ++r) {
        const double b = s.to[(size_t)r];
        if (b == 0 || r == me) continue;
        const bool inter = node(r) != node(me);
        t.sent += b;
        if (inter) t.sent_inter += b;
        both[2 * (size_t)r] = b;
        both[2 * (size_t)r + 1] = inter ? b : 0.0;
    }
    double got[2] = {0, 0};
    MPI_Reduce_scatter_block(both.data(), got, 2, MPI_DOUBLE, MPI_SUM, world);
    t.recv = got[0];
    t.recv_inter = got[1];
    return t;
}

} // namespace wire

#endif /* COLLAGE_WIRE_HPP */
