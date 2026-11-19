#pragma once
#ifndef COLLAGE_REPLAY_HPP
#define COLLAGE_REPLAY_HPP

/*
 * Host-driven trace replayer (--replay DIR; the format is in trace.hpp).
 *
 * Every recorded operation goes through the same CollBase wrappers the
 * algorithms use, so a replay runs on any transport (mpi, cray, mpich, acx,
 * ucp, shmem) and memory space (--gpu or not), whatever the trace was recorded
 * on. One communicator is duplicated per recorded communicator id, created
 * collectively in id order, so messages match as they did when recorded.
 */

#include <map>
#include <memory>
#include <vector>

#include "trace.hpp"
#include "utils.hpp"

namespace trc {

class Replayer {
    /* a CollBase whose wrappers the replayer may call */
    struct Chan : CollBase {
        explicit Chan(MPI_Comm c) : CollBase(c) {}
        void s(const void* b, size_t n, MPI_Datatype d, int p, int t, Request* r) const { isend(b, n, d, p, t, r); }
        void z(const void* b, size_t n, MPI_Datatype d, int p, int t, Request* r) const { issend(b, n, d, p, t, r); }
        void r(void* b, size_t n, MPI_Datatype d, int p, int t, Request* q) const { irecv(b, n, d, p, t, q); }
        static void w(Request* q) { wait(q); }
        static void wa(int n, Request* q) { waitall(n, q); }
        static void wy(int n, Request* q, int* i) { waitany(n, q, i); }
        static void x(const void* a, void* b, size_t n, MPI_Datatype d, MPI_Op o) { reduce_local(a, b, n, d, o); }
    };

public:
    /* collective over comm */
    Replayer(MPI_Comm comm, const Trace& t, MPI_Datatype bdt, MPI_Op op) : t_(t), bdt_(bdt), op_(op)
    {
        int maxc = -1;
        for (const Call& c : t.calls)
            for (int x : c.cids) maxc = std::max(maxc, x);
        MPI_Allreduce(MPI_IN_PLACE, &maxc, 1, MPI_INT, MPI_MAX, comm);
        std::vector<int> used((size_t)(maxc + 1), 0);
        for (const Call& c : t.calls)
            for (int x : c.cids) used[(size_t)x] = 1;
        if (maxc >= 0) MPI_Allreduce(MPI_IN_PLACE, used.data(), maxc + 1, MPI_INT, MPI_MAX, comm);
        for (int x = 0; x <= maxc; ++x)
            if (used[(size_t)x]) chans_[x] = std::make_unique<Chan>(comm);
        /* derived datatypes: built once, outside the timed region */
        dts_.resize(t.calls.size());
        for (size_t ci = 0; ci < t.calls.size(); ++ci) {
            const Call& c = t.calls[ci];
            dts_[ci].assign(c.ops.size(), MPI_DATATYPE_NULL);
            for (size_t oi = 0; oi < c.ops.size(); ++oi) {
                const Op& o = c.ops[oi];
                if ((o.k == 'S' || o.k == 'Z' || o.k == 'R') && o.dt == 'D') dts_[ci][oi] = make_type(o.blocks);
            }
        }
    }
    ~Replayer()
    {
        for (auto& v : dts_)
            for (MPI_Datatype& d : v)
                if (d != MPI_DATATYPE_NULL) MPI_Type_free(&d);
    }
    Replayer(const Replayer&) = delete;
    Replayer& operator=(const Replayer&) = delete;

    /* run one recorded call on sbuf / rbuf */
    void run(const Call& c, void* sbuf, void* rbuf) const
    {
        const size_t ci = (size_t)(&c - t_.calls.data());
        std::vector<Buf<char>> tmp((size_t)c.nbuf);
        std::vector<Request> rq(c.nreq);
        auto base = [&](int id) -> char* {
            if (id == 0) return static_cast<char*>(sbuf);
            if (id == 1) return static_cast<char*>(rbuf);
            if (id < 0) return nullptr;
            return tmp[(size_t)id].data();
        };
        for (size_t oi = 0; oi < c.ops.size(); ++oi) {
            const Op& o = c.ops[oi];
            switch (o.k) {
                case 'A': tmp[(size_t)o.a].resize((size_t)o.n); break;
                case 'F': tmp[(size_t)o.a] = Buf<char>(); break;
                case 'S': case 'Z': case 'R': {
                    char* p = o.buf < 0 ? nullptr : base(o.buf) + o.off;
                    MPI_Datatype d = o.dt == 'T' ? bdt_ : o.dt == 'B' ? MPI_BYTE : dts_[ci][oi];
                    const size_t n = (size_t)o.n;
                    const int peer = o.c < 0 ? MPI_ANY_SOURCE : o.c;
                    const Chan& ch = *chans_.at(o.b);
                    if (o.k == 'S') ch.s(p, n, d, peer, o.d, &rq[(size_t)o.a]);
                    else if (o.k == 'Z') ch.z(p, n, d, peer, o.d, &rq[(size_t)o.a]);
                    else ch.r(p, n, d, peer, o.d, &rq[(size_t)o.a]);
                    break;
                }
                case 'W':
                    if (o.reqs.size() == 1) {
                        Chan::w(&rq[o.reqs[0]]);
                    } else {
                        std::vector<Request> g;
                        for (uint32_t i : o.reqs) g.push_back(rq[i]);
                        Chan::wa((int)g.size(), g.data());
                        for (size_t j = 0; j < o.reqs.size(); ++j) rq[o.reqs[j]] = g[j];
                    }
                    break;
                case 'Y': {
                    std::vector<Request> g;
                    for (uint32_t i : o.reqs) g.push_back(rq[i]);
                    int idx = MPI_UNDEFINED;
                    Chan::wy((int)g.size(), g.data(), &idx);
                    for (size_t j = 0; j < o.reqs.size(); ++j) rq[o.reqs[j]] = g[j];
                    break;
                }
                case 'X': Chan::x(base(o.a) + o.off, base(o.c) + o.off2, (size_t)o.n, bdt_, op_); break;
                case 'C': mem::copy(base(o.a) + o.off, base(o.c) + o.off2, (size_t)o.n); break;
                default: break;
            }
        }
        /* a recorded call leaves nothing in flight; be safe anyway */
        std::vector<Request> left;
        for (Request& r : rq)
            if (r.k != Request::None) left.push_back(r);
        if (!left.empty()) Chan::wa((int)left.size(), left.data());
    }

private:
    const Trace& t_;
    MPI_Datatype bdt_;
    MPI_Op op_;
    std::map<int, std::unique_ptr<Chan>> chans_;
    std::vector<std::vector<MPI_Datatype>> dts_;

    /* byte blocks -> an indexed type over MPI_BYTE (what the transports flatten) */
    static MPI_Datatype make_type(const std::vector<long long>& b)
    {
        const size_t n = b.size() / 2;
        MPI_Datatype t;
#if defined(MPI_VERSION) && MPI_VERSION >= 4 && !defined(COLLAGE_NO_LARGE_COUNT)
        std::vector<MPI_Count> bl(n), dp(n);
        for (size_t i = 0; i < n; ++i) { dp[i] = (MPI_Count)b[2 * i]; bl[i] = (MPI_Count)b[2 * i + 1]; }
        MPI_CHECK(MPI_Type_indexed_c((MPI_Count)n, bl.data(), dp.data(), MPI_BYTE, &t));
#else
        std::vector<int> bl(n), dp(n);
        for (size_t i = 0; i < n; ++i) { dp[i] = (int)b[2 * i]; bl[i] = (int)b[2 * i + 1]; }
        MPI_CHECK(MPI_Type_indexed((int)n, bl.data(), dp.data(), MPI_BYTE, &t));
#endif
        MPI_CHECK(MPI_Type_commit(&t));
        return t;
    }
};

} // namespace trc

#endif /* COLLAGE_REPLAY_HPP */
