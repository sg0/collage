#pragma once
#ifndef COLLAGE_VERIFY_HPP
#define COLLAGE_VERIFY_HPP

/*
 * Deep verification: find WHERE a collective went wrong, not only that it did.
 *
 *   --check-steps  every local reduction step of every algorithm is verified
 *                  against a host reference computed from the same inputs
 *                  (catches a PE that computes wrong: kernel, plugin, memory)
 *   --check-msgs   every point-to-point message is checksummed when posted
 *                  (sender) and when completed (receiver); after the call the
 *                  ranks exchange their records and compare them per ordered
 *                  pair, communicator, tag and sequence number (catches data
 *                  corrupted in transit: NIC, network, IPC, transport bug)
 *   --inject SPEC  deliberately corrupt one reduction result or one received
 *                  message, to prove that the checks catch and attribute it
 *
 * Both only run during the verification pass of --check (never while timing).
 * To read device data at the right point the host waits for the stream at
 * each checked step; that turns stream-triggered and device-initiated
 * transports into host-blocking semantics for the check pass, which every
 * algorithm is correct under (it is how plain MPI behaves).
 *
 * The final-result check (bench.hpp) adds an analytic reference and a
 * rank x block error map on top; see README, "Verification".
 */

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "memory.hpp"

namespace vfy {

/* position-sensitive 64-bit checksum of a byte stream (word-wise splitmix):
 * any changed bit, swapped word or shifted block changes it */
inline uint64_t mix64(uint64_t z)
{
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
inline uint64_t checksum(const void* p, size_t bytes)
{
    const unsigned char* b = static_cast<const unsigned char*>(p);
    uint64_t h = mix64(bytes);
    size_t w = 0;
    for (; (w + 1) * 8 <= bytes; ++w) {
        uint64_t v;
        std::memcpy(&v, b + w * 8, 8);
        h += mix64(v ^ mix64(w));
    }
    if (w * 8 < bytes) {
        uint64_t v = 0;
        std::memcpy(&v, b + w * 8, bytes - w * 8);
        h += mix64(v ^ mix64(w) ^ 0xa5a5a5a5ull);
    }
    return h;
}

struct Inject {
    enum Kind { None, Reduce, Msg } kind = None;
    int rank = -1, src = -1;  /* Reduce: rank; Msg: src -> rank (receiver) */
    uint64_t k = 1;           /* the k-th reduction / message from src on that rank (1-based) */
    uint64_t at = 0;          /* element (Reduce) or byte (Msg) to corrupt */
    bool fired = false;
};

/* one message as seen by one side */
struct MsgRec {
    int32_t cid, peer, tag, pad;
    uint64_t seq, bytes, hash;
};

struct PendingRecv {
    void* buf;
    size_t count;
    MPI_Datatype dt;
    MsgRec rec;
    bool done;
};

struct Region {
    const char* base;
    size_t bytes;
    std::string name;
};

struct State {
    bool steps = false, msgs = false; /* requested */
    bool active = false;              /* inside a verification pass */
    int rank = 0;                     /* rank in the benchmark communicator */
    std::string prefix;               /* "group 3: " with --split-mask */
    std::string label;                /* "ring, 4 KiB" */
    Inject inject;
    /* element formatting of the benchmark type */
    size_t esize = 0;
    std::string (*fmt)(const void*) = nullptr;
    std::vector<Region> regions;      /* named user buffers (sbuf, rbuf) */
    /* per pass */
    uint64_t nreduce = 0, nreduce_bad = 0;
    std::vector<std::string> errors;  /* this rank's findings (capped) */
    size_t errors_total = 0;
    std::vector<MsgRec> sent, recvd;
    std::vector<std::vector<char>> recvd_data; /* what arrived (for messages shorter than the receive) */
    std::vector<PendingRecv> pend;
    std::map<std::tuple<int, int, int>, uint64_t> sseq, rseq; /* (cid, peer, tag) -> next seq */
    std::map<std::pair<int, int>, uint64_t> rcount;           /* (cid, src) -> messages completed */
    uint64_t msgs_checked = 0, msgs_unverified = 0;
    int next_cid = 0;
};

inline State& st()
{
    static State s;
    return s;
}

inline bool steps_on() { return st().active && st().steps; }
inline bool msgs_on() { return st().active && st().msgs; }

inline void error(const std::string& e)
{
    State& s = st();
    ++s.errors_total;
    if (s.errors.size() < 16) s.errors.push_back(s.prefix + e);
}

/* "rbuf[12]" / "temporary+0x40 [3]" */
inline std::string where(const void* p, size_t elem)
{
    const State& s = st();
    const char* c = static_cast<const char*>(p);
    for (const Region& r : s.regions)
        if (c >= r.base && c < r.base + r.bytes && s.esize) {
            const size_t first = (size_t)(c - r.base) / s.esize;
            return r.name + "[" + std::to_string(first + elem) + "]";
        }
    return "temporary[" + std::to_string(elem) + "]";
}

inline std::string fmt_elem(const void* p)
{
    return st().fmt ? st().fmt(p) : std::string("?");
}

/* begin / end of one verification pass (all ranks of the communicator) */
inline void begin(const std::string& label)
{
    State& s = st();
    s.active = s.steps || s.msgs || s.inject.kind != Inject::None;
    s.label = label;
    s.nreduce = s.nreduce_bad = 0;
    s.errors.clear();
    s.errors_total = 0;
    s.sent.clear();
    s.recvd.clear();
    s.recvd_data.clear();
    s.pend.clear();
    s.sseq.clear();
    s.rseq.clear();
    s.rcount.clear();
    s.msgs_checked = s.msgs_unverified = 0;
    s.inject.fired = false;
}

/* ------------------------------------------------------------------------- */
/* step checks: local reductions                                              */
/* ------------------------------------------------------------------------- */
/* runs instead of mem::reduce while a pass with --check-steps or a reduce
 * injection is active (installed as mem::st().reduce_check) */
inline void checked_reduce(const void* in, void* inout, size_t count, MPI_Datatype dt, MPI_Op op)
{
    State& s = st();
    int tsize = 0;
    MPI_Type_size(dt, &tsize);
    const size_t bytes = count * (size_t)tsize;
    const uint64_t step = ++s.nreduce;
    std::vector<char> a, b, ref;
    if (s.steps) {
        a.resize(bytes);
        b.resize(bytes);
        mem::download(a.data(), in, bytes);   /* waits for the stream */
        mem::download(b.data(), inout, bytes);
        ref = b;
        mem::reduce_local_mpi(a.data(), ref.data(), count, dt, op); /* host reference */
    }
    mem::reduce_impl(in, inout, count, dt, op);  /* the real thing: kernel, plugin, MPI */

    if (s.inject.kind == Inject::Reduce && !s.inject.fired && s.inject.rank == s.rank && s.inject.k == step) {
        const size_t e = s.inject.at < count ? (size_t)s.inject.at : 0;
        std::vector<char> v((size_t)tsize);
        char* tgt = static_cast<char*>(inout) + e * (size_t)tsize;
        mem::download(v.data(), tgt, (size_t)tsize);
        for (int i = 0; i < tsize; ++i) v[(size_t)i] ^= (char)(i == 0 ? 0x5a : 0x01); /* a faulty PE */
        mem::upload(tgt, v.data(), (size_t)tsize);
        s.inject.fired = true;
    }
    if (!s.steps) return;

    std::vector<char> got(bytes);
    mem::download(got.data(), inout, bytes);
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < count; ++i)
        if (std::memcmp(got.data() + i * tsize, ref.data() + i * tsize, (size_t)tsize) != 0) {
            if (!bad) first = i;
            ++bad;
        }
    if (bad) {
        ++s.nreduce_bad;
        const size_t o = first * (size_t)tsize;
        error("rank " + std::to_string(s.rank) + ": compute: local reduction #" + std::to_string(step) + " (" +
              std::to_string(count) + " elements into " + where(inout, 0) + "): " + std::to_string(bad) +
              " wrong, first " + where(inout, first) + ": " + fmt_elem(a.data() + o) + " (op) " +
              fmt_elem(b.data() + o) + " gave " + fmt_elem(got.data() + o) + ", expected " + fmt_elem(ref.data() + o));
    }
}

/* predefined datatype? (large-count derived types need the _c query) */
inline bool is_named(MPI_Datatype dt)
{
    int comb = 0;
#if defined(MPI_VERSION) && MPI_VERSION >= 4 && !defined(COLLAGE_NO_LARGE_COUNT)
    MPI_Count ni = 0, na = 0, nc = 0, nd = 0;
    MPI_Type_get_envelope_c(dt, &ni, &na, &nc, &nd, &comb);
#else
    int ni = 0, na = 0, nd = 0;
    MPI_Type_get_envelope(dt, &ni, &na, &nd, &comb);
#endif
    return comb == MPI_COMBINER_NAMED;
}

/* ------------------------------------------------------------------------- */
/* message checks                                                             */
/* ------------------------------------------------------------------------- */
/* the bytes of `count` elements of `dt` at `buf`, packed (host copy) */
/* (a receive may name more elements than its buffer holds when the message
 * is shorter, as upstream's ring allreduce does: never read past the end of
 * a device block) */
inline std::vector<char> packed(const void* buf, size_t count, MPI_Datatype dt)
{
    std::vector<char> out;
    if (count == 0 || buf == nullptr) return out;
#if defined(MPI_VERSION) && MPI_VERSION >= 4 && !defined(COLLAGE_NO_LARGE_COUNT)
    MPI_Count lb = 0, ext = 0, tlb = 0, text = 0;
    MPI_Type_get_extent_c(dt, &lb, &ext);
    MPI_Type_get_true_extent_c(dt, &tlb, &text);
#else
    MPI_Aint lb = 0, ext = 0, tlb = 0, text = 0;
    MPI_Type_get_extent(dt, &lb, &ext);
    MPI_Type_get_true_extent(dt, &tlb, &text);
#endif
    const size_t room = mem::device() ? mem::remaining(buf) : SIZE_MAX;
    if (is_named(dt)) {
        out.resize(count * (size_t)ext);
        mem::download(out.data(), buf, std::min(out.size(), room));
        return out;
    }
    /* derived: copy the span it touches, then pack on the host */
    const size_t span = (size_t)((long long)(count - 1) * (long long)ext + (long long)text);
    std::vector<char> raw(span);
    const size_t room2 = mem::device() ? mem::remaining(static_cast<const char*>(buf) + tlb) : SIZE_MAX;
    mem::download(raw.data(), static_cast<const char*>(buf) + tlb, std::min(span, room2));
    char* base = raw.data() - tlb;
#if defined(MPI_VERSION) && MPI_VERSION >= 4 && !defined(COLLAGE_NO_LARGE_COUNT)
    MPI_Count psize = 0, pos = 0;
    MPI_Pack_size_c((MPI_Count)count, dt, MPI_COMM_SELF, &psize);
    out.resize((size_t)psize);
    MPI_Pack_c(base, (MPI_Count)count, dt, out.data(), psize, &pos, MPI_COMM_SELF);
#else
    int psize = 0, pos = 0;
    MPI_Pack_size((int)count, dt, MPI_COMM_SELF, &psize);
    out.resize((size_t)psize);
    MPI_Pack(base, (int)count, dt, out.data(), psize, &pos, MPI_COMM_SELF);
#endif
    out.resize((size_t)pos);
    return out;
}

/* a send is posted: its buffer holds exactly what will be sent */
inline void on_send(int cid, const void* buf, size_t count, MPI_Datatype dt, int dest, int tag)
{
    State& s = st();
    std::vector<char> p = packed(buf, count, dt);
    MsgRec r{};
    r.cid = cid;
    r.peer = dest;
    r.tag = tag;
    r.seq = s.sseq[std::make_tuple(cid, dest, tag)]++;
    r.bytes = p.size();
    r.hash = checksum(p.data(), p.size());
    s.sent.push_back(r);
}

/* a receive is posted (matching order = posting order); returns its index */
inline int on_recv_post(int cid, void* buf, size_t count, MPI_Datatype dt, int src, int tag)
{
    State& s = st();
    if (src == MPI_ANY_SOURCE || src == MPI_PROC_NULL || tag == MPI_ANY_TAG) return -1;
    PendingRecv pr{};
    pr.buf = buf;
    pr.count = count;
    /* the algorithm may free a derived datatype right after posting (legal
     * MPI): keep our own handle until the receive completes */
    if (is_named(dt)) pr.dt = dt;
    else MPI_Type_dup(dt, &pr.dt);
    pr.rec.cid = cid;
    pr.rec.peer = src;
    pr.rec.tag = tag;
    pr.rec.seq = s.rseq[std::make_tuple(cid, src, tag)]++;
    pr.done = false;
    s.pend.push_back(pr);
    return (int)s.pend.size() - 1;
}

/* a receive completed: checksum what arrived (after an injected corruption);
 * got = elements received, when the status told (else the posted count) */
inline void on_recv_done(int idx, long long got = -1)
{
    State& s = st();
    if (idx < 0 || (size_t)idx >= s.pend.size() || s.pend[(size_t)idx].done) return;
    PendingRecv& pr = s.pend[(size_t)idx];
    pr.done = true;
    if (got >= 0 && (size_t)got < pr.count) pr.count = (size_t)got;
    const uint64_t nth = ++s.rcount[std::make_pair(pr.rec.cid, pr.rec.peer)];
    std::vector<char> p = packed(pr.buf, pr.count, pr.dt);
    if (s.inject.kind == Inject::Msg && !s.inject.fired && s.inject.rank == s.rank && s.inject.src == pr.rec.peer &&
        s.inject.k == nth && !p.empty()) {
        /* flip a bit in the received data (in the user's buffer too) */
        const size_t at = s.inject.at < p.size() ? (size_t)s.inject.at : 0;
        p[at] ^= 0x10;
        if (is_named(pr.dt)) {
            mem::upload(static_cast<char*>(pr.buf) + at, p.data() + at, 1);
        } else {
            /* derived: corrupt the first byte of the buffer's span instead */
#if defined(MPI_VERSION) && MPI_VERSION >= 4 && !defined(COLLAGE_NO_LARGE_COUNT)
            MPI_Count tlb = 0, text = 0;
            MPI_Type_get_true_extent_c(pr.dt, &tlb, &text); /* the span starts here */
#else
            MPI_Aint tlb = 0, text = 0;
            MPI_Type_get_true_extent(pr.dt, &tlb, &text);
#endif
            char c;
            mem::download(&c, static_cast<char*>(pr.buf) + tlb, 1);
            c ^= 0x10;
            mem::upload(static_cast<char*>(pr.buf) + tlb, &c, 1);
            p = packed(pr.buf, pr.count, pr.dt);
        }
        s.inject.fired = true;
    }
    pr.rec.bytes = p.size();
    pr.rec.hash = checksum(p.data(), p.size());
    if (!is_named(pr.dt)) MPI_Type_free(&pr.dt);
    s.recvd.push_back(pr.rec);
    s.recvd_data.push_back(std::move(p));
}

/* after the call: exchange the records over `comm` and compare per pair */
inline void exchange(MPI_Comm comm)
{
    State& s = st();
    int P = 1, me = 0;
    MPI_Comm_size(comm, &P);
    MPI_Comm_rank(comm, &me);
    for (const PendingRecv& pr : s.pend)
        if (!pr.done)
            error("rank " + std::to_string(me) + ": transit: receive from rank " + std::to_string(pr.rec.peer) +
                  " (tag " + std::to_string(pr.rec.tag) + ", #" + std::to_string(pr.rec.seq) + ") never completed");
    /* send records to their destinations */
    std::vector<int> scount((size_t)P, 0), rcount((size_t)P, 0), sdisp((size_t)P, 0), rdisp((size_t)P, 0);
    std::vector<MsgRec> out(s.sent);
    std::stable_sort(out.begin(), out.end(), [](const MsgRec& a, const MsgRec& b) { return a.peer < b.peer; });
    for (const MsgRec& r : out)
        if (r.peer >= 0 && r.peer < P) scount[(size_t)r.peer] += (int)sizeof(MsgRec);
    MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, comm);
    int stot = 0, rtot = 0;
    for (int i = 0; i < P; ++i) {
        sdisp[(size_t)i] = stot;
        rdisp[(size_t)i] = rtot;
        stot += scount[(size_t)i];
        rtot += rcount[(size_t)i];
    }
    std::vector<MsgRec> in((size_t)rtot / sizeof(MsgRec));
    MPI_Alltoallv(out.data(), scount.data(), sdisp.data(), MPI_BYTE, in.data(), rcount.data(), rdisp.data(), MPI_BYTE,
                  comm);
    /* in: what each sender says it sent me (peer field = me); key by sender */
    std::map<std::tuple<int, int, int, uint64_t>, MsgRec> sent_to_me;
    for (int src = 0; src < P; ++src) {
        const size_t b = (size_t)rdisp[(size_t)src] / sizeof(MsgRec), n = (size_t)rcount[(size_t)src] / sizeof(MsgRec);
        for (size_t i = 0; i < n; ++i) {
            MsgRec r = in[b + i];
            r.peer = src;
            sent_to_me[std::make_tuple(r.cid, r.peer, r.tag, r.seq)] = r;
        }
    }
    for (size_t ri = 0; ri < s.recvd.size(); ++ri) {
        const MsgRec& r = s.recvd[ri];
        auto it = sent_to_me.find(std::make_tuple(r.cid, r.peer, r.tag, r.seq));
        const std::string who = "rank " + std::to_string(r.peer) + " -> rank " + std::to_string(me) + " (tag " +
                                std::to_string(r.tag) + ", message #" + std::to_string(r.seq) + " of that tag";
        if (it == sent_to_me.end()) {
            error("rank " + std::to_string(me) + ": transit: " + who + "): received, but the sender has no record");
            continue;
        }
        const MsgRec& snd = it->second;
        uint64_t got = r.hash;
        if (snd.bytes < r.bytes) /* a shorter message into a larger receive: compare what it filled */
            got = checksum(s.recvd_data[ri].data(), (size_t)snd.bytes);
        if (snd.bytes > r.bytes) {
            error("rank " + std::to_string(me) + ": transit: " + who + "): " + std::to_string(snd.bytes) +
                  " bytes sent into a " + std::to_string(r.bytes) + "-byte receive");
        } else if (snd.hash != got) {
            error("rank " + std::to_string(me) + ": transit: " + who + ", " + std::to_string(snd.bytes) +
                  " bytes): corrupted (checksum sent " + std::to_string(snd.hash) + ", received " +
                  std::to_string(got) + ")");
        } else {
            ++s.msgs_checked;
        }
        sent_to_me.erase(it);
    }
    /* sends nobody received (posted without a matching completed receive) */
    for (const auto& kv : sent_to_me) {
        const MsgRec& r = kv.second;
        error("rank " + std::to_string(me) + ": transit: rank " + std::to_string(r.peer) + " sent tag " +
              std::to_string(r.tag) + " message #" + std::to_string(r.seq) + " (" + std::to_string(r.bytes) +
              " bytes) that was never received here");
    }
}

/* ------------------------------------------------------------------------- */
/* reporting                                                                  */
/* ------------------------------------------------------------------------- */
/* gather every rank's findings to rank 0 of comm (collective); returns the
 * global number of findings, and the lines on rank 0 */
inline size_t collect(MPI_Comm comm, std::vector<std::string>& lines, uint64_t* msgs_checked = nullptr,
                      uint64_t* msgs_unverified = nullptr, uint64_t* reductions = nullptr)
{
    State& s = st();
    int me = 0, P = 1;
    MPI_Comm_rank(comm, &me);
    MPI_Comm_size(comm, &P);
    unsigned long long mine[4] = {(unsigned long long)s.errors_total, s.msgs_checked, s.msgs_unverified, s.nreduce};
    unsigned long long tot[4] = {0, 0, 0, 0};
    MPI_Allreduce(mine, tot, 4, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    if (msgs_checked) *msgs_checked = tot[1];
    if (msgs_unverified) *msgs_unverified = tot[2];
    if (reductions) *reductions = tot[3];
    if (tot[0] == 0) return 0;
    std::string text;
    for (const std::string& e : s.errors) text += e + "\n";
    if (s.errors_total > s.errors.size())
        text += "rank " + std::to_string(me) + ": ... " + std::to_string(s.errors_total - s.errors.size()) + " more\n";
    int len = (int)text.size();
    std::vector<int> lens((size_t)P, 0), disp((size_t)P, 0);
    MPI_Gather(&len, 1, MPI_INT, lens.data(), 1, MPI_INT, 0, comm);
    std::string all;
    if (me == 0) {
        int t = 0;
        for (int i = 0; i < P; ++i) {
            disp[(size_t)i] = t;
            t += lens[(size_t)i];
        }
        all.resize((size_t)t);
    }
    MPI_Gatherv(text.data(), len, MPI_CHAR, me == 0 ? &all[0] : nullptr, lens.data(), disp.data(), MPI_CHAR, 0, comm);
    if (me == 0) {
        size_t a = 0;
        while (a < all.size()) {
            size_t b = all.find('\n', a);
            if (b == std::string::npos) b = all.size();
            if (b > a) lines.push_back(all.substr(a, b - a));
            a = b + 1;
        }
    }
    return (size_t)tot[0];
}

inline void end()
{
    st().active = false;
}

/* --inject reduce:RANK:K[:ELEM] | msg:SRC:DST:K[:BYTE] */
inline Inject parse_inject(const std::string& spec)
{
    Inject in;
    std::vector<std::string> f;
    size_t a = 0;
    while (a <= spec.size()) {
        size_t b = spec.find(':', a);
        if (b == std::string::npos) b = spec.size();
        f.push_back(spec.substr(a, b - a));
        a = b + 1;
    }
    auto num = [&](size_t i) -> long long {
        if (i >= f.size() || f[i].empty()) throw std::invalid_argument("--inject " + spec + ": missing field");
        return std::stoll(f[i]);
    };
    if (f[0] == "reduce") {
        in.kind = Inject::Reduce;
        in.rank = (int)num(1);
        in.k = (uint64_t)num(2);
        in.at = f.size() > 3 ? (uint64_t)num(3) : 0;
    } else if (f[0] == "msg") {
        in.kind = Inject::Msg;
        in.src = (int)num(1);
        in.rank = (int)num(2);
        in.k = (uint64_t)num(3);
        in.at = f.size() > 4 ? (uint64_t)num(4) : 0;
    } else {
        throw std::invalid_argument("--inject: use reduce:RANK:K[:ELEM] or msg:SRC:DST:K[:BYTE]");
    }
    if (in.k < 1) throw std::invalid_argument("--inject: K counts from 1");
    return in;
}

} // namespace vfy

#endif /* COLLAGE_VERIFY_HPP */
