#pragma once
#ifndef COLLAGE_TRACE_HPP
#define COLLAGE_TRACE_HPP

/*
 * Communication traces: record what an algorithm does, replay it later.
 *
 *   --trace DIR    one extra pass per (algorithm, size), before the
 *                  verification pass: every rank writes the operations it
 *                  performs to DIR/rank-NNNNN.trace
 *   --replay DIR   instead of the algorithms, run the recorded traces: each
 *                  becomes a variant "replay:<algorithm>", timed and verified
 *                  like any other, on whatever transport and memory space
 *                  this run uses
 *
 * A trace is the algorithm with all of its decisions taken: what to send to
 * whom and when, what to wait for, what to reduce and copy, in program order.
 * Replaying it needs no algorithm code, so the same message pattern can be
 * moved between transports (record on the host, replay with --gpu
 * --transport cray), MPI libraries or machines of the same size.
 *
 * Recorded per rank, in program order (text, one operation per line):
 *
 *   call LABEL BASE BYTES           one (algorithm, size); BASE = algorithm name
 *   A id bytes / F id               a temporary buffer is allocated / freed
 *   S|Z|R req cid peer tag buf off dt count [nblocks (off len)...]
 *                                   isend / issend (synchronous) / irecv;
 *                                   blocking calls appear as the nonblocking
 *                                   call + W. peer -1 = MPI_ANY_SOURCE.
 *                                   dt: T (the benchmark type), B (bytes),
 *                                   D (a derived type: byte blocks relative
 *                                   to buf+off; count is 1)
 *   W n req...                      wait for these requests (n > 1: waitall)
 *   Y n req...                      wait for any one of these (waitany)
 *   X in off io off count           local reduction io = in (op) io, in the
 *                                   benchmark type and op
 *   C dst off src off bytes         local copy
 *   end
 *
 * Buffers: 0 = sbuf, 1 = rbuf, 2... = temporaries, -1 = none. Requests and
 * temporaries are numbered per call; cid numbers the communicators (each
 * collective object has its own duplicate) in creation order.
 *
 * What a replay reproduces and what it does not: the operation sequence,
 * sizes, peers, tags, communicators and dependencies, exactly (and
 * --check verifies the result). Persistent requests replay as one
 * nonblocking operation per start; MPI_Waitany replays as a waitany over the
 * same requests (the operations that followed it are the ones recorded);
 * algorithms whose control flow depends on received counts replay the path
 * they took.
 */

#include <mpi.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "memory.hpp"

namespace trc {

struct State {
    bool active = false;
    FILE* f = nullptr;
    /* the call being recorded */
    MPI_Datatype bdt = MPI_DATATYPE_NULL; /* the benchmark type */
    const char* sbase = nullptr;
    size_t sbytes = 0;
    const char* rbase = nullptr;
    size_t rbytes = 0;
    std::map<const char*, std::pair<int, size_t>> bufs; /* temporaries: base -> (id, bytes) */
    int next_buf = 2;
    uint32_t next_req = 0;
    uint64_t nops = 0;
    uint64_t total_ops = 0, total_calls = 0;
};

inline State& st()
{
    static State s;
    return s;
}
inline bool on() { return st().active; }

/* pointer -> (buffer id, offset) */
inline std::pair<int, long long> where(const void* p)
{
    State& s = st();
    if (!p) return {-1, 0};
    const char* c = static_cast<const char*>(p);
    if (s.rbase && c >= s.rbase && c < s.rbase + s.rbytes) return {1, (long long)(c - s.rbase)};
    if (s.sbase && c >= s.sbase && c < s.sbase + s.sbytes) return {0, (long long)(c - s.sbase)};
    auto it = s.bufs.upper_bound(c);
    if (it != s.bufs.begin()) {
        --it;
        if (c >= it->first && c <= it->first + it->second.second) return {it->second.first, (long long)(c - it->first)};
    }
    /* one past the end of a user buffer (a zero-byte message at the end) */
    if (s.rbase && c == s.rbase + s.rbytes) return {1, (long long)s.rbytes};
    if (s.sbase && c == s.sbase + s.sbytes) return {0, (long long)s.sbytes};
    return {-2, 0};
}

inline void line(const std::string& l)
{
    State& s = st();
    std::fputs(l.c_str(), s.f);
    std::fputc('\n', s.f);
    ++s.nops;
}

/* predefined or indexed-of-predefined types, as byte blocks (off, len) */
inline bool flatten(MPI_Datatype dt, size_t count, std::vector<long long>& blocks, uint64_t& bytes)
{
    int comb = 0;
#if defined(MPI_VERSION) && MPI_VERSION >= 4 && !defined(COLLAGE_NO_LARGE_COUNT)
    MPI_Count ni = 0, na = 0, nc = 0, nd = 0;
    MPI_Type_get_envelope_c(dt, &ni, &na, &nc, &nd, &comb);
#else
    int ni = 0, na = 0, nd = 0, nc = 0;
    MPI_Type_get_envelope(dt, &ni, &na, &nd, &comb);
#endif
    if (comb == MPI_COMBINER_NAMED) {
        int sz = 0;
        MPI_Type_size(dt, &sz);
        bytes = (uint64_t)sz * count;
        return false;
    }
    if (comb != MPI_COMBINER_INDEXED) throw std::runtime_error("--trace: only predefined and indexed datatypes");
    std::vector<int> ints((size_t)ni);
    std::vector<MPI_Aint> aints((size_t)na);
    std::vector<MPI_Datatype> types((size_t)nd);
    std::vector<long long> cnts;
#if defined(MPI_VERSION) && MPI_VERSION >= 4 && !defined(COLLAGE_NO_LARGE_COUNT)
    std::vector<MPI_Count> counts((size_t)nc);
    MPI_Type_get_contents_c(dt, ni, na, nc, nd, ints.data(), aints.data(), counts.data(), types.data());
    if (nc > 0) cnts.assign(counts.begin(), counts.end());
    else cnts.assign(ints.begin(), ints.end());
#else
    MPI_Type_get_contents(dt, ni, na, nd, ints.data(), aints.data(), types.data());
    cnts.assign(ints.begin(), ints.end());
#endif
    MPI_Aint olb = 0, oext = 0, lb = 0, ext = 0;
    MPI_Type_get_extent(types[0], &olb, &oext);
    MPI_Type_get_extent(dt, &lb, &ext);
    const long long n = cnts[0];
    blocks.clear();
    bytes = 0;
    for (size_t rep = 0; rep < count; ++rep)
        for (long long i = 0; i < n; ++i) {
            const long long len = cnts[(size_t)(1 + i)] * (long long)oext;
            const long long off = (long long)rep * (long long)ext + cnts[(size_t)(1 + n + i)] * (long long)oext;
            if (len == 0) continue;
            if (!blocks.empty() && blocks[blocks.size() - 2] + blocks.back() == off) blocks.back() += len;
            else {
                blocks.push_back(off);
                blocks.push_back(len);
            }
            bytes += (uint64_t)len;
        }
    return true;
}

/* ---- recording ---------------------------------------------------------- */
inline void rec_p2p(char kind, int cid, const void* buf, size_t count, MPI_Datatype dt, int peer, int tag,
                    uint32_t* req_id)
{
    State& s = st();
    const uint32_t id = s.next_req++;
    *req_id = id;
    auto w = where(buf);
    std::string l = std::string(1, kind) + " " + std::to_string(id) + " " + std::to_string(cid) + " " +
                    std::to_string(peer == MPI_ANY_SOURCE ? -1 : peer) + " " + std::to_string(tag) + " ";
    std::vector<long long> blocks;
    uint64_t bytes = 0;
    const bool derived = flatten(dt, count, blocks, bytes);
    if (bytes > 0 && w.first == -2) throw std::runtime_error("--trace: a buffer outside sbuf, rbuf and temporaries");
    if (bytes == 0) w = {-1, 0};
    l += std::to_string(w.first) + " " + std::to_string(w.second) + " ";
    if (derived) {
        l += "D 1 " + std::to_string(blocks.size() / 2);
        for (long long b : blocks) l += " " + std::to_string(b);
    } else if (dt == s.bdt) {
        l += "T " + std::to_string(count);
    } else {
        l += "B " + std::to_string(bytes);
    }
    line(l);
}
inline void rec_wait(const std::vector<uint32_t>& ids)
{
    if (ids.empty()) return;
    std::string l = "W " + std::to_string(ids.size());
    for (uint32_t i : ids) l += " " + std::to_string(i);
    line(l);
}
inline void rec_waitany(const std::vector<uint32_t>& ids)
{
    if (ids.empty()) return;
    std::string l = "Y " + std::to_string(ids.size());
    for (uint32_t i : ids) l += " " + std::to_string(i);
    line(l);
}

inline void h_alloc(const void* p, size_t bytes)
{
    State& s = st();
    const int id = s.next_buf++;
    s.bufs[static_cast<const char*>(p)] = {id, bytes};
    line("A " + std::to_string(id) + " " + std::to_string(bytes));
}
inline void h_release(const void* p)
{
    State& s = st();
    auto it = s.bufs.find(static_cast<const char*>(p));
    if (it == s.bufs.end()) return;
    line("F " + std::to_string(it->second.first));
    s.bufs.erase(it);
}
inline void h_copy(void* dst, const void* src, size_t bytes)
{
    auto d = where(dst), c = where(src);
    if (d.first < 0 || c.first < 0) throw std::runtime_error("--trace: a copy outside sbuf, rbuf and temporaries");
    line("C " + std::to_string(d.first) + " " + std::to_string(d.second) + " " + std::to_string(c.first) + " " +
         std::to_string(c.second) + " " + std::to_string(bytes));
}
inline void h_reduce(const void* in, void* inout, size_t count, MPI_Datatype dt, MPI_Op)
{
    if (dt != st().bdt) throw std::runtime_error("--trace: a reduction in another datatype");
    auto a = where(in), b = where(inout);
    if (a.first < 0 || b.first < 0) throw std::runtime_error("--trace: a reduction outside sbuf, rbuf and temporaries");
    line("X " + std::to_string(a.first) + " " + std::to_string(a.second) + " " + std::to_string(b.first) + " " +
         std::to_string(b.second) + " " + std::to_string(count));
}
inline const mem::State::TraceHooks* hooks()
{
    static const mem::State::TraceHooks h{&h_alloc, &h_release, &h_copy, &h_reduce};
    return &h;
}

/* open DIR/rank-NNNNN.trace and write the header (every rank) */
inline void open(const std::string& dir, int wrank, const std::string& meta)
{
    char name[64];
    std::snprintf(name, sizeof name, "/rank-%05d.trace", wrank);
    const std::string path = dir + name;
    ::mkdir(dir.c_str(), 0755); /* created if missing (every rank: node-local filesystems too) */
    st().f = std::fopen(path.c_str(), "w");
    if (!st().f) throw std::runtime_error("--trace: cannot write " + path);
    std::fprintf(st().f, "collage-trace 1\nmeta %s\n", meta.c_str());
}
inline void close()
{
    if (st().f) std::fclose(st().f);
    st().f = nullptr;
}

/* one call: begin before the algorithm runs, end after it drained */
inline void begin(const std::string& label, const std::string& base, size_t bytes, MPI_Datatype bdt, const void* s,
                  size_t sb, const void* r, size_t rb)
{
    State& t = st();
    t.bdt = bdt;
    t.sbase = static_cast<const char*>(s);
    t.sbytes = sb;
    t.rbase = static_cast<const char*>(r);
    t.rbytes = rb;
    t.bufs.clear();
    t.next_buf = 2;
    t.next_req = 0;
    t.nops = 0;
    std::fprintf(t.f, "call %s %s %zu\n", label.c_str(), base.c_str(), bytes);
    t.active = true;
    mem::st().trace = hooks();
}
inline void end()
{
    State& t = st();
    mem::st().trace = nullptr;
    if (!t.active) return;
    t.active = false;
    std::fprintf(t.f, "end\n");
    std::fflush(t.f);
    t.total_ops += t.nops;
    ++t.total_calls;
}
inline void abort_call() /* the algorithm did not apply (n/a) */
{
    State& t = st();
    mem::st().trace = nullptr;
    if (!t.active) return;
    t.active = false;
    std::fprintf(t.f, "abort\n");
    std::fflush(t.f);
}

/* ---- reading ------------------------------------------------------------ */
struct Op {
    char k = 0;
    int a = 0, b = 0, c = 0, d = 0; /* S/Z/R: req cid peer tag; A/F: id; X/C: buf ids in a, c */
    int buf = -1;
    long long off = 0, off2 = 0;
    uint64_t n = 0;                  /* count / bytes */
    char dt = 'T';
    std::vector<long long> blocks;   /* D */
    std::vector<uint32_t> reqs;      /* W */
};
struct Call {
    std::string label, base;
    size_t bytes = 0;
    std::vector<Op> ops;
    uint32_t nreq = 0;
    int nbuf = 2;
    std::vector<int> cids;
};
struct Trace {
    std::map<std::string, std::string> meta;
    std::vector<Call> calls;
    std::vector<std::string> labels; /* in first-seen order */
    std::map<std::string, std::string> base_of;
    const Call* find(const std::string& label, size_t bytes) const
    {
        for (const Call& c : calls)
            if (c.label == label && c.bytes == bytes) return &c;
        return nullptr;
    }
};

inline Trace load(const std::string& dir, int wrank)
{
    char name[64];
    std::snprintf(name, sizeof name, "/rank-%05d.trace", wrank);
    const std::string path = dir + name;
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) throw std::runtime_error("--replay: cannot read " + path);
    Trace t;
    std::vector<char> lb(1 << 16);
    std::string ln;
    auto getline = [&](std::string& out) {
        out.clear();
        while (std::fgets(lb.data(), (int)lb.size(), f)) {
            out += lb.data();
            if (!out.empty() && out.back() == '\n') {
                out.pop_back();
                return true;
            }
        }
        return !out.empty();
    };
    auto toks = [](const std::string& s) {
        std::vector<std::string> v;
        size_t a = 0;
        while (a < s.size()) {
            while (a < s.size() && s[a] == ' ') ++a;
            size_t b = s.find(' ', a);
            if (b == std::string::npos) b = s.size();
            if (b > a) v.push_back(s.substr(a, b - a));
            a = b;
        }
        return v;
    };
    if (!getline(ln) || ln != "collage-trace 1") {
        std::fclose(f);
        throw std::runtime_error("--replay: " + path + " is not a collage trace (version 1)");
    }
    Call* cur = nullptr;
    while (getline(ln)) {
        auto v = toks(ln);
        if (v.empty()) continue;
        const std::string& k = v[0];
        auto L = [&](size_t i) { return std::stoll(v.at(i)); };
        if (k == "meta") {
            for (size_t i = 1; i < v.size(); ++i) {
                size_t e = v[i].find('=');
                if (e != std::string::npos) t.meta[v[i].substr(0, e)] = v[i].substr(e + 1);
            }
        } else if (k == "call") {
            t.calls.emplace_back();
            cur = &t.calls.back();
            cur->label = v.at(1);
            cur->base = v.at(2);
            cur->bytes = (size_t)L(3);
        } else if (k == "abort") {
            t.calls.pop_back();
            cur = nullptr;
        } else if (k == "end") {
            if (cur && !t.base_of.count(cur->label)) {
                t.labels.push_back(cur->label);
                t.base_of[cur->label] = cur->base;
            }
            cur = nullptr;
        } else if (cur) {
            Op o;
            o.k = k[0];
            switch (o.k) {
                case 'S': case 'Z': case 'R': {
                    o.a = (int)L(1); o.b = (int)L(2); o.c = (int)L(3); o.d = (int)L(4);
                    o.buf = (int)L(5); o.off = L(6); o.dt = v.at(7)[0]; o.n = (uint64_t)L(8);
                    if (o.dt == 'D') {
                        const size_t nb = (size_t)L(9);
                        for (size_t i = 0; i < 2 * nb; ++i) o.blocks.push_back(L(10 + i));
                    }
                    cur->nreq = std::max<uint32_t>(cur->nreq, (uint32_t)o.a + 1);
                    if (std::find(cur->cids.begin(), cur->cids.end(), o.b) == cur->cids.end()) cur->cids.push_back(o.b);
                    break;
                }
                case 'W': case 'Y': {
                    const size_t n = (size_t)L(1);
                    for (size_t i = 0; i < n; ++i) o.reqs.push_back((uint32_t)L(2 + i));
                    break;
                }
                case 'A': o.a = (int)L(1); o.n = (uint64_t)L(2); cur->nbuf = std::max(cur->nbuf, o.a + 1); break;
                case 'F': o.a = (int)L(1); break;
                case 'X': o.a = (int)L(1); o.off = L(2); o.c = (int)L(3); o.off2 = L(4); o.n = (uint64_t)L(5); break;
                case 'C': o.a = (int)L(1); o.off = L(2); o.c = (int)L(3); o.off2 = L(4); o.n = (uint64_t)L(5); break;
                default:
                    std::fclose(f);
                    throw std::runtime_error("--replay: " + path + ": unknown record '" + k + "'");
            }
            cur->ops.push_back(std::move(o));
        }
    }
    std::fclose(f);
    return t;
}

} // namespace trc

#endif /* COLLAGE_TRACE_HPP */
