#pragma once
#ifndef COLLAGE_BENCH_HPP
#define COLLAGE_BENCH_HPP

/*
 * Benchmark harness: one Runner per collective. A runner owns the user
 * buffers for the current message size, knows how to initialise them, how to
 * call every algorithm (and the native MPI collective as a baseline), and how
 * to verify a result against the native MPI collective.
 *
 * The harness (only the harness, never an algorithm) uses MPI collectives for
 * synchronisation, timing statistics and result agreement.
 */

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "allgather.hpp"
#include "allgatherv.hpp"
#include "allreduce.hpp"
#include "alltoall.hpp"
#include "alltoallv.hpp"
#include "barrier.hpp"
#include "bcast.hpp"
#include "gather.hpp"
#include "reduce.hpp"
#include "reduce_scatter.hpp"
#include "reduce_scatter_block.hpp"
#include "scatter.hpp"
#include "utils.hpp"

namespace bench {

/* ------------------------------------------------------------------------- */
/* options                                                                   */
/* ------------------------------------------------------------------------- */
struct Options {
    std::string coll;                       /* collective name */
    std::vector<std::string> algos;         /* empty = all */
    std::string type = "double";
    std::string op = "sum";
    size_t min_bytes = 0;
    size_t max_bytes = 4u << 20;            /* 4 MiB */
    double factor = 2.0;
    std::vector<size_t> sizes;              /* explicit list overrides min/max/factor */
    int iters = 100, warmup = 10;
    int iters_large = 20, warmup_large = 5;
    size_t large_threshold = 64u << 10;     /* bytes */
    int root = 0;
    std::vector<size_t> segsize = {0};      /* bytes, 0 = no segmentation */
    std::vector<int> fanout = {4};          /* number of chains */
    std::vector<int> radix = {4};           /* k-nomial / k-Bruck radix */
    std::vector<int> max_reqs = {0};        /* 0 = unlimited */
    std::vector<size_t> first_seg = {1024}; /* gather linear_sync, bytes */
    std::string vdist = "uniform";          /* uniform | random | linear */
    unsigned seed = 42;
    bool check = false, check_only = false;
    bool native = true;                     /* include the native MPI baseline */
    size_t max_mem = (size_t)2 << 30;       /* per-rank buffer limit, bytes */
    std::string csv;                        /* write full statistics */
    bool quiet = false;
};

/* parameters handed to one algorithm variant */
struct AlgoParams {
    size_t segsize = 0;
    int fanout = 4;
    int radix = 4;
    int max_reqs = 0;
    size_t first_seg = 1024;
    int root = 0;
    MPI_Op op = MPI_SUM;
};

enum : unsigned { P_SEG = 1, P_FANOUT = 2, P_RADIX = 4, P_MAXREQ = 8, P_FIRSTSEG = 16 };

struct AlgoDef {
    std::string name;
    unsigned params = 0;                                 /* which tunables it uses */
    std::function<void(const AlgoParams&)> fn;
    std::function<size_t(size_t)> extra_mem;             /* bytes, given count (optional) */
};

/* ------------------------------------------------------------------------- */
/* helpers                                                                    */
/* ------------------------------------------------------------------------- */
inline std::string human_bytes(size_t b)
{
    char buf[32];
    if (b >= (1u << 20) && b % (1u << 20) == 0) std::snprintf(buf, sizeof buf, "%zuM", b >> 20);
    else if (b >= 1024 && b % 1024 == 0) std::snprintf(buf, sizeof buf, "%zuK", b >> 10);
    else std::snprintf(buf, sizeof buf, "%zu", b);
    return buf;
}

enum class Fill { Data, Sum, Prod, MinMax, Logical, Bitwise, First };

inline Fill fill_for(const OpInfo& o)
{
    if (o.name == "user_nc") return Fill::First;
    switch (o.kind) {
        case OpKind::Arithmetic: return o.name == "prod" ? Fill::Prod : Fill::Sum;
        case OpKind::MinMax: return Fill::MinMax;
        case OpKind::Logical: return Fill::Logical;
        case OpKind::Bitwise: return Fill::Bitwise;
        case OpKind::User: return Fill::Sum;
    }
    return Fill::Sum;
}

/* deterministic input value of element i on rank r (small integers so that
 * floating point reductions are exact irrespective of the order) */
template <typename T>
inline T gen_value(Fill f, int r, size_t i)
{
    switch (f) {
        case Fill::Data:
        case Fill::First: return (T)((uint64_t)r * 1000003ull + (uint64_t)i * 7919ull + 1);
        case Fill::Sum: return (T)((int)((r * 7 + i * 13) % 11) - 5);
        case Fill::Prod: return (T)(((r + i) % 17 == 0) ? 2 : (((r + 2 * i) % 3 == 0) ? -1 : 1));
        case Fill::MinMax: return (T)((int)((r * 31 + i * 17) % 101) - 50);
        case Fill::Logical: return (T)(((r * (i + 1) + i) % 4) != 0 ? 1 : 0);
        case Fill::Bitwise:
            return (T)((((uint64_t)(r + 1) * 0x9E3779B97F4A7C15ull) >> (i % 23)) ^ (uint64_t)i);
    }
    return T(0);
}

template <typename T>
inline bool values_equal(T a, T b)
{
    if constexpr (std::is_floating_point<T>::value) {
        if (std::isnan(a) && std::isnan(b)) return true;
        if (std::isinf(a) || std::isinf(b)) return a == b;
        const double tol = sizeof(T) == 4 ? 1e-5 : 1e-12;
        return std::fabs((double)a - (double)b) <= tol * (1.0 + std::fabs((double)b));
    } else {
        return a == b;
    }
}

/* large-count aware native calls --------------------------------------------- */
#if COLLAGE_LARGE_COUNT
using ncount_t = MPI_Count;
using ndisp_t = MPI_Aint;
#define NATIVE(fn) fn##_c
#else
using ncount_t = int;
using ndisp_t = int;
#define NATIVE(fn) fn
#endif

inline std::vector<ncount_t> to_ncounts(const std::vector<size_t>& v)
{
    std::vector<ncount_t> r(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
#if !COLLAGE_LARGE_COUNT
        if (v[i] > (size_t)INT_MAX) throw std::overflow_error("count exceeds INT_MAX");
#endif
        r[i] = (ncount_t)v[i];
    }
    return r;
}
inline std::vector<ndisp_t> to_ndispls(const std::vector<size_t>& v)
{
    std::vector<ndisp_t> r(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
#if !COLLAGE_LARGE_COUNT
        if (v[i] > (size_t)INT_MAX) throw std::overflow_error("displacement exceeds INT_MAX");
#endif
        r[i] = (ndisp_t)v[i];
    }
    return r;
}
inline ncount_t ncnt(size_t c)
{
#if !COLLAGE_LARGE_COUNT
    if (c > (size_t)INT_MAX) throw std::overflow_error("count exceeds INT_MAX");
#endif
    return (ncount_t)c;
}

/* ------------------------------------------------------------------------- */
/* Runner base                                                               */
/* ------------------------------------------------------------------------- */
template <typename T>
class Runner {
public:
    Runner(MPI_Comm comm, const Options& o, const OpInfo& op)
        : comm_(comm), opt_(o), op_(op), dt_(TypeMap<T>())
    {
        MPI_Comm_rank(comm, &rank_);
        MPI_Comm_size(comm, &size_);
    }
    virtual ~Runner() = default;

    virtual bool sized() const { return true; }   /* false for barrier */
    virtual bool reduction() const { return false; }
    virtual std::vector<AlgoDef> algorithms() = 0;
    /* what "message size" means for this collective (printed in the header) */
    virtual const char* size_semantics() const = 0;
    /* bytes of the user buffers (sbuf+rbuf+expected) for `count` elements */
    virtual size_t mem_bytes(size_t count) const = 0;
    /* bytes moved per rank and per call, used for the bandwidth column */
    virtual double payload_bytes(size_t count) const { return (double)count * sizeof(T); }
    /* allocate and initialise inputs, poison outputs */
    virtual void setup(size_t count) = 0;
    /* called before a verified run (e.g. to make rbuf hold the in-place input) */
    virtual void prepare(const std::string& /*algo*/) {}
    virtual void native() = 0;
    /* the part of the output that is defined on this rank */
    virtual std::pair<const T*, size_t> result() const = 0;

    void snapshot()
    {
        auto r = result();
        expected_.assign(r.first, r.first + r.second);
    }
    /* returns the number of mismatching elements on this rank */
    size_t compare() const
    {
        auto r = result();
        if (r.second != expected_.size()) return std::max(r.second, expected_.size());
        size_t bad = 0;
        for (size_t i = 0; i < r.second; ++i)
            if (!values_equal(r.first[i], expected_[i])) ++bad;
        return bad;
    }

protected:
    MPI_Comm comm_;
    const Options& opt_;
    const OpInfo& op_;
    MPI_Datatype dt_;
    int rank_ = 0, size_ = 1;
    size_t count_ = 0;
    std::vector<T> sbuf_, rbuf_, expected_;

    Fill fill() const { return reduction() ? fill_for(op_) : Fill::Data; }
    void fill_input(std::vector<T>& v, size_t n, size_t offset = 0)
    {
        v.resize(std::max<size_t>(n, 1));
        const Fill f = fill();
        for (size_t i = 0; i < n; ++i) v[i] = gen_value<T>(f, rank_, offset + i);
    }
    static void poison(std::vector<T>& v, size_t n)
    {
        v.assign(std::max<size_t>(n, 1), (T)0);
        if (n) std::memset(v.data(), 0x5a, n * sizeof(T));
    }
};

/* count distributions for the v-variants: element count of block (i, j) */
inline size_t vcount(const Options& o, size_t base, int i, int j, int size)
{
    if (o.vdist == "uniform" || base == 0) return base;
    if (o.vdist == "random") {
        uint64_t h = (uint64_t)o.seed * 0x9E3779B97F4A7C15ull ^ ((uint64_t)i << 32 | (uint32_t)j);
        h ^= h >> 33; h *= 0xff51afd7ed558ccdull; h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ull; h ^= h >> 33;
        return (size_t)(h % (2 * base + 1));  /* mean == base */
    }
    if (o.vdist == "linear") {
        size_t k = (size_t)((i + j) % size) + 1;
        return 2 * base * k / (size_t)(size + 1);  /* mean ~ base */
    }
    throw std::invalid_argument("unknown --vdist '" + o.vdist + "'");
}

/* ------------------------------------------------------------------------- */
/* runners                                                                    */
/* ------------------------------------------------------------------------- */
template <typename T>
class BcastRunner : public Runner<T> {
    using B = Runner<T>;
    Bcast<T> c_;
public:
    BcastRunner(MPI_Comm comm, const Options& o, const OpInfo& op) : B(comm, o, op), c_(comm) {}
    const char* size_semantics() const override { return "bytes broadcast from the root"; }
    size_t mem_bytes(size_t n) const override { return 2 * n * sizeof(T); }
    void setup(size_t n) override
    {
        this->count_ = n;
        if (this->rank_ == this->opt_.root) this->fill_input(this->rbuf_, n);
        else this->poison(this->rbuf_, n);
    }
    void native() override
    {
        MPI_CHECK(NATIVE(MPI_Bcast)(this->rbuf_.data(), ncnt(this->count_), this->dt_, this->opt_.root, this->comm_));
    }
    std::pair<const T*, size_t> result() const override { return {this->rbuf_.data(), this->count_}; }
    std::vector<AlgoDef> algorithms() override
    {
        auto buf = [this]() { return this->rbuf_.data(); };
        auto n = [this]() { return this->count_; };
        return {
            {"bintree", P_SEG, [=](const AlgoParams& p) { c_.bintree(buf(), n(), p.root, p.segsize); }, {}},
            {"pipeline", P_SEG, [=](const AlgoParams& p) { c_.pipeline(buf(), n(), p.root, p.segsize); }, {}},
            {"chain", P_SEG | P_FANOUT, [=](const AlgoParams& p) { c_.chain(buf(), n(), p.root, p.segsize, p.fanout); }, {}},
            {"binomial", P_SEG, [=](const AlgoParams& p) { c_.binomial(buf(), n(), p.root, p.segsize); }, {}},
            {"split_bintree", P_SEG, [=](const AlgoParams& p) { c_.split_bintree(buf(), n(), p.root, p.segsize); }, {}},
            {"linear", 0, [=](const AlgoParams& p) { c_.linear(buf(), n(), p.root); }, {}},
            {"knomial", P_SEG | P_RADIX, [=](const AlgoParams& p) { c_.knomial(buf(), n(), p.root, p.segsize, p.radix); }, {}},
            {"scatter_allgather", 0, [=](const AlgoParams& p) { c_.scatter_allgather(buf(), n(), p.root, p.segsize); }, {}},
            {"scatter_allgather_ring", 0, [=](const AlgoParams& p) { c_.scatter_allgather_ring(buf(), n(), p.root, p.segsize); }, {}},
        };
    }
};

template <typename T>
class ReduceRunner : public Runner<T> {
    using B = Runner<T>;
    Reduce<T> c_;
public:
    ReduceRunner(MPI_Comm comm, const Options& o, const OpInfo& op) : B(comm, o, op), c_(comm) {}
    bool reduction() const override { return true; }
    const char* size_semantics() const override { return "bytes reduced per rank"; }
    size_t mem_bytes(size_t n) const override { return 3 * n * sizeof(T); }
    void setup(size_t n) override
    {
        this->count_ = n;
        this->fill_input(this->sbuf_, n);
        this->poison(this->rbuf_, n);
    }
    void native() override
    {
        MPI_CHECK(NATIVE(MPI_Reduce)(this->sbuf_.data(), this->rbuf_.data(), ncnt(this->count_), this->dt_,
                                     this->op_.op, this->opt_.root, this->comm_));
    }
    std::pair<const T*, size_t> result() const override
    {
        if (this->rank_ != this->opt_.root) return {this->rbuf_.data(), 0};
        return {this->rbuf_.data(), this->count_};
    }
    std::vector<AlgoDef> algorithms() override
    {
        auto s = [this]() { return (const T*)this->sbuf_.data(); };
        auto r = [this]() { return this->rbuf_.data(); };
        auto n = [this]() { return this->count_; };
        const int P = this->size_;
        return {
            {"chain", P_SEG | P_FANOUT | P_MAXREQ, [=](const AlgoParams& p) { c_.chain(s(), r(), n(), p.op, p.root, p.segsize, p.fanout, p.max_reqs); }, {}},
            {"pipeline", P_SEG | P_MAXREQ, [=](const AlgoParams& p) { c_.pipeline(s(), r(), n(), p.op, p.root, p.segsize, p.max_reqs); }, {}},
            {"binary", P_SEG | P_MAXREQ, [=](const AlgoParams& p) { c_.binary(s(), r(), n(), p.op, p.root, p.segsize, p.max_reqs); }, {}},
            {"binomial", P_SEG | P_MAXREQ, [=](const AlgoParams& p) { c_.binomial(s(), r(), n(), p.op, p.root, p.segsize, p.max_reqs); }, {}},
            {"in_order_binary", P_SEG | P_MAXREQ, [=](const AlgoParams& p) { c_.in_order_binary(s(), r(), n(), p.op, p.root, p.segsize, p.max_reqs); }, {}},
            {"linear", 0, [=](const AlgoParams& p) { c_.linear(s(), r(), n(), p.op, p.root); }, {}},
            {"redscat_gather", 0, [=](const AlgoParams& p) { c_.redscat_gather(s(), r(), n(), p.op, p.root); }, {}},
            {"knomial", P_RADIX, [=](const AlgoParams& p) { c_.knomial(s(), r(), n(), p.op, p.root, p.segsize, p.max_reqs, p.radix); },
             [=](size_t c) { return c * sizeof(T) * (size_t)(2 + std::log2(std::max(P, 2)) * 3); }},
        };
    }
};

template <typename T>
class AllreduceRunner : public Runner<T> {
    using B = Runner<T>;
    Allreduce<T> c_;
public:
    AllreduceRunner(MPI_Comm comm, const Options& o, const OpInfo& op) : B(comm, o, op), c_(comm) {}
    bool reduction() const override { return true; }
    const char* size_semantics() const override { return "bytes reduced per rank"; }
    size_t mem_bytes(size_t n) const override { return 3 * n * sizeof(T); }
    void setup(size_t n) override
    {
        this->count_ = n;
        this->fill_input(this->sbuf_, n);
        this->poison(this->rbuf_, n);
    }
    void native() override
    {
        MPI_CHECK(NATIVE(MPI_Allreduce)(this->sbuf_.data(), this->rbuf_.data(), ncnt(this->count_), this->dt_,
                                        this->op_.op, this->comm_));
    }
    std::pair<const T*, size_t> result() const override { return {this->rbuf_.data(), this->count_}; }
    std::vector<AlgoDef> algorithms() override
    {
        auto s = [this]() { return (const T*)this->sbuf_.data(); };
        auto r = [this]() { return this->rbuf_.data(); };
        auto n = [this]() { return this->count_; };
        const size_t P = (size_t)this->size_;
        return {
            {"nonoverlapping", P_SEG | P_MAXREQ, [=](const AlgoParams& p) { c_.nonoverlapping(s(), r(), n(), p.op, p.segsize, p.max_reqs); }, {}},
            {"recursivedoubling", 0, [=](const AlgoParams& p) { c_.recursivedoubling(s(), r(), n(), p.op); }, {}},
            {"ring", 0, [=](const AlgoParams& p) { c_.ring(s(), r(), n(), p.op); }, {}},
            {"ring_segmented", P_SEG, [=](const AlgoParams& p) { c_.ring_segmented(s(), r(), n(), p.op, p.segsize); }, {}},
            {"linear", 0, [=](const AlgoParams& p) { c_.linear(s(), r(), n(), p.op); }, {}},
            {"redscat_allgather", 0, [=](const AlgoParams& p) { c_.redscat_allgather(s(), r(), n(), p.op); }, {}},
            {"allgather_reduce", 0, [=](const AlgoParams& p) { c_.allgather_reduce(s(), r(), n(), p.op); },
             [=](size_t c) { return (P + 1) * c * sizeof(T); }},
        };
    }
};

template <typename T>
class AllgatherRunner : public Runner<T> {
    using B = Runner<T>;
    Allgather<T> c_;
public:
    AllgatherRunner(MPI_Comm comm, const Options& o, const OpInfo& op) : B(comm, o, op), c_(comm) {}
    const char* size_semantics() const override { return "bytes contributed per rank"; }
    size_t mem_bytes(size_t n) const override { return (1 + 2 * (size_t)this->size_) * n * sizeof(T); }
    double payload_bytes(size_t n) const override { return (double)n * sizeof(T) * this->size_; }
    void setup(size_t n) override
    {
        this->count_ = n;
        this->fill_input(this->sbuf_, n);
        this->poison(this->rbuf_, n * this->size_);
    }
    void native() override
    {
        MPI_CHECK(NATIVE(MPI_Allgather)(this->sbuf_.data(), ncnt(this->count_), this->dt_, this->rbuf_.data(),
                                        ncnt(this->count_), this->dt_, this->comm_));
    }
    std::pair<const T*, size_t> result() const override { return {this->rbuf_.data(), this->count_ * this->size_}; }
    std::vector<AlgoDef> algorithms() override
    {
        auto s = [this]() { return (const T*)this->sbuf_.data(); };
        auto r = [this]() { return this->rbuf_.data(); };
        auto n = [this]() { return this->count_; };
        return {
            {"recursivedoubling", 0, [=](const AlgoParams&) { c_.recursivedoubling(s(), n(), r(), n()); }, {}},
            {"sparbit", 0, [=](const AlgoParams&) { c_.sparbit(s(), n(), r(), n()); }, {}},
            {"ring", 0, [=](const AlgoParams&) { c_.ring(s(), n(), r(), n()); }, {}},
            {"neighborexchange", 0, [=](const AlgoParams&) { c_.neighborexchange(s(), n(), r(), n()); }, {}},
            {"linear", 0, [=](const AlgoParams&) { c_.linear(s(), n(), r(), n()); }, {}},
            {"k_bruck", P_RADIX, [=](const AlgoParams& p) { c_.k_bruck(s(), n(), r(), n(), p.radix); }, {}},
            {"direct_messaging", 0, [=](const AlgoParams&) { c_.direct_messaging(s(), n(), r(), n()); }, {}},
        };
    }
};

template <typename T>
class AllgathervRunner : public Runner<T> {
    using B = Runner<T>;
    Allgatherv<T> c_;
    std::vector<size_t> rcounts_, rdispls_;
    size_t scount_ = 0, total_ = 0;
public:
    AllgathervRunner(MPI_Comm comm, const Options& o, const OpInfo& op) : B(comm, o, op), c_(comm) {}
    const char* size_semantics() const override { return "average bytes contributed per rank (--vdist)"; }
    size_t mem_bytes(size_t n) const override { return (1 + 4 * (size_t)this->size_) * n * sizeof(T); }
    double payload_bytes(size_t n) const override { return (double)n * sizeof(T) * this->size_; }
    void setup(size_t n) override
    {
        const int P = this->size_;
        this->count_ = n;
        rcounts_.resize(P);
        rdispls_.resize(P);
        total_ = 0;
        for (int i = 0; i < P; ++i) {
            rcounts_[i] = vcount(this->opt_, n, i, 0, P);
            rdispls_[i] = total_;
            total_ += rcounts_[i];
        }
        scount_ = rcounts_[this->rank_];
        this->fill_input(this->sbuf_, scount_);
        this->poison(this->rbuf_, total_);
    }
    void native() override
    {
        auto rc = to_ncounts(rcounts_);
        auto rd = to_ndispls(rdispls_);
        MPI_CHECK(NATIVE(MPI_Allgatherv)(this->sbuf_.data(), ncnt(scount_), this->dt_, this->rbuf_.data(), rc.data(),
                                         rd.data(), this->dt_, this->comm_));
    }
    std::pair<const T*, size_t> result() const override { return {this->rbuf_.data(), total_}; }
    std::vector<AlgoDef> algorithms() override
    {
        auto s = [this]() { return (const T*)this->sbuf_.data(); };
        auto r = [this]() { return this->rbuf_.data(); };
        return {
            {"bruck", 0, [=](const AlgoParams&) { c_.bruck(s(), scount_, r(), rcounts_, rdispls_); }, {}},
            {"sparbit", 0, [=](const AlgoParams&) { c_.sparbit(s(), scount_, r(), rcounts_, rdispls_); }, {}},
            {"ring", 0, [=](const AlgoParams&) { c_.ring(s(), scount_, r(), rcounts_, rdispls_); }, {}},
            {"neighborexchange", 0, [=](const AlgoParams&) { c_.neighborexchange(s(), scount_, r(), rcounts_, rdispls_); }, {}},
            {"two_procs", 0, [=](const AlgoParams&) { c_.two_procs(s(), scount_, r(), rcounts_, rdispls_); }, {}},
            {"basic_default", 0, [=](const AlgoParams&) { c_.basic_default(s(), scount_, r(), rcounts_, rdispls_); }, {}},
        };
    }
};

template <typename T>
class AlltoallRunner : public Runner<T> {
    using B = Runner<T>;
    Alltoall<T> c_;
public:
    AlltoallRunner(MPI_Comm comm, const Options& o, const OpInfo& op) : B(comm, o, op), c_(comm) {}
    const char* size_semantics() const override { return "bytes sent to each peer"; }
    size_t mem_bytes(size_t n) const override { return 4 * (size_t)this->size_ * n * sizeof(T); }
    double payload_bytes(size_t n) const override { return (double)n * sizeof(T) * this->size_; }
    void setup(size_t n) override
    {
        const int P = this->size_;
        this->count_ = n;
        this->sbuf_.resize(std::max<size_t>(n * P, 1));
        /* block j goes to rank j: encode (sender, receiver, index) */
        for (int j = 0; j < P; ++j)
            for (size_t i = 0; i < n; ++i)
                this->sbuf_[(size_t)j * n + i] = gen_value<T>(Fill::Data, this->rank_, (size_t)j * 65537 + i);
        this->poison(this->rbuf_, n * P);
    }
    void prepare(const std::string& algo) override
    {
        if (algo.rfind("inplace", 0) == 0) this->rbuf_ = this->sbuf_;
    }
    void native() override
    {
        MPI_CHECK(NATIVE(MPI_Alltoall)(this->sbuf_.data(), ncnt(this->count_), this->dt_, this->rbuf_.data(),
                                       ncnt(this->count_), this->dt_, this->comm_));
    }
    std::pair<const T*, size_t> result() const override { return {this->rbuf_.data(), this->count_ * this->size_}; }
    std::vector<AlgoDef> algorithms() override
    {
        auto s = [this]() { return (const T*)this->sbuf_.data(); };
        auto r = [this]() { return this->rbuf_.data(); };
        auto n = [this]() { return this->count_; };
        const size_t P = (size_t)this->size_;
        return {
            {"inplace", 0, [=](const AlgoParams&) { c_.inplace(r(), n()); }, {}},
            {"pairwise", 0, [=](const AlgoParams&) { c_.pairwise(s(), n(), r(), n()); }, {}},
            {"bruck", 0, [=](const AlgoParams&) { c_.bruck(s(), n(), r(), n()); }, [=](size_t c) { return P * c * sizeof(T); }},
            {"linear_sync", P_MAXREQ, [=](const AlgoParams& p) { c_.linear_sync(s(), n(), r(), n(), p.max_reqs); }, {}},
            {"two_procs", 0, [=](const AlgoParams&) { c_.two_procs(s(), n(), r(), n()); }, {}},
            {"linear", 0, [=](const AlgoParams&) { c_.linear(s(), n(), r(), n()); }, {}},
        };
    }
};

template <typename T>
class AlltoallvRunner : public Runner<T> {
    using B = Runner<T>;
    Alltoallv<T> c_;
    std::vector<size_t> scounts_, sdispls_, rcounts_, rdispls_;
    size_t stotal_ = 0, rtotal_ = 0;
public:
    AlltoallvRunner(MPI_Comm comm, const Options& o, const OpInfo& op) : B(comm, o, op), c_(comm) {}
    const char* size_semantics() const override { return "average bytes sent to each peer (--vdist)"; }
    size_t mem_bytes(size_t n) const override { return 6 * (size_t)this->size_ * n * sizeof(T); }
    double payload_bytes(size_t n) const override { return (double)n * sizeof(T) * this->size_; }
    void setup(size_t n) override
    {
        const int P = this->size_, me = this->rank_;
        this->count_ = n;
        scounts_.resize(P); sdispls_.resize(P); rcounts_.resize(P); rdispls_.resize(P);
        stotal_ = rtotal_ = 0;
        for (int j = 0; j < P; ++j) {
            scounts_[j] = vcount(this->opt_, n, me, j, P);
            rcounts_[j] = vcount(this->opt_, n, j, me, P);
            sdispls_[j] = stotal_;
            rdispls_[j] = rtotal_;
            stotal_ += scounts_[j];
            rtotal_ += rcounts_[j];
        }
        this->sbuf_.resize(std::max<size_t>(stotal_, 1));
        for (int j = 0; j < P; ++j)
            for (size_t i = 0; i < scounts_[j]; ++i)
                this->sbuf_[sdispls_[j] + i] = gen_value<T>(Fill::Data, me, (size_t)j * 65537 + i);
        this->poison(this->rbuf_, rtotal_);
    }
    void prepare(const std::string& algo) override
    {
        if (algo.rfind("inplace", 0) == 0) {
            if (this->opt_.vdist != "uniform" && this->count_ > 0)
                throw NotApplicable("inplace needs symmetric counts (use --vdist uniform)");
            this->rbuf_ = this->sbuf_;
        }
    }
    void native() override
    {
        auto sc = to_ncounts(scounts_), rc = to_ncounts(rcounts_);
        auto sd = to_ndispls(sdispls_), rd = to_ndispls(rdispls_);
        MPI_CHECK(NATIVE(MPI_Alltoallv)(this->sbuf_.data(), sc.data(), sd.data(), this->dt_, this->rbuf_.data(),
                                        rc.data(), rd.data(), this->dt_, this->comm_));
    }
    std::pair<const T*, size_t> result() const override { return {this->rbuf_.data(), rtotal_}; }
    std::vector<AlgoDef> algorithms() override
    {
        auto s = [this]() { return (const T*)this->sbuf_.data(); };
        auto r = [this]() { return this->rbuf_.data(); };
        return {
            {"inplace", 0, [=](const AlgoParams&) {
                 if (this->opt_.vdist != "uniform" && this->count_ > 0)
                     throw NotApplicable("inplace needs symmetric counts (use --vdist uniform)");
                 c_.inplace(r(), rcounts_, rdispls_);
             }, {}},
            {"pairwise", 0, [=](const AlgoParams&) { c_.pairwise(s(), scounts_, sdispls_, r(), rcounts_, rdispls_); }, {}},
            {"linear", 0, [=](const AlgoParams&) { c_.linear(s(), scounts_, sdispls_, r(), rcounts_, rdispls_); }, {}},
        };
    }
};

template <typename T>
class GatherRunner : public Runner<T> {
    using B = Runner<T>;
    Gather<T> c_;
public:
    GatherRunner(MPI_Comm comm, const Options& o, const OpInfo& op) : B(comm, o, op), c_(comm) {}
    const char* size_semantics() const override { return "bytes contributed per rank"; }
    size_t mem_bytes(size_t n) const override { return (1 + 3 * (size_t)this->size_) * n * sizeof(T); }
    double payload_bytes(size_t n) const override { return (double)n * sizeof(T) * this->size_; }
    void setup(size_t n) override
    {
        this->count_ = n;
        this->fill_input(this->sbuf_, n);
        this->poison(this->rbuf_, n * this->size_);
    }
    void native() override
    {
        MPI_CHECK(NATIVE(MPI_Gather)(this->sbuf_.data(), ncnt(this->count_), this->dt_, this->rbuf_.data(),
                                     ncnt(this->count_), this->dt_, this->opt_.root, this->comm_));
    }
    std::pair<const T*, size_t> result() const override
    {
        if (this->rank_ != this->opt_.root) return {this->rbuf_.data(), 0};
        return {this->rbuf_.data(), this->count_ * this->size_};
    }
    std::vector<AlgoDef> algorithms() override
    {
        auto s = [this]() { return (const T*)this->sbuf_.data(); };
        auto r = [this]() { return this->rbuf_.data(); };
        auto n = [this]() { return this->count_; };
        return {
            {"binomial", 0, [=](const AlgoParams& p) { c_.binomial(s(), n(), r(), n(), p.root); }, {}},
            {"linear_sync", P_FIRSTSEG, [=](const AlgoParams& p) { c_.linear_sync(s(), n(), r(), n(), p.root, p.first_seg); }, {}},
            {"linear", 0, [=](const AlgoParams& p) { c_.linear(s(), n(), r(), n(), p.root); }, {}},
        };
    }
};

template <typename T>
class ScatterRunner : public Runner<T> {
    using B = Runner<T>;
    Scatter<T> c_;
public:
    ScatterRunner(MPI_Comm comm, const Options& o, const OpInfo& op) : B(comm, o, op), c_(comm) {}
    const char* size_semantics() const override { return "bytes received per rank"; }
    size_t mem_bytes(size_t n) const override { return (2 + 2 * (size_t)this->size_) * n * sizeof(T); }
    double payload_bytes(size_t n) const override { return (double)n * sizeof(T) * this->size_; }
    void setup(size_t n) override
    {
        this->count_ = n;
        if (this->rank_ == this->opt_.root) this->fill_input(this->sbuf_, n * this->size_);
        else this->sbuf_.assign(1, T(0));
        this->poison(this->rbuf_, n);
    }
    void native() override
    {
        MPI_CHECK(NATIVE(MPI_Scatter)(this->sbuf_.data(), ncnt(this->count_), this->dt_, this->rbuf_.data(),
                                      ncnt(this->count_), this->dt_, this->opt_.root, this->comm_));
    }
    std::pair<const T*, size_t> result() const override { return {this->rbuf_.data(), this->count_}; }
    std::vector<AlgoDef> algorithms() override
    {
        auto s = [this]() { return (const T*)this->sbuf_.data(); };
        auto r = [this]() { return this->rbuf_.data(); };
        auto n = [this]() { return this->count_; };
        return {
            {"binomial", 0, [=](const AlgoParams& p) { c_.binomial(s(), n(), r(), n(), p.root); }, {}},
            {"linear", 0, [=](const AlgoParams& p) { c_.linear(s(), n(), r(), n(), p.root); }, {}},
            {"linear_nb", P_MAXREQ, [=](const AlgoParams& p) { c_.linear_nb(s(), n(), r(), n(), p.root, p.max_reqs); }, {}},
        };
    }
};

template <typename T>
class BarrierRunner : public Runner<T> {
    using B = Runner<T>;
    Barrier c_;
public:
    BarrierRunner(MPI_Comm comm, const Options& o, const OpInfo& op) : B(comm, o, op), c_(comm) {}
    bool sized() const override { return false; }
    const char* size_semantics() const override { return "(no data)"; }
    size_t mem_bytes(size_t) const override { return 0; }
    double payload_bytes(size_t) const override { return 0; }
    void setup(size_t) override { this->count_ = 0; }
    void native() override { MPI_CHECK(MPI_Barrier(this->comm_)); }
    std::pair<const T*, size_t> result() const override { return {nullptr, 0}; }
    std::vector<AlgoDef> algorithms() override
    {
        return {
            {"doublering", 0, [=](const AlgoParams&) { c_.doublering(); }, {}},
            {"recursivedoubling", 0, [=](const AlgoParams&) { c_.recursivedoubling(); }, {}},
            {"bruck", 0, [=](const AlgoParams&) { c_.bruck(); }, {}},
            {"two_procs", 0, [=](const AlgoParams&) { c_.two_procs(); }, {}},
            {"linear", 0, [=](const AlgoParams&) { c_.linear(); }, {}},
            {"tree", 0, [=](const AlgoParams&) { c_.tree(); }, {}},
        };
    }
};

template <typename T>
class ReduceScatterRunner : public Runner<T> {
    using B = Runner<T>;
    ReduceScatter<T> c_;
    std::vector<size_t> rcounts_;
    size_t total_ = 0;
public:
    ReduceScatterRunner(MPI_Comm comm, const Options& o, const OpInfo& op) : B(comm, o, op), c_(comm) {}
    bool reduction() const override { return true; }
    const char* size_semantics() const override { return "average bytes received per rank (--vdist)"; }
    size_t mem_bytes(size_t n) const override { return 5 * (size_t)this->size_ * n * sizeof(T); }
    double payload_bytes(size_t n) const override { return (double)n * sizeof(T) * this->size_; }
    void setup(size_t n) override
    {
        const int P = this->size_;
        this->count_ = n;
        rcounts_.resize(P);
        total_ = 0;
        for (int i = 0; i < P; ++i) total_ += (rcounts_[i] = vcount(this->opt_, n, i, 0, P));
        this->fill_input(this->sbuf_, total_);
        this->poison(this->rbuf_, rcounts_[this->rank_]);
    }
    void native() override
    {
        auto rc = to_ncounts(rcounts_);
        MPI_CHECK(NATIVE(MPI_Reduce_scatter)(this->sbuf_.data(), this->rbuf_.data(), rc.data(), this->dt_,
                                             this->op_.op, this->comm_));
    }
    std::pair<const T*, size_t> result() const override { return {this->rbuf_.data(), rcounts_[this->rank_]}; }
    std::vector<AlgoDef> algorithms() override
    {
        auto s = [this]() { return (const T*)this->sbuf_.data(); };
        auto r = [this]() { return this->rbuf_.data(); };
        return {
            {"nonoverlapping", P_SEG | P_MAXREQ, [=](const AlgoParams& p) { c_.nonoverlapping(s(), r(), rcounts_, p.op, p.segsize, p.max_reqs); }, {}},
            {"recursivehalving", 0, [=](const AlgoParams& p) { c_.recursivehalving(s(), r(), rcounts_, p.op); }, {}},
            {"ring", 0, [=](const AlgoParams& p) { c_.ring(s(), r(), rcounts_, p.op); }, {}},
            {"butterfly", 0, [=](const AlgoParams& p) { c_.butterfly(s(), r(), rcounts_, p.op); }, {}},
        };
    }
};

template <typename T>
class ReduceScatterBlockRunner : public Runner<T> {
    using B = Runner<T>;
    ReduceScatterBlock<T> c_;
public:
    ReduceScatterBlockRunner(MPI_Comm comm, const Options& o, const OpInfo& op) : B(comm, o, op), c_(comm) {}
    bool reduction() const override { return true; }
    const char* size_semantics() const override { return "bytes received per rank"; }
    size_t mem_bytes(size_t n) const override { return 5 * (size_t)this->size_ * n * sizeof(T); }
    double payload_bytes(size_t n) const override { return (double)n * sizeof(T) * this->size_; }
    void setup(size_t n) override
    {
        this->count_ = n;
        this->fill_input(this->sbuf_, n * this->size_);
        this->poison(this->rbuf_, n);
    }
    void native() override
    {
        MPI_CHECK(NATIVE(MPI_Reduce_scatter_block)(this->sbuf_.data(), this->rbuf_.data(), ncnt(this->count_),
                                                   this->dt_, this->op_.op, this->comm_));
    }
    std::pair<const T*, size_t> result() const override { return {this->rbuf_.data(), this->count_}; }
    std::vector<AlgoDef> algorithms() override
    {
        auto s = [this]() { return (const T*)this->sbuf_.data(); };
        auto r = [this]() { return this->rbuf_.data(); };
        auto n = [this]() { return this->count_; };
        return {
            {"basic_linear", 0, [=](const AlgoParams& p) { c_.basic_linear(s(), r(), n(), p.op); }, {}},
            {"recursivedoubling", 0, [=](const AlgoParams& p) { c_.recursivedoubling(s(), r(), n(), p.op); }, {}},
            {"recursivehalving", 0, [=](const AlgoParams& p) { c_.recursivehalving(s(), r(), n(), p.op); }, {}},
            {"butterfly", 0, [=](const AlgoParams& p) { c_.butterfly(s(), r(), n(), p.op); }, {}},
        };
    }
};

inline std::vector<std::string> collective_names()
{
    return {"allgather", "allgatherv", "allreduce", "alltoall", "alltoallv", "barrier",
            "bcast", "gather", "reduce", "reduce_scatter", "reduce_scatter_block", "scatter"};
}

template <typename T>
std::unique_ptr<Runner<T>> make_runner(const std::string& c, MPI_Comm comm, const Options& o, const OpInfo& op)
{
    if (c == "allgather") return std::make_unique<AllgatherRunner<T>>(comm, o, op);
    if (c == "allgatherv") return std::make_unique<AllgathervRunner<T>>(comm, o, op);
    if (c == "allreduce") return std::make_unique<AllreduceRunner<T>>(comm, o, op);
    if (c == "alltoall") return std::make_unique<AlltoallRunner<T>>(comm, o, op);
    if (c == "alltoallv") return std::make_unique<AlltoallvRunner<T>>(comm, o, op);
    if (c == "barrier") return std::make_unique<BarrierRunner<T>>(comm, o, op);
    if (c == "bcast") return std::make_unique<BcastRunner<T>>(comm, o, op);
    if (c == "gather") return std::make_unique<GatherRunner<T>>(comm, o, op);
    if (c == "reduce") return std::make_unique<ReduceRunner<T>>(comm, o, op);
    if (c == "reduce_scatter") return std::make_unique<ReduceScatterRunner<T>>(comm, o, op);
    if (c == "reduce_scatter_block") return std::make_unique<ReduceScatterBlockRunner<T>>(comm, o, op);
    if (c == "scatter") return std::make_unique<ScatterRunner<T>>(comm, o, op);
    throw std::invalid_argument("unknown collective '" + c + "'");
}

/* ------------------------------------------------------------------------- */
/* harness                                                                    */
/* ------------------------------------------------------------------------- */
struct Variant {
    std::string label;
    std::string base;   /* algorithm name */
    AlgoParams params;
    const AlgoDef* def = nullptr;  /* nullptr = native */
};

struct Cell {
    enum State { Ok, NA, Skip, Fail } state = Ok;
    double avg = 0, min = 0, max = 0;  /* microseconds */
    std::string note;
};

/* expand every algorithm into one variant per combination of the tunables it uses */
inline std::vector<Variant> expand(const std::vector<AlgoDef>& defs, const Options& o, MPI_Op op,
                                   bool include_native)
{
    std::vector<Variant> out;
    auto selected = [&](const std::string& n) {
        if (o.algos.empty()) return true;
        return std::find(o.algos.begin(), o.algos.end(), n) != o.algos.end();
    };
    for (auto& d : defs) {
        if (!selected(d.name)) continue;
        std::vector<size_t> segs = (d.params & P_SEG) ? o.segsize : std::vector<size_t>{0};
        std::vector<int> fans = (d.params & P_FANOUT) ? o.fanout : std::vector<int>{4};
        std::vector<int> rads = (d.params & P_RADIX) ? o.radix : std::vector<int>{4};
        std::vector<int> mrs = (d.params & P_MAXREQ) ? o.max_reqs : std::vector<int>{0};
        std::vector<size_t> fss = (d.params & P_FIRSTSEG) ? o.first_seg : std::vector<size_t>{1024};
        for (size_t sg : segs)
            for (int fo : fans)
                for (int rx : rads)
                    for (int mr : mrs)
                        for (size_t fs : fss) {
                            Variant v;
                            v.base = d.name;
                            v.def = &d;
                            v.params.segsize = sg; v.params.fanout = fo; v.params.radix = rx;
                            v.params.max_reqs = mr; v.params.first_seg = fs;
                            v.params.root = o.root; v.params.op = op;
                            std::string l = d.name;
                            std::vector<std::string> tags;
                            if ((d.params & P_SEG) && o.segsize.size() > 1) tags.push_back("seg=" + human_bytes(sg));
                            if ((d.params & P_FANOUT) && o.fanout.size() > 1) tags.push_back("fo=" + std::to_string(fo));
                            if ((d.params & P_RADIX) && o.radix.size() > 1) tags.push_back("k=" + std::to_string(rx));
                            if ((d.params & P_MAXREQ) && o.max_reqs.size() > 1) tags.push_back("mr=" + std::to_string(mr));
                            if ((d.params & P_FIRSTSEG) && o.first_seg.size() > 1) tags.push_back("fs=" + human_bytes(fs));
                            if (!tags.empty()) {
                                l += "[";
                                for (size_t t = 0; t < tags.size(); ++t) l += (t ? "," : "") + tags[t];
                                l += "]";
                            }
                            v.label = l;
                            out.push_back(v);
                        }
    }
    if (include_native && (o.algos.empty() || selected("native"))) {
        Variant v;
        v.base = v.label = "native";
        v.params.root = o.root;
        v.params.op = op;
        out.push_back(v);
    }
    return out;
}

inline std::vector<size_t> size_list(const Options& o, size_t elem)
{
    std::vector<size_t> raw;
    if (!o.sizes.empty()) {
        raw = o.sizes;
    } else {
        if (o.min_bytes == 0) raw.push_back(0);
        double s = (double)std::max(o.min_bytes, elem);
        while ((size_t)s <= o.max_bytes) {
            raw.push_back((size_t)s);
            double nx = s * o.factor;
            if ((size_t)nx == (size_t)s) nx = s + 1;
            s = nx;
        }
    }
    /* round to whole elements and de-duplicate */
    std::vector<size_t> counts;
    for (size_t b : raw) {
        size_t c = b / elem;
        if (b > 0 && c == 0) continue;
        if (std::find(counts.begin(), counts.end(), c) == counts.end()) counts.push_back(c);
    }
    return counts;
}

template <typename T>
int run(MPI_Comm world, const Options& o, const std::string& type_name)
{
    int rank, size;
    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &size);
    if (o.root < 0 || o.root >= size) throw std::invalid_argument("--root out of range");

    OpInfo op = make_op<T>(o.op);
    int exit_code = 0;
    {
        auto runner = make_runner<T>(o.coll, world, o, op);
        auto defs = runner->algorithms();
        if (!o.algos.empty()) {
            for (auto& a : o.algos) {
                bool found = (a == "native");
                for (auto& d : defs) found = found || d.name == a;
                if (!found) throw std::invalid_argument("collective '" + o.coll + "' has no algorithm '" + a + "'");
            }
        }
        auto variants = expand(defs, o, op.op, o.native);
        std::vector<size_t> counts = runner->sized() ? size_list(o, sizeof(T)) : std::vector<size_t>{0};

        if (rank == 0 && !o.quiet) {
            char ver[MPI_MAX_LIBRARY_VERSION_STRING];
            int vlen = 0;
            MPI_Get_library_version(ver, &vlen);
            std::string v(ver, vlen);
            v = v.substr(0, v.find('\n'));
            std::printf("# collage: point-to-point collective algorithm benchmark\n");
            std::printf("# MPI library  : %s\n", v.c_str());
            std::printf("# collective   : %s (%d ranks)\n", o.coll.c_str(), size);
            std::printf("# size means   : %s\n", runner->size_semantics());
            std::printf("# datatype     : %s (%zu bytes)\n", type_name.c_str(), sizeof(T));
            if (runner->reduction())
                std::printf("# operation    : %s (%s)\n", op.name.c_str(), op.commute ? "commutative" : "non-commutative");
            if (o.coll == "bcast" || o.coll == "reduce" || o.coll == "gather" || o.coll == "scatter")
                std::printf("# root         : %d\n", o.root);
            if (o.coll == "alltoallv" || o.coll == "allgatherv" || o.coll == "reduce_scatter")
                std::printf("# distribution : %s (seed %u)\n", o.vdist.c_str(), o.seed);
            std::printf("# iterations   : %d (warmup %d), %d (warmup %d) above %s\n", o.iters, o.warmup,
                        o.iters_large, o.warmup_large, human_bytes(o.large_threshold).c_str());
            std::printf("# check        : %s\n", o.check || o.check_only ? "against native MPI" : "off");
            std::printf("# times        : average latency in microseconds (mean over ranks)\n");
            std::fflush(stdout);
        }

        /* results[count index][variant index] */
        std::vector<std::vector<Cell>> results(counts.size(), std::vector<Cell>(variants.size()));
        size_t failures = 0;

        for (size_t ci = 0; ci < counts.size(); ++ci) {
            const size_t n = counts[ci];
            const size_t bytes = n * sizeof(T);
            const bool large = bytes > o.large_threshold;
            const int iters = large ? o.iters_large : o.iters;
            const int warm = large ? o.warmup_large : o.warmup;

            bool fits = runner->mem_bytes(n) <= o.max_mem;
            if (fits) runner->setup(n);

            /* reference result for verification */
            bool have_ref = false;
            if (fits && (o.check || o.check_only) && runner->sized()) {
                runner->native();
                runner->snapshot();
                have_ref = true;
            }

            for (size_t vi = 0; vi < variants.size(); ++vi) {
                const Variant& v = variants[vi];
                Cell& cell = results[ci][vi];
                size_t extra = (v.def && v.def->extra_mem) ? v.def->extra_mem(n) : 0;
                if (!fits || runner->mem_bytes(n) + extra > o.max_mem) {
                    cell.state = Cell::Skip;
                    cell.note = "exceeds --max-mem";
                    continue;
                }
                auto call = [&]() {
                    if (v.def) v.def->fn(v.params);
                    else runner->native();
                };
                try {
                    if (have_ref) {
                        runner->setup(n);
                        runner->prepare(v.base);
                        call();
                        size_t bad = runner->compare();
                        unsigned long long gbad = bad;
                        MPI_Allreduce(MPI_IN_PLACE, &gbad, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, world);
                        if (gbad) {
                            cell.state = Cell::Fail;
                            cell.note = std::to_string(gbad) + " wrong elements";
                            ++failures;
                        }
                        runner->setup(n);
                    } else if (o.check || o.check_only) {
                        /* barrier: just make sure it completes */
                        call();
                    }
                    if (o.check_only) continue;

                    runner->prepare(v.base);
                    for (int w = 0; w < warm; ++w) call();
                    double sum = 0;
                    for (int it = 0; it < iters; ++it) {
                        MPI_Barrier(world);
                        double t0 = MPI_Wtime();
                        call();
                        sum += MPI_Wtime() - t0;
                    }
                    double mine = iters > 0 ? sum / iters * 1e6 : 0.0;
                    double tot = 0, mn = 0, mx = 0;
                    MPI_Allreduce(&mine, &tot, 1, MPI_DOUBLE, MPI_SUM, world);
                    MPI_Allreduce(&mine, &mn, 1, MPI_DOUBLE, MPI_MIN, world);
                    MPI_Allreduce(&mine, &mx, 1, MPI_DOUBLE, MPI_MAX, world);
                    if (cell.state != Cell::Fail) cell.state = Cell::Ok;
                    cell.avg = tot / size;
                    cell.min = mn;
                    cell.max = mx;
                } catch (const NotApplicable& e) {
                    cell.state = Cell::NA;
                    cell.note = e.what();
                }
                MPI_Barrier(world);
            }

            /* progressive output: one line per size */
            if (rank == 0 && !o.quiet) {
                if (ci == 0) {
                    std::printf("%-12s", runner->sized() ? "# bytes" : "#");
                    for (auto& v : variants) std::printf(" %*s", (int)std::max<size_t>(12, v.label.size()), v.label.c_str());
                    std::printf("\n");
                }
                if (runner->sized()) std::printf("%-12zu", bytes);
                else std::printf("%-12s", "-");
                for (size_t vi = 0; vi < variants.size(); ++vi) {
                    const Cell& c = results[ci][vi];
                    int w = (int)std::max<size_t>(12, variants[vi].label.size());
                    char buf[64];
                    switch (c.state) {
                        case Cell::Ok:
                            if (o.check_only) std::snprintf(buf, sizeof buf, "ok");
                            else std::snprintf(buf, sizeof buf, "%.2f", c.avg);
                            break;
                        case Cell::NA: std::snprintf(buf, sizeof buf, "n/a"); break;
                        case Cell::Skip: std::snprintf(buf, sizeof buf, "skip"); break;
                        case Cell::Fail:
                            if (o.check_only) std::snprintf(buf, sizeof buf, "FAIL");
                            else std::snprintf(buf, sizeof buf, "FAIL:%.2f", c.avg);
                            break;
                    }
                    std::printf(" %*s", w, buf);
                }
                std::printf("\n");
                std::fflush(stdout);
            }
        }

        if (rank == 0 && !o.quiet) {
            /* explain n/a cells once */
            std::map<std::string, std::string> notes;
            for (auto& row : results)
                for (size_t vi = 0; vi < variants.size(); ++vi)
                    if (row[vi].state == Cell::NA) notes[variants[vi].label] = row[vi].note;
            for (auto& kv : notes) std::printf("# n/a %s: %s\n", kv.first.c_str(), kv.second.c_str());
            if (o.check || o.check_only)
                std::printf("# verification: %s\n", failures ? (std::to_string(failures) + " FAILED case(s)").c_str() : "all passed");
        }

        if (rank == 0 && !o.csv.empty()) {
            FILE* f = std::fopen(o.csv.c_str(), "w");
            if (!f) {
                std::fprintf(stderr, "cannot open %s for writing\n", o.csv.c_str());
            } else {
                std::fprintf(f, "collective,algorithm,variant,type,op,ranks,root,bytes,count,status,"
                                "avg_us,min_us,max_us,algbw_MBps,note\n");
                for (size_t ci = 0; ci < counts.size(); ++ci)
                    for (size_t vi = 0; vi < variants.size(); ++vi) {
                        const Cell& c = results[ci][vi];
                        const char* st = c.state == Cell::Ok ? "ok" : c.state == Cell::NA ? "n/a"
                                         : c.state == Cell::Skip ? "skip" : "FAIL";
                        double bw = 0;
                        size_t bytes = counts[ci] * sizeof(T);
                        double pb = runner->payload_bytes(counts[ci]);
                        if ((c.state == Cell::Ok || c.state == Cell::Fail) && c.avg > 0) bw = pb / c.avg;  /* B/us == MB/s */
                        std::string note = c.note;
                        std::replace(note.begin(), note.end(), ',', ';');
                        std::fprintf(f, "%s,%s,%s,%s,%s,%d,%d,%zu,%zu,%s,%.3f,%.3f,%.3f,%.2f,%s\n", o.coll.c_str(),
                                     variants[vi].base.c_str(), variants[vi].label.c_str(), type_name.c_str(),
                                     runner->reduction() ? op.name.c_str() : "-", size, o.root, bytes, counts[ci], st,
                                     c.avg, c.min, c.max, bw, note.c_str());
                    }
                std::fclose(f);
                if (!o.quiet) std::printf("# statistics written to %s\n", o.csv.c_str());
            }
        }
        exit_code = failures ? 2 : 0;
    }
    free_op(op);
    return exit_code;
}

}  // namespace bench

#endif
