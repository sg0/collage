#pragma once
#ifndef COLLAGE_MEMORY_HPP
#define COLLAGE_MEMORY_HPP

/*
 * Memory layer: lets every algorithm run unchanged on host or on GPU buffers.
 *
 * The algorithms touch buffers outside MPI in three ways: local copies,
 * temporary buffers, and local reductions. All three go through this file:
 *
 *   Buf<T>          temporary buffer in the current memory space (replaces
 *                   std::vector<T>); device memory comes from a caching pool so
 *                   that per-call temporaries do not cost a cudaMalloc each time
 *   mem::copy       overlap-safe copy (memmove semantics) in the current space
 *   mem::reduce     inout = in (op) inout, in the current space:
 *                     host   -> MPI_Reduce_local
 *                     device -> CUDA kernel (--gpu-reduce kernel, default),
 *                               MPI_Reduce_local on device pointers (mpi), or
 *                               staged through pinned host memory (host).
 *                               User-defined ops are always staged: their
 *                               function is host code.
 *
 * The space is process-wide and chosen once, before any collective object is
 * built (mem::configure). Without COLLAGE_CUDA only the host space exists and
 * the behaviour is the plain host one.
 *
 * By default every device operation is synchronous with respect to the host
 * (the stream is synchronised before returning), because the next step is
 * usually an MPI call on the same buffer and MPI does not know about our
 * stream. With a stream-triggered transport (--transport mpich|acx|cray, see
 * transport.hpp) the MPI calls are queued on the same stream, so copies and
 * kernel reductions are only queued too (async mode): the stream orders them,
 * and the host waits only where it has to (sync()).
 */

#include <mpi.h>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(COLLAGE_CUDA)
#include <cuda_runtime.h>
#include "gpu_ops.hpp"
#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        cudaError_t _e = (call);                                                               \
        if (_e != cudaSuccess) {                                                               \
            std::fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__,                \
                         cudaGetErrorString(_e));                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                      \
        }                                                                                      \
    } while (0)
#endif

namespace mem {

enum class Space { Host, Device };
enum class ReduceMode { Kernel, Mpi, Host };

inline bool cuda_built()
{
#if defined(COLLAGE_CUDA)
    return true;
#else
    return false;
#endif
}

struct State {
    Space space = Space::Host;
    ReduceMode rmode = ReduceMode::Kernel;
#if defined(COLLAGE_CUDA)
    cudaStream_t stream = nullptr;
    std::map<size_t, std::vector<void*>> pool;  /* capacity -> free device blocks */
    std::map<void*, size_t> live;                /* device block -> capacity */
    void* stage[2] = {nullptr, nullptr};         /* pinned host staging */
    size_t stage_bytes = 0;
    int device = -1;
#endif
    /* async mode (stream-triggered transports): queue device work, do not wait */
    bool async = false;
    bool pending = false;              /* work queued since the last sync() */
    void (*before_stream_op)() = nullptr; /* transport hook, run before queueing work */
    void (*before_sync)() = nullptr;      /* transport hook, run before waiting */
    /* statistics, for the run header / debugging */
    size_t reductions_kernel = 0, reductions_mpi = 0, reductions_staged = 0;
};

inline State& st()
{
    static State s;
    return s;
}

inline bool device() { return st().space == Space::Device; }

/* ------------------------------------------------------------------------- */
/* configuration                                                              */
/* ------------------------------------------------------------------------- */
#if defined(COLLAGE_CUDA)
/* Pick the GPU for this process. Call before MPI_Init when the local rank is
 * known from the launcher, otherwise after (see bench.cpp).
 * Without a map: device = local_rank % visible devices.
 * With a map (--gpu-map 3,2,1,0): device = map[local_rank % map size], e.g. to
 * give each rank the GPU closest to the cores it is bound to. */
inline int select_device(int local_rank, const std::vector<int>& map = {})
{
    int n = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n));
    if (n <= 0) throw std::runtime_error("--gpu: no CUDA device visible to this process");
    const int lr = local_rank >= 0 ? local_rank : 0;
    int dev = map.empty() ? lr % n : map[(size_t)lr % map.size()];
    if (dev < 0 || dev >= n)
        throw std::runtime_error("--gpu-map: device " + std::to_string(dev) + " does not exist (" +
                                 std::to_string(n) + " visible)");
    CUDA_CHECK(cudaSetDevice(dev));
    st().device = dev;
    return dev;
}

inline std::string device_name()
{
    cudaDeviceProp p;
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    char b[320];
    std::snprintf(b, sizeof b, "%s (device %d, PCI %04x:%02x:%02x)", p.name, dev, p.pciDomainID, p.pciBusID,
                  p.pciDeviceID);
    return b;
}

inline int device_count()
{
    int n = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n));
    return n;
}
#endif

inline void configure(Space s, ReduceMode r)
{
    if (s == Space::Device && !cuda_built())
        throw std::invalid_argument("--gpu: this binary was built without CUDA support (rebuild with `make CUDA=1`)");
    st().space = s;
    st().rmode = r;
#if defined(COLLAGE_CUDA)
    if (s == Space::Device && !st().stream) CUDA_CHECK(cudaStreamCreateWithFlags(&st().stream, cudaStreamNonBlocking));
#endif
}

/* ------------------------------------------------------------------------- */
/* allocation                                                                 */
/* ------------------------------------------------------------------------- */
#if defined(COLLAGE_CUDA)
inline size_t pool_class(size_t bytes)
{
    size_t c = 256;
    while (c < bytes) c <<= 1;
    return c;
}
inline void* device_alloc(size_t bytes)
{
    if (bytes == 0) return nullptr;
    State& s = st();
    size_t cap = pool_class(bytes);
    auto& fl = s.pool[cap];
    void* p = nullptr;
    if (!fl.empty()) {
        p = fl.back();
        fl.pop_back();
    } else {
        CUDA_CHECK(cudaMalloc(&p, cap));
    }
    s.live[p] = cap;
    return p;
}
inline void device_free(void* p)
{
    if (!p) return;
    State& s = st();
    auto it = s.live.find(p);
    if (it == s.live.end()) {
        std::fprintf(stderr, "collage: freeing unknown device block %p\n", p);
        std::abort();
    }
    s.pool[it->second].push_back(p);
    s.live.erase(it);
}
#endif

/* release cached device memory and the stream (call before MPI_Finalize) */
inline void shutdown()
{
#if defined(COLLAGE_CUDA)
    State& s = st();
    for (auto& kv : s.pool)
        for (void* p : kv.second) cudaFree(p);
    s.pool.clear();
    for (int i = 0; i < 2; ++i)
        if (s.stage[i]) cudaFreeHost(s.stage[i]);
    s.stage[0] = s.stage[1] = nullptr;
    s.stage_bytes = 0;
    if (s.stream) cudaStreamDestroy(s.stream);
    s.stream = nullptr;
#endif
}

/* ------------------------------------------------------------------------- */
/* copies                                                                     */
/* ------------------------------------------------------------------------- */
inline bool overlap(const void* a, const void* b, size_t bytes)
{
    const char* x = static_cast<const char*>(a);
    const char* y = static_cast<const char*>(b);
    return x < y + bytes && y < x + bytes;
}

inline void sync()
{
#if defined(COLLAGE_CUDA)
    if (device()) {
        if (st().before_sync) st().before_sync();
        CUDA_CHECK(cudaStreamSynchronize(st().stream));
    }
#endif
    st().pending = false;
}

/* async mode: something was queued on the stream */
inline void mark_pending() { st().pending = true; }
inline bool pending() { return st().pending; }
/* wait for queued work, if any (before the host touches device data) */
inline void sync_if_pending()
{
    if (st().pending) sync();
}
/* before queueing a copy or a kernel */
inline void before_stream_op()
{
    if (st().before_stream_op) st().before_stream_op();
}
/* after queueing work: wait for it, or only note it in async mode */
inline void queued()
{
    if (st().async) mark_pending();
    else sync();
}

/* memmove semantics in the current memory space. With --gpu both pointers
 * are device memory (algorithms only hold device buffers). */
inline void copy(void* dst, const void* src, size_t bytes)
{
    if (bytes == 0 || dst == src) return;
#if defined(COLLAGE_CUDA)
    if (device()) {
        before_stream_op();
        if (overlap(dst, src, bytes)) {
            /* cudaMemcpy is undefined for overlapping ranges: go through a temporary.
             * In async mode the block goes back to the pool before the copies
             * run; that is safe because every later user of device memory is
             * either queued behind them or waits for the stream first. */
            void* t = device_alloc(bytes);
            CUDA_CHECK(cudaMemcpyAsync(t, src, bytes, cudaMemcpyDefault, st().stream));
            CUDA_CHECK(cudaMemcpyAsync(dst, t, bytes, cudaMemcpyDefault, st().stream));
            queued();
            device_free(t);
        } else {
            CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, st().stream));
            queued();
        }
        return;
    }
#endif
    std::memmove(dst, src, bytes);
}

/* host <-> current space (used by the harness only) */
inline void upload(void* dst, const void* host_src, size_t bytes)
{
    if (bytes == 0) return;
#if defined(COLLAGE_CUDA)
    if (device()) {
        CUDA_CHECK(cudaMemcpyAsync(dst, host_src, bytes, cudaMemcpyHostToDevice, st().stream));
        sync();
        return;
    }
#endif
    if (dst != host_src) std::memmove(dst, host_src, bytes);
}
/* (upload/download wait for the stream, so they also order behind queued work) */
inline void download(void* host_dst, const void* src, size_t bytes)
{
    if (bytes == 0) return;
#if defined(COLLAGE_CUDA)
    if (device()) {
        CUDA_CHECK(cudaMemcpyAsync(host_dst, src, bytes, cudaMemcpyDeviceToHost, st().stream));
        sync();
        return;
    }
#endif
    if (host_dst != src) std::memmove(host_dst, src, bytes);
}

/* ------------------------------------------------------------------------- */
/* local reductions                                                           */
/* ------------------------------------------------------------------------- */
/* MPI_Reduce_local on whatever memory the pointers refer to, large-count aware */
inline void reduce_local_mpi(const void* in, void* inout, size_t count, MPI_Datatype dt, MPI_Op op)
{
    if (0 == count) return;
#if defined(MPI_VERSION) && (MPI_VERSION >= 4) && !defined(COLLAGE_NO_LARGE_COUNT)
    int rc = MPI_Reduce_local_c(in, inout, (MPI_Count)count, dt, op);
    if (rc != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc);
#else
    int tsize = 0;
    MPI_Type_size(dt, &tsize);
    const char* s = static_cast<const char*>(in);
    char* t = static_cast<char*>(inout);
    while (count > 0) {
        int c = (int)std::min(count, (size_t)INT_MAX);
        int rc = MPI_Reduce_local(s, t, c, dt, op);
        if (rc != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc);
        s += (size_t)c * tsize;
        t += (size_t)c * tsize;
        count -= c;
    }
#endif
}

#if defined(COLLAGE_CUDA)
inline collage_gpu::DType dtype_code(MPI_Datatype dt)
{
    using collage_gpu::DType;
    if (dt == MPI_INT8_T) return DType::I8;
    if (dt == MPI_UINT8_T) return DType::U8;
    if (dt == MPI_INT16_T) return DType::I16;
    if (dt == MPI_UINT16_T) return DType::U16;
    if (dt == MPI_INT32_T) return DType::I32;
    if (dt == MPI_UINT32_T) return DType::U32;
    if (dt == MPI_INT64_T) return DType::I64;
    if (dt == MPI_UINT64_T) return DType::U64;
    if (dt == MPI_FLOAT) return DType::F32;
    if (dt == MPI_DOUBLE) return DType::F64;
    return DType::Unknown;
}
#endif

/* is `op` one of the predefined operations with a device kernel? */
inline int op_code(MPI_Op op)
{
    const MPI_Op ops[] = {MPI_SUM, MPI_PROD, MPI_MAX, MPI_MIN, MPI_LAND, MPI_LOR, MPI_LXOR, MPI_BAND, MPI_BOR, MPI_BXOR};
    for (int i = 0; i < 10; ++i)
        if (op == ops[i]) return i;  /* same order as collage_gpu::OpCode */
    return -1;
}

#if defined(COLLAGE_CUDA)
inline void reduce_staged(const void* in, void* inout, size_t count, MPI_Datatype dt, MPI_Op op)
{
    State& s = st();
    int tsize = 0;
    MPI_Type_size(dt, &tsize);
    const size_t bytes = count * (size_t)tsize;
    if (bytes > s.stage_bytes) {
        for (int i = 0; i < 2; ++i) {
            if (s.stage[i]) CUDA_CHECK(cudaFreeHost(s.stage[i]));
            CUDA_CHECK(cudaMallocHost(&s.stage[i], bytes));
        }
        s.stage_bytes = bytes;
    }
    before_stream_op();
    CUDA_CHECK(cudaMemcpyAsync(s.stage[0], in, bytes, cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.stage[1], inout, bytes, cudaMemcpyDeviceToHost, s.stream));
    sync();
    reduce_local_mpi(s.stage[0], s.stage[1], count, dt, op);
    CUDA_CHECK(cudaMemcpyAsync(inout, s.stage[1], bytes, cudaMemcpyHostToDevice, s.stream));
    sync();
    ++s.reductions_staged;
}
#endif

/* inout = in (op) inout, in the current memory space */
inline void reduce(const void* in, void* inout, size_t count, MPI_Datatype dt, MPI_Op op)
{
    if (0 == count) return;
#if defined(COLLAGE_CUDA)
    if (device()) {
        State& s = st();
        const int oc = op_code(op);
        if (oc < 0 || s.rmode == ReduceMode::Host) {
            reduce_staged(in, inout, count, dt, op);  /* user op, or forced */
            return;
        }
        if (s.rmode == ReduceMode::Mpi) {
            sync_if_pending();  /* MPI_Reduce_local does not know our stream */
            reduce_local_mpi(in, inout, count, dt, op);
            ++s.reductions_mpi;
            return;
        }
        before_stream_op();
        if (collage_gpu::device_reduce(dtype_code(dt), (collage_gpu::OpCode)oc, in, inout, count, s.stream)) {
            queued();
            ++s.reductions_kernel;
            return;
        }
        reduce_staged(in, inout, count, dt, op);  /* no kernel for this type */
        return;
    }
#endif
    reduce_local_mpi(in, inout, count, dt, op);
}

inline const char* reduce_mode_name(ReduceMode m)
{
    return m == ReduceMode::Kernel ? "kernel" : m == ReduceMode::Mpi ? "mpi" : "host";
}

}  // namespace mem

/* ------------------------------------------------------------------------- */
/* Buf<T>: temporary buffer in the current memory space                       */
/* ------------------------------------------------------------------------- */
/* Only what the algorithms use: construction with a size, resize (contents
 * are not preserved), data(), size(). Host buffers are zero-initialised like
 * std::vector; device buffers are not. Host code must never dereference
 * data() of a device Buf. */
template <typename T>
class Buf {
public:
    Buf() = default;
    explicit Buf(size_t n) { resize(n); }
    ~Buf() { release(); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    Buf(Buf&& o) noexcept : p_(o.p_), n_(o.n_), dev_(o.dev_) { o.p_ = nullptr; o.n_ = 0; }
    Buf& operator=(Buf&& o) noexcept
    {
        if (this != &o) {
            release();
            p_ = o.p_; n_ = o.n_; dev_ = o.dev_;
            o.p_ = nullptr; o.n_ = 0;
        }
        return *this;
    }

    void resize(size_t n)
    {
        if (n == n_ && p_) return;
        release();
        if (n == 0) return;
        dev_ = mem::device();
#if defined(COLLAGE_CUDA)
        if (dev_) {
            p_ = static_cast<T*>(mem::device_alloc(n * sizeof(T)));
            n_ = n;
            return;
        }
#endif
        p_ = new T[n]();
        n_ = n;
    }
    T* data() { return p_; }
    const T* data() const { return p_; }
    size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }

private:
    T* p_ = nullptr;
    size_t n_ = 0;
    bool dev_ = false;

    void release()
    {
        if (!p_) return;
#if defined(COLLAGE_CUDA)
        if (dev_) mem::device_free(p_);
        else delete[] p_;
#else
        delete[] p_;
#endif
        p_ = nullptr;
        n_ = 0;
    }
};

#endif /* COLLAGE_MEMORY_HPP */
