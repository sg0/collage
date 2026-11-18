#pragma once
#ifndef COLLAGE_UCP_TRANSPORT_HPP
#define COLLAGE_UCP_TRANSPORT_HPP

/*
 * Host side of the UCP device transport (--transport ucp, build ST=ucp):
 * device-initiated one-sided communication with the UCX GPU device API
 * (UCX >= 1.20; GDA lanes = rc_gda on ConnectX InfiniBand/RoCE, CUDA-IPC lanes
 * on a node). See ucp_proto.hpp for the protocol.
 *
 * MPI is used only to bootstrap: exchange worker addresses, rkeys, arena and
 * control-area addresses. At run time the host only enqueues kernels; the
 * GPU triggers every transfer and signal itself.
 *
 *  - One registered device arena per process (--ucp-heap); the device memory
 *    pool of memory.hpp draws from it, so every buffer an algorithm uses is
 *    reachable by peers as (peer arena base + offset).
 *  - One control area per communicator (CollBase): rts/data counters and
 *    descriptor rings (--ucp-slots receives in flight per peer).
 *  - Each operation gets one device copy for the whole call (kernels keep
 *    its state there). isend records the operation (and packs a derived
 *    datatype); irecv launches the receiver-ready kernel at once; wait/waitall
 *    launch one progress kernel that completes whichever operation is ready
 *    first. A receiver-ready descriptor whose ring slot is still busy is
 *    posted later by whichever kernel runs next: no kernel ever waits for a
 *    slot, so a full ring cannot block the stream.
 *  - Errors (tag mismatch, truncation, timeout, UCX error) are written by the
 *    GPU to mapped host memory and raised at the end of the call.
 */

#if defined(COLLAGE_ST_UCP)

#include <cuda_runtime.h>
#include <mpi.h>

#include <cstdio>
#include <cstring>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "memory.hpp"
#include "ucp_proto.hpp"

#if defined(COLLAGE_UCP_ARENA_HOOK)
/* test seam: the emulation provides process-shared arena memory */
void* collage_ucp_arena_alloc(size_t bytes);
void collage_ucp_arena_free(void* p, size_t bytes);
#endif

namespace ucpx {

using ucpp::Op;

inline void ucheck(ucs_status_t st, const char* what)
{
    if (st != UCS_OK)
        throw std::runtime_error(std::string("UCX: ") + what + ": " + ucs_status_string(st));
}
inline void ccheck(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + what + ": " + cudaGetErrorString(e));
}

/* every rank must succeed, or all stop together (no rank left in a collective) */
inline void agree(const std::string& err, MPI_Comm comm = MPI_COMM_WORLD)
{
    int bad = err.empty() ? 0 : 1;
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, comm);
    if (bad) throw std::runtime_error(err.empty() ? std::string("--transport ucp: setup failed on another rank") : err);
}

/* first-fit allocator over the registered arena */
class Arena {
public:
    void reset(char* base, size_t bytes)
    {
        base_ = base;
        free_.clear();
        used_.clear();
        free_[0] = bytes;
        bytes_ = bytes;
    }
    void* alloc(size_t bytes)
    {
        bytes = (bytes + 255) & ~(size_t)255;
        if (bytes == 0) bytes = 256;
        for (auto it = free_.begin(); it != free_.end(); ++it) {
            if (it->second < bytes) continue;
            size_t off = it->first, len = it->second;
            free_.erase(it);
            if (len > bytes) free_[off + bytes] = len - bytes;
            used_[off] = bytes;
            in_use_ += bytes;
            if (in_use_ > peak_) peak_ = in_use_;
            return base_ + off;
        }
        throw std::runtime_error("--transport ucp: device arena exhausted (" + std::to_string(bytes_ >> 20) +
                                 " MiB in use); raise --ucp-heap or lower --max-mem");
    }
    void free(void* p)
    {
        size_t off = (size_t)((char*)p - base_);
        auto u = used_.find(off);
        if (u == used_.end()) throw std::logic_error("ucp arena: free of an unknown block");
        size_t len = u->second;
        used_.erase(u);
        in_use_ -= len;
        auto next = free_.lower_bound(off);
        if (next != free_.end() && off + len == next->first) {
            len += next->second;
            next = free_.erase(next);
        }
        if (next != free_.begin()) {
            auto prev = std::prev(next);
            if (prev->first + prev->second == off) {
                prev->second += len;
                return;
            }
        }
        free_[off] = len;
    }
    bool contains(const void* p) const { return p >= base_ && (const char*)p < base_ + bytes_; }
    size_t peak() const { return peak_; }

private:
    char* base_ = nullptr;
    size_t bytes_ = 0, in_use_ = 0, peak_ = 0;
    std::map<size_t, size_t> free_, used_;
};

/* pinned host -> device staging for kernel arguments (ops, block lists);
 * reset once per call, after the stream has drained */
class Ring {
public:
    struct Slot { void* host; void* dev; };
    Slot take(size_t bytes)
    {
        bytes = (bytes + 63) & ~(size_t)63;
        for (; cur_ < chunks_.size(); ++cur_) {
            Chunk& c = chunks_[cur_];
            if (c.used + bytes <= c.cap) {
                Slot s{(char*)c.host + c.used, (char*)c.dev + c.used};
                c.used += bytes;
                return s;
            }
        }
        Chunk c;
        c.cap = bytes > (1u << 20) ? bytes : (1u << 20);
        ccheck(cudaMallocHost(&c.host, c.cap), "staging (host)");
        ccheck(cudaMalloc(&c.dev, c.cap), "staging (device)");
        c.used = bytes;
        chunks_.push_back(c);
        cur_ = chunks_.size() - 1;
        return Slot{c.host, c.dev};
    }
    void reset()
    {
        for (Chunk& c : chunks_) c.used = 0;
        cur_ = 0;
    }
    void release()
    {
        for (Chunk& c : chunks_) {
            cudaFreeHost(c.host);
            cudaFree(c.dev);
        }
        chunks_.clear();
        cur_ = 0;
    }

private:
    struct Chunk { void* host = nullptr; void* dev = nullptr; size_t cap = 0, used = 0; };
    std::vector<Chunk> chunks_;
    size_t cur_ = 0;
};

struct Comm {
    int p = 0, me = 0;
    std::vector<uint64_t> sseq, rseq; /* messages sent to / receives posted from each peer */
    void* ctl = nullptr;
    ucpp::DevComm* d = nullptr;
    uint64_t* d_peer_ctl = nullptr;
    uint32_t* d_world = nullptr;
};

struct HostOp {
    Op op;
    Op* dev = nullptr;          /* the device copy (kernel state, `got`) */
    char* unpack_dst = nullptr; /* derived receive: scatter scratch -> user buffer */
    char* scratch = nullptr;
    const uint64_t* dev_blocks = nullptr;
    uint32_t nblocks = 0;
};

struct Engine {
    bool up = false;
    int wrank = 0, wsize = 1;
    ucp_context_h ctx = nullptr;
    ucp_worker_h worker = nullptr;
    std::vector<ucp_ep_h> eps;
    std::vector<ucp_rkey_h> rkeys;
    ucp_mem_h memh = nullptr;
    char* arena = nullptr;
    size_t arena_bytes = 0;
    Arena alloc;
#if COLLAGE_UCP_API == 2
    ucp_device_local_mem_list_h lh = nullptr;
    ucp_device_remote_mem_list_h rh = nullptr;
#else
    std::vector<ucp_device_mem_list_handle_h> handles;
    ucp_device_mem_list_handle_h* d_handles = nullptr;
#endif
    ucpp::Global g{};
    uint64_t* err_host = nullptr;
    uint32_t R = 16;
    cudaStream_t stream = nullptr;
    Ring op_ring, blk_ring;
    std::vector<HostOp> ops;
    std::vector<void*> deferred; /* pack/unpack scratch, freed at drain */
    /* the call's receives, in posting order (device array), and the cursor
     * past the prefix whose receiver-ready descriptors are posted */
    Op** d_recvs = nullptr;
    uint32_t recv_cap = 0, nrecv = 0;
    uint32_t* d_cursor = nullptr;
    std::vector<void*> old_recv_arrays; /* outgrown, freed at drain */
    unsigned long long n_send = 0, n_recv = 0, n_kernels = 0;
};

inline Engine& eng()
{
    static Engine e;
    return e;
}

/* memory.hpp hooks: the device pool draws from the arena */
inline void* arena_alloc(size_t bytes) { return eng().alloc.alloc(bytes); }
inline void arena_free(void* p) { eng().alloc.free(p); }

inline const char* api_name()
{
#if COLLAGE_UCP_API == 2
    return "UCX GPU device API (local/remote mem lists, UCX >= 1.21)";
#else
    return "UCX GPU device API (per-endpoint mem lists, UCX 1.20)";
#endif
}

/* ------------------------------------------------------------------------- */
/* setup (collective over MPI_COMM_WORLD)                                     */
/* ------------------------------------------------------------------------- */
inline std::vector<char> allgatherv_bytes(const void* mine, int len, std::vector<int>& lens,
                                          std::vector<int>& displs)
{
    Engine& e = eng();
    lens.assign((size_t)e.wsize, 0);
    MPI_Allgather(&len, 1, MPI_INT, lens.data(), 1, MPI_INT, MPI_COMM_WORLD);
    displs.assign((size_t)e.wsize, 0);
    int total = 0;
    for (int r = 0; r < e.wsize; ++r) {
        displs[(size_t)r] = total;
        total += lens[(size_t)r];
    }
    std::vector<char> all((size_t)total);
    MPI_Allgatherv(mine, len, MPI_BYTE, all.data(), lens.data(), displs.data(), MPI_BYTE, MPI_COMM_WORLD);
    return all;
}

inline void init(size_t heap_bytes, uint32_t slots, double timeout_s, cudaStream_t stream)
{
    Engine& e = eng();
    MPI_Comm_rank(MPI_COMM_WORLD, &e.wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &e.wsize);
    e.R = slots < 1 ? 1 : (slots > ucpp::MAX_SLOTS ? ucpp::MAX_SLOTS : slots);
    e.stream = stream;
    std::string err;

    /* context and worker */
    try {
        ucp_config_t* cfg = nullptr;
        ucheck(ucp_config_read(nullptr, nullptr, &cfg), "ucp_config_read");
        ucp_params_t params;
        std::memset(&params, 0, sizeof params);
        params.field_mask = UCP_PARAM_FIELD_FEATURES;
        params.features = UCP_FEATURE_RMA | UCP_FEATURE_AMO64 | UCP_FEATURE_DEVICE;
        ucs_status_t st = ucp_init(&params, cfg, &e.ctx);
        ucp_config_release(cfg);
        ucheck(st, "ucp_init (features RMA | AMO64 | DEVICE)");
        ucp_worker_params_t wp;
        std::memset(&wp, 0, sizeof wp);
        wp.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
        wp.thread_mode = UCS_THREAD_MODE_SINGLE;
        ucheck(ucp_worker_create(e.ctx, &wp, &e.worker), "ucp_worker_create");
    } catch (const std::exception& x) {
        err = x.what();
    }
    agree(err);

    /* endpoints to every other rank */
    {
        ucp_address_t* addr = nullptr;
        size_t alen = 0;
        try { ucheck(ucp_worker_get_address(e.worker, &addr, &alen), "ucp_worker_get_address"); } catch (const std::exception& x) { err = x.what(); }
        agree(err);
        std::vector<int> lens, displs;
        std::vector<char> all = allgatherv_bytes(addr, (int)alen, lens, displs);
        ucp_worker_release_address(e.worker, addr);
        e.eps.assign((size_t)e.wsize, nullptr);
        try {
            for (int r = 0; r < e.wsize; ++r) {
                if (r == e.wrank) continue;
                ucp_ep_params_t ep;
                std::memset(&ep, 0, sizeof ep);
                ep.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
                ep.address = (const ucp_address_t*)(all.data() + displs[(size_t)r]);
                ucheck(ucp_ep_create(e.worker, &ep, &e.eps[(size_t)r]), "ucp_ep_create");
            }
        } catch (const std::exception& x) {
            err = x.what();
        }
        agree(err);
    }

    /* the registered arena */
    e.arena_bytes = heap_bytes;
    try {
#if defined(COLLAGE_UCP_ARENA_HOOK)
        e.arena = (char*)collage_ucp_arena_alloc(heap_bytes);
#else
        ccheck(cudaMalloc((void**)&e.arena, heap_bytes), "arena (cudaMalloc of --ucp-heap)");
#endif
        ccheck(cudaMemset(e.arena, 0, heap_bytes), "arena memset");
        ucp_mem_map_params_t mp;
        std::memset(&mp, 0, sizeof mp);
        mp.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                        UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
        mp.address = e.arena;
        mp.length = heap_bytes;
        mp.memory_type = UCS_MEMORY_TYPE_CUDA;
        ucheck(ucp_mem_map(e.ctx, &mp, &e.memh), "ucp_mem_map (arena)");
    } catch (const std::exception& x) {
        err = x.what();
    }
    agree(err);
    e.alloc.reset(e.arena, heap_bytes);

    /* rkeys and arena bases of every rank */
    std::vector<uint64_t> base((size_t)e.wsize);
    {
        void* rbuf = nullptr;
        size_t rlen = 0;
        try { ucheck(ucp_rkey_pack(e.ctx, e.memh, &rbuf, &rlen), "ucp_rkey_pack"); } catch (const std::exception& x) { err = x.what(); }
        agree(err);
        std::vector<int> lens, displs;
        std::vector<char> all = allgatherv_bytes(rbuf, (int)rlen, lens, displs);
        ucp_rkey_buffer_release(rbuf);
        e.rkeys.assign((size_t)e.wsize, nullptr);
        try {
            for (int r = 0; r < e.wsize; ++r)
                if (r != e.wrank)
                    ucheck(ucp_ep_rkey_unpack(e.eps[(size_t)r], all.data() + displs[(size_t)r], &e.rkeys[(size_t)r]),
                           "ucp_ep_rkey_unpack");
        } catch (const std::exception& x) {
            err = x.what();
        }
        agree(err);
        uint64_t mine = (uint64_t)(uintptr_t)e.arena;
        MPI_Allgather(&mine, 1, MPI_UINT64_T, base.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);
    }

    /* device descriptor lists: my arena -> every other rank's arena */
    try {
        const double t0 = MPI_Wtime();
#if COLLAGE_UCP_API == 2
        ucp_device_mem_list_elem_t le;
        std::memset(&le, 0, sizeof le);
        le.field_mask = UCP_DEVICE_MEM_LIST_ELEM_FIELD_MEMH | UCP_DEVICE_MEM_LIST_ELEM_FIELD_LOCAL_ADDR |
                        UCP_DEVICE_MEM_LIST_ELEM_FIELD_LENGTH;
        le.memh = e.memh;
        le.local_addr = e.arena;
        le.length = heap_bytes;
        ucp_device_mem_list_params_t lp;
        std::memset(&lp, 0, sizeof lp);
        lp.field_mask = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS | UCP_DEVICE_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS |
                        UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE | UCP_DEVICE_MEM_LIST_PARAMS_FIELD_WORKER;
        lp.element_size = sizeof le;
        lp.num_elements = 1;
        lp.elements = &le;
        lp.worker = e.worker;
        ucheck(ucp_device_local_mem_list_create(&lp, &e.lh), "ucp_device_local_mem_list_create");
        if (e.wsize > 1) {
            std::vector<ucp_device_mem_list_elem_t> re;
            for (int r = 0; r < e.wsize; ++r) {
                if (r == e.wrank) continue;
                ucp_device_mem_list_elem_t x;
                std::memset(&x, 0, sizeof x);
                x.field_mask = UCP_DEVICE_MEM_LIST_ELEM_FIELD_EP | UCP_DEVICE_MEM_LIST_ELEM_FIELD_REMOTE_ADDR |
                               UCP_DEVICE_MEM_LIST_ELEM_FIELD_RKEY;
                x.ep = e.eps[(size_t)r];
                x.remote_addr = base[(size_t)r];
                x.rkey = e.rkeys[(size_t)r];
                re.push_back(x);
            }
            ucp_device_mem_list_params_t rp;
            std::memset(&rp, 0, sizeof rp);
            rp.field_mask = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS | UCP_DEVICE_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS |
                            UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE;
            rp.element_size = sizeof(ucp_device_mem_list_elem_t);
            rp.num_elements = re.size();
            rp.elements = re.data();
            ucs_status_t st;
            do { /* wire-up completes in the background */
                ucp_worker_progress(e.worker);
                st = ucp_device_remote_mem_list_create(&rp, &e.rh);
            } while (st == UCS_ERR_NOT_CONNECTED && MPI_Wtime() - t0 < 120.0);
            if (st == UCS_ERR_NO_DEVICE)
                throw std::runtime_error("UCX found no GPU device lanes (rc_gda for the network, cuda_ipc on a "
                                         "node): check UCX_TLS, nvidia_peermem/dmabuf and a ConnectX NIC");
            ucheck(st, "ucp_device_remote_mem_list_create");
        }
#else
        e.handles.assign((size_t)e.wsize, nullptr);
        for (int r = 0; r < e.wsize; ++r) {
            if (r == e.wrank) continue;
            ucp_device_mem_list_elem_t x;
            std::memset(&x, 0, sizeof x);
            x.field_mask = UCP_DEVICE_MEM_LIST_ELEM_FIELD_MEMH | UCP_DEVICE_MEM_LIST_ELEM_FIELD_RKEY |
                           UCP_DEVICE_MEM_LIST_ELEM_FIELD_LOCAL_ADDR | UCP_DEVICE_MEM_LIST_ELEM_FIELD_REMOTE_ADDR |
                           UCP_DEVICE_MEM_LIST_ELEM_FIELD_LENGTH;
            x.memh = e.memh;
            x.rkey = e.rkeys[(size_t)r];
            x.local_addr = e.arena;
            x.remote_addr = base[(size_t)r];
            x.length = heap_bytes;
            ucp_device_mem_list_params_t p;
            std::memset(&p, 0, sizeof p);
            p.field_mask = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS | UCP_DEVICE_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS |
                           UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE;
            p.element_size = sizeof x;
            p.num_elements = 1;
            p.elements = &x;
            ucs_status_t st;
            do {
                ucp_worker_progress(e.worker);
                st = ucp_device_mem_list_create(e.eps[(size_t)r], &p, &e.handles[(size_t)r]);
            } while (st == UCS_ERR_NOT_CONNECTED && MPI_Wtime() - t0 < 120.0);
            if (st == UCS_ERR_NO_DEVICE)
                throw std::runtime_error("UCX found no GPU device lanes (rc_gda for the network, cuda_ipc on a "
                                         "node): check UCX_TLS, nvidia_peermem/dmabuf and a ConnectX NIC");
            ucheck(st, "ucp_device_mem_list_create");
        }
        ccheck(cudaMalloc((void**)&e.d_handles, sizeof(void*) * (size_t)e.wsize), "handle array");
        ccheck(cudaMemcpy(e.d_handles, e.handles.data(), sizeof(void*) * (size_t)e.wsize, cudaMemcpyHostToDevice),
               "handle array copy");
#endif
        /* error word in mapped host memory: the GPU writes, the host reads for free */
        ccheck(cudaHostAlloc((void**)&e.err_host, 2 * sizeof(uint64_t), cudaHostAllocMapped), "error word");
        e.err_host[0] = e.err_host[1] = 0;
        void* derr = nullptr;
        ccheck(cudaHostGetDevicePointer(&derr, e.err_host, 0), "error word (device pointer)");
        e.g.err = (volatile uint64_t*)derr;
        ccheck(cudaMalloc((void**)&e.d_cursor, sizeof(uint32_t)), "receive cursor");
        ccheck(cudaMemset(e.d_cursor, 0, sizeof(uint32_t)), "receive cursor");
    } catch (const std::exception& x) {
        err = x.what();
    }
    agree(err);

    e.g.arena = e.arena;
    e.g.timeout_ns = (uint64_t)(timeout_s * 1e9);
    e.g.me_world = (uint32_t)e.wrank;
    e.g.channel = 0;
#if COLLAGE_UCP_API == 2
    e.g.lh = e.lh;
    e.g.rh = e.rh;
#else
    e.g.handles = e.d_handles;
#endif
    mem::st().raw_alloc = &arena_alloc;
    mem::st().raw_free = &arena_free;
    e.up = true;
}

inline void finalize()
{
    Engine& e = eng();
    if (!e.up) return;
    mem::drop_pool(); /* pooled blocks are arena memory */
    mem::st().raw_alloc = nullptr;
    mem::st().raw_free = nullptr;
    e.op_ring.release();
    e.blk_ring.release();
    if (e.d_recvs) cudaFree(e.d_recvs);
    if (e.d_cursor) cudaFree(e.d_cursor);
    e.d_recvs = nullptr;
    e.d_cursor = nullptr;
    e.recv_cap = e.nrecv = 0;
#if COLLAGE_UCP_API == 2
    if (e.rh) ucp_device_mem_list_release(e.rh);
    if (e.lh) ucp_device_mem_list_release(e.lh);
    e.rh = nullptr;
    e.lh = nullptr;
#else
    for (auto h : e.handles)
        if (h) ucp_device_mem_list_release(h);
    e.handles.clear();
    if (e.d_handles) cudaFree(e.d_handles);
    e.d_handles = nullptr;
#endif
    for (auto rk : e.rkeys)
        if (rk) ucp_rkey_destroy(rk);
    e.rkeys.clear();
    if (e.memh) ucp_mem_unmap(e.ctx, e.memh);
    e.memh = nullptr;
    /* close endpoints, then let the peers do the same before the workers go */
    std::vector<ucs_status_ptr_t> closing;
    for (auto ep : e.eps)
        if (ep) {
            ucp_request_param_t rp;
            std::memset(&rp, 0, sizeof rp);
            ucs_status_ptr_t req = ucp_ep_close_nbx(ep, &rp);
            if (UCS_PTR_IS_PTR(req)) closing.push_back(req);
        }
    for (auto req : closing) {
        while (ucp_request_check_status(req) == UCS_INPROGRESS) ucp_worker_progress(e.worker);
        ucp_request_free(req);
    }
    e.eps.clear();
    MPI_Barrier(MPI_COMM_WORLD);
#if defined(COLLAGE_UCP_ARENA_HOOK)
    collage_ucp_arena_free(e.arena, e.arena_bytes);
#else
    cudaFree(e.arena);
#endif
    e.arena = nullptr;
    if (e.err_host) cudaFreeHost(e.err_host);
    e.err_host = nullptr;
    if (e.worker) ucp_worker_destroy(e.worker);
    if (e.ctx) ucp_cleanup(e.ctx);
    e.worker = nullptr;
    e.ctx = nullptr;
    e.up = false;
}

/* ------------------------------------------------------------------------- */
/* communicators (collective over comm)                                       */
/* ------------------------------------------------------------------------- */
inline Comm* comm_open(MPI_Comm comm)
{
    Engine& e = eng();
    Comm* c = new Comm;
    MPI_Comm_size(comm, &c->p);
    MPI_Comm_rank(comm, &c->me);
    c->sseq.assign((size_t)c->p, 0);
    c->rseq.assign((size_t)c->p, 0);
    ucpp::CtlLayout L{e.R};
    const size_t bytes = (size_t)c->p * L.entry();
    std::string err;
    std::vector<uint64_t> offs((size_t)c->p);
    uint64_t mine = 0;
    try {
        c->ctl = e.alloc.alloc(bytes);
        mine = (uint64_t)((char*)c->ctl - e.arena);
        ccheck(cudaMemset(c->ctl, 0, bytes), "control area");
        ccheck(cudaDeviceSynchronize(), "control area");
        ucp_device_counter_params_t cp;
        std::memset(&cp, 0, sizeof cp);
        cp.field_mask = UCP_DEVICE_COUNTER_PARAMS_FIELD_MEMH;
        cp.memh = e.memh;
        for (int j = 0; j < c->p; ++j) {
            char* en = (char*)c->ctl + (size_t)j * L.entry();
            ucheck(ucp_device_counter_init(e.worker, &cp, en + L.rts_cnt()), "ucp_device_counter_init");
            ucheck(ucp_device_counter_init(e.worker, &cp, en + L.data_cnt()), "ucp_device_counter_init");
        }
    } catch (const std::exception& x) {
        err = x.what();
    }
    agree(err, comm);
    MPI_Allgather(&mine, 1, MPI_UINT64_T, offs.data(), 1, MPI_UINT64_T, comm);
    /* world ranks of the communicator */
    MPI_Group gw, gc;
    MPI_Comm_group(MPI_COMM_WORLD, &gw);
    MPI_Comm_group(comm, &gc);
    std::vector<int> cr((size_t)c->p), wr((size_t)c->p);
    for (int i = 0; i < c->p; ++i) cr[(size_t)i] = i;
    MPI_Group_translate_ranks(gc, c->p, cr.data(), gw, wr.data());
    MPI_Group_free(&gw);
    MPI_Group_free(&gc);
    std::vector<uint32_t> w32(wr.begin(), wr.end());

    ucpp::DevComm h{};
    h.p = (uint32_t)c->p;
    h.me = (uint32_t)c->me;
    h.me_world = (uint32_t)e.wrank;
    h.L = L;
    h.ctl_off = mine;
    ccheck(cudaMalloc((void**)&c->d_peer_ctl, sizeof(uint64_t) * offs.size()), "communicator");
    ccheck(cudaMalloc((void**)&c->d_world, sizeof(uint32_t) * w32.size()), "communicator");
    ccheck(cudaMalloc((void**)&c->d, sizeof(ucpp::DevComm)), "communicator");
    ccheck(cudaMemcpy(c->d_peer_ctl, offs.data(), sizeof(uint64_t) * offs.size(), cudaMemcpyHostToDevice), "communicator");
    ccheck(cudaMemcpy(c->d_world, w32.data(), sizeof(uint32_t) * w32.size(), cudaMemcpyHostToDevice), "communicator");
    h.peer_ctl_off = c->d_peer_ctl;
    h.world = c->d_world;
    ccheck(cudaMemcpy(c->d, &h, sizeof h, cudaMemcpyHostToDevice), "communicator");
    return c;
}

inline void comm_close(Comm* c)
{
    if (!c) return;
    Engine& e = eng();
    if (e.up && c->ctl) e.alloc.free(c->ctl);
    cudaFree(c->d);
    cudaFree(c->d_peer_ctl);
    cudaFree(c->d_world);
    delete c;
}

/* ------------------------------------------------------------------------- */
/* datatypes                                                                  */
/* ------------------------------------------------------------------------- */
/* false: contiguous (bytes set); true: blocks = {offset, length} pairs */
inline bool flatten(MPI_Datatype dt, size_t count, uint64_t& bytes, std::vector<uint64_t>& blocks)
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
    if (comb != MPI_COMBINER_INDEXED)
        throw std::runtime_error("--transport ucp: only predefined and indexed datatypes are supported");
    std::vector<int> ints((size_t)ni);
    std::vector<MPI_Aint> aints((size_t)na);
    std::vector<MPI_Datatype> types((size_t)nd);
    std::vector<int64_t> cnts;
#if defined(MPI_VERSION) && MPI_VERSION >= 4 && !defined(COLLAGE_NO_LARGE_COUNT)
    std::vector<MPI_Count> counts((size_t)nc);
    MPI_Type_get_contents_c(dt, ni, na, nc, nd, ints.data(), aints.data(), counts.data(), types.data());
    if (nc > 0) cnts.assign(counts.begin(), counts.end());
    else cnts.assign(ints.begin(), ints.end());
#else
    MPI_Type_get_contents(dt, ni, na, nd, ints.data(), aints.data(), types.data());
    cnts.assign(ints.begin(), ints.end());
#endif
    MPI_Datatype old = types[0];
    int ocomb = 0;
    {
        int a = 0, b = 0, d = 0;
        MPI_Type_get_envelope(old, &a, &b, &d, &ocomb);
    }
    if (ocomb != MPI_COMBINER_NAMED)
        throw std::runtime_error("--transport ucp: indexed datatypes of derived types are not supported");
    MPI_Aint olb = 0, oext = 0, lb = 0, ext = 0;
    MPI_Type_get_extent(old, &olb, &oext);
    MPI_Type_get_extent(dt, &lb, &ext);
    const int64_t n = cnts[0];
    bytes = 0;
    blocks.clear();
    for (size_t rep = 0; rep < count; ++rep)
        for (int64_t i = 0; i < n; ++i) {
            const uint64_t len = (uint64_t)cnts[(size_t)(1 + i)] * (uint64_t)oext;
            const uint64_t off = (uint64_t)rep * (uint64_t)ext + (uint64_t)cnts[(size_t)(1 + n + i)] * (uint64_t)oext;
            if (len == 0) continue;
            if (!blocks.empty() && blocks[blocks.size() - 2] + blocks.back() == off) blocks.back() += len;
            else {
                blocks.push_back(off);
                blocks.push_back(len);
            }
            bytes += len;
        }
    return true;
}

/* ------------------------------------------------------------------------- */
/* operations (host enqueues, GPU executes)                                   */
/* ------------------------------------------------------------------------- */
inline uint64_t arena_off(const void* p, uint64_t bytes)
{
    Engine& e = eng();
    if (bytes == 0) return 0;
    if (!e.alloc.contains(p) || !e.alloc.contains((const char*)p + bytes - 1))
        throw std::runtime_error("--transport ucp: buffer is not in the registered device arena");
    return (uint64_t)((const char*)p - e.arena);
}

inline const uint64_t* stage_blocks(const std::vector<uint64_t>& blocks)
{
    Engine& e = eng();
    Ring::Slot s = e.blk_ring.take(blocks.size() * sizeof(uint64_t));
    std::memcpy(s.host, blocks.data(), blocks.size() * sizeof(uint64_t));
    ccheck(cudaMemcpyAsync(s.dev, s.host, blocks.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, e.stream),
           "block list");
    return (const uint64_t*)s.dev;
}

inline Op* stage_op(const Op& op)
{
    Engine& e = eng();
    Ring::Slot s = e.op_ring.take(sizeof(Op));
    std::memcpy(s.host, &op, sizeof(Op));
    ccheck(cudaMemcpyAsync(s.dev, s.host, sizeof(Op), cudaMemcpyHostToDevice, e.stream), "op");
    return (Op*)s.dev;
}

inline Op* const* stage_ptrs(const std::vector<Op*>& v)
{
    Engine& e = eng();
    if (v.empty()) return nullptr;
    Ring::Slot s = e.op_ring.take(v.size() * sizeof(Op*));
    std::memcpy(s.host, v.data(), v.size() * sizeof(Op*));
    ccheck(cudaMemcpyAsync(s.dev, s.host, v.size() * sizeof(Op*), cudaMemcpyHostToDevice, e.stream), "op list");
    return (Op* const*)s.dev;
}

/* append a receive to the call's device list (stream-ordered) */
inline void append_recv(Op* dev)
{
    Engine& e = eng();
    if (e.nrecv == e.recv_cap) {
        const uint32_t cap = e.recv_cap ? 2 * e.recv_cap : 1024;
        Op** a = nullptr;
        ccheck(cudaMalloc((void**)&a, sizeof(Op*) * cap), "receive list");
        if (e.nrecv)
            ccheck(cudaMemcpyAsync(a, e.d_recvs, sizeof(Op*) * e.nrecv, cudaMemcpyDeviceToDevice, e.stream),
                   "receive list");
        if (e.d_recvs) e.old_recv_arrays.push_back(e.d_recvs); /* kernels already queued may read it */
        e.d_recvs = a;
        e.recv_cap = cap;
    }
    Ring::Slot s = e.op_ring.take(sizeof(Op*));
    std::memcpy(s.host, &dev, sizeof dev);
    ccheck(cudaMemcpyAsync(e.d_recvs + e.nrecv, s.host, sizeof dev, cudaMemcpyHostToDevice, e.stream), "receive list");
    ++e.nrecv;
}

inline int isend(Comm* c, const void* buf, size_t count, MPI_Datatype dt, int peer, int tag)
{
    Engine& e = eng();
    mem::before_stream_op();
    HostOp h;
    std::vector<uint64_t> blocks;
    uint64_t bytes = 0;
    if (flatten(dt, count, bytes, blocks) && bytes > 0) {
        char* scratch = (char*)mem::device_alloc(bytes);
        e.deferred.push_back(scratch);
        ucpk::launch_blocks(e.stream, (const char*)buf, scratch, stage_blocks(blocks), (uint32_t)(blocks.size() / 2), 1);
        ++e.n_kernels;
        h.op.off = arena_off(scratch, bytes);
    } else {
        h.op.off = arena_off(buf, bytes);
    }
    h.op.seq = c->sseq[(size_t)peer]++;
    h.op.len = bytes;
    h.op.got = 0;
    h.op.comm = c->d;
    h.op.kind = ucpp::OP_SEND;
    h.op.peer = (uint32_t)peer;
    h.op.tag = (uint32_t)tag;
    h.op.done = 0;
    h.op.rts = 0;
    h.dev = stage_op(h.op);
    e.ops.push_back(h);
    ++e.n_send;
    mem::mark_pending();
    return (int)e.ops.size() - 1;
}

inline int irecv(Comm* c, void* buf, size_t count, MPI_Datatype dt, int peer, int tag)
{
    Engine& e = eng();
    mem::before_stream_op();
    HostOp h;
    std::vector<uint64_t> blocks;
    uint64_t bytes = 0;
    if (flatten(dt, count, bytes, blocks) && bytes > 0) {
        char* scratch = (char*)mem::device_alloc(bytes);
        e.deferred.push_back(scratch);
        h.scratch = scratch;
        h.unpack_dst = (char*)buf;
        h.dev_blocks = stage_blocks(blocks);
        h.nblocks = (uint32_t)(blocks.size() / 2);
        h.op.off = arena_off(scratch, bytes);
    } else {
        h.op.off = arena_off(buf, bytes);
    }
    h.op.seq = c->rseq[(size_t)peer]++;
    h.op.len = bytes; /* capacity */
    h.op.got = 0;
    h.op.comm = c->d;
    h.op.kind = ucpp::OP_RECV;
    h.op.peer = (uint32_t)peer;
    h.op.tag = (uint32_t)tag;
    h.op.done = 0;
    h.op.rts = 0;
    h.dev = stage_op(h.op);
    e.ops.push_back(h);
    ++e.n_recv;
    append_recv(h.dev);
    /* tell the sender where to put it, right away (or as soon as a slot frees) */
    ucpk::launch_rts(e.stream, e.g, e.d_recvs, e.nrecv, e.d_cursor);
    ++e.n_kernels;
    mem::mark_pending();
    return (int)e.ops.size() - 1;
}

/* the stream completes ops[idx[0..n)] (one progress kernel), then unpacks */
inline void wait(const int* idx, int n)
{
    if (n <= 0) return;
    Engine& e = eng();
    mem::before_stream_op();
    std::vector<Op*> v((size_t)n);
    for (int i = 0; i < n; ++i) v[(size_t)i] = e.ops[(size_t)idx[i]].dev;
    ucpk::launch_progress(e.stream, e.g, stage_ptrs(v), (uint32_t)n, e.d_recvs, e.nrecv, e.d_cursor);
    ++e.n_kernels;
    for (int i = 0; i < n; ++i) {
        HostOp& h = e.ops[(size_t)idx[i]];
        if (h.unpack_dst) {
            ucpk::launch_blocks(e.stream, h.scratch, h.unpack_dst, h.dev_blocks, h.nblocks, 0);
            ++e.n_kernels;
        }
    }
    mem::mark_pending();
}

/* after the stream has drained: the status of a completed receive */
inline void status(int idx, MPI_Status* st)
{
    Engine& e = eng();
    const HostOp& h = e.ops.at((size_t)idx);
    uint64_t got = 0;
    if (h.dev) ccheck(cudaMemcpy(&got, &h.dev->got, sizeof got, cudaMemcpyDeviceToHost), "status");
#if defined(MPI_VERSION) && MPI_VERSION >= 4 && !defined(COLLAGE_NO_LARGE_COUNT)
    MPI_Status_set_elements_c(st, MPI_BYTE, (MPI_Count)got);
#else
    MPI_Status_set_elements_x(st, MPI_BYTE, (MPI_Count)got);
#endif
    st->MPI_SOURCE = (int)h.op.peer;
    st->MPI_TAG = (int)h.op.tag;
}

inline std::string describe_error(uint64_t code, uint64_t info)
{
    const unsigned peer = (unsigned)(info >> 32), low = (unsigned)(info & 0xffffffffu);
    switch (code) {
        case ucpp::ERR_TAG:
            return "message from/to communicator rank " + std::to_string(peer) + " does not match the posted " +
                   "receive (tag " + std::to_string(low) + "): the two sides post their messages in different orders";
        case ucpp::ERR_TRUNCATE:
            return "message to communicator rank " + std::to_string(peer) + " is larger than the posted receive";
        case ucpp::ERR_TIMEOUT:
            return std::string("timeout waiting for communicator rank ") + std::to_string(peer) +
                   (low == 1 ? " to consume earlier messages before a receive could be posted"
                             : low == 2 ? " to post its receive" : " to deliver data") +
                   " (--ucp-timeout)";
        case ucpp::ERR_UCX:
            return "UCX device operation to communicator rank " + std::to_string(peer) + " failed: " +
                   ucs_status_string((ucs_status_t)(-(int)low));
        default:
            return "unknown error " + std::to_string(code);
    }
}

/* end of a call (stream drained): raise device errors, recycle staging */
inline void after_drain()
{
    Engine& e = eng();
    if (!e.up) return;
    if (e.err_host[0]) {
        const uint64_t code = e.err_host[0], info = e.err_host[1];
        e.err_host[0] = e.err_host[1] = 0;
        throw std::runtime_error("--transport ucp: " + describe_error(code, info));
    }
    for (void* p : e.deferred) mem::device_free(p);
    e.deferred.clear();
    for (void* p : e.old_recv_arrays) cudaFree(p);
    e.old_recv_arrays.clear();
    e.nrecv = 0;
    ccheck(cudaMemset(e.d_cursor, 0, sizeof(uint32_t)), "receive cursor");
    e.ops.clear();
    e.op_ring.reset();
    e.blk_ring.reset();
}

} // namespace ucpx

#endif /* COLLAGE_ST_UCP */
#endif /* COLLAGE_UCP_TRANSPORT_HPP */
