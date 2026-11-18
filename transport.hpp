#pragma once
#ifndef COLLAGE_TRANSPORT_HPP
#define COLLAGE_TRANSPORT_HPP

/*
 * Transport layer: how the point-to-point messages of the algorithms are
 * issued. Selected at run time with --transport (stream-triggered ones only
 * with --gpu), compiled in with `make CUDA=1 ST=mpich|acx|cray`.
 *
 *   mpi    plain (CUDA-aware) MPI, host-driven; the default. Every device
 *          copy/kernel is waited for before the next MPI call.
 *   mpich  MPICH >= 4.1 GPU-stream enqueue API: MPIX_Stream bound to our CUDA
 *          stream, MPIX_{Isend,Irecv,Wait,Waitall}_enqueue on a stream
 *          communicator (MPIX_Stream_comm_create).
 *   acx    NVIDIA MPI-ACX (github.com/NVIDIA/mpi-acx), layered on any MPI
 *          (e.g. Open MPI / HPC-X on InfiniBand or RoCE):
 *          MPIX_{Isend,Irecv,Wait,Waitall}_enqueue with MPIX_QUEUE_CUDA_STREAM.
 *   cray   HPE Cray MPICH stream-triggered (ST) operations on Slingshot:
 *          MPIX_Queue bound to our CUDA stream, MPIX_Enqueue_{send,recv},
 *          MPIX_Enqueue_start (trigger), MPIX_Enqueue_wait.
 *   ucp    UCX GPU device API (UCX >= 1.20, ucp_transport.hpp): no MPI after
 *          bootstrap; GPU kernels issue one-sided puts and remote counter
 *          increments themselves (GDA on ConnectX InfiniBand/RoCE, CUDA IPC
 *          on a node). Every send/recv is a rendezvous of device-initiated
 *          puts (see ucp_proto.hpp).
 *
 * With a stream-triggered transport an algorithm runs *in stream order*: its
 * sends, receives and waits are queued on the CUDA stream together with the
 * local copies and reduction kernels (memory.hpp async mode), and the host
 * only enqueues. The benchmark harness waits for the stream once per call
 * (drain()), inside the timed region.
 *
 * Only messages to peers on *other nodes* are stream-triggered by default.
 * Peers on the same node keep the plain path: CUDA-aware MPI, which moves the
 * data with CUDA IPC over NVLink/PCIe; before such a call the host waits for
 * the stream (the call reads or writes device data the stream may still be
 * using). --st-peers all queues every message instead.
 *
 * Request: the algorithms hold Request (not MPI_Request) so that one handle
 * type covers plain requests, enqueued ones and emulated persistent ones
 * (MPI_Send_init/MPI_Recv_init have no enqueue counterpart; in stream mode a
 * persistent request is a descriptor, started with startall()).
 *
 * Semantics worth knowing (documented in the README too):
 *  - blocking send/recv = enqueued isend/irecv + enqueued wait: the *stream*
 *    blocks, the host does not.
 *  - MPI_Waitany: the host cannot see which queued operation finishes
 *    first, so the stream waits for every active request at once and the
 *    completed ones are reported one per call (alltoall linear_sync then
 *    proceeds window by window).
 *  - MPI_Ssend / MPI_Issend have no enqueue form: they become standard sends
 *    on the stream-triggered path.
 *  - a receive's MPI_Status is only known once the stream has run: it is
 *    filled into storage owned by this layer and copied out when the
 *    algorithm asks for it (get_count), after waiting for the stream. Cray
 *    queues give no status, so algorithms that need one are n/a there.
 *  - wildcard receives (MPI_ANY_SOURCE) cannot match messages queued on a
 *    stream communicator (and Cray queues reject them): such algorithms are
 *    n/a in stream mode.
 *  - cray: MPIX_Enqueue_wait waits for everything started on a queue, so
 *    one shared queue would make a wait cover operations that must stay
 *    outstanding (a ring's send would wait for the next step's pre-posted
 *    receive -> deadlock). Every operation therefore gets its own queue from
 *    a pool; queues return to the pool, and requests are released (MPI_Wait
 *    on the host), at drain(). The pool grows to the largest number of
 *    operations one collective call issues.
 *  - acx and cray take int counts (no MPI-4 large-count calls).
 *  - derived datatypes (indexed types of Bruck & co.) are freed by the
 *    algorithms right after the call; a queued operation gets a duplicate
 *    (MPI_Type_dup), released at drain().
 */

#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "memory.hpp"

#if (defined(COLLAGE_ST_MPICH) + defined(COLLAGE_ST_ACX) + defined(COLLAGE_ST_CRAY) + defined(COLLAGE_ST_UCP)) > 1
#error "build with at most one stream-triggered transport (ST=mpich, ST=acx, ST=cray or ST=ucp)"
#endif
#if (defined(COLLAGE_ST_MPICH) || defined(COLLAGE_ST_ACX) || defined(COLLAGE_ST_CRAY) || defined(COLLAGE_ST_UCP)) && \
    !defined(COLLAGE_CUDA)
#error "stream-triggered transports need GPU support (make CUDA=1 ST=...)"
#endif
#if defined(COLLAGE_ST_MPICH) && (!defined(MPICH_NUMVERSION) || MPICH_NUMVERSION < 40100000)
#error "ST=mpich needs MPICH >= 4.1 built with CUDA (MPIX_Stream and the *_enqueue functions)"
#endif
#if defined(COLLAGE_ST_ACX)
#include <mpi-acx.h>
#endif
#if defined(COLLAGE_ST_UCP)
#include "ucp_transport.hpp"
#endif

/* Cray MPICH: the documented synopsis is MPIX_Create_queue(stream, &queue);
 * the example on the same man page also passes a communicator. Override if
 * your Cray MPICH differs, e.g. -DCOLLAGE_CRAY_CREATE_QUEUE(s,q)=MPIX_Create_queue(MPI_COMM_WORLD,s,q) */
#if defined(COLLAGE_ST_CRAY)
#ifndef COLLAGE_CRAY_CREATE_QUEUE
#define COLLAGE_CRAY_CREATE_QUEUE(stream, queue) MPIX_Create_queue((void*)(stream), (queue))
#endif
#ifndef COLLAGE_CRAY_FREE_QUEUE
#define COLLAGE_CRAY_FREE_QUEUE(queue) MPIX_Free_queue(queue)
#endif
#endif

namespace xport {

enum class Kind { Mpi, Mpich, Acx, Cray, Ucp };

inline const char* kind_name(Kind k)
{
    switch (k) {
        case Kind::Mpi: return "mpi";
        case Kind::Mpich: return "mpich";
        case Kind::Acx: return "acx";
        case Kind::Cray: return "cray";
        case Kind::Ucp: return "ucp";
    }
    return "?";
}

inline bool parse_kind(const std::string& s, Kind& k)
{
    if (s == "mpi") k = Kind::Mpi;
    else if (s == "mpich") k = Kind::Mpich;
    else if (s == "acx") k = Kind::Acx;
    else if (s == "cray") k = Kind::Cray;
    else if (s == "ucp") k = Kind::Ucp;
    else return false;
    return true;
}

/* is this transport compiled into the binary? */
inline bool kind_built(Kind k)
{
    switch (k) {
        case Kind::Mpi: return true;
#if defined(COLLAGE_ST_MPICH)
        case Kind::Mpich: return true;
#endif
#if defined(COLLAGE_ST_ACX)
        case Kind::Acx: return true;
#endif
#if defined(COLLAGE_ST_CRAY)
        case Kind::Cray: return true;
#endif
#if defined(COLLAGE_ST_UCP)
        case Kind::Ucp: return true;
#endif
        default: return false;
    }
}

inline std::string built_list()
{
    std::string s = "mpi";
    for (Kind k : {Kind::Mpich, Kind::Acx, Kind::Cray, Kind::Ucp})
        if (kind_built(k)) s += std::string(", ") + kind_name(k);
    return s;
}

/* ------------------------------------------------------------------------- */
/* requests                                                                   */
/* ------------------------------------------------------------------------- */
struct Persist;

struct Request {
    /* Done: completed by a stream-mode waitany batch, not yet reported */
    enum K : unsigned char { None = 0, Host, Enq, Pers, Done };
    K k = None;
    MPI_Request h = MPI_REQUEST_NULL; /* Host; Enq for mpich and cray */
    void* x = nullptr;                /* Enq for acx (MPIX_Request) */
    Persist* p = nullptr;             /* Pers: emulated persistent request */
    int q = -1;                       /* Enq for cray: index of its queue */
    int u = -1;                       /* Enq for ucp: index of its operation */
};

inline bool operator==(const Request& a, const Request& b)
{
    return a.k == b.k && a.h == b.h && a.x == b.x && a.p == b.p && a.q == b.q && a.u == b.u;
}
inline bool operator!=(const Request& a, const Request& b) { return !(a == b); }

/* persistent request in stream mode: the arguments, and the operation the
 * last start() issued */
struct Persist {
    bool is_send = true;
    const void* buf = nullptr;
    size_t count = 0;
    MPI_Datatype dt = MPI_DATATYPE_NULL;
    int peer = 0, tag = 0;
    Request active;
};

/* ------------------------------------------------------------------------- */
/* process-wide state                                                         */
/* ------------------------------------------------------------------------- */
struct State {
    Kind kind = Kind::Mpi;
    bool all_peers = false;     /* --st-peers all */
    bool progress_thread = false;
#if defined(COLLAGE_CUDA)
    cudaStream_t stream = nullptr;
#endif
#if defined(COLLAGE_ST_MPICH)
    MPIX_Stream mstream = MPIX_STREAM_NULL;
#endif
#if defined(COLLAGE_ST_CRAY)
    std::vector<MPIX_Queue> queues;      /* pool: one queue per operation of a call */
    size_t queues_used = 0;              /* queues[0..used) belong to the current call */
    std::vector<int> unstarted;          /* queues holding descriptors not yet started */
    std::vector<MPI_Request> to_release; /* waited on the stream, released at drain */
#endif
    /* derived datatypes used by queued operations: the algorithms free theirs
     * right after the call, before the stream has run the operation */
    std::vector<MPI_Datatype> held_types;
    /* receive statuses filled by the stream: user status -> our storage */
    std::deque<MPI_Status> slots;
    std::unordered_map<const MPI_Status*, MPI_Status*> slot_of;
#if defined(COLLAGE_ST_UCP)
    std::unordered_map<const MPI_Status*, int> ucp_status_of; /* receive status -> operation */
    size_t ucp_heap = (size_t)4 << 30;
    uint32_t ucp_slots = 16;
    double ucp_timeout = 60.0;
#endif
    /* statistics */
    unsigned long long enqueued = 0, plain = 0, host_waits = 0, drains = 0;
};

inline State& st()
{
    static State s;
    return s;
}

inline bool streamed() { return st().kind != Kind::Mpi; }
inline Kind kind() { return st().kind; }

/* MPI-4 large counts on the stream-triggered path? */
inline bool large_count()
{
    return !(st().kind == Kind::Acx || st().kind == Kind::Cray);
}

/* thread level MPI_Init_thread must provide: the stream-triggered libraries
 * call MPI from their own threads (MPI-ACX proxy, MPICH GPU host functions or
 * progress thread) while the main thread keeps issuing MPI calls */
inline int required_thread_level(Kind k)
{
    /* ucp: MPI is only the bootstrap, called from the main thread */
    return (k == Kind::Mpi || k == Kind::Ucp) ? MPI_THREAD_SINGLE : MPI_THREAD_MULTIPLE;
}

inline int icount(size_t c)
{
    if (c > (size_t)INT_MAX)
        throw std::overflow_error(std::string("message count exceeds INT_MAX: the ") + kind_name(st().kind) +
                                  " transport has no large-count calls");
    return (int)c;
}

#define XPORT_CHECK(call)                                                                      \
    do {                                                                                       \
        int _rc = (call);                                                                      \
        if (_rc != MPI_SUCCESS) {                                                              \
            std::fprintf(stderr, "%s:%d: %s failed (%d)\n", __FILE__, __LINE__, #call, _rc);   \
            MPI_Abort(MPI_COMM_WORLD, _rc ? _rc : 1);                                          \
        }                                                                                      \
    } while (0)

/* Cray: trigger the descriptors queued so far, at the current stream position */
inline void flush()
{
#if defined(COLLAGE_ST_CRAY)
    State& s = st();
    if (s.kind == Kind::Cray) {
        for (int q : s.unstarted) XPORT_CHECK(MPIX_Enqueue_start(s.queues[(size_t)q]));
        s.unstarted.clear();
    }
#endif
}

/* hook for memory.hpp: queued copies/kernels come after the triggers of the
 * messages queued before them */
inline void flush_hook() { flush(); }

/* ------------------------------------------------------------------------- */
/* setup / teardown                                                           */
/* ------------------------------------------------------------------------- */
/* call after MPI_Init_thread and after the device and its stream exist
 * (mem::configure). Collective over MPI_COMM_WORLD only for acx (MPIX_Init). */
inline void configure(Kind k, bool all_peers, bool progress_thread)
{
    State& s = st();
    if (!kind_built(k))
        throw std::invalid_argument(std::string("--transport ") + kind_name(k) +
                                    ": not built into this binary (built: " + built_list() + ")");
    s.kind = k;
    s.all_peers = all_peers;
    s.progress_thread = progress_thread;
    if (k == Kind::Mpi) return;
    if (!mem::device()) throw std::invalid_argument("stream-triggered transports need --gpu");
    int provided = MPI_THREAD_SINGLE;
    MPI_Query_thread(&provided);
    if (provided < required_thread_level(k))
        throw std::runtime_error(std::string("--transport ") + kind_name(k) +
                                 " needs MPI_THREAD_MULTIPLE, the MPI library provides only level " +
                                 std::to_string(provided));
#if defined(COLLAGE_CUDA)
    s.stream = mem::st().stream;
#endif
#if defined(COLLAGE_ST_MPICH)
    if (k == Kind::Mpich) {
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Info_set(info, "type", "cudaStream_t");
        XPORT_CHECK(MPIX_Info_set_hex(info, "value", &s.stream, sizeof(s.stream)));
        /* report a failure ourselves: the usual cause is a missing VCI */
        MPI_Comm_set_errhandler(MPI_COMM_SELF, MPI_ERRORS_RETURN);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
        int rc = MPIX_Stream_create(info, &s.mstream);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
        MPI_Comm_set_errhandler(MPI_COMM_SELF, MPI_ERRORS_ARE_FATAL);
        MPI_Info_free(&info);
        if (rc != MPI_SUCCESS)
            throw std::runtime_error("MPIX_Stream_create failed: set MPIR_CVAR_CH4_RESERVE_VCIS=1 (or more), and "
                                     "check that MPICH was built with CUDA (ch4 device)");
        if (progress_thread) XPORT_CHECK(MPIX_Start_progress_thread(s.mstream));
    }
#endif
#if defined(COLLAGE_ST_ACX)
    if (k == Kind::Acx) XPORT_CHECK(MPIX_Init());
#endif
#if defined(COLLAGE_ST_CRAY)
    /* cray: queues are created on demand (cray_queue()) */
#endif
#if defined(COLLAGE_ST_UCP)
    if (k == Kind::Ucp) ucpx::init(s.ucp_heap, s.ucp_slots, s.ucp_timeout, s.stream);
#endif
    mem::st().async = true;
    mem::st().before_stream_op = &flush_hook;
    mem::st().before_sync = &flush_hook;
}

inline void drain();

/* before MPI_Finalize, after every collective object is gone, before mem::shutdown */
inline void shutdown()
{
    State& s = st();
    if (s.kind == Kind::Mpi) return;
    drain();
#if defined(COLLAGE_ST_MPICH)
    if (s.kind == Kind::Mpich && s.mstream != MPIX_STREAM_NULL) {
        if (s.progress_thread) MPIX_Stop_progress_thread(s.mstream);
        MPIX_Stream_free(&s.mstream);
    }
#endif
#if defined(COLLAGE_ST_ACX)
    if (s.kind == Kind::Acx) MPIX_Finalize();
#endif
#if defined(COLLAGE_ST_CRAY)
    if (s.kind == Kind::Cray) {
        for (MPIX_Queue& q : s.queues) COLLAGE_CRAY_FREE_QUEUE(q);
        s.queues.clear();
        s.queues_used = 0;
    }
#endif
#if defined(COLLAGE_ST_UCP)
    if (s.kind == Kind::Ucp) ucpx::finalize();
#endif
    mem::st().async = false;
    mem::st().before_stream_op = nullptr;
    mem::st().before_sync = nullptr;
    s.kind = Kind::Mpi;
}

/* wait for everything queued by the last collective (the harness calls this
 * once per collective, inside the timed region) */
inline void drain()
{
    State& s = st();
    if (s.kind == Kind::Mpi) return;
    flush();
    mem::sync();
#if defined(COLLAGE_ST_CRAY)
    for (MPI_Request& r : s.to_release)
        if (r != MPI_REQUEST_NULL) MPI_CHECK(MPI_Wait(&r, MPI_STATUS_IGNORE));
    s.to_release.clear();
    s.queues_used = 0; /* every queue's operations are complete: reuse them */
#endif
    for (MPI_Datatype& t : s.held_types) MPI_Type_free(&t);
    s.held_types.clear();
    s.slots.clear();
    s.slot_of.clear();
#if defined(COLLAGE_ST_UCP)
    s.ucp_status_of.clear();
    if (s.kind == Kind::Ucp) ucpx::after_drain(); /* raises errors the GPU reported */
#endif
    ++s.drains;
}

/* the host is about to call MPI on device data: wait for the stream first */
inline void before_plain()
{
    if (st().kind != Kind::Mpi && mem::pending()) {
        mem::sync();
        ++st().host_waits;
    }
    ++st().plain;
}

/* algorithms that cannot run on the stream-triggered path (decided before
 * the call, identically on every rank). Returns nullptr when fine. */
inline const char* unsupported(const std::string& coll, const std::string& algo)
{
    if (st().kind == Kind::Mpi) return nullptr;
    if (coll == "barrier" && algo == "linear")
        return "wildcard receive (MPI_ANY_SOURCE) cannot match stream-queued messages";
    if (st().kind == Kind::Cray) {
        if (coll == "bcast" && (algo == "scatter_allgather" || algo == "scatter_allgather_ring"))
            return "needs receive statuses (MPI_Get_count), which Cray queues do not provide";
        if (coll == "scatter" && algo == "binomial")
            return "needs receive statuses (MPI_Get_count), which Cray queues do not provide";
    }
    return nullptr;
}

/* ------------------------------------------------------------------------- */
/* per-communicator channel                                                   */
/* ------------------------------------------------------------------------- */
/* node identity: the processor name (the host name with every common MPI;
 * the same source as the benchmark's layout report) */
inline uint64_t host_hash()
{
    char name[MPI_MAX_PROCESSOR_NAME + 1] = {0};
    int len = 0;
    MPI_Get_processor_name(name, &len);
    uint64_t h = 1469598103934665603ULL; /* FNV-1a */
    for (const char* c = name; *c; ++c) h = (h ^ (unsigned char)*c) * 1099511628211ULL;
    return h;
}

class Channel {
public:
    Channel() = default;
    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;
    ~Channel() { close(); }

    /* collective over comm (in stream mode) */
    void open(MPI_Comm comm)
    {
        comm_ = scomm_ = comm;
        if (!streamed()) return;
        int size = 0, rank = 0;
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);
        std::vector<uint64_t> h((size_t)size);
        uint64_t mine = host_hash();
        MPI_CHECK(MPI_Allgather(&mine, 1, MPI_UINT64_T, h.data(), 1, MPI_UINT64_T, comm));
        remote_.assign((size_t)size, 0);
        for (int r = 0; r < size; ++r) {
            remote_[(size_t)r] = h[(size_t)r] != mine;
            if (remote_[(size_t)r]) ++nremote_;
        }
#if defined(COLLAGE_ST_MPICH)
        if (st().kind == Kind::Mpich) {
            MPI_CHECK(MPIX_Stream_comm_create(comm, st().mstream, &scomm_));
            own_ = true;
        }
#endif
#if defined(COLLAGE_ST_UCP)
        if (st().kind == Kind::Ucp) uc_ = ucpx::comm_open(comm); /* control area */
#endif
    }
    void close()
    {
#if defined(COLLAGE_ST_UCP)
        if (uc_) {
            ucpx::comm_close(uc_);
            uc_ = nullptr;
        }
#endif
        if (own_) {
            int fin = 0;
            MPI_Finalized(&fin);
            if (!fin) MPI_Comm_free(&scomm_);
            own_ = false;
        }
        scomm_ = comm_;
    }

    /* is a message to/from this peer queued on the stream? */
    bool enq(int peer) const
    {
        if (remote_.empty() || peer < 0 || peer >= (int)remote_.size()) return false;
        return st().all_peers || remote_[(size_t)peer];
    }
    MPI_Comm comm() const { return comm_; }   /* plain path */
    MPI_Comm scomm() const { return scomm_; } /* stream path */
    int remote_peers() const { return nremote_; }
#if defined(COLLAGE_ST_UCP)
    ucpx::Comm* ucomm() const { return uc_; }
#endif

private:
    MPI_Comm comm_ = MPI_COMM_NULL, scomm_ = MPI_COMM_NULL;
    std::vector<char> remote_;
    int nremote_ = 0;
    bool own_ = false;
#if defined(COLLAGE_ST_UCP)
    ucpx::Comm* uc_ = nullptr;
#endif
};

/* ------------------------------------------------------------------------- */
/* stream-triggered operations                                                */
/* ------------------------------------------------------------------------- */
inline MPI_Status* status_slot(MPI_Status* user)
{
    if (user == MPI_STATUS_IGNORE) return MPI_STATUS_IGNORE;
    State& s = st();
    s.slots.emplace_back();
    MPI_Status* p = &s.slots.back();
    std::memset(p, 0, sizeof *p);
    s.slot_of[user] = p;
    return p;
}

/* datatype for a queued operation: predefined ones as they are, derived ones
 * duplicated (the duplicate lives until drain) */
inline MPI_Datatype hold_type(MPI_Datatype dt)
{
    int comb = 0;
#if defined(MPI_VERSION) && MPI_VERSION >= 4 && !defined(COLLAGE_NO_LARGE_COUNT)
    MPI_Count ni = 0, na = 0, nc = 0, nd = 0; /* large-count types need the _c query */
    MPI_CHECK(MPI_Type_get_envelope_c(dt, &ni, &na, &nc, &nd, &comb));
#else
    int ni = 0, na = 0, nd = 0;
    MPI_CHECK(MPI_Type_get_envelope(dt, &ni, &na, &nd, &comb));
#endif
    if (comb == MPI_COMBINER_NAMED) return dt;
    MPI_Datatype d;
    MPI_CHECK(MPI_Type_dup(dt, &d));
    st().held_types.push_back(d);
    return d;
}

#if defined(COLLAGE_ST_CRAY)
/* a queue of its own for the next operation (see the note at the top) */
inline int cray_queue()
{
    State& s = st();
    if (s.queues_used == s.queues.size()) {
        MPIX_Queue q{};
        XPORT_CHECK(COLLAGE_CRAY_CREATE_QUEUE(s.stream, &q));
        s.queues.push_back(q);
    }
    int q = (int)s.queues_used++;
    s.unstarted.push_back(q);
    return q;
}
#endif

inline void enq_isend(const Channel& ch, const void* buf, size_t count, MPI_Datatype dt, int dest, int tag,
                      Request* r)
{
    State& s = st();
#if !defined(COLLAGE_ST_UCP)
    dt = hold_type(dt); /* (ucp flattens the type at once) */
#endif
    *r = Request{};
    r->k = Request::Enq;
    (void)ch; (void)buf; (void)count; (void)dt; (void)dest; (void)tag; (void)s;
#if defined(COLLAGE_ST_UCP)
    r->u = ucpx::isend(ch.ucomm(), buf, count, dt, dest, tag);
#elif defined(COLLAGE_ST_MPICH)
#if defined(MPI_VERSION) && MPI_VERSION >= 4 && !defined(COLLAGE_NO_LARGE_COUNT)
    MPI_CHECK(MPIX_Isend_enqueue_c(buf, (MPI_Count)count, dt, dest, tag, ch.scomm(), &r->h));
#else
    MPI_CHECK(MPIX_Isend_enqueue(buf, icount(count), dt, dest, tag, ch.scomm(), &r->h));
#endif
#elif defined(COLLAGE_ST_ACX)
    XPORT_CHECK(MPIX_Isend_enqueue(buf, icount(count), dt, dest, tag, ch.comm(), (MPIX_Request*)&r->x,
                                   MPIX_QUEUE_CUDA_STREAM, &s.stream));
#elif defined(COLLAGE_ST_CRAY)
    r->q = cray_queue();
    MPI_CHECK(MPIX_Enqueue_send(buf, icount(count), dt, dest, tag, ch.comm(), s.queues[(size_t)r->q], &r->h));
#else
    throw std::logic_error("no stream-triggered transport built");
#endif
    mem::mark_pending();
    ++s.enqueued;
}

inline void enq_irecv(const Channel& ch, void* buf, size_t count, MPI_Datatype dt, int src, int tag, Request* r)
{
    State& s = st();
#if !defined(COLLAGE_ST_UCP)
    dt = hold_type(dt);
#endif
    *r = Request{};
    r->k = Request::Enq;
    (void)ch; (void)buf; (void)count; (void)dt; (void)src; (void)tag; (void)s;
#if defined(COLLAGE_ST_UCP)
    r->u = ucpx::irecv(ch.ucomm(), buf, count, dt, src, tag);
#elif defined(COLLAGE_ST_MPICH)
#if defined(MPI_VERSION) && MPI_VERSION >= 4 && !defined(COLLAGE_NO_LARGE_COUNT)
    MPI_CHECK(MPIX_Irecv_enqueue_c(buf, (MPI_Count)count, dt, src, tag, ch.scomm(), &r->h));
#else
    MPI_CHECK(MPIX_Irecv_enqueue(buf, icount(count), dt, src, tag, ch.scomm(), &r->h));
#endif
#elif defined(COLLAGE_ST_ACX)
    XPORT_CHECK(MPIX_Irecv_enqueue(buf, icount(count), dt, src, tag, ch.comm(), (MPIX_Request*)&r->x,
                                   MPIX_QUEUE_CUDA_STREAM, &s.stream));
#elif defined(COLLAGE_ST_CRAY)
    r->q = cray_queue();
    MPI_CHECK(MPIX_Enqueue_recv(buf, icount(count), dt, src, tag, ch.comm(), s.queues[(size_t)r->q], &r->h));
#else
    throw std::logic_error("no stream-triggered transport built");
#endif
    mem::mark_pending();
    ++s.enqueued;
}

/* the stream waits for r; the host does not */
inline void enq_wait(Request* r, MPI_Status* user_status)
{
    State& s = st();
    (void)s;
#if defined(COLLAGE_ST_UCP)
    ucpx::wait(&r->u, 1);
    if (user_status != MPI_STATUS_IGNORE) s.ucp_status_of[user_status] = r->u;
#elif defined(COLLAGE_ST_MPICH)
    MPI_CHECK(MPIX_Wait_enqueue(&r->h, status_slot(user_status)));
#elif defined(COLLAGE_ST_ACX)
    XPORT_CHECK(MPIX_Wait_enqueue((MPIX_Request*)&r->x, status_slot(user_status), MPIX_QUEUE_CUDA_STREAM,
                                  &s.stream));
#elif defined(COLLAGE_ST_CRAY)
    flush();
    XPORT_CHECK(MPIX_Enqueue_wait(s.queues[(size_t)r->q]));
    s.to_release.push_back(r->h);
    if (user_status != MPI_STATUS_IGNORE) s.slot_of[user_status] = nullptr; /* no status */
#else
    (void)user_status;
    throw std::logic_error("no stream-triggered transport built");
#endif
    *r = Request{};
    mem::mark_pending();
}

/* the stream waits for all of reqs[0..n) (all Enq) */
inline void enq_waitall(int n, Request** reqs)
{
    if (n <= 0) return;
    State& s = st();
    (void)s;
#if defined(COLLAGE_ST_UCP)
    std::vector<int> u((size_t)n);
    for (int i = 0; i < n; ++i) u[(size_t)i] = reqs[i]->u;
    ucpx::wait(u.data(), n);
#elif defined(COLLAGE_ST_MPICH)
    std::vector<MPI_Request> h((size_t)n);
    for (int i = 0; i < n; ++i) h[(size_t)i] = reqs[i]->h;
    MPI_CHECK(MPIX_Waitall_enqueue(n, h.data(), MPI_STATUSES_IGNORE));
#elif defined(COLLAGE_ST_ACX)
    std::vector<MPIX_Request> x((size_t)n);
    for (int i = 0; i < n; ++i) x[(size_t)i] = (MPIX_Request)reqs[i]->x;
    XPORT_CHECK(MPIX_Waitall_enqueue(n, x.data(), MPI_STATUSES_IGNORE, MPIX_QUEUE_CUDA_STREAM, &s.stream));
#elif defined(COLLAGE_ST_CRAY)
    flush();
    for (int i = 0; i < n; ++i) {
        XPORT_CHECK(MPIX_Enqueue_wait(s.queues[(size_t)reqs[i]->q]));
        s.to_release.push_back(reqs[i]->h);
    }
#else
    throw std::logic_error("no stream-triggered transport built");
#endif
    for (int i = 0; i < n; ++i) *reqs[i] = Request{};
    mem::mark_pending();
}

/* a plain receive is about to fill this status: drop any record of an
 * earlier queued receive that used the same status object */
inline void forget_status(MPI_Status* user)
{
    if (user != MPI_STATUS_IGNORE && st().kind != Kind::Mpi) {
        st().slot_of.erase(user);
#if defined(COLLAGE_ST_UCP)
        st().ucp_status_of.erase(user);
#endif
    }
}

/* status of a receive completed on the stream: wait for the stream, copy it
 * into the algorithm's status object. False when the layer has no record of
 * this status (it came from the plain path and is already filled). */
inline bool fetch_status(MPI_Status* user)
{
    State& s = st();
    if (s.kind == Kind::Mpi) return false;
#if defined(COLLAGE_ST_UCP)
    if (s.kind == Kind::Ucp) {
        auto u = s.ucp_status_of.find(user);
        if (u == s.ucp_status_of.end()) return false;
        mem::sync();
        ucpx::status(u->second, user);
        s.ucp_status_of.erase(u);
        return true;
    }
#endif
    auto it = s.slot_of.find(user);
    if (it == s.slot_of.end()) return false;
    if (!it->second)
        throw std::logic_error(std::string("the ") + kind_name(s.kind) + " transport does not return receive statuses");
    mem::sync();
    *user = *it->second;
    s.slot_of.erase(it);
    return true;
}

} // namespace xport

#endif /* COLLAGE_TRANSPORT_HPP */
