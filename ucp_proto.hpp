#pragma once
#ifndef COLLAGE_UCP_PROTO_HPP
#define COLLAGE_UCP_PROTO_HPP

/*
 * UCP device transport (--transport ucp): data layouts shared by the host
 * engine (ucp_transport.hpp) and the GPU kernels (ucp_device.cu).
 *
 * Protocol: every point-to-point message of an algorithm becomes a zero-copy
 * rendezvous made of GPU-initiated one-sided operations (UCX GPU device API):
 *
 *   receiver GPU   put  Rts{arena offset, capacity, tag} -> sender's control area
 *                  inc  sender's rts counter
 *   sender GPU     wait for the rts counter, read the Rts (local memory)
 *                  put  payload -> receiver's buffer (its arena offset)
 *                  put  length  -> receiver's control area
 *                  inc  receiver's data counter        (after both puts completed)
 *   receiver GPU   wait for the data counter
 *
 * Message k between an ordered pair (sender s, receiver r) is matched with
 * receive k from s at r: both sides count their messages in program order.
 * The tag travels with the Rts and is checked by the sender (a mismatch is
 * reported as an error rather than silently mismatching).
 *
 * Every buffer the algorithms touch lives in one registered device arena per
 * process (the device memory pool draws from it), so a peer buffer is just an
 * offset into the peer's arena: no per-message registration or rkey.
 */

#include <stddef.h>
#include <stdint.h>

#include <ucp/api/ucp.h>
#include <ucp/api/device/ucp_host.h>

#if UCP_API_VERSION >= UCP_VERSION(1, 21)
#define COLLAGE_UCP_API 2 /* local/remote mem lists, ucp_device_put (UCX >= 1.21) */
#elif UCP_API_VERSION >= UCP_VERSION(1, 20)
#define COLLAGE_UCP_API 1 /* per-endpoint mem list, ucp_device_put_single (UCX 1.20.x) */
#else
#error "--transport ucp needs UCX >= 1.20 (GPU device API)"
#endif

#if defined(__CUDACC__)
#define UCPP_HD __host__ __device__
#else
#define UCPP_HD
#endif

namespace ucpp {

constexpr uint32_t OP_SEND = 1, OP_RECV = 2;
constexpr uint32_t MAX_SLOTS = 256;

/* receiver-ready descriptor: where the sender may put its message */
struct Rts {
    uint64_t off; /* offset in the receiver's arena */
    uint64_t cap; /* capacity in bytes */
    uint32_t tag;
    uint32_t pad;
};

/* control area of one rank, one entry per peer j of the communicator.
 * The entry lives at a fixed size computed from the slot count R:
 *   uint64 rts_cnt          incremented by j: j posted its k-th receive from me
 *   uint64 data_cnt         incremented by j: j's k-th message to me arrived
 *   uint64 posted           mine: how many receive-ready descriptors I posted to j
 *   Rts    rts[R]           written by j (where my k-th message to j goes)
 *   uint64 len[R]           written by j (length of j's k-th message to me)
 *   Rts    rsrc[R]          my staging for the Rts I put to j
 *   uint64 lsrc[R]          my staging for the lengths I put to j
 */
struct CtlLayout {
    uint32_t R;
    UCPP_HD static constexpr uint64_t rts_cnt() { return 0; }
    UCPP_HD static constexpr uint64_t data_cnt() { return 8; }
    UCPP_HD static constexpr uint64_t posted() { return 16; }
    UCPP_HD uint64_t rts(uint64_t k) const { return 24 + (k % R) * sizeof(Rts); }
    UCPP_HD uint64_t len(uint64_t k) const { return 24 + (uint64_t)R * sizeof(Rts) + (k % R) * 8; }
    UCPP_HD uint64_t rsrc(uint64_t k) const { return 24 + (uint64_t)R * (sizeof(Rts) + 8) + (k % R) * sizeof(Rts); }
    UCPP_HD uint64_t lsrc(uint64_t k) const { return 24 + (uint64_t)R * (2 * sizeof(Rts) + 8) + (k % R) * 8; }
    UCPP_HD uint64_t entry() const
    {
        uint64_t e = 24 + (uint64_t)R * (2 * sizeof(Rts) + 16);
        return (e + 63) & ~(uint64_t)63;
    }
};

/* one communicator, device-resident */
struct DevComm {
    uint32_t p, me;         /* size, my rank */
    uint32_t me_world, pad;
    CtlLayout L;
    uint32_t pad2;
    uint64_t ctl_off;       /* my control area: offset in my arena */
    const uint64_t* peer_ctl_off; /* [p] each peer's control area offset in its arena */
    const uint32_t* world;  /* [p] world rank of each communicator rank */
};

/* one point-to-point operation, as the kernels see it */
struct Op {
    uint64_t seq;  /* k: message number of this ordered pair */
    uint64_t off;  /* my buffer: offset in my arena */
    uint64_t len;  /* send: bytes; recv: capacity */
    uint64_t got;  /* recv: bytes received (written by the kernel) */
    const DevComm* comm;
    uint32_t kind; /* OP_SEND / OP_RECV */
    uint32_t peer; /* communicator rank */
    uint32_t tag;
    uint32_t done; /* completed (kernel state) */
    uint32_t rts;  /* recv: receiver-ready descriptor posted (kernel state) */
    uint32_t pad;
};

/* error word (mapped host memory): code, details */
constexpr uint64_t ERR_TAG = 1, ERR_TRUNCATE = 2, ERR_TIMEOUT = 3, ERR_UCX = 4;

/* everything a kernel needs, passed by value at every launch */
struct Global {
    char* arena;               /* local arena base */
    volatile uint64_t* err;    /* [0] code (first error wins), [1] info */
    uint64_t timeout_ns;
#if COLLAGE_UCP_API == 2
    ucp_device_local_mem_list_h lh;  /* one element: my arena */
    ucp_device_remote_mem_list_h rh; /* one element per other world rank */
#else
    ucp_device_mem_list_handle_h* handles; /* device array [world size], my arena -> rank's arena */
#endif
    uint32_t me_world;
    uint32_t channel;
};

} // namespace ucpp

/* launchers (ucp_device.cu, or an emulation for tests) */
namespace ucpk {
/* Every op has one device copy for the whole call (kernel state persists).
 * recvs[0..nrecv) lists the call's receives in the order they were posted;
 * every kernel posts the receiver-ready descriptors that can go out now
 * (never waiting), starting at *cursor, which it advances past the posted
 * prefix. */
/* receiver-ready: post the descriptors that can be posted now */
void launch_rts(void* stream, const ucpp::Global& g, ucpp::Op* const* recvs, uint32_t nrecv, uint32_t* cursor);
/* complete ops[0..n): sends wait for their Rts and put; receives wait for data */
void launch_progress(void* stream, const ucpp::Global& g, ucpp::Op* const* ops, uint32_t n, ucpp::Op* const* recvs,
                     uint32_t nrecv, uint32_t* cursor);
/* gather (pack) or scatter (unpack) byte blocks {offset, length} of a derived datatype */
void launch_blocks(void* stream, const char* src, char* dst, const uint64_t* dev_blocks, uint32_t nblocks,
                   int pack);
} // namespace ucpk

#endif /* COLLAGE_UCP_PROTO_HPP */
