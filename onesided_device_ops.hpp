#pragma once
#ifndef COLLAGE_ONESIDED_DEVICE_OPS_HPP
#define COLLAGE_ONESIDED_DEVICE_OPS_HPP

/*
 * GPU side of the one-sided transports: the bodies of the three kernels
 * (receiver-ready, progress, pack/unpack), written once against
 *
 *  - a small execution interface that the includer defines:
 *      OSD_FN         function qualifier (e.g. __device__ __forceinline__)
 *      OSD_TID        thread index in the block,   OSD_NTHR  threads in the block
 *      OSD_SYNC()     block barrier                OSD_FENCE() system-scope fence
 *      OSD_NOW()      nanoseconds (a monotonic clock)
 *      OSD_LEVEL      ucp: cooperation level of the UCX calls (UCS_DEVICE_LEVEL_BLOCK)
 *
 *  - three communication primitives, one implementation per back end:
 *      put_rts    put an Rts into the sender's ring, then bump its rts counter
 *      put_msg    put length + payload into the receiver, then bump its data counter
 *      flush      make every operation issued so far complete (sources reusable)
 *    and local counter access (read_counter / write_counter).
 *
 * Control decisions are taken by thread 0 and broadcast through Shared;
 * every communication call is made by all threads of the block (UCX block
 * level, NVSHMEM *_block variants). onesided_device.cu instantiates these in
 * real kernels; the test emulation runs them on the host with one thread.
 */

#include "onesided_proto.hpp"

namespace osd {

using namespace osp;

struct Shared {
#if defined(COLLAGE_OS_UCX)
    ucp_device_request_t req;
#endif
    uint64_t dst_off, cap, aux;
    uint32_t ready, remaining, abort, tag, start, issued;
};

OSD_FN void set_error(const Global& g, uint64_t code, uint64_t info)
{
    if (g.err[0] == 0) {
        g.err[1] = info;
        OSD_FENCE();
        g.err[0] = code;
    }
    OSD_FENCE();
}

OSD_FN char* ctl(const Global& g, const DevComm* c, uint32_t peer)
{
    return g.arena + c->ctl_off + (uint64_t)peer * c->L.entry();
}
/* offset, in the arena of peer rank `peer`, of that peer's entry about me */
OSD_FN uint64_t remote_ctl(const DevComm* c, uint32_t peer)
{
    return c->peer_ctl_off[peer] + (uint64_t)c->me * c->L.entry();
}

/* ======================================================================== */
#if defined(COLLAGE_OS_UCX)
/* ---- UCX GPU device API, both generations ------------------------------- */
OSD_FN uint64_t read_counter(const char* p) { return ucp_device_counter_read<OSD_LEVEL>(p); }
OSD_FN void write_counter(char* p, uint64_t v) { ucp_device_counter_write<OSD_LEVEL>(p, v); }

OSD_FN int complete(Shared* sh, ucs_status_t st)
{
    /* UCS_INPROGRESS: progress the request until done (every thread) */
    while (st == UCS_INPROGRESS) st = ucp_device_progress_req<OSD_LEVEL>(&sh->req);
    return st == UCS_OK ? 0 : -(int)st;
}

OSD_FN int dput(const Global& g, Shared* sh, uint32_t wpeer, uint64_t src_off, uint64_t dst_off, uint64_t len)
{
#if COLLAGE_UCP_API == 2
    const unsigned elem = wpeer < g.me_world ? wpeer : wpeer - 1; /* no element for myself */
    ucs_status_t st = ucp_device_put<OSD_LEVEL>(g.lh, 0, src_off, g.rh, elem, dst_off, len, g.channel,
                                                UCP_DEVICE_FLAG_NODELAY, &sh->req);
#else
    ucs_status_t st = ucp_device_put_single<OSD_LEVEL>(g.handles[wpeer], 0, src_off, dst_off, len, g.channel,
                                                       UCP_DEVICE_FLAG_NODELAY, &sh->req);
#endif
    return complete(sh, st);
}

OSD_FN int dinc(const Global& g, Shared* sh, uint32_t wpeer, uint64_t dst_off)
{
#if COLLAGE_UCP_API == 2
    const unsigned elem = wpeer < g.me_world ? wpeer : wpeer - 1;
    ucs_status_t st = ucp_device_counter_inc<OSD_LEVEL>(1, g.rh, elem, dst_off, g.channel,
                                                        UCP_DEVICE_FLAG_NODELAY, &sh->req);
#else
    ucs_status_t st = ucp_device_counter_inc<OSD_LEVEL>(g.handles[wpeer], 0, 1, dst_off, g.channel,
                                                        UCP_DEVICE_FLAG_NODELAY, &sh->req);
#endif
    return complete(sh, st);
}

/* every UCX operation is completed before the next one is issued: the
 * counter goes up only once the data it announces has landed */
OSD_FN int put_rts(const Global& g, Shared* sh, uint32_t wpeer, uint64_t src_off, uint64_t rts_off, uint64_t cnt_off)
{
    int rc = dput(g, sh, wpeer, src_off, rts_off, sizeof(Rts));
    if (rc == 0) rc = dinc(g, sh, wpeer, cnt_off);
    return rc;
}

OSD_FN int put_msg(const Global& g, Shared* sh, uint32_t wpeer, uint64_t src_off, uint64_t dst_off, uint64_t len,
                   uint64_t lsrc_off, uint64_t len_off, uint64_t cnt_off)
{
    int rc = 0;
    if (len > 0) rc = dput(g, sh, wpeer, src_off, dst_off, len);
    if (rc == 0) rc = dput(g, sh, wpeer, lsrc_off, len_off, 8);
    if (rc == 0) rc = dinc(g, sh, wpeer, cnt_off);
    return rc;
}

OSD_FN void flush(const Global&, Shared*) {} /* already complete */

#else
/* ---- NVSHMEM device API -------------------------------------------------- */
/* The arena is one symmetric object: (arena + offset) names the same bytes on
 * every PE, and PE = world rank. */
OSD_FN uint64_t read_counter(const char* p) { return nvshmem_signal_fetch((uint64_t*)const_cast<char*>(p)); }
OSD_FN void write_counter(char* p, uint64_t v) { *(volatile uint64_t*)p = v; } /* self entry: local only */

/* put-with-signal: the signal is delivered after the data */
OSD_FN int put_rts(const Global& g, Shared* sh, uint32_t pe, uint64_t src_off, uint64_t rts_off, uint64_t cnt_off)
{
    nvshmemx_putmem_signal_nbi_block(g.arena + rts_off, g.arena + src_off, sizeof(Rts),
                                     (uint64_t*)(g.arena + cnt_off), 1, NVSHMEM_SIGNAL_ADD, (int)pe);
    if (OSD_TID == 0) sh->issued = 1;
    return 0;
}

/* length first; a fence orders it before payload + signal */
OSD_FN int put_msg(const Global& g, Shared* sh, uint32_t pe, uint64_t src_off, uint64_t dst_off, uint64_t len,
                   uint64_t lsrc_off, uint64_t len_off, uint64_t cnt_off)
{
    uint64_t* sig = (uint64_t*)(g.arena + cnt_off);
    nvshmemx_putmem_nbi_block(g.arena + len_off, g.arena + lsrc_off, 8, (int)pe);
    nvshmem_fence();
    if (len > 0) {
        nvshmemx_putmem_signal_nbi_block(g.arena + dst_off, g.arena + src_off, len, sig, 1, NVSHMEM_SIGNAL_ADD,
                                         (int)pe);
    } else {
        if (OSD_TID == 0) nvshmemx_signal_op(sig, 1, NVSHMEM_SIGNAL_ADD, (int)pe);
        OSD_SYNC();
    }
    if (OSD_TID == 0) sh->issued = 1;
    return 0;
}

/* complete what this block issued (once per pass, so puts batch up): send
 * buffers and staging become reusable, and nbi operations are guaranteed to
 * progress while the kernel spins */
OSD_FN void flush(const Global&, Shared* sh)
{
    OSD_SYNC();
    if (sh->issued) {
        nvshmem_quiet();
        OSD_SYNC();
        if (OSD_TID == 0) sh->issued = 0;
    }
    OSD_SYNC();
}
#endif
/* ======================================================================== */

/* cooperative copy (self messages, pack/unpack) */
OSD_FN void copy_bytes(char* dst, const char* src, uint64_t n)
{
    const uint64_t t = OSD_TID, nt = OSD_NTHR;
    if ((((uintptr_t)dst | (uintptr_t)src) & 15) == 0) {
        const uint64_t nv = n / 16;
        for (uint64_t i = t; i < nv; i += nt) {
            struct V { uint64_t a, b; };
            ((V*)dst)[i] = ((const V*)src)[i];
        }
        for (uint64_t i = nv * 16 + t; i < n; i += nt) dst[i] = src[i];
    } else {
        for (uint64_t i = t; i < n; i += nt) dst[i] = src[i];
    }
}

/* ---- receiver-ready -------------------------------------------------------- */
/* Post the receiver-ready descriptor of one receive, if it can be posted now:
 * descriptors to a peer go out in sequence order (the sender reads slot k
 * once its counter says k + 1), and slot k % R is free once message k - R has
 * arrived. Never waits: a descriptor that cannot go out yet stays pending and
 * a later kernel posts it, so a full ring never blocks the stream. */
OSD_FN void try_post(const Global& g, Op* op, Shared* sh)
{
    const DevComm* c = op->comm;
    const CtlLayout& L = c->L;
    const uint32_t peer = op->peer;
    char* mine = ctl(g, c, peer);
    if (OSD_TID == 0) {
        uint32_t go = 0;
        if (!op->rts && *(volatile uint64_t*)(mine + L.posted()) == op->seq &&
            (op->seq < L.R || read_counter(mine + L.data_cnt()) >= op->seq - L.R + 1))
            go = 1;
        sh->ready = go;
    }
    OSD_SYNC();
    if (!sh->ready) return;
    if (peer == c->me) {
        if (OSD_TID == 0) {
            Rts* r = (Rts*)(mine + L.rts(op->seq));
            r->off = op->off;
            r->cap = op->len;
            r->tag = op->tag;
            OSD_FENCE();
            write_counter(mine + L.rts_cnt(), read_counter(mine + L.rts_cnt()) + 1);
        }
    } else {
        if (OSD_TID == 0) {
            Rts* r = (Rts*)(mine + L.rsrc(op->seq));
            r->off = op->off;
            r->cap = op->len;
            r->tag = op->tag;
            OSD_FENCE();
        }
        OSD_SYNC();
        const uint32_t w = c->world[peer];
        const uint64_t base = remote_ctl(c, peer);
        const uint64_t src = c->ctl_off + (uint64_t)peer * L.entry() + L.rsrc(op->seq);
        const int rc = put_rts(g, sh, w, src, base + L.rts(op->seq), base + L.rts_cnt());
        if (rc != 0 && OSD_TID == 0) set_error(g, ERR_XPORT, ((uint64_t)peer << 32) | (uint32_t)rc);
    }
    if (OSD_TID == 0) {
        op->rts = 1;
        *(volatile uint64_t*)(mine + L.posted()) = op->seq + 1;
        OSD_FENCE();
    }
    OSD_SYNC();
}

/* post every descriptor that can go out (in order per peer), starting at the
 * first receive not yet posted */
OSD_FN void post_pending(const Global& g, Op* const* recvs, uint32_t nrecv, uint32_t* cursor, Shared* sh)
{
    /* a pass can unblock a later receive of the same peer: repeat while it helps */
    for (;;) {
        if (OSD_TID == 0) {
            sh->aux = 0;
            uint32_t c = *(volatile uint32_t*)cursor;
            while (c < nrecv && recvs[c]->rts) ++c; /* skip the posted prefix */
            *(volatile uint32_t*)cursor = c;
            sh->start = c;
        }
        OSD_SYNC();
        for (uint32_t i = sh->start; i < nrecv; ++i) {
            Op* op = recvs[i];
            if (op->rts) continue;
            try_post(g, op, sh);
            if (OSD_TID == 0 && op->rts) sh->aux = 1;
            OSD_SYNC();
        }
        flush(g, sh);
        if (!sh->aux) break;
    }
}

OSD_FN void rts_kernel_body(const Global& g, Op* const* recvs, uint32_t nrecv, uint32_t* cursor, Shared* sh)
{
    if (OSD_TID == 0) sh->issued = 0;
    OSD_SYNC();
    post_pending(g, recvs, nrecv, cursor, sh);
}

/* ---- sends and receives ----------------------------------------------------- */
/* thread 0: is op ready? (fills sh for a send) 0 = not yet, 1 = go, 2 = failed */
OSD_FN uint32_t ready(const Global& g, Op* op, Shared* sh)
{
    const DevComm* c = op->comm;
    const CtlLayout& L = c->L;
    char* mine = ctl(g, c, op->peer);
    if (op->kind == OP_RECV) {
        if (!op->rts || read_counter(mine + L.data_cnt()) < op->seq + 1) return 0;
        OSD_FENCE();
        op->got = *(volatile uint64_t*)(mine + L.len(op->seq));
        return 1;
    }
    if (read_counter(mine + L.rts_cnt()) < op->seq + 1) return 0;
    OSD_FENCE();
    volatile Rts* r = (volatile Rts*)(mine + L.rts(op->seq));
    sh->dst_off = r->off;
    sh->cap = r->cap;
    sh->tag = r->tag;
    if (sh->tag != op->tag) {
        set_error(g, ERR_TAG, ((uint64_t)op->peer << 32) | op->tag);
        return 2;
    }
    if (op->len > sh->cap) {
        set_error(g, ERR_TRUNCATE, ((uint64_t)op->peer << 32) | (op->len & 0xffffffffu));
        return 2;
    }
    return 1;
}

OSD_FN void send_body(const Global& g, Op* op, Shared* sh)
{
    const DevComm* c = op->comm;
    const CtlLayout& L = c->L;
    const uint32_t peer = op->peer;
    if (peer == c->me) {
        copy_bytes(g.arena + sh->dst_off, g.arena + op->off, op->len);
        OSD_SYNC();
        if (OSD_TID == 0) {
            char* mine = ctl(g, c, peer);
            *(volatile uint64_t*)(mine + L.len(op->seq)) = op->len;
            OSD_FENCE();
            write_counter(mine + L.data_cnt(), read_counter(mine + L.data_cnt()) + 1);
            OSD_FENCE();
        }
        OSD_SYNC();
        return;
    }
    const uint32_t w = c->world[peer];
    const uint64_t base = remote_ctl(c, peer);
    if (OSD_TID == 0) {
        *(volatile uint64_t*)(ctl(g, c, peer) + L.lsrc(op->seq)) = op->len;
        OSD_FENCE();
    }
    OSD_SYNC();
    const uint64_t lsrc = c->ctl_off + (uint64_t)peer * L.entry() + L.lsrc(op->seq);
    const int rc = put_msg(g, sh, w, op->off, sh->dst_off, op->len, lsrc, base + L.len(op->seq), base + L.data_cnt());
    if (rc != 0 && OSD_TID == 0) set_error(g, ERR_XPORT, ((uint64_t)peer << 32) | (uint32_t)rc);
    OSD_SYNC();
}

/* complete every op; whichever is ready goes first, like MPI_Waitall */
OSD_FN void progress_body(const Global& g, Op* const* ops, uint32_t n, Op* const* recvs, uint32_t nrecv,
                          uint32_t* cursor, Shared* sh)
{
    const uint64_t t0 = OSD_NOW();
    if (OSD_TID == 0) sh->issued = 0;
    OSD_SYNC();
    for (;;) {
        post_pending(g, recvs, nrecv, cursor, sh);
        if (OSD_TID == 0) {
            sh->remaining = 0;
            sh->abort = 0;
        }
        OSD_SYNC();
        for (uint32_t i = 0; i < n; ++i) {
            Op* op = ops[i];
            if (op->done) continue; /* written by thread 0 before a barrier */
            if (OSD_TID == 0) sh->ready = ready(g, op, sh);
            OSD_SYNC();
            const uint32_t r = sh->ready;
            if (r == 1 && op->kind == OP_SEND) send_body(g, op, sh);
            if (OSD_TID == 0) {
                if (r) op->done = 1;
                else ++sh->remaining;
            }
            OSD_SYNC();
        }
        flush(g, sh); /* sends of this pass complete before the kernel can end */
        if (sh->remaining == 0) break;
        if (OSD_TID == 0 && OSD_NOW() - t0 > g.timeout_ns) {
            for (uint32_t i = 0; i < n; ++i)
                if (!ops[i]->done) {
                    const uint32_t why = ops[i]->kind == OP_SEND ? 2 : (ops[i]->rts ? 3 : 1);
                    set_error(g, ERR_TIMEOUT, ((uint64_t)ops[i]->peer << 32) | why);
                    break;
                }
            sh->abort = 1;
        }
        OSD_SYNC();
        if (sh->abort) break;
    }
}

/* derived datatypes: blocks = {offset in the user buffer, length} pairs */
OSD_FN void blocks_body(const char* src, char* dst, const uint64_t* blocks, uint32_t nblocks, int pack)
{
    uint64_t pos = 0;
    for (uint32_t b = 0; b < nblocks; ++b) {
        const uint64_t off = blocks[2 * b], len = blocks[2 * b + 1];
        if (pack) copy_bytes(dst + pos, src + off, len);
        else copy_bytes(dst + off, src + pos, len);
        pos += len;
    }
    OSD_SYNC();
}

} // namespace osd

#endif /* COLLAGE_ONESIDED_DEVICE_OPS_HPP */
