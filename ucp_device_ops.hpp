#pragma once
#ifndef COLLAGE_UCP_DEVICE_OPS_HPP
#define COLLAGE_UCP_DEVICE_OPS_HPP

/*
 * GPU side of the UCP device transport: the bodies of the three kernels
 * (receiver-ready, progress, pack/unpack), written once against a small
 * execution interface that the includer defines:
 *
 *   UCPD_FN         function qualifier (e.g. __device__ __forceinline__)
 *   UCPD_TID        thread index in the block,   UCPD_NTHR  threads in the block
 *   UCPD_SYNC()     block barrier                UCPD_FENCE() system-scope fence
 *   UCPD_NOW()      nanoseconds (a monotonic clock)
 *   UCPD_LEVEL      cooperation level of the UCX calls (UCS_DEVICE_LEVEL_BLOCK:
 *                   every thread of the block calls them with the same arguments)
 *
 * Control decisions are taken by thread 0 and broadcast through Shared;
 * every UCX call is made by all threads of the block.
 * ucp_device.cu instantiates these in real kernels; the test emulation runs
 * them on the host with one thread.
 */

#include "ucp_proto.hpp"

namespace ucpd {

using namespace ucpp;

struct Shared {
    ucp_device_request_t req;
    uint64_t dst_off, cap, aux;
    uint32_t ready, remaining, abort, tag, start;
};

UCPD_FN void set_error(const Global& g, uint64_t code, uint64_t info)
{
    if (g.err[0] == 0) {
        g.err[1] = info;
        UCPD_FENCE();
        g.err[0] = code;
    }
    UCPD_FENCE();
}

UCPD_FN char* ctl(const Global& g, const DevComm* c, uint32_t peer)
{
    return g.arena + c->ctl_off + (uint64_t)peer * c->L.entry();
}
/* offset, in the arena of peer rank `peer`, of that peer's entry about me */
UCPD_FN uint64_t remote_ctl(const DevComm* c, uint32_t peer)
{
    return c->peer_ctl_off[peer] + (uint64_t)c->me * c->L.entry();
}
UCPD_FN uint64_t read_counter(const char* p) { return ucp_device_counter_read<UCPD_LEVEL>(p); }

/* ---- UCX device API, both generations ---------------------------------- */
UCPD_FN ucs_status_t complete(Shared* sh, ucs_status_t st)
{
    /* UCS_INPROGRESS: progress the request until done (every thread) */
    while (st == UCS_INPROGRESS) st = ucp_device_progress_req<UCPD_LEVEL>(&sh->req);
    return st;
}

UCPD_FN ucs_status_t dput(const Global& g, Shared* sh, uint32_t wpeer, uint64_t src_off, uint64_t dst_off,
                          uint64_t len)
{
#if COLLAGE_UCP_API == 2
    const unsigned elem = wpeer < g.me_world ? wpeer : wpeer - 1; /* no element for myself */
    ucs_status_t st = ucp_device_put<UCPD_LEVEL>(g.lh, 0, src_off, g.rh, elem, dst_off, len, g.channel,
                                                 UCP_DEVICE_FLAG_NODELAY, &sh->req);
#else
    ucs_status_t st = ucp_device_put_single<UCPD_LEVEL>(g.handles[wpeer], 0, src_off, dst_off, len, g.channel,
                                                        UCP_DEVICE_FLAG_NODELAY, &sh->req);
#endif
    return complete(sh, st);
}

UCPD_FN ucs_status_t dinc(const Global& g, Shared* sh, uint32_t wpeer, uint64_t dst_off)
{
#if COLLAGE_UCP_API == 2
    const unsigned elem = wpeer < g.me_world ? wpeer : wpeer - 1;
    ucs_status_t st = ucp_device_counter_inc<UCPD_LEVEL>(1, g.rh, elem, dst_off, g.channel,
                                                         UCP_DEVICE_FLAG_NODELAY, &sh->req);
#else
    ucs_status_t st = ucp_device_counter_inc<UCPD_LEVEL>(g.handles[wpeer], 0, 1, dst_off, g.channel,
                                                         UCP_DEVICE_FLAG_NODELAY, &sh->req);
#endif
    return complete(sh, st);
}

/* cooperative copy (self messages, pack/unpack) */
UCPD_FN void copy_bytes(char* dst, const char* src, uint64_t n)
{
    const uint64_t t = UCPD_TID, nt = UCPD_NTHR;
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
UCPD_FN void try_post(const Global& g, Op* op, Shared* sh)
{
    const DevComm* c = op->comm;
    const CtlLayout& L = c->L;
    const uint32_t peer = op->peer;
    char* mine = ctl(g, c, peer);
    if (UCPD_TID == 0) {
        uint32_t go = 0;
        if (!op->rts && *(volatile uint64_t*)(mine + L.posted()) == op->seq &&
            (op->seq < L.R || read_counter(mine + L.data_cnt()) >= op->seq - L.R + 1))
            go = 1;
        sh->ready = go;
    }
    UCPD_SYNC();
    if (!sh->ready) return;
    if (peer == c->me) {
        if (UCPD_TID == 0) {
            Rts* r = (Rts*)(mine + L.rts(op->seq));
            r->off = op->off;
            r->cap = op->len;
            r->tag = op->tag;
            UCPD_FENCE();
            ucp_device_counter_write<UCPD_LEVEL>(mine + L.rts_cnt(), read_counter(mine + L.rts_cnt()) + 1);
        }
    } else {
        if (UCPD_TID == 0) {
            Rts* r = (Rts*)(mine + L.rsrc(op->seq));
            r->off = op->off;
            r->cap = op->len;
            r->tag = op->tag;
            UCPD_FENCE();
        }
        UCPD_SYNC();
        const uint32_t w = c->world[peer];
        const uint64_t base = remote_ctl(c, peer);
        const uint64_t src = c->ctl_off + (uint64_t)peer * L.entry() + L.rsrc(op->seq);
        ucs_status_t st = dput(g, sh, w, src, base + L.rts(op->seq), sizeof(Rts));
        if (st == UCS_OK) st = dinc(g, sh, w, base + L.rts_cnt());
        if (st != UCS_OK && UCPD_TID == 0) set_error(g, ERR_UCX, ((uint64_t)peer << 32) | (uint32_t)(-st));
    }
    if (UCPD_TID == 0) {
        op->rts = 1;
        *(volatile uint64_t*)(mine + L.posted()) = op->seq + 1;
        UCPD_FENCE();
    }
    UCPD_SYNC();
}

/* post every descriptor that can go out (in order per peer), starting at the
 * first receive not yet posted */
UCPD_FN void post_pending(const Global& g, Op* const* recvs, uint32_t nrecv, uint32_t* cursor, Shared* sh)
{
    /* a pass can unblock a later receive of the same peer: repeat while it helps */
    for (;;) {
        if (UCPD_TID == 0) {
            sh->aux = 0;
            uint32_t c = *(volatile uint32_t*)cursor;
            while (c < nrecv && recvs[c]->rts) ++c; /* skip the posted prefix */
            *(volatile uint32_t*)cursor = c;
            sh->start = c;
        }
        UCPD_SYNC();
        for (uint32_t i = sh->start; i < nrecv; ++i) {
            Op* op = recvs[i];
            if (op->rts) continue;
            try_post(g, op, sh);
            if (UCPD_TID == 0 && op->rts) sh->aux = 1;
            UCPD_SYNC();
        }
        if (!sh->aux) break;
    }
}

UCPD_FN void rts_kernel_body(const Global& g, Op* const* recvs, uint32_t nrecv, uint32_t* cursor, Shared* sh)
{
    post_pending(g, recvs, nrecv, cursor, sh);
}

/* ---- sends and receives ----------------------------------------------------- */
/* thread 0: is op ready? (fills sh for a send) 0 = not yet, 1 = go, 2 = failed */
UCPD_FN uint32_t ready(const Global& g, Op* op, Shared* sh)
{
    const DevComm* c = op->comm;
    const CtlLayout& L = c->L;
    char* mine = ctl(g, c, op->peer);
    if (op->kind == OP_RECV) {
        if (!op->rts || read_counter(mine + L.data_cnt()) < op->seq + 1) return 0;
        UCPD_FENCE();
        op->got = *(volatile uint64_t*)(mine + L.len(op->seq));
        return 1;
    }
    if (read_counter(mine + L.rts_cnt()) < op->seq + 1) return 0;
    UCPD_FENCE();
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

UCPD_FN void send_body(const Global& g, Op* op, Shared* sh)
{
    const DevComm* c = op->comm;
    const CtlLayout& L = c->L;
    const uint32_t peer = op->peer;
    if (peer == c->me) {
        copy_bytes(g.arena + sh->dst_off, g.arena + op->off, op->len);
        UCPD_SYNC();
        if (UCPD_TID == 0) {
            char* mine = ctl(g, c, peer);
            *(volatile uint64_t*)(mine + L.len(op->seq)) = op->len;
            UCPD_FENCE();
            ucp_device_counter_write<UCPD_LEVEL>(mine + L.data_cnt(), read_counter(mine + L.data_cnt()) + 1);
            UCPD_FENCE();
        }
        UCPD_SYNC();
        return;
    }
    const uint32_t w = c->world[peer];
    const uint64_t base = remote_ctl(c, peer);
    ucs_status_t st = UCS_OK;
    if (op->len > 0) st = dput(g, sh, w, op->off, sh->dst_off, op->len);
    if (UCPD_TID == 0) {
        *(volatile uint64_t*)(ctl(g, c, peer) + L.lsrc(op->seq)) = op->len;
        UCPD_FENCE();
    }
    UCPD_SYNC();
    const uint64_t lsrc = c->ctl_off + (uint64_t)peer * L.entry() + L.lsrc(op->seq);
    /* the counter goes up only after payload and length have completed */
    if (st == UCS_OK) st = dput(g, sh, w, lsrc, base + L.len(op->seq), 8);
    if (st == UCS_OK) st = dinc(g, sh, w, base + L.data_cnt());
    if (st != UCS_OK && UCPD_TID == 0) set_error(g, ERR_UCX, ((uint64_t)peer << 32) | (uint32_t)(-st));
    UCPD_SYNC();
}

/* complete every op; whichever is ready goes first, like MPI_Waitall */
UCPD_FN void progress_body(const Global& g, Op* const* ops, uint32_t n, Op* const* recvs, uint32_t nrecv,
                           uint32_t* cursor, Shared* sh)
{
    const uint64_t t0 = UCPD_NOW();
    for (;;) {
        post_pending(g, recvs, nrecv, cursor, sh);
        if (UCPD_TID == 0) {
            sh->remaining = 0;
            sh->abort = 0;
        }
        UCPD_SYNC();
        for (uint32_t i = 0; i < n; ++i) {
            Op* op = ops[i];
            if (op->done) continue; /* written by thread 0 before a barrier */
            if (UCPD_TID == 0) sh->ready = ready(g, op, sh);
            UCPD_SYNC();
            const uint32_t r = sh->ready;
            if (r == 1 && op->kind == OP_SEND) send_body(g, op, sh);
            if (UCPD_TID == 0) {
                if (r) op->done = 1;
                else ++sh->remaining;
            }
            UCPD_SYNC();
        }
        if (sh->remaining == 0) break;
        if (UCPD_TID == 0 && UCPD_NOW() - t0 > g.timeout_ns) {
            for (uint32_t i = 0; i < n; ++i)
                if (!ops[i]->done) {
                    const uint32_t why = ops[i]->kind == OP_SEND ? 2 : (ops[i]->rts ? 3 : 1);
                    set_error(g, ERR_TIMEOUT, ((uint64_t)ops[i]->peer << 32) | why);
                    break;
                }
            sh->abort = 1;
        }
        UCPD_SYNC();
        if (sh->abort) break;
    }
}

/* derived datatypes: blocks = {offset in the user buffer, length} pairs */
UCPD_FN void blocks_body(const char* src, char* dst, const uint64_t* blocks, uint32_t nblocks, int pack)
{
    uint64_t pos = 0;
    for (uint32_t b = 0; b < nblocks; ++b) {
        const uint64_t off = blocks[2 * b], len = blocks[2 * b + 1];
        if (pack) copy_bytes(dst + pos, src + off, len);
        else copy_bytes(dst + off, src + pos, len);
        pos += len;
    }
    UCPD_SYNC();
}

} // namespace ucpd

#endif /* COLLAGE_UCP_DEVICE_OPS_HPP */
