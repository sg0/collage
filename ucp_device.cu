/*
 * GPU kernels of the UCP device transport (--transport ucp, build ST=ucp).
 * The protocol is in ucp_device_ops.hpp; this file supplies the CUDA
 * execution interface and the launchers. Every kernel is one block of
 * UCPD_THREADS threads, so UCX transfers use the block cooperation level
 * (the CUDA-IPC lanes copy with all threads; GDA lanes post from one).
 */
#include <cuda_runtime.h>
#include <ucp/api/device/ucp_device_impl.h>

#ifndef UCPD_THREADS
#define UCPD_THREADS 128
#endif

#define UCPD_FN __device__ __forceinline__ static
#define UCPD_TID threadIdx.x
#define UCPD_NTHR blockDim.x
#define UCPD_SYNC() __syncthreads()
#define UCPD_FENCE() __threadfence_system()
#define UCPD_LEVEL UCS_DEVICE_LEVEL_BLOCK

__device__ __forceinline__ static uint64_t ucpd_now()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
#define UCPD_NOW() ucpd_now()

#include "ucp_device_ops.hpp"

#include <cstdio>
#include <cstdlib>

namespace {

__global__ void k_rts(ucpp::Global g, ucpp::Op* const* recvs, uint32_t nrecv, uint32_t* cursor)
{
    __shared__ ucpd::Shared sh;
    ucpd::rts_kernel_body(g, recvs, nrecv, cursor, &sh);
}

__global__ void k_progress(ucpp::Global g, ucpp::Op* const* ops, uint32_t n, ucpp::Op* const* recvs, uint32_t nrecv,
                           uint32_t* cursor)
{
    __shared__ ucpd::Shared sh;
    ucpd::progress_body(g, ops, n, recvs, nrecv, cursor, &sh);
}

__global__ void k_blocks(const char* src, char* dst, const uint64_t* blocks, uint32_t nblocks, int pack)
{
    ucpd::blocks_body(src, dst, blocks, nblocks, pack);
}

void check(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) {
        std::fprintf(stderr, "collage ucp: %s: %s\n", what, cudaGetErrorString(e));
        std::abort();
    }
}

} // namespace

namespace ucpk {

void launch_rts(void* stream, const ucpp::Global& g, ucpp::Op* const* recvs, uint32_t nrecv, uint32_t* cursor)
{
    k_rts<<<1, UCPD_THREADS, 0, (cudaStream_t)stream>>>(g, recvs, nrecv, cursor);
    check(cudaGetLastError(), "receiver-ready kernel");
}

void launch_progress(void* stream, const ucpp::Global& g, ucpp::Op* const* ops, uint32_t n, ucpp::Op* const* recvs,
                     uint32_t nrecv, uint32_t* cursor)
{
    k_progress<<<1, UCPD_THREADS, 0, (cudaStream_t)stream>>>(g, ops, n, recvs, nrecv, cursor);
    check(cudaGetLastError(), "progress kernel");
}

void launch_blocks(void* stream, const char* src, char* dst, const uint64_t* dev_blocks, uint32_t nblocks, int pack)
{
    k_blocks<<<1, 512, 0, (cudaStream_t)stream>>>(src, dst, dev_blocks, nblocks, pack);
    check(cudaGetLastError(), pack ? "pack kernel" : "unpack kernel");
}

} // namespace ucpk
