/*
 * GPU kernels of the one-sided transports (build ST=ucp or ST=nvshmem).
 * The protocol is in onesided_device_ops.hpp; this file supplies the CUDA
 * execution interface and the launchers. Every kernel is one block of
 * OSD_THREADS threads; communication calls are block-cooperative
 * (UCX: UCS_DEVICE_LEVEL_BLOCK, NVSHMEM: the *_block variants).
 *
 *   ST=ucp      compiled as ordinary CUDA (the UCX device API is header-only)
 *   ST=nvshmem  compiled with -rdc=true and device-linked with the NVSHMEM
 *               device library; the NVSHMEM host calls collage needs are
 *               wrapped here too, so that the host compiler never includes
 *               nvshmem.h
 */
#include <cuda_runtime.h>

#if defined(COLLAGE_ST_UCP)
#include <ucp/api/device/ucp_device_impl.h>
#define OSD_LEVEL UCS_DEVICE_LEVEL_BLOCK
#elif defined(COLLAGE_ST_NVSHMEM)
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

#ifndef OSD_THREADS
#define OSD_THREADS 128
#endif

#define OSD_FN __device__ __forceinline__ static
#define OSD_TID threadIdx.x
#define OSD_NTHR blockDim.x
#define OSD_SYNC() __syncthreads()
#define OSD_FENCE() __threadfence_system()

__device__ __forceinline__ static uint64_t osd_now()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
#define OSD_NOW() osd_now()

#include "onesided_device_ops.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

__global__ void k_rts(osp::Global g, osp::Op* const* recvs, uint32_t nrecv, uint32_t* cursor)
{
    __shared__ osd::Shared sh;
    osd::rts_kernel_body(g, recvs, nrecv, cursor, &sh);
}

__global__ void k_progress(osp::Global g, osp::Op* const* ops, uint32_t n, osp::Op* const* recvs, uint32_t nrecv,
                           uint32_t* cursor)
{
    __shared__ osd::Shared sh;
    osd::progress_body(g, ops, n, recvs, nrecv, cursor, &sh);
}

__global__ void k_blocks(const char* src, char* dst, const uint64_t* blocks, uint32_t nblocks, int pack)
{
    osd::blocks_body(src, dst, blocks, nblocks, pack);
}

void check(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) {
        std::fprintf(stderr, "collage one-sided transport: %s: %s\n", what, cudaGetErrorString(e));
        std::abort();
    }
}

} // namespace

namespace osk {

void launch_rts(void* stream, const osp::Global& g, osp::Op* const* recvs, uint32_t nrecv, uint32_t* cursor)
{
    k_rts<<<1, OSD_THREADS, 0, (cudaStream_t)stream>>>(g, recvs, nrecv, cursor);
    check(cudaGetLastError(), "receiver-ready kernel");
}

void launch_progress(void* stream, const osp::Global& g, osp::Op* const* ops, uint32_t n, osp::Op* const* recvs,
                     uint32_t nrecv, uint32_t* cursor)
{
    k_progress<<<1, OSD_THREADS, 0, (cudaStream_t)stream>>>(g, ops, n, recvs, nrecv, cursor);
    check(cudaGetLastError(), "progress kernel");
}

void launch_blocks(void* stream, const char* src, char* dst, const uint64_t* dev_blocks, uint32_t nblocks, int pack)
{
    k_blocks<<<1, 512, 0, (cudaStream_t)stream>>>(src, dst, dev_blocks, nblocks, pack);
    check(cudaGetLastError(), pack ? "pack kernel" : "unpack kernel");
}

#if defined(COLLAGE_ST_NVSHMEM)
int shmem_init_mpi(void* mpi_comm)
{
#if defined(NVSHMEMX_INIT_ATTR_INITIALIZER)
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
#else
    nvshmemx_init_attr_t attr;
    std::memset(&attr, 0, sizeof attr);
#endif
    attr.mpi_comm = mpi_comm;
    return nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
}
int shmem_my_pe() { return nvshmem_my_pe(); }
int shmem_n_pes() { return nvshmem_n_pes(); }
void* shmem_malloc(size_t bytes) { return nvshmem_malloc(bytes); }
void shmem_free(void* p) { nvshmem_free(p); }
void shmem_finalize() { nvshmem_finalize(); }
void shmem_info(char* buf, size_t len)
{
    char name[NVSHMEM_MAX_NAME_LEN];
    nvshmem_info_get_name(name);
    std::snprintf(buf, len, "%s", name);
}
#endif

} // namespace osk
