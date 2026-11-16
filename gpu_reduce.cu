/*
 * CUDA kernels for local reductions on device buffers (built with `make CUDA=1`).
 * One grid-stride kernel per (type, operator); see gpu_ops.hpp.
 */
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "gpu_ops.hpp"

namespace collage_gpu {

template <class T, class Op>
__global__ void reduce_kernel(const T* __restrict__ in, T* __restrict__ inout, size_t n)
{
    Op op;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        inout[i] = op(in[i], inout[i]);
}

struct CudaLauncher {
    cudaStream_t stream;
    template <class T, class Op>
    void run(const T* in, T* inout, size_t n)
    {
        if (n == 0) return;
        const int threads = 256;
        /* enough blocks to fill the device, grid-stride for the rest */
        size_t blocks = std::min<size_t>((n + threads - 1) / threads, 65535);
        reduce_kernel<T, Op><<<(unsigned)blocks, threads, 0, stream>>>(in, inout, n);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "collage: reduction kernel launch failed: %s\n", cudaGetErrorString(e));
            std::abort();
        }
    }
};

bool device_reduce(DType t, OpCode op, const void* in, void* inout, size_t n, void* stream)
{
    CudaLauncher l{static_cast<cudaStream_t>(stream)};
    return dispatch(t, op, in, inout, n, l);
}

}  // namespace collage_gpu
