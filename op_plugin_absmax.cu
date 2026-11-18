/*
 * Example reduction-operator plugin for collage (-o plugin:./libabsmax.so):
 * element-wise absolute maximum, inout[i] = max(|in[i]|, |inout[i]|), for
 * float, double, float16 and bfloat16, with a host MPI function and a CUDA
 * kernel. See collage_op_plugin.h for the contract.
 *
 *   nvcc -O3 -std=c++17 -arch=sm_80 -shared -Xcompiler -fPIC \
 *        -I$MPI_HOME/include -I.. -o libabsmax.so op_plugin_absmax.cu
 *   srun ... ./collage -c allreduce -t bfloat16 -o plugin:./libabsmax.so --gpu --check
 */
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>

#include "collage_op_plugin.h"
#include "lowp.hpp" /* collage's float16 / bfloat16 storage types */

namespace {

enum Kind { NONE, F32, F64, F16, BF16 };
Kind g_kind = NONE; /* set by init(); one element type per run */

template <class T>
struct AbsMax {
    __host__ __device__ T operator()(T a, T b) const
    {
        const float x = fabsf((float)a), y = fabsf((float)b);
        return T(x > y ? x : y);
    }
};
template <>
struct AbsMax<double> {
    __host__ __device__ double operator()(double a, double b) const { return fmax(fabs(a), fabs(b)); }
};

template <class T>
__global__ void absmax_kernel(const T* __restrict__ in, T* __restrict__ inout, size_t n)
{
    AbsMax<T> op;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)blockDim.x * gridDim.x)
        inout[i] = op(in[i], inout[i]);
}

template <class T>
void host_loop(const void* in, void* inout, size_t n)
{
    AbsMax<T> op;
    const T* a = (const T*)in;
    T* b = (T*)inout;
    for (size_t i = 0; i < n; ++i) b[i] = op(a[i], b[i]);
}

template <class T>
int launch(const void* in, void* inout, size_t n, cudaStream_t s)
{
    const unsigned threads = 256;
    size_t blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    absmax_kernel<T><<<(unsigned)blocks, threads, 0, s>>>((const T*)in, (T*)inout, n);
    return cudaGetLastError() == cudaSuccess ? 0 : 1; /* nonzero: collage uses host_fn */
}

/* MPI user function (host): the native baseline and staged reductions */
void host_fn(void* in, void* inout, int* len, MPI_Datatype*)
{
    switch (g_kind) {
        case F32: host_loop<float>(in, inout, (size_t)*len); break;
        case F64: host_loop<double>(in, inout, (size_t)*len); break;
        case F16: host_loop<lowp::float16>(in, inout, (size_t)*len); break;
        case BF16: host_loop<lowp::bfloat16>(in, inout, (size_t)*len); break;
        default: break;
    }
}

/* device launcher: queue on collage's stream, never synchronise */
int device_fn(const void* in, void* inout, size_t n, MPI_Datatype, void* stream)
{
    if (n == 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    switch (g_kind) {
        case F32: return launch<float>(in, inout, n, s);
        case F64: return launch<double>(in, inout, n, s);
        case F16: return launch<lowp::float16>(in, inout, n, s);
        case BF16: return launch<lowp::bfloat16>(in, inout, n, s);
        default: return 1;
    }
}

int init(const char* type, size_t)
{
    g_kind = !strcmp(type, "float") ? F32 : !strcmp(type, "double") ? F64 : !strcmp(type, "float16") ? F16
           : !strcmp(type, "bfloat16") ? BF16 : NONE;
    return g_kind == NONE; /* reject other types */
}

const collage_op_plugin plugin = {COLLAGE_OP_PLUGIN_ABI, "absmax", 1, &host_fn, &device_fn, &init};

} // namespace

extern "C" const collage_op_plugin* collage_op_plugin_get(void) { return &plugin; }
