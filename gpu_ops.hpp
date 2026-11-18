#pragma once
#ifndef COLLAGE_GPU_OPS_HPP
#define COLLAGE_GPU_OPS_HPP

/*
 * Element-wise reduction operators for device buffers, and the table that
 * maps (datatype, operation) to them. Shared by the CUDA kernels
 * (gpu_reduce.cu) and by any host build, so the operator semantics are
 * defined once. Semantics follow MPI_Reduce_local:  inout[i] = in[i] op inout[i].
 *
 * Only predefined operations have device kernels. User-defined operations are
 * host functions and are always reduced on the host (see memory.hpp). The
 * low-precision types (lowp.hpp) reuse these operators: their values convert
 * to float, and the result is rounded back to the storage type. Their MPI ops
 * are collage's own (utils.hpp); memory.hpp maps them back to a kernel here.
 */

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "lowp.hpp"

#if defined(__CUDACC__)
#define COLLAGE_HD __host__ __device__ __forceinline__
#else
#define COLLAGE_HD inline
#endif

namespace collage_gpu {

enum class DType : int { I8, U8, I16, U16, I32, U32, I64, U64, F32, F64, F16, BF16, F8E4M3, F8E5M2, Unknown };
/* First: f(a, b) = a, the non-commutative user_nc of the benchmark */
enum class OpCode : int { Sum, Prod, Max, Min, LAnd, LOr, LXor, BAnd, BOr, BXor, First, Unknown };

template <typename T> struct OpSum  { COLLAGE_HD T operator()(T a, T b) const { return (T)(a + b); } };
template <typename T> struct OpProd { COLLAGE_HD T operator()(T a, T b) const { return (T)(a * b); } };
template <typename T> struct OpMax  { COLLAGE_HD T operator()(T a, T b) const { return a > b ? a : b; } };
template <typename T> struct OpMin  { COLLAGE_HD T operator()(T a, T b) const { return a < b ? a : b; } };
template <typename T> struct OpLAnd { COLLAGE_HD T operator()(T a, T b) const { return (T)((a != T(0)) && (b != T(0))); } };
template <typename T> struct OpLOr  { COLLAGE_HD T operator()(T a, T b) const { return (T)((a != T(0)) || (b != T(0))); } };
template <typename T> struct OpLXor { COLLAGE_HD T operator()(T a, T b) const { return (T)((a != T(0)) != (b != T(0))); } };
template <typename T> struct OpBAnd { COLLAGE_HD T operator()(T a, T b) const { return (T)(a & b); } };
template <typename T> struct OpBOr  { COLLAGE_HD T operator()(T a, T b) const { return (T)(a | b); } };
template <typename T> struct OpBXor { COLLAGE_HD T operator()(T a, T b) const { return (T)(a ^ b); } };
template <typename T> struct OpFirst { COLLAGE_HD T operator()(T a, T) const { return a; } };

/* Launcher concept:  template <class T, class Op> void run(const T* in, T* inout, size_t n);
 * Returns false when there is no kernel for this (type, op). */
template <class T, class L>
bool dispatch_op(OpCode op, const void* in, void* inout, size_t n, L& launch)
{
    const T* a = static_cast<const T*>(in);
    T* b = static_cast<T*>(inout);
    switch (op) {
        case OpCode::Sum:  launch.template run<T, OpSum<T>>(a, b, n); return true;
        case OpCode::Prod: launch.template run<T, OpProd<T>>(a, b, n); return true;
        case OpCode::Max:  launch.template run<T, OpMax<T>>(a, b, n); return true;
        case OpCode::Min:  launch.template run<T, OpMin<T>>(a, b, n); return true;
        case OpCode::LAnd: launch.template run<T, OpLAnd<T>>(a, b, n); return true;
        case OpCode::LOr:  launch.template run<T, OpLOr<T>>(a, b, n); return true;
        case OpCode::LXor: launch.template run<T, OpLXor<T>>(a, b, n); return true;
        case OpCode::First: launch.template run<T, OpFirst<T>>(a, b, n); return true;
        default: break;
    }
    if constexpr (std::is_integral<T>::value) {
        switch (op) {
            case OpCode::BAnd: launch.template run<T, OpBAnd<T>>(a, b, n); return true;
            case OpCode::BOr:  launch.template run<T, OpBOr<T>>(a, b, n); return true;
            case OpCode::BXor: launch.template run<T, OpBXor<T>>(a, b, n); return true;
            default: break;
        }
    }
    return false;
}

template <class L>
bool dispatch(DType t, OpCode op, const void* in, void* inout, size_t n, L& launch)
{
    switch (t) {
        case DType::I8:  return dispatch_op<int8_t>(op, in, inout, n, launch);
        case DType::U8:  return dispatch_op<uint8_t>(op, in, inout, n, launch);
        case DType::I16: return dispatch_op<int16_t>(op, in, inout, n, launch);
        case DType::U16: return dispatch_op<uint16_t>(op, in, inout, n, launch);
        case DType::I32: return dispatch_op<int32_t>(op, in, inout, n, launch);
        case DType::U32: return dispatch_op<uint32_t>(op, in, inout, n, launch);
        case DType::I64: return dispatch_op<int64_t>(op, in, inout, n, launch);
        case DType::U64: return dispatch_op<uint64_t>(op, in, inout, n, launch);
        case DType::F32: return dispatch_op<float>(op, in, inout, n, launch);
        case DType::F64: return dispatch_op<double>(op, in, inout, n, launch);
        case DType::F16: return dispatch_op<lowp::float16>(op, in, inout, n, launch);
        case DType::BF16: return dispatch_op<lowp::bfloat16>(op, in, inout, n, launch);
        case DType::F8E4M3: return dispatch_op<lowp::float8_e4m3>(op, in, inout, n, launch);
        case DType::F8E5M2: return dispatch_op<lowp::float8_e5m2>(op, in, inout, n, launch);
        default: return false;
    }
}

/* Implemented in gpu_reduce.cu: launch the kernel on `stream` (a cudaStream_t)
 * without synchronising. Returns false when there is no kernel. */
bool device_reduce(DType t, OpCode op, const void* in, void* inout, size_t n, void* stream);

}  // namespace collage_gpu

#endif
