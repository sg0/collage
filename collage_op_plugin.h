/*
 * collage reduction-operator plugin ABI (-o plugin:PATH).
 *
 * A plugin is a shared library that exports
 *
 *     extern "C" const collage_op_plugin* collage_op_plugin_get(void);
 *
 * It defines a reduction operator once for MPI (a host MPI_User_function,
 * used by the native baseline and for host-side reductions) and, optionally,
 * as a device kernel that collage launches for every local reduction step of
 * every algorithm when the buffers are on the GPU (--gpu, --gpu-reduce kernel).
 *
 * Semantics follow MPI_Reduce_local:  inout[i] = in[i] (op) inout[i], where
 * `in` holds the contribution of the lower-ranked processes. The host and
 * device versions must compute the same function. The operator must be
 * associative; declare commute = 0 if it is not commutative, and the
 * algorithms that need commutativity report n/a.
 *
 * Datatypes. dt is the MPI datatype collage passes for the element type
 * (-t). The low-precision types travel as MPI_UINT16_T / MPI_UINT8_T, so a
 * plugin that wants to tell float16 from bfloat16 implements init(), which
 * receives collage's type name ("float16", "bfloat16", "float8_e4m3",
 * "float8_e5m2", "float", "double", "int32", ...). Element layouts of the
 * low-precision types are in lowp.hpp.
 *
 * Device launcher contract:
 *   - queue the work on `stream` (a cudaStream_t) and return; do NOT
 *     synchronise, and do not call MPI. With the stream-triggered and the
 *     one-sided transports the reduction is part of a stream-ordered
 *     sequence: an implicit sync would stall it, a sync on another stream
 *     would break it.
 *   - return 0 when the work was queued; any other value means "not for
 *     this datatype/count", and collage falls back to the host function
 *     (staged through pinned host memory).
 *   - in and inout never overlap; count may be large (size_t).
 *
 * Build, e.g.:
 *   nvcc -O3 -arch=sm_80 -std=c++17 -shared -Xcompiler -fPIC \
 *        -I$MPI_HOME/include -I/path/to/collage -o libmyop.so myop.cu
 * (examples/op_plugin_absmax.cu is a complete example.)
 */
#ifndef COLLAGE_OP_PLUGIN_H
#define COLLAGE_OP_PLUGIN_H

#include <mpi.h>
#include <stddef.h>

#define COLLAGE_OP_PLUGIN_ABI 1

#ifdef __cplusplus
extern "C" {
#endif

typedef struct collage_op_plugin {
    int abi;                     /* COLLAGE_OP_PLUGIN_ABI */
    const char* name;            /* short name, shown as plugin:<name> */
    int commute;                 /* 1: commutative, 0: not */
    MPI_User_function* host_fn;  /* required */
    /* optional (NULL: host only): queue inout = in op inout on `stream` */
    int (*device_fn)(const void* in, void* inout, size_t count, MPI_Datatype dt, void* stream);
    /* optional: called once with collage's element type name and size;
     * return nonzero to reject the type */
    int (*init)(const char* type_name, size_t elem_size);
} collage_op_plugin;

typedef const collage_op_plugin* (*collage_op_plugin_get_fn)(void);

#ifdef __cplusplus
}
#endif

#endif /* COLLAGE_OP_PLUGIN_H */
