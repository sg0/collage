#pragma once
#ifndef COLLAGE_UTILS_HPP
#define COLLAGE_UTILS_HPP

/*
 * Shared infrastructure for the standalone collective benchmark.
 *
 *  - TypeMap<T>()              : C++ type -> MPI datatype
 *  - op_from_string()/...      : user-selectable reduction operations
 *  - CollBase                  : communicator handling + point-to-point wrappers
 *                                (size_t counts, MPI-4 large-count aware)
 *  - Tree / build_*            : port of ompi/mca/coll/base/coll_base_topo.c
 *  - misc helpers              : next_poweroftwo, rounddown, mirror_perm, segcount
 *
 * Everything here only uses MPI point-to-point calls, MPI_Reduce_local (a purely
 * local operation) and derived-datatype helpers. No MPI collective is used by
 * any algorithm.
 */

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "memory.hpp"
#include <type_traits>

#if defined(MPI_VERSION) && (MPI_VERSION >= 4) && !defined(COLLAGE_NO_LARGE_COUNT)
#define COLLAGE_LARGE_COUNT 1
#else
#define COLLAGE_LARGE_COUNT 0
#endif

/* ------------------------------------------------------------------------- */
/* error handling                                                            */
/* ------------------------------------------------------------------------- */
#define MPI_CHECK(call)                                                        \
    do {                                                                       \
        int _rc = (call);                                                      \
        if (_rc != MPI_SUCCESS) {                                              \
            char _msg[MPI_MAX_ERROR_STRING];                                   \
            int _len = 0;                                                      \
            MPI_Error_string(_rc, _msg, &_len);                                \
            std::fprintf(stderr, "%s:%d: MPI error %d: %s\n", __FILE__,        \
                         __LINE__, _rc, _msg);                                 \
            MPI_Abort(MPI_COMM_WORLD, _rc);                                    \
        }                                                                      \
    } while (0)

/* thrown when an algorithm cannot run for the given configuration
 * (e.g. a two-process-only algorithm on 4 ranks). The benchmark reports n/a. */
struct NotApplicable : public std::runtime_error {
    explicit NotApplicable(const std::string& w) : std::runtime_error(w) {}
};

#include "transport.hpp"

/* request handles held by the algorithms (see transport.hpp) */
using Request = xport::Request;
#define REQUEST_NULL (::xport::Request{})


/* ------------------------------------------------------------------------- */
/* tags (one communicator is dup'ed per collective object, so tags only need  */
/* to be unique within an algorithm)                                          */
/* ------------------------------------------------------------------------- */
#define TAG_BASE                   (1000)
#define TAG_ALLTOALLV              (101)
#define TAG_ALLTOALL               (102)
#define TAG_ALLREDUCE              (103)
#define TAG_ALLGATHERV             (104)
#define TAG_ALLGATHER              (105)
#define TAG_BARRIER                (106)
#define TAG_BCAST                  (107)
#define TAG_REDUCE                 (108)
#define TAG_SCATTER                (109)
#define TAG_GATHER                 (110)
#define TAG_SCATTERV               (111)
#define TAG_GATHERV                (112)
#define TAG_REDUCE_SCATTER         (201)
#define TAG_REDUCE_SCATTER_BLOCK   (202)

/* ------------------------------------------------------------------------- */
/* type map                                                                  */
/* ------------------------------------------------------------------------- */
template <typename T> MPI_Datatype TypeMap();
template <> inline MPI_Datatype TypeMap<int8_t>()   { return MPI_INT8_T; }
template <> inline MPI_Datatype TypeMap<uint8_t>()  { return MPI_UINT8_T; }
template <> inline MPI_Datatype TypeMap<int16_t>()  { return MPI_INT16_T; }
template <> inline MPI_Datatype TypeMap<uint16_t>() { return MPI_UINT16_T; }
template <> inline MPI_Datatype TypeMap<int32_t>()  { return MPI_INT32_T; }
template <> inline MPI_Datatype TypeMap<uint32_t>() { return MPI_UINT32_T; }
template <> inline MPI_Datatype TypeMap<int64_t>()  { return MPI_INT64_T; }
template <> inline MPI_Datatype TypeMap<uint64_t>() { return MPI_UINT64_T; }
template <> inline MPI_Datatype TypeMap<float>()    { return MPI_FLOAT; }
template <> inline MPI_Datatype TypeMap<double>()   { return MPI_DOUBLE; }

/* ------------------------------------------------------------------------- */
/* reduction operations                                                      */
/* ------------------------------------------------------------------------- */
/* Two user-defined operations are provided in addition to the predefined ones:
 *   user    : element-wise sum implemented as a user op (commutative), measures
 *             the cost of a non-builtin reduction kernel.
 *   user_nc : NON-commutative but associative op  f(a,b) = a  (keeps the value
 *             of the lowest rank). Exercises the non-commutative code paths
 *             of the algorithms (in-order trees, fallbacks, ...). */
namespace detail {
template <typename T>
inline void user_sum_fn(void* in, void* inout, int* len, MPI_Datatype*)
{
    const T* a = static_cast<const T*>(in);
    T* b = static_cast<T*>(inout);
    for (int i = 0; i < *len; ++i) b[i] = (T)(a[i] + b[i]);
}
/* MPI: inout[i] = in[i] op inout[i]; "in" always comes from the lower ranks */
inline void user_first_fn(void* in, void* inout, int* len, MPI_Datatype* dt)
{
    int tsize = 0;
    MPI_Type_size(*dt, &tsize);
    std::memcpy(inout, in, (size_t)(*len) * (size_t)tsize);
}
}  // namespace detail

enum class OpKind { Arithmetic, Logical, Bitwise, MinMax, User };

struct OpInfo {
    std::string name;
    MPI_Op op = MPI_OP_NULL;
    OpKind kind = OpKind::Arithmetic;
    bool user = false;     /* created with MPI_Op_create -> must be freed */
    bool commute = true;
};

inline std::vector<std::string> op_names()
{
    return {"sum", "prod", "max", "min", "land", "lor", "lxor",
            "band", "bor", "bxor", "user", "user_nc"};
}

/* Create the MPI_Op for a given name and element type T. */
template <typename T>
inline OpInfo make_op(const std::string& name)
{
    OpInfo o;
    o.name = name;
    if      (name == "sum")  { o.op = MPI_SUM;  o.kind = OpKind::Arithmetic; }
    else if (name == "prod") { o.op = MPI_PROD; o.kind = OpKind::Arithmetic; }
    else if (name == "max")  { o.op = MPI_MAX;  o.kind = OpKind::MinMax; }
    else if (name == "min")  { o.op = MPI_MIN;  o.kind = OpKind::MinMax; }
    else if (name == "land") { o.op = MPI_LAND; o.kind = OpKind::Logical; }
    else if (name == "lor")  { o.op = MPI_LOR;  o.kind = OpKind::Logical; }
    else if (name == "lxor") { o.op = MPI_LXOR; o.kind = OpKind::Logical; }
    else if (name == "band") { o.op = MPI_BAND; o.kind = OpKind::Bitwise; }
    else if (name == "bor")  { o.op = MPI_BOR;  o.kind = OpKind::Bitwise; }
    else if (name == "bxor") { o.op = MPI_BXOR; o.kind = OpKind::Bitwise; }
    else if (name == "user") {
        MPI_CHECK(MPI_Op_create(&detail::user_sum_fn<T>, 1, &o.op));
        o.kind = OpKind::User; o.user = true;
    } else if (name == "user_nc") {
        MPI_CHECK(MPI_Op_create(&detail::user_first_fn, 0, &o.op));
        o.kind = OpKind::User; o.user = true;
    } else {
        throw std::invalid_argument("unknown reduction operation '" + name + "'");
    }
    if (o.kind == OpKind::Bitwise && std::is_floating_point<T>::value)
        throw std::invalid_argument("bitwise operation '" + name +
                                    "' is not defined for floating point types");
    int c = 1;
    MPI_CHECK(MPI_Op_commutative(o.op, &c));
    o.commute = (c != 0);
    return o;
}

inline void free_op(OpInfo& o)
{
    if (o.user && o.op != MPI_OP_NULL) MPI_Op_free(&o.op);
    o.op = MPI_OP_NULL;
}

inline bool op_is_commute(MPI_Op op)
{
    int c = 1;
    MPI_Op_commutative(op, &c);
    return c != 0;
}

/* ------------------------------------------------------------------------- */
/* misc helpers (opal/util/bit_ops.h, coll_base_util.c)                       */
/* ------------------------------------------------------------------------- */
/* smallest power of two strictly greater than value (opal_next_poweroftwo) */
inline int next_poweroftwo(int value)
{
    int power2;
    if (0 == value) return 1;
    for (power2 = 1; value > 0; value >>= 1, power2 <<= 1) {}
    return power2;
}
/* smallest power of two >= value (opal_next_poweroftwo_inclusive) */
inline int next_poweroftwo_inclusive(int value)
{
    if (value <= 1) return 1;
    return next_poweroftwo(value - 1);
}
/* index of the highest set bit, -1 for 0 (opal_hibit with start=31) */
inline int hibit(int value, int start = 31)
{
    unsigned int mask;
    --start;
    mask = 1u << start;
    for (; start >= 0; --start, mask >>= 1)
        if ((unsigned)value & mask) break;
    return start;
}
inline bool is_poweroftwo(int size) { return (size != 0) && !(size & (size - 1)); }
inline int rounddown(int num, int factor) { num /= factor; return num * factor; }
inline unsigned int mirror_perm(unsigned int x, int nbits)
{
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    x = ((x >> 16) | (x << 16));
    return x >> (sizeof(x) * CHAR_BIT - nbits);
}

/* COLL_BASE_COMPUTED_SEGCOUNT: number of elements that fit in segsize bytes;
 * segcount must be initialised to the full count by the caller. */
inline void computed_segcount(size_t segsize, size_t typelng, size_t& segcount)
{
    if ((segsize >= typelng) && (segsize < (typelng * segcount))) {
        size_t residual;
        segcount = segsize / typelng;
        residual = segsize - segcount * typelng;
        if (residual > (typelng >> 1)) segcount++;
    }
}

/* COLL_BASE_COMPUTE_BLOCKCOUNT */
inline void compute_blockcount(size_t count, size_t num_blocks, size_t& split_index,
                               size_t& early_block_count, size_t& late_block_count)
{
    early_block_count = late_block_count = count / num_blocks;
    split_index = count % num_blocks;
    if (0 != split_index) early_block_count = early_block_count + 1;
}

/* ------------------------------------------------------------------------- */
/* topology (port of coll_base_topo.c)                                        */
/* ------------------------------------------------------------------------- */
#define MAXTREEFANOUT 32

struct Tree {
    int tree_root = MPI_UNDEFINED;
    int tree_fanout = 0;
    int tree_bmtree = 0;
    int tree_prev = -1;
    int tree_nextsize = 0;
    std::vector<int> tree_next;
};

namespace topo {

inline int pown(int fanout, int num)
{
    int j, p = 1;
    if (num < 0) return 0;
    if (1 == num) return fanout;
    if (2 == fanout) return p << num;
    for (j = 0; j < num; j++) p *= fanout;
    return p;
}

inline int calculate_level(int fanout, int rank)
{
    int level, num;
    if (rank < 0) return -1;
    for (level = 0, num = 0; num <= rank; level++) num += pown(fanout, level);
    return level - 1;
}

inline int calculate_num_nodes_up_to_level(int fanout, int level)
{
    if (1 == fanout) return level;
    return ((pown(fanout, level) - 1) / (fanout - 1));
}

/* generic k-ary tree, e.g. fanout = 2, size = 7:  0 -> {1,2}; 1 -> {3,5}; 2 -> {4,6} */
inline Tree build_tree(int fanout, int rank, int size, int root)
{
    if (fanout < 1 || fanout > MAXTREEFANOUT)
        throw std::invalid_argument("build_tree: invalid fanout " + std::to_string(fanout));
    Tree tree;
    int schild, sparent, shiftedrank, i, level, delta, slimit;
    tree.tree_next.assign(MAXTREEFANOUT, -1);
    tree.tree_fanout = fanout;
    tree.tree_bmtree = 0;
    tree.tree_root = root;
    tree.tree_prev = -1;
    tree.tree_nextsize = 0;
    if (size < 2) return tree;

    shiftedrank = rank - root;
    if (shiftedrank < 0) shiftedrank += size;

    level = calculate_level(fanout, shiftedrank);
    delta = pown(fanout, level);

    for (i = 0; i < fanout; i++) {
        schild = shiftedrank + delta * (i + 1);
        if (schild < size) {
            tree.tree_next[i] = (schild + root) % size;
            tree.tree_nextsize = tree.tree_nextsize + 1;
        } else {
            break;
        }
    }

    slimit = calculate_num_nodes_up_to_level(fanout, level);
    sparent = shiftedrank;
    if (sparent < fanout) {
        sparent = 0;
    } else {
        while (sparent >= slimit) sparent -= delta / fanout;
    }
    tree.tree_prev = (sparent + root) % size;
    return tree;
}

/* in-order binary tree rooted at size-1 (for non-commutative reductions) */
inline Tree build_in_order_bintree(int rank, int size)
{
    Tree tree;
    int myrank, rightsize, delta, parent, lchild, rchild;
    tree.tree_next.assign(MAXTREEFANOUT, -1);
    tree.tree_fanout = 2;
    tree.tree_bmtree = 0;
    tree.tree_root = size - 1;
    tree.tree_prev = -1;
    tree.tree_nextsize = 0;

    myrank = rank;
    parent = size - 1;
    delta = 0;

    while (1) {
        rightsize = size >> 1;
        lchild = -1;
        rchild = -1;
        if (size - 1 > 0) {
            lchild = parent - 1;
            if (lchild > 0) rchild = rightsize - 1;
        }
        if (myrank == parent) {
            if (lchild >= 0) tree.tree_next[0] = lchild + delta;
            if (rchild >= 0) tree.tree_next[1] = rchild + delta;
            break;
        }
        if (myrank > rchild) {
            if (myrank == lchild) tree.tree_prev = parent + delta;
            size = size - rightsize - 1;
            delta = delta + rightsize;
            myrank = myrank - rightsize;
            parent = size - 1;
        } else {
            if (myrank == rchild) tree.tree_prev = parent + delta;
            size = rightsize;
            parent = rchild;
        }
    }
    if (tree.tree_next[0] >= 0) tree.tree_nextsize = 1;
    if (tree.tree_next[1] >= 0) tree.tree_nextsize += 1;
    return tree;
}

/* binomial tree:  size = 8 -> 0:{1,2,4}, 2:{3}, 4:{5,6}, 6:{7} */
inline Tree build_bmtree(int rank, int size, int root)
{
    Tree bmtree;
    int childs = 0, mask = 1, index, remote;
    index = rank - root;
    bmtree.tree_bmtree = 1;
    bmtree.tree_next.assign(MAXTREEFANOUT, -1);
    if (index < 0) index += size;
    mask = next_poweroftwo(index);
    if (root == rank) {
        bmtree.tree_prev = root;
    } else {
        remote = (index ^ (mask >> 1)) + root;
        if (remote >= size) remote -= size;
        bmtree.tree_prev = remote;
    }
    while (mask < size) {
        remote = (index ^ mask);
        if (remote >= size) break;
        remote += root;
        if (remote >= size) remote -= size;
        if (childs == MAXTREEFANOUT) throw std::runtime_error("build_bmtree: fanout overflow");
        bmtree.tree_next[childs] = remote;
        mask <<= 1;
        childs++;
    }
    bmtree.tree_nextsize = childs;
    bmtree.tree_root = root;
    return bmtree;
}

/* in-order binomial tree (gather/scatter):  size = 8 -> 0:{1,2,4}, 2:{3}, 4:{5,6}, 6:{7} */
inline Tree build_in_order_bmtree(int rank, int size, int root)
{
    Tree bmtree;
    int childs = 0, vrank, mask = 1, remote;
    vrank = (rank - root + size) % size;
    bmtree.tree_bmtree = 1;
    bmtree.tree_next.assign(MAXTREEFANOUT, -1);
    if (root == rank) bmtree.tree_prev = root;
    while (mask < size) {
        remote = vrank ^ mask;
        if (remote < vrank) {
            bmtree.tree_prev = (remote + root) % size;
            break;
        } else if (remote < size) {
            bmtree.tree_next[childs] = (remote + root) % size;
            childs++;
            if (childs == MAXTREEFANOUT) throw std::runtime_error("build_in_order_bmtree: fanout overflow");
        }
        mask <<= 1;
    }
    bmtree.tree_nextsize = childs;
    bmtree.tree_root = root;
    return bmtree;
}

/* k-nomial tree */
inline Tree build_kmtree(int rank, int comm_size, int root, int radix)
{
    if (radix < 2) throw std::invalid_argument("build_kmtree: radix must be >= 2");
    int log_radix = 0;
    for (int i = 1; i < comm_size; i *= radix) log_radix++;
    int nchilds_max = (radix - 1) * log_radix;

    int vrank = (rank - root + comm_size) % comm_size;
    Tree kmtree;
    kmtree.tree_next.assign(std::max(nchilds_max, 1), -1);
    kmtree.tree_bmtree = 0;
    kmtree.tree_root = root;
    kmtree.tree_prev = MPI_PROC_NULL;
    kmtree.tree_nextsize = 0;

    int mask = 0x1;
    while (mask < comm_size) {
        if (vrank % (radix * mask)) {
            kmtree.tree_prev = vrank / (radix * mask) * (radix * mask);
            kmtree.tree_prev = (kmtree.tree_prev + root) % comm_size;
            break;
        }
        mask *= radix;
    }
    mask /= radix;
    int nchilds = 0;
    while (mask > 0) {
        for (int r = 1; r < radix; r++) {
            int child = vrank + mask * r;
            if (child < comm_size) {
                child = (child + root) % comm_size;
                kmtree.tree_next[nchilds] = child;
                nchilds++;
            }
        }
        mask /= radix;
    }
    kmtree.tree_nextsize = nchilds;
    return kmtree;
}

/* "fanout" chains hanging off root (fanout == 1 is a pipeline) */
inline Tree build_chain(int fanout, int rank, int size, int root)
{
    int i, maxchainlen, mark, head, len, srank;
    Tree chain;
    if (fanout < 1) fanout = 1;
    if (fanout > MAXTREEFANOUT) fanout = MAXTREEFANOUT;
    chain.tree_next.assign(MAXTREEFANOUT, -1);
    chain.tree_root = root;
    chain.tree_fanout = fanout;
    if ((size - 1) < fanout) {
        chain.tree_nextsize = size - 1;
        fanout = size - 1;
    } else {
        chain.tree_nextsize = fanout;
    }
    srank = rank - root;
    if (srank < 0) srank += size;

    if (fanout == 1) {
        if (srank == 0) chain.tree_prev = -1;
        else chain.tree_prev = (srank - 1 + root) % size;
        if ((srank + 1) >= size) {
            chain.tree_next[0] = -1;
            chain.tree_nextsize = 0;
        } else {
            chain.tree_next[0] = (srank + 1 + root) % size;
            chain.tree_nextsize = 1;
        }
        return chain;
    }
    if (size == 1) {
        chain.tree_next[0] = -1;
        chain.tree_nextsize = 0;
        chain.tree_prev = -1;
        return chain;
    }
    maxchainlen = (size - 1) / fanout;
    if ((size - 1) % fanout != 0) {
        maxchainlen++;
        mark = (size - 1) % fanout;
    } else {
        mark = fanout + 1;
    }
    if (srank != 0) {
        int column;
        if (srank - 1 < (mark * maxchainlen)) {
            column = (srank - 1) / maxchainlen;
            head = 1 + column * maxchainlen;
            len = maxchainlen;
        } else {
            column = mark + (srank - 1 - mark * maxchainlen) / (maxchainlen - 1);
            head = mark * maxchainlen + 1 + (column - mark) * (maxchainlen - 1);
            len = maxchainlen - 1;
        }
        if (srank == head) chain.tree_prev = 0;
        else chain.tree_prev = srank - 1;
        if (srank == (head + len - 1)) {
            chain.tree_next[0] = -1;
            chain.tree_nextsize = 0;
        } else {
            if ((srank + 1) < size) {
                chain.tree_next[0] = srank + 1;
                chain.tree_nextsize = 1;
            } else {
                chain.tree_next[0] = -1;
                chain.tree_nextsize = 0;
            }
        }
        chain.tree_prev = (chain.tree_prev + root) % size;
        if (chain.tree_next[0] != -1) chain.tree_next[0] = (chain.tree_next[0] + root) % size;
    } else {
        chain.tree_prev = -1;
        chain.tree_next[0] = (root + 1) % size;
        for (i = 1; i < fanout; i++) {
            chain.tree_next[i] = chain.tree_next[i - 1] + maxchainlen;
            if (i > mark) chain.tree_next[i]--;
            chain.tree_next[i] %= size;
        }
        chain.tree_nextsize = fanout;
    }
    return chain;
}

}  // namespace topo

/* ------------------------------------------------------------------------- */
/* CollBase: communicator + point-to-point wrappers                           */
/* ------------------------------------------------------------------------- */
class CollBase {
public:
    explicit CollBase(MPI_Comm comm)
    {
        MPI_CHECK(MPI_Comm_dup(comm, &comm_));
        MPI_CHECK(MPI_Comm_size(comm_, &size_));
        MPI_CHECK(MPI_Comm_rank(comm_, &rank_));
        ch_.open(comm_); /* collective: node locality, stream communicator */
    }
    virtual ~CollBase()
    {
        int finalized = 0;
        MPI_Finalized(&finalized);
        ch_.close();
        if (!finalized && comm_ != MPI_COMM_NULL) MPI_Comm_free(&comm_);
    }
    CollBase(const CollBase&) = delete;
    CollBase& operator=(const CollBase&) = delete;

    MPI_Comm comm() const { return comm_; }
    int rank() const { return rank_; }
    int size() const { return size_; }

protected:
    using Request = xport::Request;

    MPI_Comm comm_ = MPI_COMM_NULL;
    int rank_ = 0, size_ = 1;
    xport::Channel ch_;

#if COLLAGE_LARGE_COUNT
    static MPI_Count cnt(size_t c) { return (MPI_Count)c; }
#else
    static int cnt(size_t c)
    {
        if (c > (size_t)INT_MAX)
            throw std::overflow_error("message count exceeds INT_MAX and the MPI library "
                                      "has no MPI-4 large-count support");
        return (int)c;
    }
#endif

        /* --------------------------------------------------------------------
     * Point-to-point wrappers: the MCA_PML_CALL(...) equivalents.
     * Every message goes through the transport layer (transport.hpp): plain
     * (CUDA-aware) MPI by default; with a stream-triggered transport,
     * messages to peers on other nodes are queued on the CUDA stream and the
     * others stay on the plain path, after the host waits for the stream.
     * -------------------------------------------------------------------- */
    void send(const void* buf, size_t count, MPI_Datatype dt, int dest, int tag) const
    {
        if (ch_.enq(dest)) {
            Request r;
            xport::enq_isend(ch_, buf, count, dt, dest, tag, &r);
            xport::enq_wait(&r, MPI_STATUS_IGNORE);
            return;
        }
        xport::before_plain();
#if COLLAGE_LARGE_COUNT
        MPI_CHECK(MPI_Send_c(buf, cnt(count), dt, dest, tag, comm_));
#else
        MPI_CHECK(MPI_Send(buf, cnt(count), dt, dest, tag, comm_));
#endif
    }
    /* MCA_PML_BASE_SEND_SYNCHRONOUS (a standard send on the stream path:
     * the enqueue APIs have no synchronous mode) */
    void ssend(const void* buf, size_t count, MPI_Datatype dt, int dest, int tag) const
    {
        if (ch_.enq(dest)) {
            send(buf, count, dt, dest, tag);
            return;
        }
        xport::before_plain();
#if COLLAGE_LARGE_COUNT
        MPI_CHECK(MPI_Ssend_c(buf, cnt(count), dt, dest, tag, comm_));
#else
        MPI_CHECK(MPI_Ssend(buf, cnt(count), dt, dest, tag, comm_));
#endif
    }
    void recv(void* buf, size_t count, MPI_Datatype dt, int src, int tag,
              MPI_Status* st = MPI_STATUS_IGNORE) const
    {
        if (ch_.enq(src)) {
            Request r;
            xport::enq_irecv(ch_, buf, count, dt, src, tag, &r);
            xport::enq_wait(&r, st);
            return;
        }
        xport::before_plain();
        xport::forget_status(st);
#if COLLAGE_LARGE_COUNT
        MPI_CHECK(MPI_Recv_c(buf, cnt(count), dt, src, tag, comm_, st));
#else
        MPI_CHECK(MPI_Recv(buf, cnt(count), dt, src, tag, comm_, st));
#endif
    }
    void isend(const void* buf, size_t count, MPI_Datatype dt, int dest, int tag,
               Request* req) const
    {
        if (ch_.enq(dest)) {
            xport::enq_isend(ch_, buf, count, dt, dest, tag, req);
            return;
        }
        xport::before_plain();
        *req = Request{};
        req->k = Request::Host;
#if COLLAGE_LARGE_COUNT
        MPI_CHECK(MPI_Isend_c(buf, cnt(count), dt, dest, tag, comm_, &req->h));
#else
        MPI_CHECK(MPI_Isend(buf, cnt(count), dt, dest, tag, comm_, &req->h));
#endif
    }
    /* synchronous-mode isend (reduce flow control); standard on the stream path */
    void issend(const void* buf, size_t count, MPI_Datatype dt, int dest, int tag,
                Request* req) const
    {
        if (ch_.enq(dest)) {
            xport::enq_isend(ch_, buf, count, dt, dest, tag, req);
            return;
        }
        xport::before_plain();
        *req = Request{};
        req->k = Request::Host;
#if COLLAGE_LARGE_COUNT
        MPI_CHECK(MPI_Issend_c(buf, cnt(count), dt, dest, tag, comm_, &req->h));
#else
        MPI_CHECK(MPI_Issend(buf, cnt(count), dt, dest, tag, comm_, &req->h));
#endif
    }
    void irecv(void* buf, size_t count, MPI_Datatype dt, int src, int tag,
               Request* req) const
    {
        if (ch_.enq(src)) {
            xport::enq_irecv(ch_, buf, count, dt, src, tag, req);
            return;
        }
        xport::before_plain();
        *req = Request{};
        req->k = Request::Host;
#if COLLAGE_LARGE_COUNT
        MPI_CHECK(MPI_Irecv_c(buf, cnt(count), dt, src, tag, comm_, &req->h));
#else
        MPI_CHECK(MPI_Irecv(buf, cnt(count), dt, src, tag, comm_, &req->h));
#endif
    }
    /* persistent requests: real ones on the plain transport; in stream mode a
     * descriptor that startall() turns into isend/irecv on the right path */
    void send_init(const void* buf, size_t count, MPI_Datatype dt, int dest, int tag,
                   Request* req) const
    {
        *req = Request{};
        if (xport::streamed()) {
            req->k = Request::Pers;
            req->p = new xport::Persist{true, buf, count, dt, dest, tag, Request{}};
            return;
        }
        req->k = Request::Host;
#if COLLAGE_LARGE_COUNT
        MPI_CHECK(MPI_Send_init_c(buf, cnt(count), dt, dest, tag, comm_, &req->h));
#else
        MPI_CHECK(MPI_Send_init(buf, cnt(count), dt, dest, tag, comm_, &req->h));
#endif
    }
    void recv_init(void* buf, size_t count, MPI_Datatype dt, int src, int tag,
                   Request* req) const
    {
        *req = Request{};
        if (xport::streamed()) {
            req->k = Request::Pers;
            req->p = new xport::Persist{false, buf, count, dt, src, tag, Request{}};
            return;
        }
        req->k = Request::Host;
#if COLLAGE_LARGE_COUNT
        MPI_CHECK(MPI_Recv_init_c(buf, cnt(count), dt, src, tag, comm_, &req->h));
#else
        MPI_CHECK(MPI_Recv_init(buf, cnt(count), dt, src, tag, comm_, &req->h));
#endif
    }
    void startall(int n, Request* reqs) const
    {
        std::vector<MPI_Request> h;
        for (int i = 0; i < n; ++i) {
            Request& r = reqs[i];
            if (r.k == Request::Pers) {
                xport::Persist& p = *r.p;
                if (p.is_send) isend(p.buf, p.count, p.dt, p.peer, p.tag, &p.active);
                else irecv(const_cast<void*>(p.buf), p.count, p.dt, p.peer, p.tag, &p.active);
            } else if (r.k == Request::Host) {
                h.push_back(r.h);
            }
        }
        if (!h.empty()) {
            xport::before_plain();
            MPI_CHECK(MPI_Startall((int)h.size(), h.data()));
        }
    }
    static void request_free(Request* r)
    {
        if (r->k == Request::Pers) delete r->p;
        else if (r->k == Request::Host && r->h != MPI_REQUEST_NULL) MPI_Request_free(&r->h);
        *r = Request{};
    }
    static void wait(Request* req, MPI_Status* st = MPI_STATUS_IGNORE)
    {
        switch (req->k) {
            case Request::None:
                return;
            case Request::Done:
                *req = Request{};
                return;
            case Request::Host:
                xport::forget_status(st);
                MPI_CHECK(MPI_Wait(&req->h, st));
                if (req->h == MPI_REQUEST_NULL) *req = Request{}; /* persistent ones stay */
                return;
            case Request::Enq:
                xport::enq_wait(req, st);
                return;
            case Request::Pers:
                wait(&req->p->active, st);
                return;
        }
    }
    static void waitall(int n, Request* reqs)
    {
        if (n <= 0) return;
        std::vector<Request*> enq;
        std::vector<int> host;
        std::vector<MPI_Request> h;
        for (int i = 0; i < n; ++i) {
            Request* r = &reqs[i];
            if (r->k == Request::Pers) r = &r->p->active;
            if (r->k == Request::Done) *r = Request{};
            else if (r->k == Request::Enq) enq.push_back(r);
            else if (r->k == Request::Host) {
                host.push_back(i);
                h.push_back(r->h);
            }
        }
        /* the stream waits first (does not block the host), then the host waits */
        xport::enq_waitall((int)enq.size(), enq.data());
        if (!h.empty()) {
            MPI_CHECK(MPI_Waitall((int)h.size(), h.data(), MPI_STATUSES_IGNORE));
            for (size_t j = 0; j < host.size(); ++j) {
                Request* r = &reqs[host[j]];
                if (r->k == Request::Pers) r = &r->p->active;
                r->h = h[j];
                if (r->h == MPI_REQUEST_NULL) *r = Request{};
            }
        }
    }
    /* MPI_Waitany; in stream mode: the first active request (a legal outcome) */
    static void waitany(int n, Request* reqs, int* idx)
    {
        *idx = MPI_UNDEFINED;
        if (!xport::streamed()) {
            std::vector<MPI_Request> h((size_t)n, MPI_REQUEST_NULL);
            for (int i = 0; i < n; ++i)
                if (reqs[i].k == Request::Host) h[(size_t)i] = reqs[i].h;
            MPI_CHECK(MPI_Waitany(n, h.data(), idx, MPI_STATUS_IGNORE));
            if (*idx != MPI_UNDEFINED) {
                reqs[*idx].h = h[(size_t)*idx];
                if (reqs[*idx].h == MPI_REQUEST_NULL) reqs[*idx] = Request{};
            }
            return;
        }
        /* Stream mode: the host cannot observe which queued operation
         * completes first. Wait for every active request (the stream waits
         * for the queued ones), then report the completed ones one per call;
         * waiting for one particular request instead could deadlock (alltoall
         * linear_sync: every rank would wait for a receive whose sender is
         * itself waiting for a receive). */
        for (int pass = 0; pass < 2; ++pass) {
            for (int i = 0; i < n; ++i)
                if (reqs[i].k == Request::Done) {
                    reqs[i] = Request{};
                    *idx = i;
                    return;
                }
            if (pass) return;
            std::vector<int> act;
            for (int i = 0; i < n; ++i) {
                Request* r = &reqs[i];
                if (r->k == Request::Pers) r = &r->p->active;
                if (r->k != Request::None) act.push_back(i);
            }
            if (act.empty()) return;
            std::vector<Request> tmp;
            for (int i : act) tmp.push_back(reqs[i]);
            waitall((int)tmp.size(), tmp.data());
            for (int i : act) {
                if (reqs[i].k == Request::Pers) reqs[i].p->active = Request{};
                else reqs[i] = Request{};
                reqs[i].k = reqs[i].k == Request::Pers ? Request::Pers : Request::Done;
            }
        }
    }
    static size_t get_count(MPI_Status* st, MPI_Datatype dt)
    {
        xport::fetch_status(st); /* stream path: wait for the stream, copy the status */
#if COLLAGE_LARGE_COUNT
        MPI_Count c = 0;
        MPI_CHECK(MPI_Get_count_c(st, dt, &c));
#else
        int c = 0;
        MPI_CHECK(MPI_Get_count(st, dt, &c));
#endif
        return (size_t)c;
    }

    /* ompi_coll_base_sendrecv_actual: irecv, blocking send, wait */
    void sendrecv_actual(const void* sbuf, size_t scount, MPI_Datatype sdt, int dest, int stag,
                         void* rbuf, size_t rcount, MPI_Datatype rdt, int source, int rtag,
                         MPI_Status* st = MPI_STATUS_IGNORE) const
    {
        Request req;
        irecv(rbuf, rcount, rdt, source, rtag, &req);
        send(sbuf, scount, sdt, dest, stag);
        wait(&req, st);
    }

    /* ompi_coll_base_sendrecv: like above, with a local-copy shortcut for self.
     * The self path is only valid for contiguous predefined types. */
    void sendrecv(const void* sbuf, size_t scount, MPI_Datatype sdt, int dest, int stag,
                  void* rbuf, size_t rcount, MPI_Datatype rdt, int source, int rtag,
                  MPI_Status* st = MPI_STATUS_IGNORE) const
    {
        if (dest == source && source == rank_) {
            int tsize = 0;
            MPI_Type_size(sdt, &tsize);
            size_t n = std::min(scount, rcount) * (size_t)tsize;
            if (n && sbuf != rbuf) mem::copy(rbuf, sbuf, n);  /* host or device */
            return;
        }
        sendrecv_actual(sbuf, scount, sdt, dest, stag, rbuf, rcount, rdt, source, rtag, st);
    }

    /* ompi_coll_base_sendrecv_zero (barrier helper) */
    void sendrecv_zero(int dest, int stag, int source, int rtag) const
    {
        Request req;
        irecv(nullptr, 0, MPI_BYTE, source, rtag, &req);
        send(nullptr, 0, MPI_BYTE, dest, stag);
        wait(&req);
    }

    /* ompi_op_reduce: target = source (op) target, in the current memory
     * space (MPI_Reduce_local on the host; kernel / MPI / staged on a GPU) */
    static void reduce_local(const void* source, void* target, size_t count, MPI_Datatype dt,
                             MPI_Op op)
    {
        mem::reduce(source, target, count, dt, op);
    }

    /* ompi_3buff_op_reduce: target = source1 (op) source2 */
    static void reduce_3buf(const void* source1, const void* source2, void* target,
                            size_t count, MPI_Datatype dt, MPI_Op op)
    {
        if (0 == count) return;
        int tsize = 0;
        MPI_Type_size(dt, &tsize);
        if (target != source2) mem::copy(target, source2, count * (size_t)tsize);
        reduce_local(source1, target, count, dt, op);
    }
};

/* local typed copy (ompi_datatype_copy_content_same_ddt / ompi_datatype_sndrcv),
 * memmove semantics, in the current memory space */
template <typename T>
inline void copy_elems(T* dst, const T* src, size_t count)
{
    if (count && dst != src) mem::copy(dst, src, count * sizeof(T));
}

#endif /* COLLAGE_UTILS_HPP */
