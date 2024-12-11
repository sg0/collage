#pragma once
#ifndef UTILS_HPP
#define UTILS_HPP

#include <bitset>
typedef std::bitset<8> byte;

template<typename T> MPI_Datatype TypeMap();
template<> inline MPI_Datatype TypeMap<byte>() { return MPI_UNSIGNED_CHAR; }
template<> inline MPI_Datatype TypeMap<char>() { return MPI_CHAR; }
template<> inline MPI_Datatype TypeMap<int>() { return MPI_INT; }
template<> inline MPI_Datatype TypeMap<unsigned>() { return MPI_UNSIGNED; }
template<> inline MPI_Datatype TypeMap<long int>() { return MPI_LONG_INT; }
template<> inline MPI_Datatype TypeMap<long unsigned>() { return MPI_UNSIGNED_LONG; }
template<> inline MPI_Datatype TypeMap<long long int>() { return MPI_LONG_LONG_INT; }
template<> inline MPI_Datatype TypeMap<unsigned long long>() { return MPI_UNSIGNED_LONG_LONG; }
template<> inline MPI_Datatype TypeMap<float>() { return MPI_FLOAT; }
template<> inline MPI_Datatype TypeMap<double>() { return MPI_DOUBLE; }

struct Op
{
    MPI_Op op;
    explicit Op(MPI_Op mpiop = MPI_OP_NULL) : op(mpiop) {}
};

const Op MAX = static_cast<Op>(MPI_MAX);
const Op MIN = static_cast<Op>(MPI_MIN);
const Op MAXLOC = static_cast<Op>(MPI_MAXLOC);
const Op MINLOC = static_cast<Op>(MPI_MINLOC);
const Op PROD = static_cast<Op>(MPI_PROD);
const Op SUM = static_cast<Op>(MPI_SUM);
const Op REPLACE = static_cast<Op>(MPI_REPLACE);
const Op LAND = static_cast<Op>(MPI_LAND);
const Op LOR = static_cast<Op>(MPI_LOR);
const Op LXOR = static_cast<Op>(MPI_LXOR);
const Op BAND = static_cast<Op>(MPI_BAND);
const Op BOR = static_cast<Op>(MPI_BOR);
const Op BXOR = static_cast<Op>(MPI_BXOR);

#define SETUP(comm) \
do { \
        MPI_Comm_dup(comm, comm_); \
        MPI_Comm_size(comm_, &size_); \
        MPI_Comm_rank(comm_, &rank_); \
    } while(0)

#define FREE() \
do { \
    MPI_Comm_free(&comm_); \
    } while(0)

#ifndef TAG_ALLTOALLV
#define TAG_ALLTOALLV              (101)
#endif

#ifndef TAG_ALLTOALL
#define TAG_ALLTOALL               (102)
#endif

#ifndef TAG_ALLREDUCE
#define TAG_ALLREDUCE              (103) 
#endif

#ifndef TAG_ALLGATHERV
#define TAG_ALLGATHERV             (104)
#endif

#ifndef TAG_ALLGATHER
#define TAG_ALLGATHER              (105)
#endif

#ifndef TAG_BARRIER
#define TAG_BARRIER                (106)
#endif

#ifndef TAG_BCAST
#define TAG_BCAST                  (107)
#endif

#ifndef TAG_REDUCE
#define TAG_REDUCE                 (108)
#endif

#ifndef TAG_SCATTER
#define TAG_SCATTER                (109)
#endif

#ifndef TAG_REDUCE_SCATTER
#define TAG_REDUCE_SCATTER         (201)
#endif

#ifndef TAG_REDUCE_SCATTER_BLOCK
#define TAG_REDUCE_SCATTER_BLOCK   (202)
#endif

#endif
