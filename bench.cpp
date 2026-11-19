/*
 * collage -- standalone benchmark for the Open MPI coll/base collective
 * algorithms, re-implemented as sends, receives and local reductions over a
 * pluggable transport (MPI, stream-triggered, or device-initiated one-sided).
 *
 *   mpirun -np 64 ./collage -c allreduce -o sum -t float --check
 *   mpirun -np 64 ./collage -c bcast -a binomial,pipeline,native -s 0,8K,64K
 *
 * Run with --help for all options.
 */
#include <cstdio>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "bench.hpp"

namespace {

using bench::Options;

std::vector<std::string> split(const std::string& s, char sep = ',')
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep))
        if (!item.empty()) out.push_back(item);
    return out;
}

/* "64", "8K", "8KB", "1M", "1MiB", "2G" -> bytes */
size_t parse_bytes(const std::string& s)
{
    char* end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || v < 0) throw std::invalid_argument("bad size '" + s + "'");
    std::string suf(end);
    for (auto& ch : suf) ch = (char)std::toupper((unsigned char)ch);
    double mult = 1;
    if (suf.empty() || suf == "B") mult = 1;
    else if (suf[0] == 'K') mult = 1024.0;
    else if (suf[0] == 'M') mult = 1024.0 * 1024;
    else if (suf[0] == 'G') mult = 1024.0 * 1024 * 1024;
    else throw std::invalid_argument("bad size suffix in '" + s + "'");
    return (size_t)(v * mult + 0.5);
}

std::vector<size_t> parse_bytes_list(const std::string& s)
{
    std::vector<size_t> out;
    for (auto& t : split(s)) out.push_back(parse_bytes(t));
    if (out.empty()) throw std::invalid_argument("empty list");
    return out;
}

std::vector<int> parse_int_list(const std::string& s)
{
    std::vector<int> out;
    for (auto& t : split(s)) out.push_back(std::stoi(t));
    if (out.empty()) throw std::invalid_argument("empty list");
    return out;
}

void usage()
{
    std::printf(
        "usage: collage -c COLLECTIVE [options]\n"
        "\n"
        "Benchmarks the Open MPI coll/base algorithms of one collective over a range\n"
        "of message sizes, on MPI, stream-triggered or device-initiated transports.\n"
        "\n"
        "selection\n"
        "  -c, --coll NAME        allgather allgatherv allreduce alltoall alltoallv barrier\n"
        "                         bcast gather reduce reduce_scatter reduce_scatter_block scatter\n"
        "  -a, --algos LIST       comma separated algorithms (default: all + native)\n"
        "      --no-native        do not run the native MPI collective as a baseline\n"
        "      --list             list collectives and their algorithms, then exit\n"
        "  -t, --type TYPE        int8 uint8 int16 uint16 int32 uint32 int64 uint64\n"
        "                         float double (default double)\n"
        "                         float16 bfloat16 float8_e4m3 (alias float8) float8_e5m2:\n"
        "                         carried as uint16/uint8, reduced in float with one\n"
        "                         rounding per step (8-bit formats saturate)\n"
        "  -o, --op OP            reduction: sum prod max min land lor lxor band bor bxor\n"
        "                         user (user-defined sum) user_nc (non-commutative)\n"
        "                         plugin:PATH (shared library, collage_op_plugin.h: a host\n"
        "                         MPI function + optional CUDA launcher) (default sum)\n"
        "\n"
        "message sizes (bytes; K/M/G suffixes accepted)\n"
        "  -m, --min BYTES        smallest size (default 0)\n"
        "  -M, --max BYTES        largest size (default 4M)\n"
        "  -f, --factor F         geometric step (default 2)\n"
        "      --sizes LIST       explicit list, e.g. 0,8,1K,64K,4M (overrides -m/-M/-f)\n"
        "      --vdist D          v-collectives: uniform | random | linear (default uniform)\n"
        "      --seed N           seed for --vdist random and --split-mask M:p (default 42)\n"
        "\n"
        "algorithm tunables (comma lists are swept, producing one column each)\n"
        "  -r, --root R           root of rooted collectives (default 0)\n"
        "  -s, --segsize LIST     segment size in bytes, 0 = none (default 0)\n"
        "      --fanout LIST      number of chains for chain algorithms (default 4)\n"
        "  -k, --radix LIST       k-nomial / k-Bruck radix (default 4)\n"
        "      --max-reqs LIST    max outstanding requests, 0 = unlimited (default 0)\n"
        "      --first-seg LIST   gather linear_sync first segment bytes (default 1K)\n"
        "\n"
        "concurrent groups\n"
        "      --split-mask M[:K|:p]\n"
        "                         split the N processes into N/M communicators of M\n"
        "                         ranks and run every algorithm on all of them at the\n"
        "                         same time (M must divide N; --root is the rank within\n"
        "                         a group). Reports the mean latency over groups and\n"
        "                         the aggregate bandwidth. Group members are:\n"
        "                           M     consecutive ranks\n"
        "                           M:K   exactly K ranks apart (N divisible by M*K)\n"
        "                           M:p   random, p in [0,1] written with a decimal\n"
        "                                 point (8:0.45, 8:1.0); low p keeps groups\n"
        "                                 nearly consecutive, high p scatters them.\n"
        "                                 Uses --seed.\n"
        "\n"
        "measurement\n"
        "  -i, --iters N          timed iterations (default 100)\n"
        "  -w, --warmup N         warm-up iterations (default 10)\n"
        "      --iters-large N    timed iterations above --large (default 20)\n"
        "      --warmup-large N   warm-up iterations above --large (default 5)\n"
        "      --large BYTES      threshold for the large-message counts (default 64K)\n"
        "      --max-mem BYTES    skip cases needing more buffer memory per rank (default 2G)\n"
        "      --check            verify every algorithm's result (analytic reference) and,\n"
        "                         on failure, print a rank x block error map\n"
        "      --check-only       verify, do not time\n"
        "      --check-ref R      analytic (default: from the generated inputs, independent of\n"
        "                         MPI) | native (what the native MPI collective produces)\n"
        "      --check-steps      also verify every local reduction step against a host\n"
        "                         reference (finds the rank and step that computed wrong)\n"
        "      --check-msgs       also checksum every message at both ends and compare them\n"
        "                         per pair after the call (finds data corrupted in transit)\n"
        "      --check-iters N    also verify every Nth timed iteration\n"
        "      --trace DIR        record every algorithm (one call per size) to\n"
        "                         DIR/rank-NNNNN.trace: its sends, receives, waits,\n"
        "                         reductions and copies, in program order\n"
        "      --trace-times      also record when every traced operation starts and ends\n"
        "                         (ns; for simulators: compute time between messages)\n"
        "      --replay DIR       run the traces in DIR instead of the algorithms\n"
        "                         (variants replay:ALGO), on this run's transport and\n"
        "                         memory space; -a filters by algorithm\n"
        "      --inject SPEC      corrupt on purpose, to test the checks:\n"
        "                         reduce:RANK:K[:ELEM]  the K-th local reduction on RANK\n"
        "                         msg:SRC:DST:K[:BYTE]  the K-th message from SRC received on DST\n"
        "                         (ranks within the group; K counts from 1)\n"
        "      --csv FILE         write per-(algorithm,size) statistics to FILE\n"
        "      --tag TEXT         label copied into every CSV row (e.g. job id, study)\n"
        "      --bw LIST          tables after the latencies, comma list (default bus,wire):\n"
        "                         alg (S/time), bus (nccl-tests' busbw), wire (bytes on the\n"
        "                         busiest rank's port / time), traffic (those bytes / S);\n"
        "                         or all, none\n"
        "      --no-wire          do not count wire bytes (skips one untimed call per cell)\n"
        "  -q, --quiet            no table on stdout\n"
        "\n"
        "GPU buffers (needs a CUDA-aware MPI and a build with `make CUDA=1`)\n"
        "      --gpu              user buffers and all algorithm temporaries in GPU\n"
        "                         memory; one GPU per process (local rank %% visible GPUs)\n"
        "      --gpu-reduce MODE  local reductions on GPU data: kernel (CUDA kernels,\n"
        "                         default) | mpi (MPI_Reduce_local on device pointers)\n"
        "                         | host (staged through pinned host memory).\n"
        "                         User-defined ops (-o user, user_nc) always use host.\n"
        "      --assume-cuda-aware  run even if the MPI library does not report CUDA\n"
        "                         awareness (for libraries that cannot be queried)\n"
        "      --gpu-map LIST     device of each local rank, map[local rank %% size],\n"
        "                         e.g. 3,2,1,0 (default: local rank %% visible GPUs)\n"
        "\n"
        "Transport of the algorithms' messages (see transport.hpp; with --gpu only)\n"
        "      --transport T      mpi (default): CUDA-aware MPI, host-driven\n"
        "                         mpich: MPICH MPIX_Stream enqueue API (build ST=mpich)\n"
        "                         acx: NVIDIA MPI-ACX enqueue API (build ST=acx)\n"
        "                         cray: Cray MPICH stream-triggered MPIX_Queue (build ST=cray)\n"
        "                         ucp: UCX GPU device API, device-initiated one-sided puts\n"
        "                         and counters, no MPI after bootstrap (build ST=ucp)\n"
        "                         shmem: NVSHMEM device API, the same protocol on\n"
        "                         put-with-signal (IBGDA, NVLink, proxy; build ST=nvshmem)\n"
        "                         Stream-triggered transports queue sends, receives,\n"
        "                         waits, copies and reduction kernels on the CUDA stream;\n"
        "                         they need MPI_THREAD_MULTIPLE (requested automatically)\n"
        "      --st-peers P       inter: queue only messages to other nodes, on-node ones\n"
        "                         use CUDA-aware MPI | all: queue every message\n"
        "                         (default inter; ucp, shmem: all)\n"
        "      --ucp-heap BYTES   ucp/shmem: device arena per rank (default 4G; alias --shmem-heap)\n"
        "      --ucp-slots R      ucp/shmem: receives in flight per peer (default 16;\n"
        "                         alias --shmem-slots)\n"
        "      --ucp-timeout S    ucp/shmem: GPU-side timeout in seconds (default 60;\n"
        "                         alias --shmem-timeout)\n"
        "      --mpich-progress-thread  start MPICH's stream progress thread\n"
        "                         (MPIX_Start_progress_thread; transport mpich)\n"
        "      --thread-multiple  initialise MPI with MPI_THREAD_MULTIPLE for the plain\n"
        "                         transport too (same thread level as the stream ones)\n");
}

/* --split-mask M        groups of M consecutive ranks
 * --split-mask M:K      groups of M ranks spaced exactly K apart (K integer >= 1)
 * --split-mask M:p      groups of M ranks, randomised with probability p in [0,1];
 *                       p is recognised by its decimal point (8:1 is stride 1,
 *                       8:1.0 is p = 1) */
void parse_split(const std::string& spec, Options& o)
{
    const std::string err = "bad --split-mask '" + spec + "' (expected M, M:K or M:p, e.g. 8, 8:4, 8:0.45)";
    auto whole = [&](const std::string& t, size_t& pos) {
        if (t.empty()) throw std::invalid_argument(err);
        double d = std::stod(t, &pos);
        if (pos != t.size()) throw std::invalid_argument(err);
        return d;
    };
    size_t colon = spec.find(':');
    std::string ms = spec.substr(0, colon);
    size_t pos = 0;
    try {
        if (ms.empty() || ms.find_first_not_of("0123456789") != std::string::npos) throw std::invalid_argument(err);
        o.split_mask = std::stoi(ms);
        o.split_stride = 1;
        o.split_p = -1.0;
        if (colon != std::string::npos) {
            std::string rest = spec.substr(colon + 1);
            if (rest.find_first_of(".eE") != std::string::npos) {
                double p = whole(rest, pos);
                if (!(p >= 0.0 && p <= 1.0))
                    throw std::invalid_argument("--split-mask '" + spec + "': p must be in [0, 1]");
                o.split_p = p;
            } else {
                if (rest.empty() || rest.find_first_not_of("0123456789") != std::string::npos)
                    throw std::invalid_argument(err);
                o.split_stride = std::stoi(rest);
                if (o.split_stride < 1)
                    throw std::invalid_argument("--split-mask '" + spec + "': stride K must be >= 1");
            }
        }
    } catch (const std::invalid_argument&) {
        throw;
    } catch (const std::exception&) {
        throw std::invalid_argument(err);
    }
    o.split_spec = spec;
}

Options parse(int argc, char** argv, bool& list, bool& help)
{
    Options o;
    list = help = false;
    auto need = [&](int& i) -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + argv[i]);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        /* allow --opt=value (and -opt=value) */
        std::string val;
        bool has_eq = false;
        if (a.rfind("-", 0) == 0 && a.find('=') != std::string::npos) {
            val = a.substr(a.find('=') + 1);
            a = a.substr(0, a.find('='));
            has_eq = true;
        }
        auto v = [&]() { return has_eq ? val : need(i); };

        if (a == "-h" || a == "--help") help = true;
        else if (a == "--list") list = true;
        else if (a == "-c" || a == "--coll") o.coll = v();
        else if (a == "-a" || a == "--algos") o.algos = split(v());
        else if (a == "--no-native") o.native = false;
        else if (a == "-t" || a == "--type") o.type = v();
        else if (a == "-o" || a == "--op") o.op = v();
        else if (a == "-m" || a == "--min") o.min_bytes = parse_bytes(v());
        else if (a == "-M" || a == "--max") o.max_bytes = parse_bytes(v());
        else if (a == "-f" || a == "--factor") o.factor = std::stod(v());
        else if (a == "--sizes") o.sizes = parse_bytes_list(v());
        else if (a == "--vdist") o.vdist = v();
        else if (a == "--seed") o.seed = (unsigned)std::stoul(v());
        else if (a == "-r" || a == "--root") o.root = std::stoi(v());
        else if (a == "-s" || a == "--segsize") o.segsize = parse_bytes_list(v());
        else if (a == "--fanout") o.fanout = parse_int_list(v());
        else if (a == "-k" || a == "--radix") o.radix = parse_int_list(v());
        else if (a == "--max-reqs") o.max_reqs = parse_int_list(v());
        else if (a == "--first-seg") o.first_seg = parse_bytes_list(v());
        else if (a == "-i" || a == "--iters") o.iters = std::stoi(v());
        else if (a == "-w" || a == "--warmup") o.warmup = std::stoi(v());
        else if (a == "--iters-large") o.iters_large = std::stoi(v());
        else if (a == "--warmup-large") o.warmup_large = std::stoi(v());
        else if (a == "--large") o.large_threshold = parse_bytes(v());
        else if (a == "--max-mem") o.max_mem = parse_bytes(v());
        else if (a == "--check") o.check = true;
        else if (a == "--check-only") o.check_only = true;
        else if (a == "--check-steps") o.check_steps = true;
        else if (a == "--check-msgs") o.check_msgs = true;
        else if (a == "--check-iters") o.check_iters = std::stoi(v());
        else if (a == "--inject") o.inject = v();
        else if (a == "--trace") o.trace_dir = v();
        else if (a == "--trace-times") o.trace_times = true;
        else if (a == "--replay") o.replay_dir = v();
        else if (a == "--check-ref") {
            o.check_ref = v();
            if (o.check_ref != "analytic" && o.check_ref != "native")
                throw std::invalid_argument("--check-ref must be analytic or native");
        }
        else if (a == "--gpu") o.gpu = true;
        else if (a == "--gpu-reduce") {
            o.gpu_reduce = v();
            if (o.gpu_reduce != "kernel" && o.gpu_reduce != "mpi" && o.gpu_reduce != "host")
                throw std::invalid_argument("--gpu-reduce must be kernel, mpi or host");
        }
        else if (a == "--assume-cuda-aware") o.assume_cuda_aware = true;
        else if (a == "--gpu-map") {
            o.gpu_map = parse_int_list(v());
            for (int d : o.gpu_map)
                if (d < 0) throw std::invalid_argument("--gpu-map: device ids must be >= 0");
        }
        else if (a == "--csv") o.csv = v();
        else if (a == "--tag") o.tag = v();
        else if (a == "--transport") {
            o.transport = v();
            xport::Kind k;
            if (!xport::parse_kind(o.transport, k))
                throw std::invalid_argument("--transport must be mpi, mpich, acx or cray");
        }
        else if (a == "--st-peers") {
            std::string p = v();
            if (p != "inter" && p != "all") throw std::invalid_argument("--st-peers must be inter or all");
            o.st_all_peers = (p == "all");
            o.st_peers_set = true;
        }
        else if (a == "--mpich-progress-thread") o.mpich_progress = true;
        else if (a == "--ucp-heap" || a == "--shmem-heap") o.os_heap = parse_bytes(v());
        else if (a == "--ucp-slots" || a == "--shmem-slots") o.os_slots = std::stoi(v());
        else if (a == "--ucp-timeout" || a == "--shmem-timeout") o.os_timeout = std::stod(v());
        else if (a == "--thread-multiple") o.thread_multiple = true;
        else if (a == "-q" || a == "--quiet") o.quiet = true;
        else if (a == "--bw") {
            o.bw = v();
            if (o.bw == "both") o.bw = "alg,bus";
            if (o.bw != "all" && o.bw != "none") {
                std::string l = o.bw + ",";
                for (size_t p = 0, q; (q = l.find(',', p)) != std::string::npos; p = q + 1) {
                    const std::string k = l.substr(p, q - p);
                    if (k != "alg" && k != "bus" && k != "wire" && k != "traffic")
                        throw std::invalid_argument("--bw: a comma list of alg, bus, wire, traffic; or all, none");
                }
            }
        }
        else if (a == "--no-wire") o.wire = false;
        else if (a == "--split-mask" || a == "-split-mask") parse_split(v(), o);
        else if (a[0] != '-' && o.coll.empty()) o.coll = a;
        else throw std::invalid_argument("unknown option '" + a + "' (see --help)");
    }
    if (o.factor <= 1.0) throw std::invalid_argument("--factor must be > 1");
    for (int k : o.radix)
        if (k < 2) throw std::invalid_argument("--radix values must be >= 2");
    for (int f : o.fanout)
        if (f < 1 || f > MAXTREEFANOUT) throw std::invalid_argument("--fanout values must be in [1, 32]");
    if (o.iters < 0 || o.warmup < 0 || o.iters_large < 0 || o.warmup_large < 0)
        throw std::invalid_argument("iteration counts must be >= 0");
    if (o.split_mask < 0) throw std::invalid_argument("--split-mask must be >= 1");
    if (o.check_iters < 0) throw std::invalid_argument("--check-iters must be >= 0");
    /* the deep checks run in the verification pass: they imply --check */
    if ((o.check_steps || o.check_msgs || !o.inject.empty() || o.check_iters > 0) && !o.check_only) o.check = true;
    if (!o.inject.empty()) vfy::parse_inject(o.inject); /* validate early */
    return o;
}

template <typename T>
int dispatch_run(const Options& o, const std::string& tname)
{
    return bench::run<T>(MPI_COMM_WORLD, o, tname);
}

int run_typed(const Options& o)
{
    const std::string& t = o.type;
    if (t == "int8" || t == "char") return dispatch_run<int8_t>(o, "int8");
    if (t == "uint8" || t == "byte") return dispatch_run<uint8_t>(o, "uint8");
    if (t == "int16" || t == "short") return dispatch_run<int16_t>(o, "int16");
    if (t == "uint16") return dispatch_run<uint16_t>(o, "uint16");
    if (t == "int32" || t == "int") return dispatch_run<int32_t>(o, "int32");
    if (t == "uint32" || t == "unsigned") return dispatch_run<uint32_t>(o, "uint32");
    if (t == "int64" || t == "long") return dispatch_run<int64_t>(o, "int64");
    if (t == "uint64") return dispatch_run<uint64_t>(o, "uint64");
    if (t == "float") return dispatch_run<float>(o, "float");
    if (t == "double") return dispatch_run<double>(o, "double");
    if (t == "float16" || t == "half" || t == "fp16") return dispatch_run<lowp::float16>(o, "float16");
    if (t == "bfloat16" || t == "bf16") return dispatch_run<lowp::bfloat16>(o, "bfloat16");
    if (t == "float8_e4m3" || t == "float8" || t == "fp8" || t == "e4m3")
        return dispatch_run<lowp::float8_e4m3>(o, "float8_e4m3");
    if (t == "float8_e5m2" || t == "e5m2") return dispatch_run<lowp::float8_e5m2>(o, "float8_e5m2");
    throw std::invalid_argument("unknown type '" + t + "'");
}

void list_algorithms(const Options& o, int rank)
{
    /* every rank builds the runners (communicators are dup'ed); rank 0 prints */
    OpInfo op = make_op<double>("sum");
    for (auto& c : bench::collective_names()) {
        if (!o.coll.empty() && c != o.coll) continue;
        auto r = bench::make_runner<double>(c, MPI_COMM_WORLD, o, op);
        if (rank != 0) continue;
        std::printf("%-22s", c.c_str());
        for (auto& d : r->algorithms()) {
            std::string tags;
            if (d.params & bench::P_SEG) tags += "s";
            if (d.params & bench::P_FANOUT) tags += "f";
            if (d.params & bench::P_RADIX) tags += "k";
            if (d.params & bench::P_MAXREQ) tags += "m";
            if (d.params & bench::P_FIRSTSEG) tags += "1";
            std::printf(" %s%s%s%s", d.name.c_str(), tags.empty() ? "" : "(", tags.c_str(), tags.empty() ? "" : ")");
        }
        std::printf(" native\n");
    }
    if (rank == 0) std::printf("\ntunables: s=--segsize f=--fanout k=--radix m=--max-reqs 1=--first-seg\n");
}

}  // namespace

/* ---- GPU setup ---------------------------------------------------------- */
#if defined(COLLAGE_CUDA) && defined(OPEN_MPI) && defined(__has_include)
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h> /* MPIX_CUDA_AWARE_SUPPORT, MPIX_Query_cuda_support */
#endif
#endif

/* does the MPI library say it can take device pointers? 1 yes, 0 no, -1 unknown */
int cuda_aware(std::string& how)
{
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    how = "Open MPI, MPIX_Query_cuda_support()";
    return MPIX_Query_cuda_support() ? 1 : 0;
#elif defined(MPIX_CUDA_AWARE_SUPPORT)
    how = "Open MPI, built without CUDA support";
    return 0;
#elif defined(MPIX_GPU_SUPPORT_CUDA)
    int ok = 0;
    how = "MPIX_GPU_query_support(MPIX_GPU_SUPPORT_CUDA)";
    if (MPIX_GPU_query_support(MPIX_GPU_SUPPORT_CUDA, &ok) != MPI_SUCCESS) return -1;
    return ok ? 1 : 0;
#else
    how = "the library offers no query";
    return -1;
#endif
}

/* node-local rank from the launcher's environment, before MPI_Init */
int env_local_rank(std::string& from)
{
    const char* vars[] = {"OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID",
                          "PMI_LOCAL_RANK", "PALS_LOCAL_RANKID", "SLURM_LOCALID", "FLUX_TASK_LOCAL_ID"};
    for (const char* v : vars)
        if (const char* e = std::getenv(v)) {
            from = v;
            return std::atoi(e);
        }
    return -1;
}

bool argv_has(int argc, char** argv, const char* opt)
{
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], opt) == 0) return true;
    return false;
}

/* value of "--opt V" or "--opt=V" before MPI_Init (empty if absent) */
std::string argv_value(int argc, char** argv, const char* opt)
{
    const size_t n = std::strlen(opt);
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], opt) == 0 && i + 1 < argc) return argv[i + 1];
        if (std::strncmp(argv[i], opt, n) == 0 && argv[i][n] == '=') return argv[i] + n + 1;
    }
    return "";
}

#if defined(COLLAGE_CUDA)
/* select the GPU (if not done before MPI_Init), check CUDA awareness,
 * configure the memory layer and prepare the header lines */
void setup_gpu(Options& o, int pre_local, const std::string& pre_from, const std::string& pre_err, int rank)
{
    int local = pre_local;
    std::string from = pre_from;
    /* Ranks whose device could not be set before MPI_Init (no launcher local
     * rank, or a bad --gpu-map entry) select it now. The split is collective,
     * so every rank takes part when any rank needs it, and a failure on some
     * ranks is agreed on before throwing, so no rank is left waiting. */
    std::string err = pre_err;  /* a device chosen before MPI_Init that failed is not retried */
    int need = (local < 0 && err.empty()) ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &need, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (need) {
        MPI_Comm node;
        int node_local = 0;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
        MPI_Comm_rank(node, &node_local);
        MPI_Comm_free(&node);
        if (local < 0 && err.empty()) {
            local = node_local;
            from = "MPI_COMM_TYPE_SHARED, after MPI_Init";
            try { mem::select_device(local, o.gpu_map); } catch (const std::exception& e) { err = e.what(); }
        }
    }
    int bad = err.empty() ? 0 : 1, first_bad = bad ? rank : INT_MAX;
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &first_bad, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (bad) {
        if (rank == first_bad)
            std::fprintf(stderr, "collage: rank %d (local rank %d): %s\n", rank, local, err.c_str());
        throw std::runtime_error("--gpu: device selection failed (see the message above)");
    }
    std::string how;
    int aware = cuda_aware(how);
    /* every rank must take the same decision */
    int all = aware, any_no = (aware == 0);
    MPI_Allreduce(MPI_IN_PLACE, &any_no, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (any_no && !o.assume_cuda_aware)
        throw std::runtime_error("--gpu: the MPI library reports that it is not CUDA-aware (" + how +
                                 "); device buffers would crash it. Use a CUDA-aware build, or pass "
                                 "--assume-cuda-aware if you know better.");
    mem::configure(mem::Space::Device, o.gpu_reduce == "kernel" ? mem::ReduceMode::Kernel
                                       : o.gpu_reduce == "mpi"  ? mem::ReduceMode::Mpi
                                                                : mem::ReduceMode::Host);
    /* how many distinct GPUs are in use per node (for the header) */
    int ndev = mem::device_count();
    o.gpu_report.push_back("# buffers      : GPU memory (CUDA), one device per process");
    o.gpu_report.push_back("#                rank 0 on " + mem::device_name());
    std::string mapping;
    if (o.gpu_map.empty()) {
        mapping = "local rank % " + std::to_string(ndev) + " visible device(s)";
    } else {
        mapping = "--gpu-map ";
        for (size_t i = 0; i < o.gpu_map.size(); ++i) mapping += (i ? "," : "") + std::to_string(o.gpu_map[i]);
        mapping += " indexed by local rank (" + std::to_string(ndev) + " visible)";
    }
    o.gpu_report.push_back("#                device = " + mapping + "; local rank from " + from);
    std::string verdict = all == 1 ? "yes" : any_no ? "NO (overridden by --assume-cuda-aware)" : "unknown";
    o.gpu_report.push_back("# CUDA-aware   : " + verdict + " (" + how + ")");
    if (all == -1 && !any_no && rank == 0)
        std::fprintf(stderr, "collage: warning: cannot tell whether the MPI library is CUDA-aware (%s); "
                             "continuing with device buffers\n", how.c_str());
}
#endif

int main(int argc, char** argv)
{
    /* With --gpu, pick the device before MPI_Init when the launcher tells us the
     * local rank: several CUDA-aware MPIs set up their CUDA context in MPI_Init. */
    const bool want_gpu = argv_has(argc, argv, "--gpu");
    (void)want_gpu;
    std::string pre_from, pre_err;
    int pre_local = -1;
#if defined(COLLAGE_CUDA)
    if (want_gpu) {
        pre_local = env_local_rank(pre_from);
        if (pre_local >= 0) {
            std::vector<int> map;
            try {
                std::string mv = argv_value(argc, argv, "--gpu-map");
                if (!mv.empty()) map = parse_int_list(mv);
                mem::select_device(pre_local, map);
            } catch (const std::exception& e) {
                pre_err = e.what(); /* bad option or device: reported by every rank after MPI_Init */
            }
        }
    }
#endif
    /* stream-triggered transports call MPI from library threads (MPI-ACX proxy,
     * MPICH GPU host functions / progress thread): MPI_THREAD_MULTIPLE */
    const std::string want_transport = argv_value(argc, argv, "--transport");
    const bool want_multiple = (!want_transport.empty() && want_transport != "mpi" && want_transport != "ucp" &&
                                want_transport != "shmem") ||
                               argv_has(argc, argv, "--thread-multiple");
    if (want_multiple) {
        int provided = MPI_THREAD_SINGLE;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    } else {
        MPI_Init(&argc, &argv);
    }
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rc = 0;
    try {
        bool list = false, help = false;
        Options o = parse(argc, argv, list, help);
        if (o.gpu && !help && !list) {
#if defined(COLLAGE_CUDA)
            setup_gpu(o, pre_local, pre_from, pre_err, rank);
#else
            (void)pre_local;
            throw std::invalid_argument("--gpu: this binary was built without CUDA support (rebuild with `make CUDA=1`)");
#endif
        }
        if (!help && !list) {
            /* ucp/shmem need no thread level beyond the default: MPI is only their bootstrap */
            if (o.thread_multiple || (o.transport != "mpi" && o.transport != "ucp" && o.transport != "shmem")) {
                int provided = MPI_THREAD_SINGLE;
                MPI_Query_thread(&provided);
                if (provided < MPI_THREAD_MULTIPLE)
                    throw std::runtime_error("MPI_THREAD_MULTIPLE requested but the MPI library provides only "
                                             "thread level " + std::to_string(provided));
            }
            if (o.transport != "mpi") {
                if (!o.gpu) throw std::invalid_argument("--transport " + o.transport + " needs --gpu");
                xport::Kind k = xport::Kind::Mpi;
                xport::parse_kind(o.transport, k);
                /* every rank must succeed, or all stop together */
                std::string err;
                /* ucp/shmem move on-node messages themselves too (CUDA IPC, NVLink), unless told otherwise */
                if (xport::onesided(k) && !o.st_peers_set) o.st_all_peers = true;
#if defined(COLLAGE_ONESIDED)
                xport::st().os_heap = o.os_heap;
                xport::st().os_slots = (uint32_t)o.os_slots;
                xport::st().os_timeout = o.os_timeout;
#endif
                try { xport::configure(k, o.st_all_peers, o.mpich_progress); } catch (const std::exception& e) { err = e.what(); }
                int bad = err.empty() ? 0 : 1, first_bad = bad ? rank : INT_MAX;
                MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, &first_bad, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
                if (bad) {
                    if (rank == first_bad) std::fprintf(stderr, "collage: rank %d: %s\n", rank, err.c_str());
                    throw std::runtime_error("--transport " + o.transport + ": setup failed (see the message above)");
                }
            }
        }
        if (help) {
            if (rank == 0) usage();
        } else if (list) {
            list_algorithms(o, rank);
        } else {
            if (o.coll.empty()) throw std::invalid_argument("no collective given (-c NAME, see --help)");
            rc = run_typed(o);
        }
    } catch (const std::exception& e) {
        if (rank == 0) std::fprintf(stderr, "collage: %s\n", e.what());
        rc = 1;
    }
    xport::shutdown(); /* MPICH stream, MPI-ACX, Cray queue (before the CUDA stream goes) */
    mem::shutdown();   /* cached device memory, staging buffers, stream */
    MPI_Finalize();
    return rc;
}
