/*
 * collage -- standalone benchmark for the Open MPI coll/base collective
 * algorithms, re-implemented on top of MPI point-to-point.
 *
 *   mpirun -np 64 ./collage -c allreduce -o sum -t float --check
 *   mpirun -np 64 ./collage -c bcast -a binomial,pipeline,native -s 0,8K,64K
 *
 * Run with --help for all options.
 */
#include <cstdio>
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
        "Benchmarks the Open MPI coll/base algorithms of one collective, implemented\n"
        "with MPI point-to-point, over a range of message sizes.\n"
        "\n"
        "selection\n"
        "  -c, --coll NAME        allgather allgatherv allreduce alltoall alltoallv barrier\n"
        "                         bcast gather reduce reduce_scatter reduce_scatter_block scatter\n"
        "  -a, --algos LIST       comma separated algorithms (default: all + native)\n"
        "      --no-native        do not run the native MPI collective as a baseline\n"
        "      --list             list collectives and their algorithms, then exit\n"
        "  -t, --type TYPE        int8 uint8 int16 uint16 int32 uint32 int64 uint64\n"
        "                         float double (default double)\n"
        "  -o, --op OP            reduction: sum prod max min land lor lxor band bor bxor\n"
        "                         user (user-defined sum) user_nc (non-commutative)\n"
        "                         (default sum)\n"
        "\n"
        "message sizes (bytes; K/M/G suffixes accepted)\n"
        "  -m, --min BYTES        smallest size (default 0)\n"
        "  -M, --max BYTES        largest size (default 4M)\n"
        "  -f, --factor F         geometric step (default 2)\n"
        "      --sizes LIST       explicit list, e.g. 0,8,1K,64K,4M (overrides -m/-M/-f)\n"
        "      --vdist D          v-collectives: uniform | random | linear (default uniform)\n"
        "      --seed N           seed for --vdist random (default 42)\n"
        "\n"
        "algorithm tunables (comma lists are swept, producing one column each)\n"
        "  -r, --root R           root of rooted collectives (default 0)\n"
        "  -s, --segsize LIST     segment size in bytes, 0 = none (default 0)\n"
        "      --fanout LIST      number of chains for chain algorithms (default 4)\n"
        "  -k, --radix LIST       k-nomial / k-Bruck radix (default 4)\n"
        "      --max-reqs LIST    max outstanding requests, 0 = unlimited (default 0)\n"
        "      --first-seg LIST   gather linear_sync first segment bytes (default 1K)\n"
        "\n"
        "measurement\n"
        "  -i, --iters N          timed iterations (default 100)\n"
        "  -w, --warmup N         warm-up iterations (default 10)\n"
        "      --iters-large N    timed iterations above --large (default 20)\n"
        "      --warmup-large N   warm-up iterations above --large (default 5)\n"
        "      --large BYTES      threshold for the large-message counts (default 64K)\n"
        "      --max-mem BYTES    skip cases needing more buffer memory per rank (default 2G)\n"
        "      --check            verify every algorithm against the native MPI collective\n"
        "      --check-only       verify, do not time\n"
        "      --csv FILE         write per-(algorithm,size) statistics to FILE\n"
        "  -q, --quiet            no table on stdout\n");
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
        /* allow --opt=value */
        std::string val;
        bool has_eq = false;
        if (a.rfind("--", 0) == 0 && a.find('=') != std::string::npos) {
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
        else if (a == "--csv") o.csv = v();
        else if (a == "-q" || a == "--quiet") o.quiet = true;
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

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rc = 0;
    try {
        bool list = false, help = false;
        Options o = parse(argc, argv, list, help);
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
    MPI_Finalize();
    return rc;
}
