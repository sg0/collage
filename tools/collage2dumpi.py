#!/usr/bin/env python3
"""collage trace -> SST DUMPI trace (SST/macro, and anything else that reads DUMPI).

    collage2dumpi.py TRACEDIR [-o OUTDIR] [--call LABEL] [--bytes N]
                     [--compute measured|model|zero] [--reduce-gbs G] [--copy-gbs G]
                     [--iters N] [--ascii2dumpi PATH] [--ascii-only]

For every recorded call it writes OUTDIR/<algorithm>-<bytes>/ holding
  <algorithm>-<bytes>-NNNN.txt   DUMPI ASCII, one per world rank
  <algorithm>-<bytes>-NNNN.bin   binary DUMPI (ascii2dumpi from sst-dumpi; skipped with --ascii-only)
  <algorithm>-<bytes>.meta       the metafile SST/macro opens
  sstmac.ini                     a minimal SST/macro parameter file for it

and runs, e.g.:
    cd OUTDIR/ring-4096 && sstmac -f sstmac.ini

Mapping:
  * S -> MPI_Isend, Z -> MPI_Issend, R -> MPI_Irecv, all on MPI_COMM_WORLD
    with MPI_BYTE (count = bytes; derived datatypes: the bytes they cover),
    peers as world ranks, tags (cid << 20) | tag so that the recorded
    communicators never cross-match.
  * W -> MPI_Wait (one request) or MPI_Waitall; Y -> MPI_Waitany, with the
    index that completed when recorded (if the trace has it).
  * Local reductions and copies are not MPI calls: they appear as time
    between calls, which SST/macro replays as compute. measured (traces taken
    with --trace-times, the default for them) keeps the recorded wall-clock
    times of every call, so the gaps are what the host spent; model / zero
    build a synthetic clock where communication calls take 1 ns and the
    reductions / copies take bytes / --reduce-gbs or --copy-gbs (or nothing).
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collage_trace as ct  # noqa: E402

T_START = 1_700_000_000  # walltime origin (s); DUMPI wants absolute times

TYPE_TABLE = [
    (0, "MPI_DATATYPE_ERROR", 0), (1, "MPI_DATATYPE_NULL", 0), (2, "MPI_CHAR", 1),
    (3, "MPI_SIGNED_CHAR", 1), (4, "MPI_UNSIGNED_CHAR", 1), (5, "MPI_BYTE", 1),
    (6, "MPI_WCHAR", 4), (7, "MPI_SHORT", 2), (8, "MPI_UNSIGNED_SHORT", 2), (9, "MPI_INT", 4),
    (10, "MPI_UNSIGNED", 4), (11, "MPI_LONG", 8), (12, "MPI_UNSIGNED_LONG", 8),
    (13, "MPI_FLOAT", 4), (14, "MPI_DOUBLE", 8), (15, "MPI_LONG_DOUBLE", 16),
    (16, "MPI_LONG_LONG_INT", 8), (17, "MPI_UNSIGNED_LONG_LONG", 8), (18, "MPI_LONG_LONG", 8),
    (19, "MPI_PACKED", 1), (20, "MPI_LB", 0), (21, "MPI_UB", 0), (22, "MPI_FLOAT_INT", 8),
    (23, "MPI_DOUBLE_INT", 12), (24, "MPI_LONG_INT", 12), (25, "MPI_SHORT_INT", 6),
    (26, "MPI_2INT", 8), (27, "MPI_LONG_DOUBLE_INT", 20),
]
INT_MAX = 2**31 - 1


class Writer:
    def __init__(self, rank, world):
        self.out = []
        self.t = 0  # ns since T_START
        self.counts = {}
        self.rank, self.world = rank, world

    @staticmethod
    def stamp(ns):
        s, f = divmod(ns, 1_000_000_000)
        return f"{T_START + s}.{f:09d}"

    def call(self, name, args, t0=None, t1=None):
        """one MPI call; times in ns (None: advance the synthetic clock by 1 ns)"""
        if t0 is None:
            t0 = self.t
        t0 = max(t0, self.t)
        if t1 is None:
            t1 = t0 + 1
        t1 = max(t1, t0)
        self.t = t1
        cpu = lambda ns: f"{ns // 1_000_000_000}.{ns % 1_000_000_000:09d}"
        self.out.append(f"{name} entering at walltime {self.stamp(t0)}, cputime {cpu(t0)} seconds in thread 0.")
        self.out.extend(args)
        self.out.append(f"{name} returning at walltime {self.stamp(t1)}, cputime {cpu(t1)} seconds in thread 0.")
        self.counts[name] = self.counts.get(name, 0) + 1

    def footer(self):
        o = ["Total keyvals: 0"]
        total = 0
        for n, c in self.counts.items():
            o.append(f"{n} called {c} times and ignored 0 times")
            total += c
        o.append(f"MPI_ALL_FUNCTIONS called {total} times and ignored 0 times")
        o.append("Performance counters: 0")
        for i, n, s in TYPE_TABLE:
            o.append(f"Datatype {i} ({n}) has size {s}")
        return o


def header(start):
    return ["version=1.0.0", f"starttime={time.asctime(time.gmtime(start))}", "hostname=collage", "username=collage",
            "meshdim=0", "meshsize=[]", "meshcrd=[]"]


def model_ns(rt, op, args):
    if args.compute == "zero":
        return 0
    nb = ct.local_bytes(rt, op)
    return int(round(nb / (args.reduce_gbs if op.k == "X" else args.copy_gbs)))


def emit_call(rt, call, w, args, t_off):
    """append one instance; returns the end time (ns) of this instance on this rank"""
    measured = args.compute == "measured"
    rid_base = 2  # DUMPI_REQUEST_ERROR = 0 and DUMPI_REQUEST_NULL = 1 are reserved
    for op in call.ops:
        k, f = op.k, op.f
        t0 = t1 = None
        if measured and op.t0 is not None:
            t0, t1 = t_off + op.t0, t_off + op.t1
        if k in ("S", "Z", "R"):
            nb = ct.msg_bytes(rt, op)
            if nb > INT_MAX:
                raise SystemExit(f"{rt.path}: a {nb}-byte message does not fit DUMPI's int count")
            peer = rt.world(f[2])
            name = {"S": "MPI_Isend", "Z": "MPI_Issend", "R": "MPI_Irecv"}[k]
            w.call(name, [f"int count={nb}", "MPI_Datatype datatype=5 (MPI_BYTE)",
                          f"int {'source' if k == 'R' else 'dest'}={peer}{' (MPI_ANY_SOURCE)' if peer == -1 else ''}", f"int tag={ct.tag_of(op)}",
                          "MPI_Comm comm=2 (MPI_COMM_WORLD)", f"MPI_Request request=[{rid_base + f[0]}]"], t0, t1)
        elif k in ("W", "Y"):
            n = f[0]
            ids = [rid_base + i for i in f[1:1 + n]]
            if k == "W" and n == 1:
                w.call("MPI_Wait", [f"MPI_Request request=[{ids[0]}]", "MPI_Status status=<IGNORED>"], t0, t1)
            elif k == "W":
                w.call("MPI_Waitall", [f"int count={n}", f"MPI_Request requests[{n}]=[{', '.join(map(str, ids))}]",
                                       f"MPI_Status statuses[{n}]=<IGNORED>"], t0, t1)
            else:
                idx = 0
                if len(f) > 1 + n and f[1 + n] >= 0 and f[1 + n] in f[1:1 + n]:
                    idx = f[1:1 + n].index(f[1 + n])
                w.call("MPI_Waitany", [f"int count={n}", f"MPI_Request requests[{n}]=[{', '.join(map(str, ids))}]",
                                       f"int index={idx}", "MPI_Status status=<IGNORED>"], t0, t1)
        elif k in ("X", "C") and not measured:
            w.t += model_ns(rt, op, args)  # shows up as compute before the next call
    if measured and call.t_end is not None:
        w.t = max(w.t, t_off + call.t_end)
    return w.t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracedir")
    ap.add_argument("-o", "--outdir", default=".")
    ap.add_argument("--call")
    ap.add_argument("--bytes", type=int)
    ap.add_argument("--compute", choices=["measured", "model", "zero"], default=None)
    ap.add_argument("--reduce-gbs", type=float, default=10.0)
    ap.add_argument("--copy-gbs", type=float, default=20.0)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--ascii2dumpi", default=shutil.which("ascii2dumpi"),
                    help="path to sst-dumpi's ascii2dumpi (default: from PATH)")
    ap.add_argument("--ascii-only", action="store_true", help="write the ASCII traces only")
    args = ap.parse_args()

    ranks = ct.load(args.tracedir)
    timed = ranks[0].meta.get("times") == "1"
    if args.compute is None:
        args.compute = "measured" if timed else "model"
    if args.compute == "measured" and not timed:
        raise SystemExit("--compute measured needs a trace recorded with --trace-times")
    if not args.ascii_only and not args.ascii2dumpi:
        raise SystemExit("ascii2dumpi not found (build sst-dumpi, pass --ascii2dumpi PATH, or use --ascii-only)")
    world = len(ranks)
    wrote = 0
    for label, nbytes in ct.calls_of(ranks):
        if args.call and label != args.call:
            continue
        if args.bytes is not None and nbytes != args.bytes:
            continue
        pre = f"{ct.safe(label)}-{nbytes}"
        d = os.path.join(args.outdir, pre)
        os.makedirs(d, exist_ok=True)
        for rt in ranks:
            w = Writer(rt.world_rank, world)
            w.call("MPI_Init", ["int argc=1", 'string argv[1]=["collage"]'])
            w.call("MPI_Comm_rank", ["MPI_Comm comm=2 (MPI_COMM_WORLD)", f"int rank={rt.world_rank}"])
            w.call("MPI_Comm_size", ["MPI_Comm comm=2 (MPI_COMM_WORLD)", f"int size={world}"])
            c = ct.find_call(rt, label, nbytes)
            t_off = w.t
            if c is not None:
                for _ in range(args.iters):
                    t_off = emit_call(rt, c, w, args, t_off)
            w.call("MPI_Finalize", [])
            txt = os.path.join(d, f"{pre}-{rt.world_rank:04d}.txt")
            with open(txt, "w") as fh:
                fh.write("\n".join(header(T_START) + w.out + w.footer()) + "\n")
            if not args.ascii_only:
                r = subprocess.run([args.ascii2dumpi, "-o", txt[:-4] + ".bin", txt], capture_output=True, text=True)
                if r.returncode != 0:
                    raise SystemExit(f"ascii2dumpi failed on {txt}:\n{r.stderr[-2000:]}")
        with open(os.path.join(d, pre + ".meta"), "w") as fh:
            fh.write(f"hostname=collage\nnumprocs={world}\nusername=collage\nstartime={T_START}\n"
                     f"fileprefix={pre}\nversion=1\nsubversion=0\nsubsubversion=0\n")
        with open(os.path.join(d, "sstmac.ini"), "w") as fh:
            fh.write(SSTMAC_INI.format(meta=pre + ".meta", n=world))
        wrote += 1
        print(d)
    if not wrote:
        raise SystemExit("no matching calls in the trace")


SSTMAC_INI = """\
# SST/macro run of a collage trace:  sstmac -f sstmac.ini
# A small packet-level (pisces) machine: {n} single-core nodes on a 1-D torus.
# Replace node/switch/topology with your machine (SST/macro ships models in
# configurations/, e.g. dragonfly.ini, fat_tree.ini).
node {{
 name = simple
 app1 {{
  name = parsedumpi
  dumpi_metaname = {meta}
  size = {n}
  indexing = block
  allocation = first_available
  start = 0ms
  # SST/macro's eager1 protocol (between max_vshort_msg_size and
  # max_eager_msg_size; 512 B..8 KiB by default) matches a message when its
  # payload arrives, so a smaller message sent after it on the same tag can
  # overtake it and break MPI's ordering. Collective algorithms rely on that
  # ordering: keep the two thresholds equal, so that messages are eager (data
  # with the header) up to the threshold and rendezvous above it.
  mpi {{
   max_vshort_msg_size = 8192B
   max_eager_msg_size = 8192B
  }}
 }}
 nic {{
  name = pisces
  injection {{
   arbitrator = cut_through
   mtu = 4096
   bandwidth = 10GB/s
   latency = 1us
   credits = 64KB
  }}
 }}
 memory {{
  name = pisces
  total_bandwidth = 10GB/s
  latency = 15ns
  mtu = 65536
  max_single_bandwidth = 10GB/s
 }}
 proc {{
  frequency = 2.1GHz
  ncores = 1
 }}
 nsockets = 1
}}
switch {{
 name = pisces
 arbitrator = cut_through
 mtu = 4096
 link {{
  bandwidth = 10GB/s
  latency = 100ns
  credits = 64KB
 }}
 xbar {{
  bandwidth = 10GB/s
 }}
 router {{
  name = torus_minimal
 }}
 logp {{
  bandwidth = 10GB/s
  out_in_latency = 1us
  hop_latency = 100ns
 }}
}}
topology {{
 name = torus
 geometry = [{n}]
 redundant = [1]
 injection_redundant = 1
 concentration = 1
}}
"""

if __name__ == "__main__":
    main()
