#!/usr/bin/env python3
"""collage trace -> GOAL schedule (LogGOPSim, ATLAHS: LogGOPSim and htsim back ends).

    collage2goal.py TRACEDIR [-o OUTDIR] [--call LABEL] [--bytes N]
                    [--compute measured|model|zero] [--reduce-gbs G] [--copy-gbs G]
                    [--gaps] [--iters N]

Writes one GOAL text file per recorded call, OUTDIR/<algorithm>-<bytes>.goal,
for all world ranks (num_ranks = the world size of the recording). Convert and
simulate with, e.g.:

    txt2bin -i ring-4096.goal -o ring-4096.bin
    LogGOPSim -f ring-4096.bin -L 3000 -o 1000 -g 1000 -G 0.04 -S 65536

What is kept, per rank, from the trace:
  * every send (S, Z) and receive (R): peer translated to its world rank,
    size in bytes (derived datatypes: the bytes they cover), and a tag that
    folds in the communicator: (cid << 20) | tag, so messages of different
    communicators never match. MPI_ANY_SOURCE stays -1.
  * the host's program order: an operation starts only after the last
    blocking point before it (a wait, a local reduction or copy). Each wait
    becomes a zero-length `calc 0` join node that requires the operations it
    waited for; a waitany requires the one that completed when the trace was
    recorded (if the trace has it, otherwise all of them, which is
    conservative).
  * local reductions (X) and copies (C) become `calc` nodes:
      measured: their recorded duration (traces taken with --trace-times);
      model:    bytes / --reduce-gbs or --copy-gbs (1 GB/s = 1 byte/ns);
      zero:     communication only.
    LogGOPSim times are in ns, so measured durations are used as recorded.
  * --gaps (needs --trace-times): host time between consecutive trace records
    that no record covers (algorithm bookkeeping, posting overhead beyond the
    model's o) becomes `calc` nodes in the dependency chain as well.

Temporaries, allocation (A/F) and buffer addresses do not matter to a network
simulator and are dropped. Recorded `--iters N` concatenates N back-to-back
instances of the call (each depends on the previous one on the same rank).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collage_trace as ct  # noqa: E402


class RankGoal:
    def __init__(self):
        self.lines = []
        self.n = 0

    def node(self, text, deps=()):
        self.n += 1
        lab = f"l{self.n}"
        self.lines.append(f"{lab}: {text}")
        for d in deps:
            if d is not None:
                self.lines.append(f"{lab} requires {d}")
        return lab


def calc_ns(rt, op, args):
    if args.compute == "zero":
        return 0
    if args.compute == "measured":
        if op.t0 is None:
            raise SystemExit("--compute measured needs a trace recorded with --trace-times "
                             "(use --compute model or zero)")
        return max(0, op.t1 - op.t0)
    nb = ct.local_bytes(rt, op)
    gbs = args.reduce_gbs if op.k == "X" else args.copy_gbs
    return int(round(nb / gbs))


def convert_rank(rt, call, g, cursor, args):
    """append one instance of `call` to g; returns the final join label"""
    pending = {}  # trace request id -> GOAL label
    last_t1 = None
    for op in call.ops:
        if args.gaps and op.t0 is not None and last_t1 is not None and op.k not in "AF":
            gap = op.t0 - last_t1
            if gap > 0:
                cursor = g.node(f"calc {gap}", [cursor])
        if op.t1 is not None and op.k not in "AF":
            last_t1 = max(last_t1 or 0, op.t1)
        k, f = op.k, op.f
        if k in ("S", "Z", "R"):
            req, peer = f[0], rt.world(f[2])
            nb = ct.msg_bytes(rt, op)
            tag = ct.tag_of(op)
            if k == "R":
                lab = g.node(f"recv {nb}b from {peer} tag {tag}", [cursor])
            else:
                lab = g.node(f"send {nb}b to {peer} tag {tag}", [cursor])
            pending[req] = lab
        elif k in ("W", "Y"):
            n = f[0]
            ids = f[1:1 + n]
            if k == "Y" and len(f) > 1 + n and f[1 + n] >= 0:
                ids = [f[1 + n]]
            deps = [pending.pop(i) for i in ids if i in pending]
            cursor = g.node("calc 0", [cursor] + deps)
        elif k in ("X", "C"):
            cursor = g.node(f"calc {calc_ns(rt, op, args)}", [cursor])
    if pending:  # nothing should be left in flight; join it anyway
        cursor = g.node("calc 0", [cursor] + list(pending.values()))
    return cursor


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracedir")
    ap.add_argument("-o", "--outdir", default=".")
    ap.add_argument("--call", help="only this algorithm label")
    ap.add_argument("--bytes", type=int, help="only this message size")
    ap.add_argument("--compute", choices=["measured", "model", "zero"], default=None,
                    help="cost of local reductions/copies (default: measured if the trace has times, else model)")
    ap.add_argument("--reduce-gbs", type=float, default=10.0, help="model: local reduction rate, GB/s (default 10)")
    ap.add_argument("--copy-gbs", type=float, default=20.0, help="model: local copy rate, GB/s (default 20)")
    ap.add_argument("--gaps", action="store_true", help="add untraced host time between records as calc")
    ap.add_argument("--iters", type=int, default=1, help="back-to-back instances per file (default 1)")
    args = ap.parse_args()

    ranks = ct.load(args.tracedir)
    timed = ranks[0].meta.get("times") == "1"
    if args.compute is None:
        args.compute = "measured" if timed else "model"
    if args.gaps and not timed:
        raise SystemExit("--gaps needs a trace recorded with --trace-times")
    os.makedirs(args.outdir, exist_ok=True)
    world = len(ranks)
    wrote = 0
    for label, nbytes in ct.calls_of(ranks):
        if args.call and label != args.call:
            continue
        if args.bytes is not None and nbytes != args.bytes:
            continue
        out = os.path.join(args.outdir, f"{ct.safe(label)}-{nbytes}.goal")
        with open(out, "w") as fh:
            m = ranks[0].meta
            fh.write(f"// collage {m.get('coll')} {label} {nbytes} bytes type={m.get('type')} "
                     f"transport={m.get('transport')} gpu={m.get('gpu')} compute={args.compute}"
                     f"{' gaps' if args.gaps else ''} iters={args.iters}\n")
            fh.write(f"num_ranks {world}\n\n")
            for rt in ranks:
                g = RankGoal()
                c = ct.find_call(rt, label, nbytes)
                if c is not None:  # a rank outside the participating group has no record
                    cur = None
                    for _ in range(args.iters):
                        cur = convert_rank(rt, c, g, cur, args)
                if not g.lines:  # LogGOPSim crashes on a schedule without operations
                    g.node("calc 0")
                fh.write(f"rank {rt.world_rank} {{\n")
                for line in g.lines:
                    fh.write(line + "\n")
                fh.write("}\n\n")
        wrote += 1
        print(out)
    if not wrote:
        raise SystemExit("no matching calls in the trace")


if __name__ == "__main__":
    main()
