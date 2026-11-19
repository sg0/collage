"""Reader for collage communication traces (collage --trace DIR).

Format: see trace.hpp. One file per world rank, DIR/rank-NNNNN.trace; each
holds the calls (algorithm, size) that rank recorded. Peers in a trace are
ranks of the rank's group (--split-mask); `members` in the header maps them
to world ranks, which is what simulators use.
"""
import glob
import os
import re


class Op:
    __slots__ = ("k", "f", "t0", "t1")

    def __init__(self, k, f, t0, t1):
        self.k, self.f, self.t0, self.t1 = k, f, t0, t1


class Call:
    def __init__(self, label, base, nbytes):
        self.label, self.base, self.bytes = label, base, nbytes
        self.ops = []
        self.t_end = None


class RankTrace:
    def __init__(self, path):
        self.path = path
        self.meta = {}
        self.calls = []
        with open(path) as fh:
            first = fh.readline().strip()
            if first != "collage-trace 1":
                raise ValueError(f"{path}: not a collage trace (version 1)")
            cur = None
            for line in fh:
                v = line.split()
                if not v:
                    continue
                t0 = t1 = None
                if "@" in v:
                    i = v.index("@")
                    t0, t1 = int(v[i + 1]), int(v[i + 2])
                    v = v[:i]
                k = v[0]
                if k == "meta":
                    for kv in v[1:]:
                        a, _, b = kv.partition("=")
                        self.meta[a] = b
                elif k == "call":
                    cur = Call(v[1], v[2], int(v[3]))
                elif k == "abort":
                    cur = None
                elif k == "end":
                    if cur is not None:
                        cur.t_end = t1
                        self.calls.append(cur)
                    cur = None
                elif cur is not None:
                    cur.ops.append(Op(k, [int(x) if re.fullmatch(r"-?\d+", x) else x for x in v[1:]], t0, t1))
        self.members = [int(x) for x in self.meta.get("members", "").split(",") if x != ""]
        self.esize = int(self.meta.get("esize", "0") or 0)
        self.world_rank = int(self.meta.get("rank", "-1"))

    def world(self, peer):
        """group rank -> world rank (-1 stays ANY_SOURCE)"""
        if peer < 0:
            return peer
        return self.members[peer] if self.members else peer


def load(tdir):
    paths = sorted(glob.glob(os.path.join(tdir, "rank-*.trace")))
    if not paths:
        raise SystemExit(f"no rank-*.trace files in {tdir}")
    ranks = [RankTrace(p) for p in paths]
    for i, r in enumerate(ranks):
        if r.world_rank < 0:
            r.world_rank = int(re.search(r"rank-(\d+)\.trace", r.path).group(1))
    ranks.sort(key=lambda r: r.world_rank)
    if not ranks[0].members:
        raise SystemExit(f"{tdir}: traces predate the members/esize header fields; record them again")
    world = int(ranks[0].meta.get("world", len(ranks)))
    if len(ranks) != world:
        raise SystemExit(f"{tdir}: {len(ranks)} rank files, but the trace was recorded on {world} ranks")
    return ranks


def calls_of(ranks):
    """(label, bytes) of every call, in the order rank 0 recorded them"""
    seen, out = set(), []
    for r in ranks:
        for c in r.calls:
            key = (c.label, c.bytes)
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def find_call(rt, label, nbytes):
    for c in rt.calls:
        if c.label == label and c.bytes == nbytes:
            return c
    return None


def msg_bytes(rt, op):
    """S/Z/R record -> payload bytes"""
    f = op.f  # req cid peer tag buf off dt count [nblocks blocks...]
    dt, n = f[6], f[7]
    if dt == "T":
        return n * rt.esize
    if dt == "B":
        return n
    nb = f[8]
    return sum(f[9 + 2 * i + 1] for i in range(nb))


def tag_of(op, cid_bits=20):
    """communicator + tag -> one tag (each collective object has its own communicator)"""
    cid, tag = op.f[1], op.f[3]
    return (cid << cid_bits) | tag


def local_bytes(rt, op):
    """X (count elements) / C (bytes) -> bytes touched"""
    return op.f[4] * rt.esize if op.k == "X" else op.f[4]


def safe(label):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
