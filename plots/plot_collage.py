#!/usr/bin/env python3
"""
plot_collage.py -- figures from collage CSV output (seaborn / matplotlib).

    python3 plots/plot_collage.py $SCRATCH/collage_study -o figs
    python3 plots/plot_collage.py study/ -o figs --kind best,scaling --coll allreduce,bcast
    python3 plots/plot_collage.py study/ -o figs --kind latency --layout world --ppn 4,128

Reads every *.csv below the given directories (or the given files), i.e. the
raw/ folders written by the Perlmutter job scripts, or any `collage --csv` file.

Figure kinds (--kind, comma separated, default all):
  latency   latency vs message size; one panel per node count; one line per algorithm
  best      heatmap nodes x size: speedup of the best algorithm over native MPI,
            with the winner's name in each cell (+ best_algorithms.csv, the table view)
  scaling   latency vs node count at a few message sizes (also covers barrier)
  layout    aggregate bandwidth vs size per algorithm, one line per group layout
            (world / node / cross / stride:K / group:M / random:P, i.e. --split-mask
            variants; the "split" study)
  random    aggregate bandwidth vs the randomisation p of --split-mask PPN:p, at a
            few sizes (p = 0 is the node layout); one line per algorithm
  tunables  per algorithm with several tunable variants (segment size, fan-out,
            radix, max requests; the "tunables" study): heatmap variant x size of
            latency relative to the fastest row at that size, native included
            (+ tunables_best.csv, the table view)
  ppn       latency vs processes per node at a few sizes (CPU studies)
  buffers   host-memory / GPU-memory latency ratio vs size (GPU studies)
  transport CUDA-aware MPI / stream-triggered latency ratio vs size, per
            transport (GPU runs with --transport cray|mpich|acx, TRANSPORTS=...)

Each figure is written for every combination of the "configuration" columns
present in the data (study, buffers, ppn, layout, type, op, ...); narrow it
down with the filter options (e.g. --study algos).
"""
import argparse
import glob
import math
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# --------------------------------------------------------------------------
# palette & chrome (validated categorical order; light surface)
# --------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
NATIVE_COLOR = INK            # the MPI library baseline: neutral ink, dashed
DIVERGING = LinearSegmentedColormap.from_list("speedup", ["#e34948", "#f0efec", "#2a78d6"])

# algorithm order per collective, as in collage --list: colours follow the
# algorithm, never its rank in a particular figure
ALGOS = {
    "allgather": "recursivedoubling sparbit ring neighborexchange linear k_bruck direct_messaging",
    "allgatherv": "bruck sparbit ring neighborexchange two_procs basic_default",
    "allreduce": "nonoverlapping recursivedoubling ring ring_segmented linear redscat_allgather allgather_reduce",
    "alltoall": "inplace pairwise bruck linear_sync two_procs linear",
    "alltoallv": "inplace pairwise linear",
    "barrier": "doublering recursivedoubling bruck two_procs linear tree",
    "bcast": "bintree pipeline chain binomial split_bintree linear knomial scatter_allgather scatter_allgather_ring",
    "gather": "binomial linear_sync linear",
    "reduce": "chain pipeline binary binomial in_order_binary linear redscat_gather knomial",
    "reduce_scatter": "nonoverlapping recursivehalving ring butterfly",
    "reduce_scatter_block": "basic_linear recursivedoubling recursivehalving butterfly",
    "scatter": "binomial linear linear_nb",
}
ALGOS = {k: v.split() for k, v in ALGOS.items()}
BUF_COLS = ["study", "transport", "ranks_per_node", "layoutname", "type", "op", "gpu_reduce"]
XPORT_COLS = ["study", "ranks_per_node", "layoutname", "type", "op", "gpu_reduce"]
CONFIG_COLS = ["study", "buffers", "gpu_reduce", "transport", "ranks_per_node", "layoutname", "type", "op", "root"]


def theme():
    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans")
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.fontsize": 8, "legend.title_fontsize": 8, "legend.frameon": False,
        "lines.linewidth": 1.5, "lines.markersize": 5.5, "text.color": INK,
    })


def layout_key(name):
    """sort order of layout names: world, node, cross, stride:K, group:M, random:P (by K, M, P)"""
    kind, _, arg = str(name).partition(":")
    rank = {"world": 0, "node": 1, "cross": 2, "stride": 3, "group": 4, "random": 5}.get(kind, 6)
    try:
        num = float(arg) if arg else (0.5 if kind == "random" else 0.0)
    except ValueError:
        num = 0.0
    return (rank, num, str(name))


def random_p(name):
    """p of a random:P layout name (None for other layouts)"""
    kind, _, arg = str(name).partition(":")
    if kind != "random":
        return None
    try:
        return float(arg) if arg else None
    except ValueError:
        return None


# random:P layouts are one magnitude (p), so they share one sequential hue,
# light (nearly consecutive) -> dark (scattered); the structural layouts take
# the categorical order after blue; world is the reference: neutral ink, dashed
RANDOM_RAMP = LinearSegmentedColormap.from_list("random_p", ["#9cc3ef", "#2a78d6", "#0b3b78"])


def layout_styles(layouts):
    styles, k = {}, 0
    rnd = sorted((l for l in layouts if str(l).startswith("random")), key=layout_key)
    for l in sorted(layouts, key=layout_key):
        if l == "world":
            styles[l] = dict(color=NATIVE_COLOR, ls=(0, (5, 2)), marker="")
        elif str(l).startswith("random"):
            t = rnd.index(l) / max(1, len(rnd) - 1)
            styles[l] = dict(color=RANDOM_RAMP(0.1 + 0.9 * t), ls="-", marker=MARKERS[rnd.index(l) % len(MARKERS)])
        else:
            pal = SERIES[1:]
            styles[l] = dict(color=pal[k % len(pal)], ls="-" if k < len(pal) else ":",
                             marker=MARKERS[(k + 3) % len(MARKERS)])
            k += 1
    return styles


def plain_log_y(ax):
    """log y axis labelled with plain numbers (1, 2, 5, 10, ...) instead of 10^k"""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.6g}"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def human_bytes(b):
    b = int(b)
    for unit, div in (("M", 1 << 20), ("K", 1 << 10)):
        if b >= div and b % div == 0:
            return f"{b // div}{unit}"
    return str(b)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "**", "*.csv"), recursive=True)
        else:
            files += glob.glob(p)
    files = [f for f in sorted(set(files)) if os.path.basename(f) != "best_algorithms.csv"]
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f, keep_default_na=False)
        except Exception as e:  # empty / partial file from an interrupted step
            print(f"skipping {f}: {e}", file=sys.stderr)
            continue
        if "collective" not in d.columns or "avg_us" not in d.columns:
            continue
        frames.append(d)
    if not frames:
        sys.exit("no collage CSV files found")
    df = pd.concat(frames, ignore_index=True)

    # older CSVs (before nodes/ppn/buffers columns) still load
    for col, default in (("nodes", 1), ("ranks_per_node", 0), ("buffers", "host"), ("gpu_reduce", "-"), ("transport", "mpi"),
                         ("layout", "-"), ("tag", ""), ("groups", 1), ("aggbw_MBps", np.nan),
                         ("group_ranks", 0)):
        if col not in df.columns:
            df[col] = default
    if (df["ranks_per_node"] == 0).any():
        df.loc[df["ranks_per_node"] == 0, "ranks_per_node"] = df["ranks"] // df["nodes"].clip(lower=1)

    # layout name: from the job tag (layoutname=...) or derived from the layout column
    def layoutname(row):
        m = re.search(r"layoutname=([^;]+)", str(row["tag"]))
        if m:
            return m.group(1)
        lay = str(row["layout"])
        if lay in ("-", ""):
            return "world"
        if lay == "consecutive":
            return "node" if int(row["group_ranks"]) == int(row["ranks_per_node"]) else f"consec{row['group_ranks']}"
        if lay.startswith("stride"):
            return "cross" if int(row["group_ranks"]) == int(row["nodes"]) else lay
        if lay.startswith("random"):
            return "random"
        return lay
    df["layoutname"] = df.apply(layoutname, axis=1)
    # study name from the job tag (study=...), "-" for files written outside the job scripts
    df["study"] = df["tag"].astype(str).str.extract(r"study=([^;]+)", expand=False).fillna("-")

    df = df[df["status"].isin(["ok"])].copy()
    for c in ("avg_us", "aggbw_MBps", "bytes", "nodes", "ranks_per_node", "ranks"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # repeated measurements of the same case (re-runs): keep the median
    keys = ["collective", "algorithm", "variant", "nodes", "bytes"] + CONFIG_COLS
    df = df.groupby(keys, as_index=False, dropna=False).agg(
        avg_us=("avg_us", "median"), aggbw_MBps=("aggbw_MBps", "median"), ranks=("ranks", "first"))
    return df


def apply_filters(df, a):
    def lst(s, conv=str):
        return [conv(x) for x in s.split(",")] if s else None
    for col, val, conv in (("study", a.study, str), ("collective", a.coll, str), ("nodes", a.nodes, int), ("ranks_per_node", a.ppn, int),
                           ("layoutname", a.layout, str), ("buffers", a.buffers, str), ("transport", a.transport, str), ("op", a.op, str),
                           ("type", a.type, str)):
        v = lst(val, conv)
        if v is not None:
            df = df[df[col].isin(v)]
    if a.algos:
        keep = set(a.algos.split(",")) | {"native"}
        df = df[df["algorithm"].isin(keep)]
    return df


# --------------------------------------------------------------------------
# series styling: colour by algorithm (fixed order per collective), marker per
# algorithm, line style per variant; native in neutral ink, dashed
# --------------------------------------------------------------------------
def series_styles(coll, variants_by_algo):
    order = ALGOS.get(coll, [])
    extra = sorted(a for a in variants_by_algo if a not in order and a != "native")
    styles = {}
    for i, algo in enumerate(order + extra):
        if algo not in variants_by_algo:
            continue
        color = SERIES[i % len(SERIES)]
        base_ls = "-" if i < len(SERIES) else ":"     # 9th+ algorithm: re-used hue, dotted
        for j, var in enumerate(sorted(variants_by_algo[algo])):
            ls = base_ls if j == 0 else [(0, (4, 1.5)), (0, (1, 1.2)), (0, (5, 1, 1, 1))][(j - 1) % 3]
            styles[var] = dict(color=color, ls=ls, marker=MARKERS[i % len(MARKERS)], lw=1.5, zorder=3)
    for var in variants_by_algo.get("native", []):
        styles[var] = dict(color=NATIVE_COLOR, ls=(0, (5, 2)), marker="", lw=1.8, zorder=4)
    return styles


def variants_of(d):
    out = {}
    for algo, var in d[["algorithm", "variant"]].drop_duplicates().itertuples(index=False):
        out.setdefault(algo, set()).add(var)
    return out


def legend_handles(styles, order):
    hs = []
    for var in order:
        st = styles[var]
        hs.append(Line2D([], [], color=st["color"], ls=st["ls"], marker=st["marker"], lw=st["lw"],
                         markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=0.8, label=var))
    return hs


def series_order(styles, coll):
    rank = {a: i for i, a in enumerate(ALGOS.get(coll, []))}
    def key(v):
        base = v.split("[")[0]
        return (base == "native", rank.get(base, 99), v)
    return sorted(styles, key=key)


def size_axis(ax, sizes):
    """message sizes as ordinal positions (0 B has no place on a log axis)"""
    pos = {b: i for i, b in enumerate(sizes)}
    step = max(1, math.ceil(len(sizes) / 8))
    ax.set_xticks(range(0, len(sizes), step))
    ax.set_xticklabels([human_bytes(sizes[i]) for i in range(0, len(sizes), step)])
    ax.set_xlim(-0.4, len(sizes) - 0.6)
    ax.set_xlabel("message size (bytes)")
    return pos


def config_groups(df, extra_drop=()):
    cols = [c for c in CONFIG_COLS if c not in extra_drop and df[c].nunique(dropna=False) >= 1]
    return df.groupby(cols, dropna=False), cols


def config_label(cols, key):
    key = key if isinstance(key, tuple) else (key,)
    parts = []
    for c, v in zip(cols, key):
        if c == "gpu_reduce" and v == "-":
            continue
        if c == "op" and v == "-":
            continue
        if c == "root" and int(v) == 0:
            continue
        if c == "study" and v == "-":
            continue
        if c == "transport" and v == "mpi":
            continue
        name = {"ranks_per_node": "ppn", "layoutname": "layout", "gpu_reduce": "reduce"}.get(c, c)
        parts.append(f"{name}={v}")
    return ", ".join(parts)


def config_slug(cols, key):
    return re.sub(r"[^A-Za-z0-9=.,+_-]+", "", config_label(cols, key).replace(", ", "+")) or "all"


def save(fig, outdir, name, formats):
    os.makedirs(outdir, exist_ok=True)
    for fmt in formats:
        fig.savefig(os.path.join(outdir, f"{name}.{fmt}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return name


# --------------------------------------------------------------------------
# figure kinds
# --------------------------------------------------------------------------
def fig_latency(df, a):
    made = []
    for coll, dc in df[df["collective"] != "barrier"].groupby("collective"):
        groups, cols = config_groups(dc)
        for key, d in groups:
            nodes = sorted(d["nodes"].unique())
            sizes = sorted(d["bytes"].unique())
            if len(sizes) < 2:
                continue
            styles = series_styles(coll, variants_of(d))
            order = series_order(styles, coll)
            ncol = min(len(nodes), a.max_cols)
            nrow = math.ceil(len(nodes) / ncol)
            fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol + 1.8, 2.7 * nrow), sharey=True, squeeze=False)
            for ax, n in zip(axes.flat, nodes):
                dn = d[d["nodes"] == n]
                pos = size_axis(ax, sizes)
                for var in order:
                    s = dn[dn["variant"] == var].sort_values("bytes")
                    if s.empty:
                        continue
                    st = styles[var]
                    ax.plot([pos[b] for b in s["bytes"]], s["avg_us"], color=st["color"], ls=st["ls"],
                            marker=st["marker"], lw=st["lw"], zorder=st["zorder"],
                            markeredgecolor=SURFACE, markeredgewidth=0.8)
                plain_log_y(ax)
                ranks = int(dn["ranks"].iloc[0]) if not dn.empty else 0
                ax.set_title(f"{n} node{'s' if n != 1 else ''} ({ranks} ranks)")
            for ax in axes.flat[len(nodes):]:
                ax.set_visible(False)
            for ax in axes[:, 0]:
                ax.set_ylabel("latency (µs)")
            fig.legend(handles=legend_handles(styles, order), loc="center left", bbox_to_anchor=(1.0, 0.5),
                       title="algorithm")
            fig.suptitle(f"{coll}: latency vs message size  ·  {config_label(cols, key)}", x=0.02, ha="left",
                         fontsize=11, color=INK)
            fig.tight_layout()
            made.append(save(fig, os.path.join(a.out, "latency"), f"{coll}__{config_slug(cols, key)}", a.formats))
    return made


def best_table(df):
    """best non-native algorithm per configuration, node count and size, vs native"""
    keys = ["collective", "nodes", "bytes"] + CONFIG_COLS
    p2p = df[df["algorithm"] != "native"]
    nat = df[df["algorithm"] == "native"][keys + ["avg_us"]].rename(columns={"avg_us": "native_us"})
    idx = p2p.groupby(keys, dropna=False)["avg_us"].idxmin()
    best = p2p.loc[idx, keys + ["variant", "avg_us"]].rename(columns={"variant": "best_algorithm",
                                                                      "avg_us": "best_us"})
    out = best.merge(nat, on=keys, how="left")
    out["speedup_vs_native"] = out["native_us"] / out["best_us"]
    # MPI libraries return immediately for empty messages: no meaningful speedup at 0 B
    out.loc[(out["bytes"] == 0) | ~np.isfinite(out["speedup_vs_native"]) | (out["speedup_vs_native"] <= 0),
            "speedup_vs_native"] = np.nan
    return out.sort_values(keys)


def fig_best(df, a):
    made = []
    table = best_table(df)
    os.makedirs(a.out, exist_ok=True)
    table.to_csv(os.path.join(a.out, "best_algorithms.csv"), index=False)
    for coll, dc in table[(table["collective"] != "barrier") & (table["bytes"] > 0)].groupby("collective"):
        groups, cols = config_groups(dc)
        for key, d in groups:
            d = d.dropna(subset=["speedup_vs_native"])
            if d.empty:
                continue
            sizes = sorted(d["bytes"].unique())
            nodes = sorted(d["nodes"].unique())
            sp = d.pivot_table(index="nodes", columns="bytes", values="speedup_vs_native", aggfunc="first")
            names = d.pivot_table(index="nodes", columns="bytes", values="best_algorithm", aggfunc="first")
            sp = sp.reindex(index=nodes, columns=sizes)
            names = names.reindex(index=nodes, columns=sizes)
            logsp = np.log2(sp.astype(float))
            lim = max(1.0, float(np.nanmax(np.abs(logsp.values))))
            fig, ax = plt.subplots(figsize=(max(6.0, 0.62 * len(sizes) + 2.4), 0.55 * len(nodes) + 1.9))
            sns.heatmap(logsp, ax=ax, cmap=DIVERGING, norm=TwoSlopeNorm(0.0, -lim, lim), linewidths=2,
                        linecolor=SURFACE, cbar_kws={"label": "speedup vs native"},
                        annot=names.map(lambda v: abbreviate(v) if isinstance(v, str) else "").values,
                        fmt="", annot_kws={"fontsize": 6.5, "color": INK})
            cb = ax.collections[0].colorbar
            ticks = list(range(-math.floor(lim), math.floor(lim) + 1))  # inside the norm range
            cb.set_ticks(ticks)
            cb.set_ticklabels([f"{2.0 ** t:g}×" for t in ticks])
            ax.set_xticklabels([human_bytes(b) for b in sizes], rotation=0)
            ax.set_yticklabels([str(int(n)) for n in nodes], rotation=0)
            ax.set_xlabel("message size (bytes)")
            ax.set_ylabel("nodes")
            ax.set_title(f"{coll}: best collage algorithm per cell\n{config_label(cols, key)}\n"
                         f"colour: its speedup over native MPI (blue faster, red slower); 0 B omitted",
                         loc="left", fontsize=8.5)
            fig.tight_layout()
            made.append(save(fig, os.path.join(a.out, "best"), f"{coll}__{config_slug(cols, key)}", a.formats))
    return made


ABBREV = {"recursivedoubling": "recdbl", "recursivehalving": "rechlv", "neighborexchange": "nbrx",
          "direct_messaging": "direct", "nonoverlapping": "nonovl", "ring_segmented": "ringseg",
          "redscat_allgather": "rsag", "allgather_reduce": "agred", "scatter_allgather_ring": "sagring",
          "scatter_allgather": "sag", "split_bintree": "splitbt", "in_order_binary": "iobin",
          "redscat_gather": "rsgath", "basic_default": "basic", "basic_linear": "linear", "linear_sync": "linsync",
          "linear_nb": "linnb", "doublering": "dring", "butterfly": "bfly", "pipeline": "pipe",
          "binomial": "binom", "knomial": "knom", "bintree": "bintree", "k_bruck": "kbruck", "two_procs": "2proc"}


def abbreviate(variant):
    base, _, rest = variant.partition("[")
    return ABBREV.get(base, base[:7]) + ("*" if rest else "")


def pick_sizes(sizes, wanted):
    if wanted:
        want = [int(float(x)) for x in wanted.split(",")]
    else:
        nz = [s for s in sizes if s > 0]
        want = [nz[0], nz[len(nz) // 2], nz[-1]] if len(nz) >= 3 else nz
    picked = []
    for w in want:
        best = min(sizes, key=lambda s: abs(math.log2(max(s, 1)) - math.log2(max(w, 1))))
        if best not in picked:
            picked.append(best)
    return picked


def fig_vs(df, a, xcol, xlabel, kind):
    """latency vs nodes (scaling) or vs ppn, at a few message sizes"""
    made = []
    drop = ("ranks_per_node",) if xcol == "ranks_per_node" else ()
    for coll, dc in df.groupby("collective"):
        groups, cols = config_groups(dc, extra_drop=drop)
        for key, d in groups:
            for fixed in (sorted(d["nodes"].unique()) if xcol == "ranks_per_node" else [None]):
                dd = d if fixed is None else d[d["nodes"] == fixed]
                if dd[xcol].nunique() < 2:
                    continue
                sizes = pick_sizes(sorted(dd["bytes"].unique()), a.sizes)
                styles = series_styles(coll, variants_of(dd))
                order = series_order(styles, coll)
                fig, axes = plt.subplots(1, len(sizes), figsize=(3.2 * len(sizes) + 1.8, 2.8), sharey=False,
                                         squeeze=False)
                xs = sorted(dd[xcol].unique())
                for ax, b in zip(axes[0], sizes):
                    db = dd[dd["bytes"] == b]
                    for var in order:
                        s = db[db["variant"] == var].sort_values(xcol)
                        if s.empty:
                            continue
                        st = styles[var]
                        ax.plot(s[xcol], s["avg_us"], color=st["color"], ls=st["ls"], marker=st["marker"],
                                lw=st["lw"], zorder=st["zorder"], markeredgecolor=SURFACE, markeredgewidth=0.8)
                    ax.set_xscale("log", base=2)
                    plain_log_y(ax)
                    ax.set_xticks(xs)
                    ax.set_xticklabels([str(int(x)) for x in xs])
                    ax.minorticks_off()
                    ax.set_xlabel(xlabel)
                    ax.set_title("no data (barrier)" if coll == "barrier" else f"{human_bytes(b)} bytes")
                axes[0][0].set_ylabel("latency (µs)")
                fig.legend(handles=legend_handles(styles, order), loc="center left", bbox_to_anchor=(1.0, 0.5),
                           title="algorithm")
                what = "scaling with node count" if xcol == "nodes" else f"processes per node ({int(fixed)} nodes)"
                fig.suptitle(f"{coll}: {what}  ·  {config_label(cols, key)}", x=0.02, ha="left", fontsize=11)
                fig.tight_layout()
                suffix = "" if fixed is None else f"__{int(fixed)}N"
                made.append(save(fig, os.path.join(a.out, kind), f"{coll}__{config_slug(cols, key)}{suffix}",
                                 a.formats))
    return made


def fig_layout(df, a):
    """aggregate bandwidth per layout, small multiples per algorithm"""
    made = []
    d0 = df[(df["collective"] != "barrier") & df["aggbw_MBps"].notna()]
    for coll, dc in d0.groupby("collective"):
        groups, cols = config_groups(dc, extra_drop=("layoutname",))
        for key, d in groups:
            if d["layoutname"].nunique() < 2:
                continue
            for n in sorted(d["nodes"].unique()):
                dn = d[(d["nodes"] == n) & (d["bytes"] > 0)]
                sizes = sorted(dn["bytes"].unique())
                if len(sizes) < 2:
                    continue
                layouts = sorted(set(dn["layoutname"]), key=layout_key)
                lst = layout_styles(layouts)
                variants = series_order(series_styles(coll, variants_of(dn)), coll)
                variants = variants[: a.max_panels]
                ncol = min(len(variants), a.max_cols)
                nrow = math.ceil(len(variants) / ncol)
                fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol + 1.6, 2.5 * nrow), sharey=True,
                                         squeeze=False)
                for ax, var in zip(axes.flat, variants):
                    dv = dn[dn["variant"] == var]
                    pos = size_axis(ax, sizes)
                    for l in layouts:
                        s = dv[dv["layoutname"] == l].sort_values("bytes")
                        s = s[s["aggbw_MBps"] > 0]
                        if s.empty:
                            continue
                        st = lst[l]
                        ax.plot([pos[b] for b in s["bytes"]], s["aggbw_MBps"], color=st["color"], ls=st["ls"],
                                marker=st["marker"], lw=1.8 if l == "world" else 1.5, markeredgecolor=SURFACE,
                                markeredgewidth=0.8, zorder=4 if l == "world" else 3)
                    plain_log_y(ax)
                    ax.set_title(var, color=NATIVE_COLOR if var == "native" else INK)
                for ax in axes.flat[len(variants):]:
                    ax.set_visible(False)
                for ax in axes[:, 0]:
                    ax.set_ylabel("aggregate bandwidth (MB/s)")
                handles = [Line2D([], [], color=lst[l]["color"], ls=lst[l]["ls"], marker=lst[l]["marker"], lw=1.5,
                                  markeredgecolor=SURFACE, label=l) for l in layouts]
                fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), title="group layout")
                fig.suptitle(f"{coll}: concurrent groups (--split-mask), {int(n)} nodes  ·  {config_label(cols, key)}",
                             x=0.02, ha="left", fontsize=11)
                fig.tight_layout()
                made.append(save(fig, os.path.join(a.out, "layout"), f"{coll}__{int(n)}N__{config_slug(cols, key)}",
                                 a.formats))
    return made


def ratio_figures(m, a, group_cols, ylabel, title, subdir):
    """line plots of m["ratio"] vs size, one panel per node count, one line per algorithm"""
    made = []
    for coll, dc in m[m["collective"] != "barrier"].groupby("collective"):
        for key, d in dc.groupby(group_cols, dropna=False):
            nodes = sorted(d["nodes"].unique())
            sizes = sorted(d["bytes"].unique())
            if len(sizes) < 2:
                continue
            styles = series_styles(coll, variants_of(d))
            order = series_order(styles, coll)
            ncol = min(len(nodes), a.max_cols)
            nrow = math.ceil(len(nodes) / ncol)
            fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol + 1.8, 2.7 * nrow), sharey=True, squeeze=False)
            for ax, n in zip(axes.flat, nodes):
                dn = d[d["nodes"] == n]
                pos = size_axis(ax, sizes)
                ax.axhline(1.0, color=AXIS, lw=1.0, zorder=1)
                for var in order:
                    s = dn[dn["variant"] == var].sort_values("bytes")
                    if s.empty:
                        continue
                    st = styles[var]
                    ax.plot([pos[b] for b in s["bytes"]], s["ratio"], color=st["color"], ls=st["ls"],
                            marker=st["marker"], lw=st["lw"], zorder=st["zorder"], markeredgecolor=SURFACE,
                            markeredgewidth=0.8)
                ax.set_yscale("log", base=2)
                lo, hi = ax.get_ylim()
                lo, hi = min(lo, 0.8), max(hi, 1.25)
                ax.set_ylim(lo, hi)
                cands = [1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 0.75, 1, 1.5, 2, 4, 8, 16, 32, 64]
                ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([c for c in cands if lo <= c <= hi]))
                ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
                ax.set_title(f"{n} node{'s' if n != 1 else ''}")
            for ax in axes.flat[len(nodes):]:
                ax.set_visible(False)
            for ax in axes[:, 0]:
                ax.set_ylabel(ylabel)
            fig.legend(handles=legend_handles(styles, order), loc="center left", bbox_to_anchor=(1.0, 0.5),
                       title="algorithm")
            label = config_label(group_cols, key)
            fig.suptitle(f"{coll}: {title}  ·  {label}", x=0.02, ha="left", fontsize=11)
            fig.tight_layout()
            slug = config_slug(group_cols, key)
            made.append(save(fig, os.path.join(a.out, subdir), f"{coll}__{slug}", a.formats))
    return made


def fig_buffers(df, a):
    """latency(host buffers) / latency(GPU buffers) per algorithm"""
    if set(df["buffers"]) < {"host", "gpu"}:
        return []
    keys = ["study", "collective", "variant", "algorithm", "nodes", "bytes", "ranks_per_node", "layoutname", "type",
            "op", "root"]
    g = df[df["buffers"] == "gpu"]
    if a.gpu_reduce:
        g = g[g["gpu_reduce"].isin(a.gpu_reduce.split(",") + ["-"])]
    h = df[(df["buffers"] == "host")][keys + ["avg_us"]].rename(columns={"avg_us": "host_us"})
    m = g.merge(h, on=keys, how="inner")
    if m.empty:
        return []
    m["ratio"] = m["host_us"] / m["avg_us"]
    return ratio_figures(m, a, BUF_COLS, "host latency / GPU latency",
                         "GPU buffers vs host buffers (>1: GPU buffers faster)", "buffers")


def fig_transport(df, a):
    """latency(plain CUDA-aware MPI) / latency(stream-triggered transport), GPU buffers"""
    # native calls the library's collective whatever the transport: left out
    g = df[(df["buffers"] == "gpu") & (df["algorithm"] != "native")]
    if g["transport"].nunique() < 2 or "mpi" not in set(g["transport"]):
        return []
    keys = ["study", "collective", "variant", "algorithm", "nodes", "bytes", "ranks_per_node", "layoutname", "type",
            "op", "root", "gpu_reduce"]
    base = g[g["transport"] == "mpi"][keys + ["avg_us"]].rename(columns={"avg_us": "mpi_us"})
    made = []
    for t in sorted(set(g["transport"]) - {"mpi"}):
        m = g[g["transport"] == t].merge(base, on=keys, how="inner")
        if m.empty:
            continue
        m["ratio"] = m["mpi_us"] / m["avg_us"]
        made += ratio_figures(m, a, XPORT_COLS, f"latency mpi / latency {t}",
                              f"stream-triggered transport {t} vs CUDA-aware MPI (>1: {t} faster)",
                              os.path.join("transport", t.replace(":", "-")))
    return made


def fig_random(df, a):
    """aggregate bandwidth vs p of --split-mask PPN:p (node layout = p 0), at a few sizes"""
    made = []
    d0 = df[(df["collective"] != "barrier") & df["aggbw_MBps"].notna() & (df["bytes"] > 0)].copy()
    d0["p"] = d0["layoutname"].map(random_p)
    d0.loc[d0["layoutname"] == "node", "p"] = 0.0
    d0 = d0.dropna(subset=["p"])
    for coll, dc in d0.groupby("collective"):
        groups, cols = config_groups(dc, extra_drop=("layoutname",))
        for key, d in groups:
            for n in sorted(d["nodes"].unique()):
                dn = d[d["nodes"] == n]
                if dn["p"].nunique() < 3:
                    continue
                sizes = pick_sizes(sorted(dn["bytes"].unique()), a.sizes)
                styles = series_styles(coll, variants_of(dn))
                order = series_order(styles, coll)
                ps = sorted(dn["p"].unique())
                fig, axes = plt.subplots(1, len(sizes), figsize=(3.2 * len(sizes) + 1.8, 2.8), squeeze=False)
                for ax, b in zip(axes[0], sizes):
                    db = dn[dn["bytes"] == b]
                    for var in order:
                        s = db[(db["variant"] == var) & (db["aggbw_MBps"] > 0)].sort_values("p")
                        if s.empty:
                            continue
                        st = styles[var]
                        ax.plot(s["p"], s["aggbw_MBps"], color=st["color"], ls=st["ls"], marker=st["marker"],
                                lw=st["lw"], zorder=st["zorder"], markeredgecolor=SURFACE, markeredgewidth=0.8)
                    plain_log_y(ax)
                    ax.set_xticks(ps)
                    ax.set_xticklabels([f"{p:g}" for p in ps])
                    ax.set_xlim(-0.04, max(ps) + 0.04)
                    ax.set_xlabel("p in --split-mask PPN:p")
                    ax.set_title(f"{human_bytes(b)} bytes")
                axes[0][0].set_ylabel("aggregate bandwidth (MB/s)")
                fig.legend(handles=legend_handles(styles, order), loc="center left", bbox_to_anchor=(1.0, 0.5),
                           title="algorithm")
                fig.suptitle(f"{coll}: randomised groups, {int(n)} nodes (p = 0: one group per node)  ·  "
                             f"{config_label(cols, key)}", x=0.02, ha="left", fontsize=11)
                fig.tight_layout()
                made.append(save(fig, os.path.join(a.out, "random"), f"{coll}__{int(n)}N__{config_slug(cols, key)}",
                                 a.formats))
    return made


# sequential single hue for "times slower than the fastest": light = fastest
SLOWDOWN = LinearSegmentedColormap.from_list("slowdown", ["#eef4fc", "#7fb0ea", "#2a78d6", "#0b3b78"])


def fig_tunables(df, a):
    """per algorithm: variants (tunable settings) x size, latency / fastest row at that size"""
    made, rows = [], []
    d0 = df[(df["collective"] != "barrier") & (df["bytes"] > 0)]
    for coll, dc in d0.groupby("collective"):
        groups, cols = config_groups(dc)
        for key, d in groups:
            for n in sorted(d["nodes"].unique()):
                dn = d[d["nodes"] == n]
                vba = variants_of(dn)
                algos = [al for al in ALGOS.get(coll, []) + sorted(set(vba) - set(ALGOS.get(coll, [])))
                         if al in vba and al != "native" and len(vba[al]) > 1]
                if not algos:
                    continue
                sizes = sorted(dn["bytes"].unique())
                nat = dn[dn["algorithm"] == "native"].set_index("bytes")["avg_us"]
                mats = []
                for al in algos[: a.max_panels]:
                    vars_ = sorted(vba[al], key=lambda v: (len(v), v))
                    m = dn[dn["algorithm"] == al].pivot_table(index="variant", columns="bytes", values="avg_us",
                                                             aggfunc="first").reindex(index=vars_, columns=sizes)
                    if not nat.empty:
                        m.loc["native"] = nat.reindex(sizes).values
                    ratio = m / m.min(axis=0)
                    mats.append((al, m, ratio))
                    for b in sizes:
                        col = m.loc[vars_, b].dropna()
                        if col.empty:
                            continue
                        rows.append(dict(zip(cols, key if isinstance(key, tuple) else (key,)),
                                         collective=coll, nodes=int(n), bytes=int(b), algorithm=al,
                                         best_variant=col.idxmin(), best_us=col.min(), worst_variant=col.idxmax(),
                                         worst_us=col.max(), spread=col.max() / col.min(),
                                         native_us=float(nat.get(b, np.nan))))
                vmax = max(2.0, min(64.0, float(np.nanmax([np.nanmax(r.values) for _, _, r in mats]))))
                heights = [len(r) for _, _, r in mats]
                fig = plt.figure(figsize=(max(6.0, 0.55 * len(sizes) + 3.2), 0.19 * sum(heights) + 0.75 * len(mats) + 1.2))
                gs = fig.add_gridspec(len(mats), 2, height_ratios=heights, width_ratios=[40, 1], hspace=0.55, wspace=0.03)
                norm = matplotlib.colors.LogNorm(1.0, vmax)
                cax = fig.add_subplot(gs[:, 1])
                for i, (al, m, ratio) in enumerate(mats):
                    ax = fig.add_subplot(gs[i, 0])
                    best = ratio.eq(1.0) & ratio.notna()
                    sns.heatmap(ratio.clip(upper=vmax), ax=ax, cmap=SLOWDOWN, norm=norm, linewidths=1.5,
                                linecolor=SURFACE, cbar=(i == 0), cbar_ax=cax if i == 0 else None,
                                cbar_kws={"label": "latency / fastest row at that size"},
                                annot=best.map(lambda v: "●" if v else "").values, fmt="",
                                annot_kws={"fontsize": 6, "color": INK})
                    labels = [v.split("[", 1)[1].rstrip("]") if "[" in v else v for v in ratio.index]
                    ax.set_yticks(np.arange(len(labels)) + 0.5)
                    ax.set_yticklabels(labels, rotation=0, fontsize=6.5)
                    for t, v in zip(ax.get_yticklabels(), ratio.index):
                        if v == "native":
                            t.set_color(MUTED)
                    ax.set_ylabel("")
                    ax.set_title(al, loc="left", fontsize=9)
                    if i == len(mats) - 1:
                        ax.set_xticklabels([human_bytes(b) for b in sizes], rotation=0)
                        ax.set_xlabel("message size (bytes)")
                    else:
                        ax.set_xticklabels([])
                        ax.set_xlabel("")
                ticks = [t for t in (1, 1.5, 2, 4, 8, 16, 32, 64) if t <= vmax]
                cax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
                cax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                cax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
                fig.suptitle(f"{coll}: tunable settings, {int(n)} nodes  ·  {config_label(cols, key)}\n"
                             f"● fastest row at that size (native included); 0 B omitted",
                             x=0.02, ha="left", fontsize=10)
                made.append(save(fig, os.path.join(a.out, "tunables"), f"{coll}__{int(n)}N__{config_slug(cols, key)}",
                                 a.formats))
    if rows:
        os.makedirs(a.out, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "tunables_best.csv"), index=False)
    return made


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="directories (searched recursively) or CSV files")
    ap.add_argument("-o", "--out", default="figs", help="output directory (default figs)")
    ap.add_argument("--kind", default="all",
                    help="latency,best,scaling,layout,random,tunables,ppn,buffers,transport or all")
    ap.add_argument("--study", help="only these studies (algos,split,tunables; from the job tag)")
    ap.add_argument("--formats", default="png,pdf", help="image formats (default png,pdf)")
    ap.add_argument("--coll", help="only these collectives (comma list)")
    ap.add_argument("--algos", help="only these algorithms (native is always kept)")
    ap.add_argument("--nodes", help="only these node counts")
    ap.add_argument("--ppn", help="only these ranks-per-node values")
    ap.add_argument("--layout", help="only these layouts (world,node,cross,stride:2,random:0.5,group:512,...)")
    ap.add_argument("--buffers", help="host and/or gpu")
    ap.add_argument("--transport", help="only these transports (mpi,cray,mpich,acx,ucp,shmem; ':all' or ':inter' suffix for a non-default --st-peers)")
    ap.add_argument("--gpu-reduce", dest="gpu_reduce", help="for --kind buffers: kernel,mpi,host")
    ap.add_argument("--op", help="only these reduction ops")
    ap.add_argument("--type", help="only these datatypes")
    ap.add_argument("--sizes", help="message sizes for scaling/ppn/random panels (bytes, comma list)")
    ap.add_argument("--max-cols", type=int, default=4, help="panels per row (default 4)")
    ap.add_argument("--max-panels", type=int, default=12, help="algorithm panels in layout/tunables figures")
    a = ap.parse_args()
    a.formats = a.formats.split(",")
    kinds = (["latency", "best", "scaling", "layout", "random", "tunables", "ppn", "buffers", "transport"]
             if a.kind == "all" else a.kind.split(","))

    theme()
    df = apply_filters(load(a.inputs), a)
    if df.empty:
        sys.exit("no data left after filtering")
    print(f"{len(df)} measurements: {df['collective'].nunique()} collectives, nodes "
          f"{sorted(int(x) for x in df['nodes'].unique())}, ppn {sorted(int(x) for x in df['ranks_per_node'].unique())}, "
          f"buffers {sorted(df['buffers'].unique())}, layouts {sorted(df['layoutname'].unique(), key=layout_key)}, "
          f"studies {sorted(df['study'].unique())}")
    total = 0
    for k in kinds:
        if k == "latency":
            made = fig_latency(df, a)
        elif k == "best":
            made = fig_best(df, a)
        elif k == "scaling":
            made = fig_vs(df, a, "nodes", "nodes", "scaling")
        elif k == "ppn":
            made = fig_vs(df, a, "ranks_per_node", "processes per node", "ppn")
        elif k == "layout":
            made = fig_layout(df, a)
        elif k == "buffers":
            made = fig_buffers(df, a)
        elif k == "transport":
            made = fig_transport(df, a)
        elif k == "random":
            made = fig_random(df, a)
        elif k == "tunables":
            made = fig_tunables(df, a)
        else:
            sys.exit(f"unknown --kind {k}")
        print(f"{k:8s}: {len(made)} figure(s) in {os.path.join(a.out, k)}")
        total += len(made)
    print(f"done: {total} figures" + (f", table view: {os.path.join(a.out, 'best_algorithms.csv')}"
                                     if "best" in kinds else ""))


if __name__ == "__main__":
    main()
