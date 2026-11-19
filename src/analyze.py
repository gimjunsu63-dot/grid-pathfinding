"""Turn results/*.csv into charts and summary tables.

    python3 src/analyze.py                 # reads results/, writes figures/charts/ + figures/RESULTS.md
    python3 src/analyze.py --results DIR   # read another results folder

Everything here is computed from the raw CSV rows written by bench.py, so the
figures can always be rebuilt from the data. Medians and quartiles are taken
over every run of a condition (maps x repetitions).

Derived measures (definitions used in the report):
  time per expanded cell   time_ms / expanded, in microseconds
  expanded share           expanded / reachable cells
  A* work ratio            A* expanded / Dijkstra expanded on the same map
  heuristic accuracy       manhattan(start, goal) / optimal cost  (h at the start)
  cost ratio               path cost / optimal cost
  slope                    least-squares slope of log(median) vs log(cells)
"""

import argparse
import ast
import csv
import inspect
import json
import math
import statistics
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bidirectional  # noqa: E402
import jps  # noqa: E402
import landmarks  # noqa: E402
import grid as grid_mod  # noqa: E402
import search  # noqa: E402
from svgchart import ALGO_COLORS, INK_2, MUTED, LinePlot, bar_chart, bar_panels, tint  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAIN = ["bfs", "dijkstra", "astar"]
NAMES = {"bfs": "BFS", "dijkstra": "Dijkstra", "astar": "A*", "bidir": "Bidirectional", "jps": "JPS"}
HQ_MAPS = {"random": "random 20%, cost 1-9", "maze": "maze", "city": "city, jams 20%"}


# ---------------------------------------------------------------------------
# loading and summaries
# ---------------------------------------------------------------------------


def load(results_dir, exp):
    path = results_dir / f"{exp}.csv"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["params"] = json.loads(r["params"])
            for k in ("size", "map_index", "seed", "attempts", "open_cells", "reachable",
                      "manhattan", "rep", "order", "found", "expanded", "stale", "max_frontier"):
                r[k] = int(r[k])
            r["time_ms"] = float(r["time_ms"])
            r["cost"] = int(r["cost"]) if r["cost"] else None
            r["optimal_cost"] = int(r["optimal_cost"]) if r["optimal_cost"] else None
            r["h_start"] = int(r["h_start"]) if r.get("h_start") else None
            r["scanned"] = int(r["scanned"]) if r.get("scanned") else 0
            r["prep_ms"] = float(r["prep_ms"]) if r.get("prep_ms") else 0.0
            r["h_accuracy"] = float(r["h_accuracy"]) if r.get("h_accuracy") else None
            rows.append(r)
    return rows


def short_label(r):
    p = r["params"]
    kind = r["kind"]
    if kind == "random":
        s = f"random {round(p.get('wall_ratio', 0) * 100)}%"
    elif kind == "maze":
        s = "maze + loops" if p.get("loop_ratio") else "maze"
    elif kind == "nopath":
        s = "no path"
    elif kind == "uwall":
        s = "U wall"
    elif kind == "city" and p.get("local_cost") == 1 and p.get("weighted"):
        return f"jams {round(p.get('jam_ratio', 0) * 100)}%"
    else:
        s = kind
    if kind != "terrain" and p.get("weighted"):
        s += "\ncost 1-9"
    if r["endpoints"] == "random":
        s += "\nrandom ends"
    return s


def quartiles(values):
    if len(values) < 2:
        v = values[0]
        return v, v, v
    q = statistics.quantiles(values, n=4)
    return q[0], statistics.median(values), q[2]


def summarize(rows):
    """condition -> algorithm -> stats, keeping CSV order."""
    groups = defaultdict(lambda: defaultdict(list))
    first = {}
    for r in rows:
        groups[r["condition"]][r["algorithm"]].append(r)
        first.setdefault(r["condition"], r)
    out = {}
    for cond, algs in groups.items():
        out[cond] = {}
        for alg, rs in algs.items():
            t = [r["time_ms"] for r in rs]
            q1, med, q3 = quartiles(t)
            ratios = [r["cost"] / r["optimal_cost"] for r in rs if r["cost"] and r["optimal_cost"]]
            per_cell = [r["time_ms"] * 1000 / r["expanded"] for r in rs if r["expanded"]]
            out[cond][alg] = dict(
                time=med, time_q1=q1, time_q3=q3,
                expanded=statistics.median(r["expanded"] for r in rs),
                share=statistics.median(r["expanded"] / r["reachable"] for r in rs if r["reachable"]),
                stale=statistics.median(r["stale"] for r in rs),
                frontier=statistics.median(r["max_frontier"] for r in rs),
                scanned=statistics.median(r["scanned"] for r in rs),
                cost_ratio=statistics.mean(ratios) if ratios else None,
                cost_ratio_max=max(ratios) if ratios else None,
                per_cell=statistics.median(per_cell) if per_cell else None,
                found=statistics.mean(r["found"] for r in rs),
                label=short_label(first[cond]), size=first[cond]["size"], row=first[cond],
            )
    return out


def per_map(rows):
    """(condition, map_index) -> algorithm -> median time, expanded, plus map info."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[(r["condition"], r["map_index"])][r["algorithm"]].append(r)
    out = {}
    for key, algs in groups.items():
        out[key] = {alg: dict(time=statistics.median(x["time_ms"] for x in rs),
                              expanded=rs[0]["expanded"], row=rs[0])
                    for alg, rs in algs.items()}
    return out


def linear_fit(xs, ys):
    """Least squares y = a + b x. Returns (a, b, r squared)."""
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys))
    return a, b, 1 - ss_res / ss_tot if ss_tot else 1.0


def slope(xs, ys):
    lx = [math.log10(x) for x in xs]
    ly = [math.log10(y) for y in ys]
    mx, my = statistics.mean(lx), statistics.mean(ly)
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den if den else float("nan")


# ---------------------------------------------------------------------------
# tables (markdown)
# ---------------------------------------------------------------------------


def md_table(headers, rows, align=None):
    align = align or ["---"] + ["---:"] * (len(headers) - 1)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def condition_table(summary, algs, names=None):
    names = names or {}
    rows = []
    sizes = {next(iter(b.values()))["size"] for b in summary.values()}
    for cond, by_alg in summary.items():
        for alg in algs:
            if alg not in by_alg:
                continue
            s = by_alg[alg]
            cr = "-" if s["cost_ratio"] is None else f"{s['cost_ratio']:.3f}"
            label = s["label"].replace("\n", ", ")
            if len(sizes) > 1:
                label += f", {s['size']}x{s['size']}"
            rows.append([label, names.get(alg, NAMES.get(alg, alg)),
                         f"{s['time']:.2f} [{s['time_q1']:.2f}-{s['time_q3']:.2f}]",
                         f"{s['expanded']:,.0f}", f"{s['share']:.0%}", f"{s['stale']:,.0f}",
                         f"{s['frontier']:,.0f}", cr,
                         "-" if s["per_cell"] is None else f"{s['per_cell']:.2f}"])
    return md_table(["condition", "algorithm", "time ms, median [IQR]", "expanded", "of reachable",
                     "stale", "max frontier", "cost / optimal", "us per cell"], rows,
                    ["---", "---"] + ["---:"] * 7)


# ---------------------------------------------------------------------------
# code size
# ---------------------------------------------------------------------------


def code_lines(fn):
    """Lines of code in a function: no blank lines, comments or docstring."""
    src = textwrap.dedent(inspect.getsource(fn))
    node = ast.parse(src).body[0]
    skip = set()
    first = node.body[0] if node.body else None
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
            and isinstance(first.value.value, str):
        skip = set(range(first.lineno, first.end_lineno + 1))
    return sum(1 for i, line in enumerate(src.splitlines(), 1)
               if line.strip() and not line.strip().startswith("#") and i not in skip)


def code_size_table():
    shared = code_lines(grid_mod.Grid.neighbors) + code_lines(grid_mod.reconstruct_path) + \
        code_lines(search._trivial)
    bf = code_lines(search.best_first)
    rows = [
        ["BFS", "`bfs`", code_lines(search.bfs)],
        ["Dijkstra", "`best_first` + `dijkstra`", bf + code_lines(search.dijkstra)],
        ["A*", "`best_first` + `astar` (+ `manhattan`)",
         bf + code_lines(search.astar) + code_lines(grid_mod.manhattan)],
        ["Bidirectional Dijkstra", "`bidirectional_dijkstra`",
         code_lines(bidirectional.bidirectional_dijkstra)],
        ["JPS", "`jump_point_search` + `_expand_path`",
         code_lines(jps.jump_point_search) + code_lines(jps._expand_path)],
        ["A* + landmarks", "`best_first` + `LandmarkHeuristic` + `choose_landmarks` + `distances_from`",
         bf + code_lines(landmarks.LandmarkHeuristic.__call__) + code_lines(landmarks.choose_landmarks)
         + code_lines(landmarks.distances_from)],
        ["shared by all", "`Grid.neighbors`, `reconstruct_path`, `_trivial`", shared],
    ]
    note = ("Dijkstra and A* share `best_first`, which also carries the options used in the "
            "extra experiments (tie-break, weight, heap choice), so their counts include those lines.")
    return md_table(["algorithm", "code", "lines"], rows, ["---", "---", "---:"]) + "\n\n" + note


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------


def algo_series(summary, algs, key, conds=None, lo_key=None, hi_key=None, scale=1.0):
    conds = conds or list(summary)
    out = []
    for a in algs:
        vals = [summary[c][a][key] * scale if a in summary[c] else None for c in conds]
        s = dict(label=NAMES.get(a, a), color=ALGO_COLORS.get(a, MUTED), values=vals)
        if lo_key:
            s["lo"] = [summary[c][a][lo_key] * scale if a in summary[c] else None for c in conds]
            s["hi"] = [summary[c][a][hi_key] * scale if a in summary[c] else None for c in conds]
        out.append(s)
    return out


def chart_size(summary, out, slopes):
    conds = list(summary)
    cells = [summary[c]["bfs"]["size"] ** 2 for c in conds]
    for key, fname, ylabel, title in (
        ("time", "A_time.svg", "time (ms, log)", "Search time vs map size"),
        ("expanded", "A_expanded.svg", "expanded cells (log)", "Expanded cells vs map size"),
    ):
        p = LinePlot(title, "cells in the map (N x N, log)", ylabel, xlog=True, ylog=True,
                     subtitle="Random obstacles 20%, unit cost. Median of 10 maps x 5 runs; bars show IQR.",
                     xticks=cells, xfmt=lambda v: f"{int(round(math.sqrt(v)))}²")
        for a in MAIN:
            ys = [summary[c][a][key] for c in conds]
            sl = slope(cells, ys)
            slopes[(key, a)] = sl
            kw = {}
            if key == "time":
                kw = dict(lo=[summary[c][a]["time_q1"] for c in conds],
                          hi=[summary[c][a]["time_q3"] for c in conds])
            p.line(cells, ys, ALGO_COLORS[a], NAMES[a], end_label=f"{NAMES[a]}  slope {sl:.2f}", **kw)
        (out / fname).write_text(p.render())


def chart_walls(summary, out):
    conds = list(summary)
    xs = [summary[c]["bfs"]["row"]["params"].get("wall_ratio", 0) * 100 for c in conds]
    for key, fname, ylabel, title, scale in (
        ("time", "B_time.svg", "time (ms)", "Search time vs obstacle ratio", 1),
        ("share", "B_share.svg", "expanded cells (% of reachable)", "Share of reachable cells expanded", 100),
    ):
        p = LinePlot(title, "obstacle ratio (%)", ylabel, xticks=xs, xfmt=lambda v: f"{v:g}%",
                     subtitle="200 x 200, unit cost, corner to corner. Median of 10 maps x 5 runs.")
        for a in MAIN:
            kw = {}
            if key == "time":
                kw = dict(lo=[summary[c][a]["time_q1"] for c in conds],
                          hi=[summary[c][a]["time_q3"] for c in conds])
            p.line(xs, [summary[c][a][key] * scale for c in conds], ALGO_COLORS[a], NAMES[a],
                   end_label=NAMES[a], **kw)
        (out / fname).write_text(p.render())


def chart_bars(summary, out, prefix, title_suffix, subtitle, algs=MAIN, log_time=False):
    cats = [summary[c][algs[0]]["label"] for c in summary]
    (out / f"{prefix}_time.svg").write_text(bar_chart(
        f"Search time {title_suffix}", cats,
        algo_series(summary, algs, "time", lo_key="time_q1", hi_key="time_q3"),
        "time (ms" + (", log)" if log_time else ")"), subtitle=subtitle, ylog=log_time))
    (out / f"{prefix}_share.svg").write_text(bar_chart(
        f"Expanded cells {title_suffix}", cats, algo_series(summary, algs, "share", scale=100),
        "expanded (% of reachable)", subtitle=subtitle, ylim=(0, 100)))


def chart_cost(summary, out):
    cats = [summary[c]["bfs"]["label"] for c in summary]
    vals = [(summary[c]["bfs"]["cost_ratio"] - 1) * 100 for c in summary]
    (out / "D_bfs_cost.svg").write_text(bar_chart(
        "How much more expensive is the BFS path?", cats,
        [dict(label="BFS", color=ALGO_COLORS["bfs"], values=vals)],
        "extra cost over optimal (%)", value_labels=True, value_fmt=lambda v: f"+{v:.0f}%",
        subtitle="Mean over 10 maps. Dijkstra and A* are always optimal (0%).", width=680))


def chart_decomposition(maps, out):
    p = LinePlot("Time = expanded cells x time per cell", "expanded cells (log)", "time (ms, log)",
                 subtitle="Every map of experiments A-E (median of 5 runs). Each band is one algorithm.",
                 xlog=True, ylog=True)
    for a in MAIN:
        pts = [(v[a]["expanded"], v[a]["time"]) for v in maps.values() if a in v]
        p.scatter([x for x, _ in pts], [y for _, y in pts], ALGO_COLORS[a], NAMES[a], r=3, opacity=0.6)
    (out / "decomposition.svg").write_text(p.render())


def chart_per_cell(per_cell, out):
    cats = [c for c in ("unit cost", "cost 1-9") if all(per_cell.get((a, c)) for a in MAIN)]
    series = []
    for a in MAIN:
        vals, lo, hi = [], [], []
        for c in cats:
            q1, med, q3 = quartiles(per_cell[(a, c)])
            vals.append(med)
            lo.append(q1)
            hi.append(q3)
        series.append(dict(label=NAMES[a], color=ALGO_COLORS[a], values=vals, lo=lo, hi=hi))
    (out / "per_cell.svg").write_text(bar_chart(
        "Time per expanded cell", cats, series, "microseconds per cell",
        subtitle="All runs of experiments A-E. Median with IQR.", width=560,
        value_labels=True, value_fmt=lambda v: f"{v:.2f}"))


def chart_heuristic(maps, out):
    pts, groups = [], defaultdict(list)
    for v in maps.values():
        if "astar" not in v or "dijkstra" not in v:
            continue
        r = v["astar"]["row"]
        if not r["optimal_cost"]:
            continue
        x = r["manhattan"] / r["optimal_cost"]
        y = v["astar"]["expanded"] / v["dijkstra"]["expanded"] * 100
        lab = short_label(r).replace("\n", ", ")
        pts.append((x, y, lab))
        groups[lab].append((x, y))
    p = LinePlot("A* saves work only when the heuristic is accurate",
                 "heuristic accuracy at the start: Manhattan distance / optimal cost",
                 "A* expanded cells (% of Dijkstra)", xlim=(0, 1), ylim=(0, 100),
                 subtitle="One dot per map, experiments A-E (no-path maps excluded).")
    p.scatter([x for x, _, _ in pts], [y for _, y, _ in pts], ALGO_COLORS["astar"], r=3.5, opacity=0.55,
              titles=[f"{lab}: accuracy {x:.2f}, {y:.0f}%" for x, y, lab in pts])
    placed = []
    for lab, xy in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        mx = statistics.median(x for x, _ in xy)
        my = statistics.median(y for _, y in xy)
        if any(abs(mx - a) < 0.08 and abs(my - b) < 7 for a, b in placed):
            continue
        placed.append((mx, my))
        p.note(mx, my, lab, anchor="middle", dy=-10)
    (out / "heuristic_accuracy.svg").write_text(p.render())


def chart_noise(rows, out):
    by = defaultdict(list)
    for r in rows:
        by[r["algorithm"]].append(r["time_ms"])
    p = LinePlot("How much does the clock wobble?", "", "deviation from median (%)",
                 subtitle="Same 200 x 200 map, each algorithm run 30 times.",
                 xticks=[1, 2, 3], xfmt=lambda v: NAMES[MAIN[int(v) - 1]] if 1 <= v <= 3 else "",
                 xlim=(0.5, 3.5), width=560)
    for i, a in enumerate(MAIN, 1):
        med = statistics.median(by[a])
        ys = [(t / med - 1) * 100 for t in by[a]]
        xs = [i + ((k % 7) - 3) * 0.05 for k in range(len(ys))]
        p.scatter(xs, ys, ALGO_COLORS[a], r=3.5, opacity=0.7)
    p.reference([0.5, 3.5], [0, 0], None)
    (out / "noise.svg").write_text(p.render())


def chart_tie(summary, out):
    cats = [summary[c]["astar_high_g"]["label"] for c in summary]
    colors = {"astar_high_g": ALGO_COLORS["astar"], "astar_low_g": INK_2, "astar_fifo": tint(MUTED, 0.35)}
    labels = {"astar_high_g": "larger g first (default)", "astar_low_g": "smaller g first",
              "astar_fifo": "insertion order"}
    series = [dict(label=labels[a], color=colors[a], values=[summary[c][a]["expanded"] for c in summary])
              for a in colors]
    (out / "T_expanded.svg").write_text(bar_chart(
        "A*: how ties in f are broken changes the work", cats, series, "expanded cells (log)",
        ylog=True, subtitle="Same maps, same optimal cost; only the order among equal f differs."))


def _pct(x):
    """+0.6% for small values, +13% for larger ones (keeps labels narrow)."""
    v = x * 100
    return f"+{v:.1f}%" if v < 10 else f"+{v:.0f}%"


def chart_weights(summary, out):
    panels = []
    for c, by in summary.items():
        d = by["dijkstra"]["expanded"]
        names = [a for a in by if a.startswith("astar_w")]
        panels.append(dict(
            title=by["dijkstra"]["label"].replace("\n", ", "),
            categories=[a.replace("astar_w", "x") for a in names],
            values=[by[a]["expanded"] / d * 100 for a in names],
            labels=[_pct(by[a]["cost_ratio"] - 1) for a in names],
        ))
    (out / "W_tradeoff.svg").write_text(bar_panels(
        "Weighted A*: less work, slightly longer paths", panels, "expanded (% of Dijkstra)",
        subtitle="Bars: A* expanded cells as % of Dijkstra. Labels: extra path cost over optimal (mean).",
        width=1000))


def chart_heap(summary, out):
    cats = [summary[c]["dijkstra_heapq"]["label"] + f"\n{summary[c]['dijkstra_heapq']['size']}²"
            for c in summary]
    spec = [("dijkstra_heapq", "Dijkstra, heapq", ALGO_COLORS["dijkstra"]),
            ("dijkstra_python", "Dijkstra, Python heap", tint(ALGO_COLORS["dijkstra"], 0.55)),
            ("astar_heapq", "A*, heapq", ALGO_COLORS["astar"]),
            ("astar_python", "A*, Python heap", tint(ALGO_COLORS["astar"], 0.55))]
    series = [dict(label=lab, color=col, values=[summary[c][a]["time"] for c in summary],
                   lo=[summary[c][a]["time_q1"] for c in summary], hi=[summary[c][a]["time_q3"] for c in summary])
              for a, lab, col in spec]
    (out / "H_time.svg").write_text(bar_chart(
        "heapq (C) vs a binary heap written in Python", cats, series, "time (ms, log)", ylog=True,
        subtitle="Same search code and the same expanded cells; only the priority queue differs."))


def chart_bidir(summary, out):
    cats = [summary[c]["dijkstra"]["label"] for c in summary]
    algs = ["dijkstra", "astar", "bidir"]
    (out / "BI_expanded.svg").write_text(bar_chart(
        "Bidirectional Dijkstra: expanded cells", cats, algo_series(summary, algs, "expanded"),
        "expanded cells (log)", ylog=True, width=900,
        subtitle="200 x 200. Median of 10 maps x 5 runs."))
    (out / "BI_time.svg").write_text(bar_chart(
        "Bidirectional Dijkstra: search time", cats,
        algo_series(summary, algs, "time", lo_key="time_q1", hi_key="time_q3"),
        "time (ms, log)", ylog=True, width=900, subtitle="200 x 200. Median of 10 maps x 5 runs; whiskers show IQR."))


def chart_moves8(summary, out):
    cats = [summary[c]["dijkstra"]["label"] for c in summary]
    algs = ["bfs", "dijkstra", "astar", "bidir", "jps"]
    sub = "8-directional moves, 200 x 200. JPS runs only where every cost is 1."
    (out / "O_expanded.svg").write_text(bar_chart(
        "8 directions: cells taken off the heap", cats, algo_series(summary, algs, "expanded"),
        "expanded cells (log)", ylog=True, width=1000, subtitle=sub + " For JPS these are jump points."))
    (out / "O_time.svg").write_text(bar_chart(
        "8 directions: search time", cats,
        algo_series(summary, algs, "time", lo_key="time_q1", hi_key="time_q3"),
        "time (ms, log)", ylog=True, width=1000, subtitle=sub + " Whiskers show IQR."))


def chart_traffic(summary, out):
    conds = list(summary)
    xs = list(range(len(conds)))
    labels = {i: summary[c]["dijkstra"]["label"].replace("jams ", "") for i, c in enumerate(conds)}
    fmt = lambda v: labels.get(int(round(v)), "") if abs(v - round(v)) < 1e-9 else ""  # noqa: E731
    p = LinePlot("Traffic jams: when does A* lose its lead?", "share of jammed road cells (not to scale)",
                 "time (ms)", xticks=xs, xfmt=fmt, xlim=(-0.3, len(xs) - 0.7),
                 subtitle="City 200 x 200, every road costs 1, a jammed cell costs 9. Median of 10 maps x 5 runs.")
    for a in ["bfs", "dijkstra", "astar", "bidir"]:
        p.line(xs, [summary[c][a]["time"] for c in conds], ALGO_COLORS[a], NAMES[a], end_label=NAMES[a],
               lo=[summary[c][a]["time_q1"] for c in conds], hi=[summary[c][a]["time_q3"] for c in conds])
    (out / "J_time.svg").write_text(p.render())
    q = LinePlot("Why: jams make the Manhattan estimate inaccurate", "share of jammed road cells (not to scale)",
                 "percent", xticks=xs, xfmt=fmt, xlim=(-0.3, len(xs) - 0.7), ylim=(0, 100),
                 subtitle="Both lines are medians over the same maps.")
    work, acc = [], []
    for c in conds:
        rows = [r["row"] for r in [summary[c]["astar"]]]
        work.append(summary[c]["astar"]["expanded"] / summary[c]["dijkstra"]["expanded"] * 100)
        r = rows[0]
        acc.append(r["manhattan"] / r["optimal_cost"] * 100 if r["optimal_cost"] else 0)
    q.line(xs, work, ALGO_COLORS["astar"], "A* cells as % of Dijkstra", end_label="A* work")
    q.line(xs, acc, MUTED, "heuristic accuracy (Manhattan / optimal cost)", end_label="accuracy", dashed=True)
    (out / "J_why.svg").write_text(q.render())


# ---------------------------------------------------------------------------
# heuristic quality (HQ)
# ---------------------------------------------------------------------------


def hq_table(rows):
    """map kind -> algorithm -> medians over maps: accuracy, work (% of Dijkstra cells), time, prep."""
    per_map = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_map[(r["kind"], r["map_index"])][r["algorithm"]].append(r)
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (kind, _), algs in per_map.items():
        d_cells = algs["dijkstra"][0]["expanded"]
        d_time = statistics.median(x["time_ms"] for x in algs["dijkstra"])
        for alg, rs in algs.items():
            o = out[kind][alg]
            o["accuracy"].append(rs[0]["h_accuracy"] or 0.0)
            o["work"].append(rs[0]["expanded"] / d_cells * 100 if d_cells else 0)
            o["time"].append(statistics.median(x["time_ms"] for x in rs))
            o["time_rel"].append(statistics.median(x["time_ms"] for x in rs) / d_time * 100)
            o["prep"].append(rs[0]["prep_ms"])
            o["per_cell"].append(statistics.median(x["time_ms"] * 1000 / x["expanded"] for x in rs if x["expanded"]))
            o["cost_ratio"].append(rs[0]["cost"] / rs[0]["optimal_cost"] if rs[0]["optimal_cost"] else 1)
    return {k: {a: {m: statistics.median(v) for m, v in d.items()} for a, d in algs.items()}
            for k, algs in out.items()}


def _hq_order(algs):
    rulers = sorted((a for a in algs if a.startswith("ruler")), key=lambda a: float(a[5:]))
    alts = sorted((a for a in algs if a.startswith("alt")), key=lambda a: int(a[3:]))
    return rulers, alts


def chart_hq_curves(table, out):
    names = []
    for kind, algs in table.items():
        rulers, alts = _hq_order(algs)
        p = LinePlot(f"Accuracy vs work: {HQ_MAPS.get(kind, kind)}",
                     "heuristic accuracy (mean of estimate / real remaining cost)",
                     "A* expanded cells (% of Dijkstra, log)", ylog=True, xlim=(0, 1.55),
                     xticks=[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5], xfmt=lambda v: f"{v:.0%}" if v else "0",
                     yticks=[0.5, 1, 2, 5, 10, 20, 50, 100], ylim=(0.5, 100), width=720,
                     subtitle="Gray: the ruler h = k x real cost (not a method). Medians of 10 maps, random start and goal.")
        rx = [0.0] + [algs[a]["accuracy"] for a in rulers]
        ry = [100.0] + [max(algs[a]["work"], 0.5) for a in rulers]
        p.line(rx, ry, MUTED, "ruler: k x real remaining cost", dashed=True)
        for a, x, y in zip(rulers, rx[1:], ry[1:]):
            if a in ("ruler0.5", "ruler0.9", "ruler1", "ruler1.5"):
                p.note(x, y, f"k={a[5:]}", anchor="start", dy=-9)
        p.scatter([0.0], [100.0], ALGO_COLORS["dijkstra"], "Dijkstra (h = 0)", r=6, opacity=1)
        m = algs["astar"]
        p.scatter([m["accuracy"]], [max(m["work"], 0.5)], ALGO_COLORS["astar"], "A* + Manhattan", r=6, opacity=1)
        p.note(m["accuracy"], max(m["work"], 0.5), "Manhattan", anchor="start", dy=-10)
        p.scatter([algs[a]["accuracy"] for a in alts], [max(algs[a]["work"], 0.5) for a in alts],
                  ALGO_COLORS["alt"], "A* + landmarks", r=6, opacity=1,
                  titles=[f"{a[3:]} landmarks: accuracy {algs[a]['accuracy']:.0%}, {algs[a]['work']:.1f}%" for a in alts])
        for a in alts:
            if a in ("alt1", "alt4", "alt16"):
                p.note(algs[a]["accuracy"] - 0.025, max(algs[a]["work"], 0.5), f"{a[3:]} LM", anchor="end", dy=4)
        p.reference([1.0, 1.0], [0.5, 100], "perfect")
        name = f"HQ_curve_{kind}.svg"
        (out / name).write_text(p.render())
        names.append(name)
    return names


def chart_hq_time(table, out):
    panels = []
    for kind, algs in table.items():
        _, alts = _hq_order(algs)
        order = ["dijkstra", "astar"] + alts
        panels.append(dict(
            title=HQ_MAPS.get(kind, kind),
            categories=["Dij", "Man"] + [a[3:] for a in alts],
            values=[algs[a]["time"] for a in order],
            labels=[f"{algs[a]['accuracy']:.0%}" for a in order],
            colors=[ALGO_COLORS["dijkstra"], ALGO_COLORS["astar"]] + [ALGO_COLORS["alt"]] * len(alts),
        ))
    (out / "HQ_time.svg").write_text(bar_panels(
        "Query time: Dijkstra, Manhattan A*, A* with 1-16 landmarks", panels, "search time (ms, median)",
        subtitle="Labels above bars: heuristic accuracy. Precomputation is not included here (see next chart).",
        width=1000, height=360))


def chart_hq_cost(table, out, slopes):
    counts = None
    p = LinePlot("Landmark precomputation: a fixed part + one Dijkstra per landmark", "landmarks",
                 "precomputation (ms)", xticks=[0, 1, 2, 4, 8, 16], xlim=(0, 16), width=640,
                 subtitle="Dashed: least-squares line a + b x landmarks. A straight line = linear in the number of landmarks.")
    q = LinePlot("Each landmark adds work to every estimate", "landmarks", "time per expanded cell (us)",
                 xticks=[0, 1, 2, 4, 8, 16], xlim=(0, 16), width=640,
                 subtitle="A* computes two bounds per landmark for every cell it looks at.")
    tints = [0.0, 0.35, 0.6]
    for i, (kind, algs) in enumerate(table.items()):
        _, alts = _hq_order(algs)
        counts = [int(a[3:]) for a in alts]
        color = tint(ALGO_COLORS["alt"], tints[i % len(tints)])
        prep = [algs[a]["prep"] for a in alts]
        a_fit, b_fit, r2 = linear_fit(counts, prep)
        slopes[kind] = (a_fit, b_fit, r2)
        p.line(counts, prep, color, HQ_MAPS.get(kind, kind), end_label=f"{b_fit:.0f} ms / landmark")
        p.reference([0, 16], [a_fit, a_fit + 16 * b_fit], None)
        q.line(counts, [algs[a]["per_cell"] for a in alts], color, HQ_MAPS.get(kind, kind),
               end_label=HQ_MAPS.get(kind, kind).split(",")[0])
    (out / "HQ_prep.svg").write_text(p.render())
    (out / "HQ_per_cell.svg").write_text(q.render())


def hq_section_table(table):
    rows = []
    for kind, algs in table.items():
        rulers, alts = _hq_order(algs)
        d_time = algs["dijkstra"]["time"]
        for a in ["dijkstra", "astar"] + alts + rulers:
            g = algs[a]
            if a.startswith("alt"):
                label = f"A* + {a[3:]} landmarks"
            elif a.startswith("ruler"):
                label = f"ruler k = {a[5:]}"
            else:
                label = {"dijkstra": "Dijkstra", "astar": "A* + Manhattan"}[a]
            saved = d_time - g["time"]
            even = f"{g['prep'] / saved:.1f}" if g["prep"] and saved > 0 and a.startswith("alt") else "-"
            rows.append([HQ_MAPS.get(kind, kind), label, f"{g['accuracy']:.0%}", f"{g['work']:.1f}%",
                         f"{g['time']:.2f}", f"{g['prep']:.1f}" if g["prep"] else "-", even,
                         f"{g['cost_ratio']:.3f}"])
    return md_table(["map", "heuristic", "accuracy", "cells vs Dijkstra", "time ms", "precompute ms",
                     "queries to pay off (vs Dijkstra)", "cost / optimal"], rows,
                    ["---", "---"] + ["---:"] * 6)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def environment_line(results_dir):
    for exp in ("A", "noise"):
        p = results_dir / f"{exp}.env.json"
        if p.exists():
            e = json.loads(p.read_text())["environment"]
            where = "GitHub Codespaces" if e.get("codespaces") else e.get("platform")
            return (f"Measured on {where}: {e['cpu_model']}, {e['cpu_count']} CPUs, "
                    f"Python {e['python']}, commit `{e['git_commit']}`"
                    + (" (with uncommitted changes)" if e.get("git_dirty") else "") + ".")
    return ""


def analyze(results_dir, out_dir, log=print):
    charts = out_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    data = {exp: load(results_dir, exp)
            for exp in ("A", "B", "C", "D", "E", "T", "W", "H", "BI", "O", "J", "HQ", "noise")}
    md = ["# Experiment results", "",
          "Generated by `make analyze` from the raw data in `results/`. "
          "Times are medians over 10 maps x 5 runs with the interquartile range (IQR) in brackets.", ""]
    env = environment_line(results_dir)
    if env:
        md += [env, ""]

    def section(title, images, table, text=None):
        md.extend([f"## {title}", ""])
        if text:
            md.extend([text, ""])
        for img in images:
            md.extend([f"![{img}](charts/{img})", ""])
        if table:
            md.extend([table, ""])

    slopes = {}
    if data["A"]:
        s = summarize(data["A"])
        chart_size(s, charts, slopes)
        slope_rows = [[NAMES[a], f"{slopes[('time', a)]:.2f}", f"{slopes[('expanded', a)]:.2f}"] for a in MAIN]
        section("A. Map size", ["A_time.svg", "A_expanded.svg"],
                md_table(["algorithm", "slope of time", "slope of expanded"], slope_rows) + "\n\n"
                + condition_table(s, MAIN),
                "Slope on the log-log chart: 1 means time grows in proportion to the number of cells.")
    if data["B"]:
        s = summarize(data["B"])
        chart_walls(s, charts)
        section("B. Obstacle ratio", ["B_time.svg", "B_share.svg"], condition_table(s, MAIN))
    if data["C"]:
        s = summarize(data["C"])
        chart_bars(s, charts, "C", "by map kind", "200 x 200, unit cost. Median of 10 maps x 5 runs; whiskers show IQR.")
        section("C. Map kind", ["C_time.svg", "C_share.svg"], condition_table(s, MAIN))
    if data["D"]:
        s = summarize(data["D"])
        chart_bars(s, charts, "D", "by cost model", "200 x 200. Same walls for unit and 1-9 costs.")
        chart_cost(s, charts)
        section("D. Cost model", ["D_time.svg", "D_share.svg", "D_bfs_cost.svg"], condition_table(s, MAIN))
    if data["E"]:
        s = summarize(data["E"])
        chart_bars(s, charts, "E", "by endpoints", "200 x 200, random obstacles 20%.")
        section("E. Fixed vs random endpoints", ["E_time.svg", "E_share.svg"], condition_table(s, MAIN))

    main_rows = [r for e in "ABCDE" for r in data[e]]
    if main_rows:
        maps = per_map(main_rows)
        chart_decomposition(maps, charts)
        per_cell = defaultdict(list)
        for r in main_rows:
            if r["expanded"]:
                weighted = r["kind"] == "terrain" or r["params"].get("weighted")
                per_cell[(r["algorithm"], "cost 1-9" if weighted else "unit cost")].append(
                    r["time_ms"] * 1000 / r["expanded"])
        chart_per_cell(per_cell, charts)
        rows = []
        for a in MAIN:
            u, w = (per_cell.get((a, k)) for k in ("unit cost", "cost 1-9"))
            rows.append([NAMES[a], f"{statistics.median(u):.2f}" if u else "-",
                         f"{statistics.median(w):.2f}" if w else "-"])
        section("Time decomposition", ["decomposition.svg", "per_cell.svg"],
                md_table(["algorithm", "us per cell, unit cost", "us per cell, cost 1-9"], rows))
        chart_heuristic(maps, charts)
        section("Heuristic accuracy", ["heuristic_accuracy.svg"], None,
                "Accuracy 1 means the Manhattan distance equals the real remaining cost; "
                "with costs 1-9 it is about 1/5, and A* expands almost as much as Dijkstra.")
    if data["noise"]:
        chart_noise(data["noise"], charts)
        rows = []
        by = defaultdict(list)
        for r in data["noise"]:
            by[r["algorithm"]].append(r["time_ms"])
        for a in MAIN:
            q1, med, q3 = quartiles(by[a])
            rows.append([NAMES[a], f"{med:.2f}", f"{(q3 - q1) / med:.1%}", f"{min(by[a]):.2f}", f"{max(by[a]):.2f}"])
        section("Measurement noise", ["noise.svg"],
                md_table(["algorithm", "median ms", "IQR / median", "min", "max"], rows))
    if data["T"]:
        s = summarize(data["T"])
        chart_tie(s, charts)
        section("T. A* tie-breaking", ["T_expanded.svg"],
                condition_table(s, ["astar_high_g", "astar_low_g", "astar_fifo"],
                                {"astar_high_g": "larger g first", "astar_low_g": "smaller g first",
                                 "astar_fifo": "insertion order"}))
    if data["W"]:
        s = summarize(data["W"])
        chart_weights(s, charts)
        algs = list(next(iter(s.values())))
        section("W. Weighted A*", ["W_tradeoff.svg"],
                condition_table(s, algs, {a: a.replace("astar_w", "A* x") for a in algs}))
    if data["H"]:
        s = summarize(data["H"])
        chart_heap(s, charts)
        rows = []
        for c, by in s.items():
            for base in ("dijkstra", "astar"):
                fast, slow = by[f"{base}_heapq"], by[f"{base}_python"]
                rows.append([fast["label"].replace("\n", ", ") + f" {fast['size']}²", NAMES[base],
                             f"{fast['time']:.2f}", f"{slow['time']:.2f}", f"{slow['time'] / fast['time']:.2f}x"])
        section("H. heapq vs plain-Python heap", ["H_time.svg"],
                md_table(["condition", "algorithm", "heapq ms", "Python heap ms", "slower by"], rows))
    if data["BI"]:
        s = summarize(data["BI"])
        chart_bidir(s, charts)
        section("BI. Bidirectional Dijkstra", ["BI_expanded.svg", "BI_time.svg"],
                condition_table(s, ["dijkstra", "astar", "bidir"]))

    if data["O"]:
        s = summarize(data["O"])
        chart_moves8(s, charts)
        jrows = [[by["jps"]["label"].replace("\n", ", "), f"{by['jps']['expanded']:,.0f}", f"{by['jps']['scanned']:,.0f}",
                  f"{by['astar']['expanded']:,.0f}", f"{by['jps']['time']:.2f}", f"{by['astar']['time']:.2f}"]
                 for by in s.values() if "jps" in by]
        section("O. 8-directional moves and Jump Point Search", ["O_expanded.svg", "O_time.svg"],
                md_table(["condition", "JPS jump points", "JPS cells scanned", "A* expanded", "JPS ms", "A* ms"], jrows)
                + "\n\n" + condition_table(s, ["bfs", "dijkstra", "astar", "bidir", "jps"]),
                "Costs are in 8-direction units (straight step 10, diagonal 14). "
                "JPS puts only jump points on the heap but scans many cells in between.")
    if data["J"]:
        s = summarize(data["J"])
        chart_traffic(s, charts)
        section("J. Traffic jams", ["J_time.svg", "J_why.svg"], condition_table(s, ["bfs", "dijkstra", "astar", "bidir"]))

    if data["HQ"]:
        table = hq_table(data["HQ"])
        curves = chart_hq_curves(table, charts)
        chart_hq_time(table, charts)
        hq_slopes = {}
        chart_hq_cost(table, charts, hq_slopes)
        section("HQ. Heuristic quality: Manhattan vs landmarks, measured with the accuracy ruler",
                curves + ["HQ_time.svg", "HQ_prep.svg", "HQ_per_cell.svg"],
                hq_section_table(table),
                "Accuracy = mean over all cells that can reach the goal of (estimate / real remaining cost). "
                "The ruler h = k x real remaining cost needs the answer in advance, so it is a measuring tool, "
                "not a competitor. Random start and goal: landmarks are computed once per map and serve any query. "
                "Queries to pay off = precomputation time / time saved per query compared with Dijkstra. "
                "Precomputation fitted as a + b x landmarks: "
                + ", ".join(f"{HQ_MAPS.get(k, k)}: {a:.0f} ms + {b:.1f} ms per landmark (R² {r2:.3f})"
                            for k, (a, b, r2) in hq_slopes.items()) + ".")

    section("Code size", [], code_size_table(),
            "Lines of code without blank lines, comments and docstrings.")
    (out_dir / "RESULTS.md").write_text("\n".join(md))
    log(f"  {len(list(charts.glob('*.svg')))} charts -> {charts}")
    log(f"  summary -> {out_dir / 'RESULTS.md'}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Draw charts and tables from results/*.csv")
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--out", type=Path, default=ROOT / "figures")
    args = p.parse_args(argv)
    analyze(args.results, args.out)


if __name__ == "__main__":
    main()
