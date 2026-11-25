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
import roads  # noqa: E402
import grid as grid_mod  # noqa: E402
import search  # noqa: E402
from svgchart import (ALGO_COLORS, FONT, INK, INK_2, MUTED, SURFACE, LinePlot, bar_chart,  # noqa: E402
                      bar_panels, tint)

ROOT = Path(__file__).resolve().parents[1]
MAIN = ["bfs", "dijkstra", "astar"]
NAMES = {"bfs": "BFS", "dijkstra": "Dijkstra", "astar": "A*", "bidir": "Bidirectional", "jps": "JPS"}
HQ_MAPS = {"random": "random 20%, cost 1-9", "maze": "maze", "city": "city, jams 20%"}
DY_ALGS = ["dijkstra", "astar", "alt_new", "alt_old"]
DY_NAMES = {"dijkstra": "Dijkstra", "astar": "A* + Manhattan", "alt_new": "landmarks, recomputed",
            "alt_old": "landmarks, old (not recomputed)"}
LP_NAMES = {"farthest": "farthest", "corners": "corners", "random": "random"}
LP_SHORT = {"farthest": "far", "corners": "corner", "random": "rand"}
RM_ALGS = ["dijkstra", "astar", "alt_now", "alt_night", "alt_rush"]
RM_NAMES = {"dijkstra": "Dijkstra", "astar": "A* + straight line", "alt_now": "landmarks measured this hour",
            "alt_night": "landmarks measured at night (3h)", "alt_rush": "landmarks measured at 8h"}
RM_SHORT = {"dijkstra": "Dijkstra", "astar": "A* + straight line", "alt_now": "landmarks (now)",
            "alt_night": "landmarks (from 3h)", "alt_rush": "landmarks (from 8h)"}


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
            r["h_over"] = float(r["h_over"]) if r.get("h_over") else None
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
# dynamic traffic (DY) and landmark placement (LP)
# ---------------------------------------------------------------------------


def dy_change(r):
    """Signed change in percent: +30 = 30% of free road cells jammed, -30 = 30% of jams cleared."""
    p = r["params"]
    return round((p.get("jam_up", 0) - p.get("jam_down", 0)) * 100)


def dy_label(ch):
    return "no change" if ch == 0 else (f"jams +{ch}%" if ch > 0 else f"jams cleared {-ch}%")


def dy_table(rows):
    """change -> algorithm -> medians over maps (time, cells, accuracy, overestimated cells, precompute)
    plus the share of queries answered with a longer path and the worst extra cost."""
    per_map = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_map[(dy_change(r), r["map_index"])][r["algorithm"]].append(r)
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (ch, _), algs in per_map.items():
        for alg, rs in algs.items():
            o = out[ch][alg]
            o["time"].append(statistics.median(x["time_ms"] for x in rs))
            o["cells"].append(rs[0]["expanded"])
            o["accuracy"].append(rs[0]["h_accuracy"] or 0.0)
            o["over"].append(rs[0]["h_over"] or 0.0)
            o["prep"].append(rs[0]["prep_ms"])
            o["extra"].append((rs[0]["cost"] / rs[0]["optimal_cost"] - 1) * 100 if rs[0]["optimal_cost"] else 0)
    table = {}
    for ch in sorted(out):
        table[ch] = {}
        for alg, d in out[ch].items():
            g = {m: statistics.median(v) for m, v in d.items() if m != "extra"}
            g["wrong"] = sum(1 for e in d["extra"] if e > 0) / len(d["extra"]) * 100
            g["worst"] = max(d["extra"])
            g["maps"] = len(d["extra"])
            table[ch][alg] = g
    return table


def _dy_style(alg):
    if alg == "alt_old":
        return tint(ALGO_COLORS["alt"], 0.45), True
    return ALGO_COLORS["alt" if alg == "alt_new" else alg], False


def chart_dy(table, out):
    changes = sorted(table)
    ticks = changes
    xfmt = lambda v: "0" if v == 0 else f"{abs(v):g}%"   # noqa: E731
    xlabel = "<- share of jams cleared            |            share of free road jammed ->"
    sub = ("City 200 x 200 with 20% jams. Old landmarks were measured before the change. "
           "Medians of 30 maps, random start and goal.")
    p = LinePlot("After the traffic changes: query time", xlabel, "search time (ms, median)",
                 subtitle=sub, xticks=ticks, xfmt=xfmt, width=760)
    for alg in DY_ALGS:
        color, dashed = _dy_style(alg)
        p.line(changes, [table[c][alg]["time"] for c in changes], color, DY_NAMES[alg], dashed=dashed)
    (out / "DY_time.svg").write_text(p.render())

    q = LinePlot("After the traffic changes: heuristic accuracy", xlabel,
                 "accuracy (mean of estimate / real remaining cost)", subtitle=sub, xticks=ticks, xfmt=xfmt,
                 width=760, yfmt=lambda v: f"{v:.0%}")
    for alg in ("astar", "alt_new", "alt_old"):
        color, dashed = _dy_style(alg)
        q.line(changes, [table[c][alg]["accuracy"] for c in changes], color, DY_NAMES[alg], dashed=dashed)
    q.reference([changes[0], changes[-1]], [1.0, 1.0], "exact")
    (out / "DY_accuracy.svg").write_text(q.render())

    cats = [dy_label(c).replace("jams cleared ", "cleared\n").replace("jams ", "jams\n") for c in changes]
    old = [table[c]["alt_old"] for c in changes]
    (out / "DY_wrong.svg").write_text(bar_chart(
        "Old landmarks: safe when jams appear, wrong when jams clear", cats, [
            dict(label="cells where the old estimate is too high (%)", color=tint(ALGO_COLORS["alt"], 0.45),
                 values=[g["over"] * 100 for g in old]),
            dict(label="queries answered with a longer path (%)", color=ALGO_COLORS["alt"],
                 values=[g["wrong"] for g in old]),
        ], "% (median cells / share of queries)", value_labels=True, value_fmt=lambda v: f"{v:.0f}",
        subtitle="Recomputed landmarks, Manhattan and Dijkstra: 0% in every condition (checked in every run).",
        ylim=(0, 105), width=820))


def dy_section_table(table):
    rows = []
    for ch, algs in table.items():
        new, old = algs["alt_new"], algs["alt_old"]
        for alg in DY_ALGS:
            g = algs[alg]
            pay = "-"
            if alg == "alt_new" and ch != 0:
                saved = old["time"] - new["time"]
                pay = f"{new['prep'] / saved:.0f}" if saved > 0 else "never (old is as fast)"
            prep = "-" if alg in ("dijkstra", "astar") else ("0 (already paid)" if alg == "alt_old" else f"{g['prep']:.0f}")
            rows.append([dy_label(ch), DY_NAMES[alg], f"{g['time']:.2f}", f"{g['cells']:,.0f}",
                         "-" if alg == "dijkstra" else f"{g['accuracy']:.0%}", f"{g['over']:.0%}",
                         f"{g['wrong']:.0f}%", f"+{g['worst']:.1f}%" if g["worst"] > 0 else "0", prep, pay])
    return md_table(["change", "method", "time ms", "cells", "accuracy", "cells overestimated",
                     "queries with a longer path", "worst extra cost", "precompute after the change (ms)",
                     "queries to pay off recomputing (vs old)"], rows, ["---", "---"] + ["---:"] * 8)


def lp_table(rows):
    """map kind -> algorithm -> medians over maps: time, accuracy, cells, speed-up over Dijkstra."""
    per_map = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_map[(r["kind"], r["map_index"])][r["algorithm"]].append(r)
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (kind, _), algs in per_map.items():
        d_time = statistics.median(x["time_ms"] for x in algs["dijkstra"])
        for alg, rs in algs.items():
            t = statistics.median(x["time_ms"] for x in rs)
            o = out[kind][alg]
            o["time"].append(t)
            o["accuracy"].append(rs[0]["h_accuracy"] or 0.0)
            o["cells"].append(rs[0]["expanded"])
            o["speedup"].append(d_time / t)
    return {k: {a: {m: statistics.median(v) for m, v in d.items()} for a, d in algs.items()}
            for k, algs in out.items()}


def lp_order(algs):
    return [a for n in (4, 8) for pl in LP_NAMES for a in [f"alt{n}_{pl}"] if a in algs]


def chart_lp(table, out):
    shades = {"farthest": 0.0, "corners": 0.35, "random": 0.65}
    panels = []
    for kind, algs in table.items():
        order = lp_order(algs)
        panels.append(dict(
            title=HQ_MAPS.get(kind, kind),
            categories=[f"{LP_SHORT[a.split('_')[1]]} {a[3:].split('_')[0]}" for a in order],
            values=[algs[a]["speedup"] for a in order],
            labels=[f"{algs[a]['accuracy']:.0%}" for a in order],
            colors=[tint(ALGO_COLORS["alt"], shades[a.split("_")[1]]) for a in order],
        ))
    (out / "LP_speedup.svg").write_text(bar_panels(
        "Where to put the landmarks: farthest-point, corners or random", panels,
        "speed-up over Dijkstra (x, median)",
        subtitle="Bars: farthest-point (dark), corners (middle), random (light); 4 then 8 landmarks. "
                 "Labels: heuristic accuracy.", width=1000, height=360))


def lp_section_table(table):
    rows = []
    for kind, algs in table.items():
        for a in lp_order(algs):
            n, pl = a[3:].split("_")
            g = algs[a]
            rows.append([HQ_MAPS.get(kind, kind), n, LP_NAMES[pl], f"{g['accuracy']:.0%}", f"{g['cells']:,.0f}",
                         f"{g['time']:.2f}", f"{g['speedup']:.1f}x"])
    return md_table(["map", "landmarks", "placement", "accuracy", "cells", "time ms", "speed-up over Dijkstra"],
                    rows, ["---", "---:", "---"] + ["---:"] * 4)


# ---------------------------------------------------------------------------
# real road map (RM)
# ---------------------------------------------------------------------------


def rm_table(rows):
    """hour -> algorithm -> medians over queries, plus share of longer paths and the worst extra cost."""
    per_q = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_q[(r["params"]["hour"], r["map_index"])][r["algorithm"]].append(r)
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (hour, _), algs in per_q.items():
        for alg, rs in algs.items():
            o = out[hour][alg]
            o["time"].append(statistics.median(x["time_ms"] for x in rs))
            o["cells"].append(rs[0]["expanded"])
            o["accuracy"].append(rs[0]["h_accuracy"] or 0.0)
            o["over"].append(rs[0]["h_over"] or 0.0)
            o["prep"].append(rs[0]["prep_ms"])
            o["extra"].append((rs[0]["cost"] / rs[0]["optimal_cost"] - 1) * 100 if rs[0]["optimal_cost"] else 0)
    table = {}
    for hour in sorted(out):
        table[hour] = {}
        for alg, d in out[hour].items():
            g = {m: statistics.median(v) for m, v in d.items() if m != "extra"}
            g["wrong"] = sum(1 for e in d["extra"] if e > 0) / len(d["extra"]) * 100
            g["worst"] = max(d["extra"])
            g["queries"] = len(d["extra"])
            table[hour][alg] = g
    return table


def _rm_style(alg):
    return {"dijkstra": (ALGO_COLORS["dijkstra"], False), "astar": (ALGO_COLORS["astar"], False),
            "alt_now": (ALGO_COLORS["alt"], False), "alt_night": (tint(ALGO_COLORS["alt"], 0.45), True),
            "alt_rush": (ALGO_COLORS["jps"], True)}[alg]


def chart_rm(table, out, network=None, prefix="RM"):
    hours = sorted(table)
    yonsei = prefix == "RY"
    who = "Real road map" if not yonsei else "Trips from Yonsei University"
    trips = "30 start/goal pairs" if not yonsei else "30 trips from Yonsei University International Campus"
    sub = (f"Songdo road network ({network.n:,} points, {len(network.eu):,} one-way segments), "
           if network else "Songdo road network, ") + f"traffic model, 8 landmarks. Medians of {trips}."
    p = LinePlot(f"{who}: query time over the day", "hour of the day", "search time (ms, median)",
                 subtitle=sub, xticks=hours, xlim=(min(hours), max(hours)), width=900)
    for alg in RM_ALGS:
        color, dashed = _rm_style(alg)
        p.line(hours, [table[h][alg]["time"] for h in hours], color, RM_SHORT[alg], dashed=dashed)
    (out / f"{prefix}_time.svg").write_text(p.render())

    q = LinePlot(f"{who}: heuristic accuracy over the day", "hour of the day",
                 "accuracy (mean of estimate / real remaining time)", subtitle=sub, xticks=hours,
                 xlim=(min(hours), max(hours)), width=900, yfmt=lambda v: f"{v:.0%}")
    for alg in ("astar", "alt_now", "alt_night", "alt_rush"):
        color, dashed = _rm_style(alg)
        q.line(hours, [table[h][alg]["accuracy"] for h in hours], color, RM_SHORT[alg], dashed=dashed)
    q.reference([min(hours), max(hours)], [1.0, 1.0], "exact")
    (out / f"{prefix}_accuracy.svg").write_text(q.render())

    rush = [table[h]["alt_rush"] for h in hours]
    title = ("Landmarks measured at 8h: wrong at almost every other hour" if not yonsei
             else "Trips from Yonsei: landmarks measured at 8h, used at other hours")
    (out / f"{prefix}_wrong.svg").write_text(bar_chart(
        title, [f"{h}h" for h in hours], [
            dict(label="points where the estimate is too high (%)", color=tint(ALGO_COLORS["jps"], 0.45),
                 values=[g["over"] * 100 for g in rush]),
            dict(label="queries answered with a longer route (%)", color=ALGO_COLORS["jps"],
                 values=[g["wrong"] for g in rush]),
        ], "% (median points / share of queries)", value_labels=True, value_fmt=lambda v: f"{v:.0f}",
        subtitle="Landmarks measured at night (free flow): 0% at every hour (checked in every run).",
        ylim=(0, 105), width=820))


# ---------------------------------------------------------------------------
# landmarks per time band (RB)
# ---------------------------------------------------------------------------

RB_ALGS = ["dijkstra", "alt_now", "alt_night", "alt_band", "alt_band_safe"]
RB_NAMES = {"dijkstra": "Dijkstra", "alt_now": "landmarks measured this hour",
            "alt_night": "landmarks measured at night", "alt_band": "landmarks of the time band",
            "alt_band_safe": "time band with 10% margin"}
RB_SHORT = {"dijkstra": "Dijkstra", "alt_now": "measured now", "alt_night": "measured at night",
            "alt_band": "time band", "alt_band_safe": "time band + 10% margin"}
RB_SETS = {"dijkstra": "-", "alt_now": "24", "alt_night": "1", "alt_band": "4", "alt_band_safe": "4"}
RB_SAFE = {"dijkstra": "always", "alt_now": "if traffic = model", "alt_night": "if traffic = model",
           "alt_band": "if traffic = model", "alt_band_safe": "if traffic within 10% of model"}


def _rb_style(alg):
    return {"dijkstra": (ALGO_COLORS["dijkstra"], False), "alt_now": (ALGO_COLORS["alt"], False),
            "alt_night": (tint(ALGO_COLORS["alt"], 0.5), True), "alt_band": (ALGO_COLORS["jps"], False),
            "alt_band_safe": (tint(ALGO_COLORS["jps"], 0.4), True)}[alg]


def rb_tables(rows):
    """error -> rm_table of the rows with that much traffic error (0.0 = traffic follows the model)."""
    split = defaultdict(list)
    for r in rows:
        split[r["params"].get("error", 0.0)].append(r)
    return {e: rm_table(rs) for e, rs in sorted(split.items())}


def rb_wrong(table):
    """algorithm -> (% of all queries over all hours answered with a longer route, worst extra %)."""
    out = {}
    for alg in RB_ALGS:
        n = sum(table[h][alg]["queries"] for h in table)
        wrong = sum(table[h][alg]["wrong"] * table[h][alg]["queries"] / 100 for h in table)
        out[alg] = (wrong / n * 100, max(table[h][alg]["worst"] for h in table))
    return out


def chart_rb(tables, out):
    for e, table in tables.items():
        hours = sorted(table)
        traffic = "traffic as the model says" if e == 0 else f"real traffic up to {e:.0%} off the model"
        p = LinePlot(f"Landmarks per time band, {traffic}", "hour of the day", "search time (ms, median)",
                     subtitle="Songdo, 8 landmarks. Bands 7-9h, 10-16h, 17-20h, 21-6h. "
                              "Medians of 30 start/goal pairs.",
                     xticks=hours, xlim=(min(hours), max(hours)), width=900)
        for alg in RB_ALGS:
            color, dashed = _rb_style(alg)
            p.line(hours, [table[h][alg]["time"] for h in hours], color, RB_SHORT[alg], dashed=dashed)
        (out / ("RB_time.svg" if e == 0 else "RB_time_off.svg")).write_text(p.render())
    algs = RB_ALGS[1:]
    series = []
    for e, shade in zip(sorted(tables), (0.55, 0.0)):
        w = rb_wrong(tables[e])
        label = "traffic = model" if e == 0 else f"traffic up to {e:.0%} off the model"
        series.append(dict(label=label, color=tint(ALGO_COLORS["alt"], shade), values=[w[a][0] for a in algs]))
    (out / "RB_wrong.svg").write_text(bar_chart(
        "When real traffic is not the model: wrong answers", [RB_SHORT[a] for a in algs], series,
        "queries answered with a longer route (%)", value_labels=True, value_fmt=lambda v: f"{v:.1f}",
        subtitle="All 12 hours x 30 start/goal pairs. Landmarks are always measured on the model.", width=760))


def rb_section_table(tables):
    exact, off = tables.get(0.0), tables[max(tables)]
    peak = 8 if 8 in off else max(off)
    we, wo = (rb_wrong(exact) if exact else {}), rb_wrong(off)
    rows = []
    for alg in RB_ALGS:
        g = off[peak][alg]
        prep = f"{g['prep']:.0f}" if alg.startswith("alt") else "-"
        rows.append([RB_NAMES[alg], RB_SAFE[alg], RB_SETS[alg],
                     f"{exact[peak][alg]['time']:.2f}" if exact else "-", f"{g['time']:.2f}", f"{g['cells']:,.0f}",
                     "-" if alg == "dijkstra" else f"{g['accuracy']:.0%}",
                     f"{we[alg][0]:.1f}%" if exact else "-", f"{wo[alg][0]:.1f}%",
                     f"+{wo[alg][1]:.1f}%" if wo[alg][1] > 0 else "0", prep])
    return md_table(["method", "exact", "landmark sets a day", f"{peak}h ms (traffic = model)",
                     f"{peak}h ms (traffic off)", f"{peak}h points (traffic off)", f"{peak}h accuracy (traffic off)",
                     "longer routes (traffic = model)", "longer routes (traffic off)", "worst extra (traffic off)",
                     "precompute ms"], rows, ["---", "---"] + ["---:"] * 9)


def chart_rm_model(network, out):
    """The traffic model itself: mean travel time vs free flow, by road group and hour."""
    hours = list(range(24))
    p = LinePlot("Time-of-day traffic model (not measured traffic)", "hour of the day",
                 "travel time / free-flow time (mean)", xticks=[0, 3, 6, 8, 12, 15, 18, 21, 23],
                 xlim=(0, 23), width=760,
                 subtitle="Peaks at 8h and 18-19h; big roads slow down more; no hour is faster than the night. "
                          + ", ".join(f"{g} {sum(1 for x in network.group if x == g):,}" for g in roads.GROUPS)
                          + " segments.")
    shades = {"highway": 0.0, "arterial": 0.25, "collector": 0.5, "local": 0.7}
    for grp in roads.GROUPS:
        idx = [i for i, g in enumerate(network.group) if g == grp]
        if not idx:
            continue
        ys = []
        for h in hours:
            m = network.multipliers(h)
            ys.append(sum(m[i] for i in idx) / len(idx))
        p.line(hours, ys, tint(ALGO_COLORS["astar"], shades[grp]), grp)
    (out / "RM_model.svg").write_text(p.render())


# congestion map: travel time / free flow in four levels, traffic-light colors as in navigation apps
# (checked for color-blind separation; the legend labels each level, so color is never the only cue)
MAP_BINS = [(1.20, "#239a3b", "free (x1.0-1.2)"), (1.50, "#d9ab00", "slow (x1.2-1.5)"),
            (2.00, "#e5590a", "congested (x1.5-2.0)"), (99.0, "#a8141c", "jammed (x2.0 or more)")]
MAP_HOURS = (3, 7, 8, 12, 18, 21)
MAP_LABELS = {3: "3h (night)", 8: "8h (morning peak)", 12: "12h (midday)", 18: "18h (evening peak)"}


def chart_rm_congestion_map(network, out, hours=MAP_HOURS, cols=3, panel=300, name="RM_map.svg",
                            title="Congestion on Songdo roads by hour (traffic model)", panels=None,
                            subtitle="Travel time / free-flow time. Each direction is drawn on its own side of the road "
                                     "(keep right); big roads thicker. Black ring = centre of Songdo."):
    """Songdo roads colored by how much slower than free flow they are at each hour.

    Each direction of a road is drawn slightly to the right of the centre line
    (cars keep right in Korea), so commute direction shows: lanes into the centre
    are redder in the morning, lanes out of it in the evening.
    """
    lat, lon = network.lat, network.lon
    lat0, lat1, lon0, lon1 = min(lat), max(lat), min(lon), max(lon)
    kx = math.cos(math.radians((lat0 + lat1) / 2))
    scale = panel / max((lon1 - lon0) * kx, lat1 - lat0)
    pw, ph = (lon1 - lon0) * kx * scale, (lat1 - lat0) * scale
    gap, top, title_h = 24, 92, 22
    panels = panels or [(h, 0, MAP_LABELS.get(h, f"{h}h")) for h in hours]   # (hour, direction filter, label)
    rows = math.ceil(len(panels) / cols)
    W, H = int(cols * pw + (cols + 1) * gap), int(top + rows * (ph + title_h + gap) + 80)
    big = {"highway", "arterial"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
             f'font-family="{FONT}" role="img" aria-label="Congestion on Songdo roads by hour">',
             f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
             f'<text x="{gap}" y="34" font-size="20" font-weight="700" fill="{INK}">'
             f'{title}</text>',
             f'<text x="{gap}" y="58" font-size="13" fill="{INK_2}">{subtitle}</text>',
             f'<text x="{gap}" y="{H - 10}" font-size="11" fill="{MUTED}">'
             f'Map data (c) OpenStreetMap contributors, ODbL 1.0. Traffic is a model (src/roads.py).</text>']
    for k, (hour, sign, label) in enumerate(panels):
        mult = network.multipliers(hour)
        shown = [m for m, c in zip(mult, network.toward) if not sign or c * sign >= 0.3]
        ox = gap + (k % cols) * (pw + gap)
        oy = top + (k // cols) * (ph + title_h + gap)
        parts.append(f'<text x="{ox:.0f}" y="{oy + 14:.0f}" font-size="14" font-weight="600" fill="{INK}">'
                     f'{label}<tspan dx="10" fill="{MUTED}" font-weight="400">'
                     f'average x{sum(shown) / len(shown):.2f}</tspan></text>')
        paths = {(b, wide): [] for b in range(len(MAP_BINS)) for wide in (False, True)}
        faint = []
        for i, (u, v) in enumerate(zip(network.eu, network.ev)):
            if sign and network.toward[i] * sign < 0.3:          # not heading this way: context only
                fx0, fy0 = ox + (lon[u] - lon0) * kx * scale, oy + title_h + (lat1 - lat[u]) * scale
                fx1, fy1 = ox + (lon[v] - lon0) * kx * scale, oy + title_h + (lat1 - lat[v]) * scale
                faint.append(f"M{fx0:.1f} {fy0:.1f}L{fx1:.1f} {fy1:.1f}")
                continue
            b = next(j for j, (hi, _, _) in enumerate(MAP_BINS) if mult[i] < hi)
            wide = network.group[i] in big
            x0, y0 = ox + (lon[u] - lon0) * kx * scale, oy + title_h + (lat1 - lat[u]) * scale
            x1, y1 = ox + (lon[v] - lon0) * kx * scale, oy + title_h + (lat1 - lat[v]) * scale
            d = math.hypot(x1 - x0, y1 - y0) or 1.0
            off = (1.3 if wide else 0.7) * panel / 300       # right-hand side of the direction of travel
            nx, ny = (0.0, 0.0) if sign else (-(y1 - y0) / d * off, (x1 - x0) / d * off)
            paths[b, wide].append(f"M{x0 + nx:.1f} {y0 + ny:.1f}L{x1 + nx:.1f} {y1 + ny:.1f}")
        if faint:
            parts.append(f'<path d="{"".join(faint)}" stroke="#ecebe6" stroke-width="1" fill="none"/>')
        for wide in (False, True):                          # green first, red last: jams stay on top
            for b in range(len(MAP_BINS)):
                if paths[b, wide]:
                    parts.append(f'<path d="{"".join(paths[b, wide])}" stroke="{MAP_BINS[b][1]}" '
                                 f'stroke-width="{(2.2 if wide else 1.0) * panel / 300:.1f}" '
                                 f'stroke-linecap="round" fill="none"/>')
        cx = ox + (roads.CENTER[1] - lon0) * kx * scale
        cy = oy + title_h + (lat1 - roads.CENTER[0]) * scale
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{5 * panel / 300:.1f}" fill="none" '
                     f'stroke="{INK}" stroke-width="2"/>')
    x, ly = gap, H - 40
    for _, color, text in MAP_BINS:
        parts.append(f'<rect x="{x}" y="{ly - 11}" width="26" height="12" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{x + 32}" y="{ly}" font-size="13" fill="{INK_2}">{text}</text>')
        x += 32 + 7 * len(text) + 20
    parts.append("</svg>")
    (out / name).write_text("\n".join(parts))


def chart_ry_trips(network, goals, out, start, panel=560):
    """Where the trips of experiment RY go: Yonsei (start) and the 30 goals on the Songdo map."""
    lat, lon = network.lat, network.lon
    lat0, lat1, lon0, lon1 = min(lat), max(lat), min(lon), max(lon)
    kx = math.cos(math.radians((lat0 + lat1) / 2))
    scale = panel / max((lon1 - lon0) * kx, lat1 - lat0)
    gap, top = 24, 100
    W, H = int((lon1 - lon0) * kx * scale + 2 * gap), int(top + (lat1 - lat0) * scale + 60)

    def xy(la, lo):
        return gap + (lo - lon0) * kx * scale, top + (lat1 - la) * scale

    big = {"highway", "arterial"}
    paths = {False: [], True: []}
    for u, v, g in zip(network.eu, network.ev, network.group):
        (x0, y0), (x1, y1) = xy(lat[u], lon[u]), xy(lat[v], lon[v])
        paths[g in big].append(f"M{x0:.1f} {y0:.1f}L{x1:.1f} {y1:.1f}")
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
             f'font-family="{FONT}" role="img" aria-label="Trips from Yonsei University">',
             f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
             f'<text x="{gap}" y="34" font-size="20" font-weight="700" fill="{INK}">'
             f'Experiment RY: {len(goals)} trips from Yonsei University</text>',
             f'<text x="{gap}" y="58" font-size="13" fill="{INK_2}">Start: campus corner on Songdogwahak-ro. '
             f'Goals: random road points</text>',
             f'<text x="{gap}" y="76" font-size="13" fill="{INK_2}">at least 1.5 km away, the same at every hour.</text>',
             f'<path d="{"".join(paths[False])}" stroke="#dcdad3" stroke-width="0.8" fill="none"/>',
             f'<path d="{"".join(paths[True])}" stroke="#c3c1b8" stroke-width="1.6" fill="none"/>']
    sx, sy = xy(lat[start], lon[start])
    for g in goals:
        gx, gy = xy(lat[g], lon[g])
        parts.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{gx:.1f}" y2="{gy:.1f}" stroke="{ALGO_COLORS["alt"]}" '
                     f'stroke-opacity="0.25" stroke-width="1"/>')
    for g in goals:
        gx, gy = xy(lat[g], lon[g])
        parts.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="4.5" fill="{ALGO_COLORS["alt"]}" '
                     f'stroke="{SURFACE}" stroke-width="2"/>')
    cx, cy = xy(*roads.CENTER)
    parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="none" stroke="{INK}" stroke-width="2"/>')
    parts.append(f'<text x="{cx + 10:.1f}" y="{cy - 8:.1f}" font-size="12" fill="{INK_2}">Central Park</text>')
    parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="8" fill="{INK}" stroke="{SURFACE}" stroke-width="2.5"/>')
    parts.append(f'<text x="{sx + 12:.1f}" y="{sy + 18:.1f}" font-size="13" font-weight="600" fill="{INK}">'
                 f'Yonsei University (start)</text>')
    parts.append(f'<text x="{gap}" y="{H - 30}" font-size="11" fill="{MUTED}">'
                 f'Thin lines join start and goal in a straight line (not the route).</text>')
    parts.append(f'<text x="{gap}" y="{H - 14}" font-size="11" fill="{MUTED}">'
                 f'Map data (c) OpenStreetMap contributors, ODbL 1.0.</text>')
    parts.append("</svg>")
    (out / "RY_trips.svg").write_text("\n".join(parts))


def rm_change(network, n_queries=30):
    """Each hour against the night map, for the same start/goal pairs as experiment RM.

    Returns rows (hour, mean segment slowdown, median trip slowdown, worst trip slowdown,
    share of queries whose fastest route changed, median and worst extra time of
    still driving the night route).
    """
    from bench import BASE_SEED
    from search import dijkstra
    night = network.graph_at(roads.NIGHT)
    queries = [roads.random_query(night, BASE_SEED + i) for i in range(n_queries)]
    base = {q: dijkstra(night, *q) for q in queries}
    night_total = sum(network.costs_at(roads.NIGHT))
    rows = []
    for hour in range(24):
        g = network.graph_at(hour)
        ratios, changed, loss = [], 0, []
        for q in queries:
            r, b = dijkstra(g, *q), base[q]
            ratios.append(r.cost / b.cost)
            changed += r.path != b.path
            keep = sum(g.move_cost(a, c) for a, c in zip(b.path, b.path[1:]))
            loss.append((keep / r.cost - 1) * 100)
        rows.append(dict(hour=hour, segment=sum(network.costs_at(hour)) / night_total,
                         trip=statistics.median(ratios), trip_max=max(ratios), changed=changed / len(queries) * 100,
                         loss=statistics.median(loss), loss_max=max(loss)))
    return rows


def chart_rm_change(rows, out):
    hours = [r["hour"] for r in rows]
    dark, mid, light = INK_2, MUTED, "#c9c7bf"
    p = LinePlot("How much slower than the night map, by hour", "hour of the day",
                 "travel time / night travel time", xticks=[0, 3, 6, 8, 12, 15, 18, 21, 23], xlim=(0, 23), width=900,
                 subtitle="Same 30 start/goal pairs as experiment RM. Trips use big roads, which slow down more than "
                          "the average road.")
    p.line(hours, [r["trip_max"] for r in rows], light, "slowest trip", dashed=True)
    p.line(hours, [r["trip"] for r in rows], dark, "typical trip (median)")
    p.line(hours, [r["segment"] for r in rows], mid, "average road segment", dashed=True)
    (out / "RM_change.svg").write_text(p.render())
    sel = [r for r in rows if roads.rush(r["hour"]) > 0]
    (out / "RM_route.svg").write_text(bar_chart(
        "Does the fastest route change during the day?", [f"{r['hour']}h" for r in sel], [
            dict(label="queries whose fastest route differs from the night (%)", color=mid,
                 values=[r["changed"] for r in sel]),
            dict(label="extra time if you keep the night route, worst query (%)", color=dark,
                 values=[r["loss_max"] for r in sel]),
        ], "%", value_labels=True, value_fmt=lambda v: f"{v:.0f}",
        subtitle=f"Typical (median) extra time of keeping the night route: at most "
                 f"{max(r['loss'] for r in sel):.1f}% at any hour.", ylim=(0, 100), width=1000))


def rm_change_table(rows):
    return md_table(["hour", "average segment vs night", "typical trip vs night", "slowest trip vs night",
                     "fastest route changed", "keep night route: typical / worst extra"],
                    [[f"{r['hour']}h", f"x{r['segment']:.2f}", f"x{r['trip']:.2f}", f"x{r['trip_max']:.2f}",
                      f"{r['changed']:.0f}%", f"+{r['loss']:.1f}% / +{r['loss_max']:.1f}%"] for r in rows],
                    ["---"] + ["---:"] * 5)


def rm_section_table(table):
    rows = []
    for hour, algs in table.items():
        for alg in RM_ALGS:
            g = algs[alg]
            prep = f"{g['prep']:.0f}" if alg.startswith("alt") else "-"
            rows.append([f"{hour}h", RM_NAMES[alg], f"{g['time']:.2f}", f"{g['cells']:,.0f}",
                         "-" if alg == "dijkstra" else f"{g['accuracy']:.0%}", f"{g['over']:.0%}",
                         f"{g['wrong']:.0f}%", f"+{g['worst']:.1f}%" if g["worst"] > 0 else "0", prep])
    return md_table(["hour", "method", "time ms", "points expanded", "accuracy", "points overestimated",
                     "queries with a longer route", "worst extra time", "landmark precompute ms"],
                    rows, ["---", "---"] + ["---:"] * 7)


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
            for exp in ("A", "B", "C", "D", "E", "T", "W", "H", "BI", "O", "J", "HQ", "DY", "LP", "RM", "RY",
                        "RB", "noise")}
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

    if data["DY"]:
        table = dy_table(data["DY"])
        chart_dy(table, charts)
        section("DY. Dynamic traffic: old landmarks vs recomputed",
                ["DY_time.svg", "DY_accuracy.svg", "DY_wrong.svg"], dy_section_table(table),
                "The landmarks are measured on the jammed city, then the traffic changes and the same "
                "landmarks are used without measuring again (old), or measured again on the new map "
                "(recomputed). Jams +x%: x% of free road cells become jammed (costs only rise). "
                "Jams cleared x%: x% of jammed cells become free (some costs fall). "
                "Old landmarks are checked to be optimal in every run where costs only rose. "
                "Queries to pay off recomputing = precomputation time / time saved per query compared with the old landmarks.")
    if data["LP"]:
        table = lp_table(data["LP"])
        chart_lp(table, charts)
        section("LP. Landmark placement", ["LP_speedup.svg"], lp_section_table(table),
                "farthest: each landmark is the cell farthest (by path cost) from those already chosen. "
                "corners: cells nearest the map corners, then edge midpoints. random: seeded random cells.")

    if data["RM"]:
        table = rm_table(data["RM"])
        quick = any(r["params"].get("network") == "songdo-center" for r in data["RM"])
        net = roads.load_network(quick=quick) if roads.data_available() else None
        chart_rm(table, charts, net)
        images = ["RM_time.svg", "RM_accuracy.svg", "RM_wrong.svg"]
        if net:
            chart_rm_model(net, charts)
            chart_rm_congestion_map(net, charts)
            chart_rm_congestion_map(
                net, charts, cols=2, panel=360, name="RM_map_direction.svg",
                title="Morning: lanes into the centre jam. Evening: lanes out of it.",
                subtitle="Travel time / free-flow time. Left: lanes heading toward the centre (black ring). "
                         "Right: lanes heading away.",
                panels=[(8, 1, "8h, lanes heading into the centre"), (8, -1, "8h, lanes heading out"),
                        (18, 1, "18h, lanes heading into the centre"), (18, -1, "18h, lanes heading out")])
            change = rm_change(net)
            chart_rm_change(change, charts)
            images = ["RM_model.svg", "RM_map.svg", "RM_map_direction.svg", "RM_change.svg", "RM_route.svg"] + images
            section("RM0. How much the Songdo map changes over the day (traffic model)", [],
                    rm_change_table(change),
                    "Same start/goal pairs as RM, fastest route at each hour compared with the night (free flow).")
        section("RM. Real road map (Songdo, OpenStreetMap) over the day", images, rm_section_table(table),
                "Map data (c) OpenStreetMap contributors, ODbL 1.0 (data/songdo_roads.json). Edge cost = travel time "
                "in tenths of a second at that hour. The traffic is a model (src/roads.py), not measured traffic. "
                "The straight-line heuristic divides the great-circle distance by the fastest free-flow speed. "
                "Landmarks (8) are measured once per hour and reused by all queries of that hour. "
                "Night landmarks are checked to be optimal in every run.")

    if data["RB"]:
        tables = rb_tables(data["RB"])
        chart_rb(tables, charts)
        section("RB. Landmarks per time band (real road map)", ["RB_time.svg", "RB_time_off.svg", "RB_wrong.svg"],
                rb_section_table(tables),
                "Bands: morning 7-9h, day 10-16h, evening 17-20h, night 21-6h. Time-band landmarks are measured "
                "once per band on the cheapest hour of every segment over the band, so no hour of the band is "
                "cheaper (exact while traffic follows the model). The 10% margin multiplies those costs by 0.9, "
                "so a day up to 10% lighter than the model is not cheaper either. 'Traffic off': every query "
                "searches one day's traffic, each segment up to +-10% off the model (seeded); landmarks are "
                "always measured on the model.")

    if data["RY"]:
        table = rm_table(data["RY"])
        quick = any(r["params"].get("network") == "songdo-center" for r in data["RY"])
        net = roads.load_network(quick=quick) if roads.data_available() else None
        chart_rm(table, charts, net, prefix="RY")
        images = ["RY_time.svg", "RY_accuracy.svg", "RY_wrong.svg"]
        if net:
            first = min(r["params"]["hour"] for r in data["RY"])
            goals = sorted({int(r["goal"]) for r in data["RY"] if r["params"]["hour"] == first})
            chart_ry_trips(net, goals, charts, int(data["RY"][0]["start"]))
            images = ["RY_trips.svg"] + images
        section("RY. Trips from Yonsei University International Campus", images, rm_section_table(table),
                "Same map, traffic model and methods as RM, but every trip starts at the campus corner on "
                "Songdogwahak-ro (roads.YONSEI). 30 goals, the same at every hour.")

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
