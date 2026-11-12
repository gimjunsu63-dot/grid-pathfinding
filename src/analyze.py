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
import grid as grid_mod  # noqa: E402
import search  # noqa: E402
from svgchart import ALGO_COLORS, INK_2, MUTED, LinePlot, bar_chart, bar_panels, tint  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAIN = ["bfs", "dijkstra", "astar"]
NAMES = {"bfs": "BFS", "dijkstra": "Dijkstra", "astar": "A*", "bidir": "Bidirectional"}


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
            s["lo"] = [summary[c][a][lo_key] * scale for c in conds]
            s["hi"] = [summary[c][a][hi_key] * scale for c in conds]
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
    cats = ["unit cost", "cost 1-9"]
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
    data = {exp: load(results_dir, exp) for exp in ("A", "B", "C", "D", "E", "T", "W", "H", "BI", "noise")}
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
            u = statistics.median(per_cell[(a, "unit cost")])
            w = statistics.median(per_cell[(a, "cost 1-9")])
            rows.append([NAMES[a], f"{u:.2f}", f"{w:.2f}"])
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
