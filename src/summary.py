"""One-page summary of all experiments: who wins where, and the key numbers.

    python3 src/summary.py                 # print to the terminal
    python3 src/summary.py --write         # also write figures/SUMMARY.md

Every number is computed from results/*.csv (nothing is typed in by hand),
so the summary changes whenever the experiments are run again.
"""

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze import MAIN, NAMES, load, md_table, quartiles, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _label(s):
    return s["label"].replace("\n", ", ") + (f", {s['size']}x{s['size']}" if s["size"] != 200 else "")


def winner_table(results_dir):
    """One row per distinct condition of experiments A-E."""
    seen = {}
    for exp in "ABCDE":
        for cond, by in summarize(load(results_dir, exp)).items():
            seen.setdefault(cond, by)
    rows, stats = [], defaultdict(int)
    for cond, by in seen.items():
        if not all(a in by for a in MAIN):
            continue
        # only answers with the optimal cost can win (BFS is fast but wrong on weighted maps)
        correct = [a for a in MAIN if by[a]["cost_ratio"] is None or by[a]["cost_ratio"] <= 1 + 1e-9]
        order = sorted(correct, key=lambda a: by[a]["time"])
        stats[order[0]] += 1
        d, a, b = by["dijkstra"], by["astar"], by["bfs"]
        speed = d["time"] / a["time"]
        if speed < 1:
            stats["astar_slower"] += 1
        bfs_extra = (b["cost_ratio"] - 1) * 100 if b["cost_ratio"] else None
        rows.append(dict(
            label=_label(b), fastest=NAMES[order[0]],
            bfs=b["time"], dij=d["time"], astar=a["time"], speed=speed,
            work=a["expanded"] / d["expanded"] * 100 if d["expanded"] else 0,
            bfs_extra=bfs_extra,
        ))
    stats["total"] = len(rows)
    return rows, stats


def build(results_dir):
    lines = ["# Summary", "",
             "Computed from `results/*.csv` by `make summary`. Times are medians (ms). "
             "Details and charts: [RESULTS.md](RESULTS.md).", ""]

    rows, st = winner_table(results_dir)
    if rows:
        lines += ["## Who is fastest where (experiments A-E)", "",
                  "Only algorithms that returned the optimal path can win a condition; "
                  "BFS is excluded where costs differ, because its path is more expensive.", ""]
        table = []
        for r in rows:
            extra = "-" if r["bfs_extra"] is None else f"+{r['bfs_extra']:.0f}%"
            table.append([r["label"], f"**{r['fastest']}**", f"{r['bfs']:.1f}", f"{r['dij']:.1f}",
                          f"{r['astar']:.1f}", f"{r['speed']:.2f}x", f"{r['work']:.0f}%", extra])
        lines += [md_table(["condition", "fastest (optimal path)", "BFS", "Dijkstra", "A*", "A* speed-up over Dijkstra",
                            "A* cells vs Dijkstra", "BFS path extra cost"], table,
                           ["---", "---"] + ["---:"] * 6), ""]
        best = max(rows, key=lambda r: r["speed"])
        worst = min(rows, key=lambda r: r["speed"])
        lines += [
            f"- Fastest overall: BFS in {st['bfs']}, Dijkstra in {st['dijkstra']}, "
            f"A* in {st['astar']} of {st['total']} conditions.",
            f"- A* is slower than Dijkstra in {st['astar_slower']} of {st['total']} conditions.",
            f"- Best case for A*: {best['label']} ({best['speed']:.1f}x faster than Dijkstra, "
            f"{best['work']:.0f}% of its cells).",
            f"- Worst case for A*: {worst['label']} ({1 / worst['speed']:.1f}x slower than Dijkstra, "
            f"{worst['work']:.0f}% of its cells).",
        ]
        extras = [r for r in rows if r["bfs_extra"]]
        if extras:
            w = max(extras, key=lambda r: r["bfs_extra"])
            lines.append(f"- BFS ignores costs: where costs differ it is still the quickest to answer, "
                         f"but its path costs up to +{w['bfs_extra']:.0f}% more ({w['label']}).")
        lines.append("")

    lines += ["## Key numbers", ""]
    a_rows = load(results_dir, "A")
    if a_rows:
        from analyze import slope
        s = summarize(a_rows)
        conds = list(s)
        cells = [s[c]["bfs"]["size"] ** 2 for c in conds]
        sl = ", ".join(f"{NAMES[a]} {slope(cells, [s[c][a]['time'] for c in conds]):.2f}" for a in MAIN)
        lines.append(f"- Growth with map size (log-log slope of time, 1 = proportional): {sl}.")
    main_rows = [r for e in "ABCDE" for r in load(results_dir, e)]
    if main_rows:
        per = {a: statistics.median(r["time_ms"] * 1000 / r["expanded"] for r in main_rows
                                    if r["algorithm"] == a and r["expanded"]) for a in MAIN}
        txt = ", ".join(f"{NAMES[a]} {per[a]:.2f}" for a in MAIN)
        lines.append(f"- Time per expanded cell (microseconds): {txt}. "
                     f"A* pays {per['astar'] / per['bfs']:.1f}x BFS per cell, so it must cut cells to win.")
    t_rows = load(results_dir, "T")
    if t_rows:
        s = summarize(t_rows)
        best = max(s.values(), key=lambda by: by["astar_low_g"]["expanded"] / by["astar_high_g"]["expanded"])
        ratio = best["astar_low_g"]["expanded"] / best["astar_high_g"]["expanded"]
        lines.append(f"- Tie-breaking: up to {ratio:.0f}x more cells when ties go to smaller g "
                     f"({_label(best['astar_high_g'])}).")
    w_rows = load(results_dir, "W")
    if w_rows:
        s = summarize(w_rows)
        parts = []
        for by in s.values():
            if "astar_w2" in by:
                parts.append(f"{_label(by['dijkstra'])}: {by['astar_w2']['expanded'] / by['dijkstra']['expanded']:.0%} "
                             f"of Dijkstra's cells, +{(by['astar_w2']['cost_ratio'] - 1) * 100:.1f}% cost")
        lines.append("- Weighted A* with weight 2: " + "; ".join(parts) + ".")
    h_rows = load(results_dir, "H")
    if h_rows:
        s = summarize(h_rows)
        ratios = [by[f"{b}_python"]["time"] / by[f"{b}_heapq"]["time"] for by in s.values() for b in ("dijkstra", "astar")]
        lines.append(f"- Plain-Python heap instead of heapq: {min(ratios):.2f}x to {max(ratios):.2f}x slower, "
                     "with exactly the same cells expanded.")
    bi_rows = load(results_dir, "BI")
    if bi_rows:
        s = summarize(bi_rows)
        r = {c: by["bidir"]["expanded"] / by["dijkstra"]["expanded"] for c, by in s.items() if by["dijkstra"]["expanded"]}
        lo, hi = min(r, key=r.get), max(r, key=r.get)
        lines.append(f"- Bidirectional Dijkstra: from {r[lo]:.1%} of Dijkstra's cells ({_label(s[lo]['bidir'])}) "
                     f"to {r[hi]:.0%} ({_label(s[hi]['bidir'])}).")
    n_rows = load(results_dir, "noise")
    if n_rows:
        by = defaultdict(list)
        for x in n_rows:
            by[x["algorithm"]].append(x["time_ms"])
        spread = ", ".join(f"{NAMES[a]} {(quartiles(by[a])[2] - quartiles(by[a])[0]) / statistics.median(by[a]):.1%}"
                           for a in MAIN if a in by)
        lines.append(f"- Measurement noise (IQR / median, same map 30 runs): {spread}.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description="Print a one-page summary of the experiments.")
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--write", action="store_true", help="also write figures/SUMMARY.md")
    p.add_argument("--out", type=Path, default=ROOT / "figures")
    args = p.parse_args(argv)
    text = build(args.results)
    print(text)
    if args.write:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "SUMMARY.md").write_text(text)
        print(f"-> {args.out / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
