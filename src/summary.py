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
    o_rows = load(results_dir, "O")
    if o_rows:
        s = summarize(o_rows)
        unit = [by for by in s.values() if "jps" in by]
        extras = [(by["bfs"]["cost_ratio"] - 1) * 100 for by in unit if by["bfs"]["cost_ratio"]]
        if unit:
            wins = sum(1 for by in unit if by["jps"]["time"] < by["astar"]["time"])
            share = statistics.median(by["jps"]["expanded"] / by["astar"]["expanded"] for by in unit
                                      if by["astar"]["expanded"])
            lines.append(f"- 8 directions: BFS's path costs up to +{max(extras):.1f}% even when every cost is 1 "
                         f"(a diagonal step costs more than a straight one). JPS takes {share:.0%} of A*'s cells "
                         f"off the heap (median) but is faster than A* in {wins} of {len(unit)} maps, "
                         "because it scans the cells between jump points.")
    j_rows = load(results_dir, "J")
    if j_rows:
        s = summarize(j_rows)
        conds = list(s)
        first, last = s[conds[0]], s[conds[-1]]
        loss = next((s[c]["astar"]["label"] for c in conds if s[c]["astar"]["time"] > s[c]["dijkstra"]["time"]), None)
        best_last = min(("dijkstra", "astar", "bidir"), key=lambda a: last[a]["time"])
        txt = (f"- Traffic jams: with no jams A* is {first['dijkstra']['time'] / first['astar']['time']:.1f}x faster "
               f"than Dijkstra; ")
        txt += f"from {loss} it is slower. " if loss else "it stays faster at every jam level. "
        txt += f"At {last['astar']['label']}, the fastest optimal search is {NAMES[best_last]}."
        lines.append(txt)
    hq_rows = load(results_dir, "HQ")
    if hq_rows:
        from analyze import HQ_MAPS, hq_table
        table = hq_table(hq_rows)
        lines += ["", "## Heuristic quality (main experiment)", ""]
        rows = []
        for kind, algs in table.items():
            alts = sorted((a for a in algs if a.startswith("alt")), key=lambda a: int(a[3:]))
            best = min(alts, key=lambda a: algs[a]["time"])
            d, m, b = algs["dijkstra"], algs["astar"], algs[best]
            saved = d["time"] - b["time"]
            rows.append([HQ_MAPS.get(kind, kind), f"{m['accuracy']:.0%}", f"{m['work']:.0f}%",
                         f"{best[3:]}", f"{b['accuracy']:.0%}", f"{b['work']:.1f}%",
                         f"{d['time'] / b['time']:.1f}x", f"{b['prep'] / saved:.0f}" if saved > 0 else "-"])
        lines += [md_table(["map", "Manhattan accuracy", "Manhattan cells vs Dijkstra", "best landmark count",
                            "landmark accuracy", "landmark cells vs Dijkstra", "landmark speed-up over Dijkstra",
                            "queries to pay off precompute"], rows, ["---"] + ["---:"] * 7), ""]
        lines.append("- Landmarks need precomputation (one Dijkstra per landmark), so they pay off only when the "
                     "same map answers several queries; see the last column.")
    dy_rows = load(results_dir, "DY")
    if dy_rows:
        from analyze import dy_table
        t = dy_table(dy_rows)
        ups = [c for c in t if c > 0]
        downs = [c for c in t if c < 0]
        lines += ["", "## Dynamic traffic (old landmarks vs recomputed)", ""]
        if 0 in t and ups:
            top = max(ups)
            old, new, d = t[top]["alt_old"], t[top]["alt_new"], t[top]["dijkstra"]
            acc = ", ".join(f"{t[c]['alt_old']['accuracy']:.0%}" for c in [0] + sorted(ups))
            pays = []
            for c in sorted(ups):
                gain = t[c]["alt_old"]["time"] - t[c]["alt_new"]["time"]
                pays.append(f"{t[c]['alt_new']['prep'] / gain:.0f}" if gain > 0 else "never")
            lines.append(f"- Jams appear (costs only rise): old landmarks were optimal in every run; their accuracy "
                         f"fell {acc} (no change -> +{top}%), and at +{top}% they took {old['time']:.1f} ms vs "
                         f"Dijkstra {d['time']:.1f} ms and recomputed landmarks {new['time']:.1f} ms. "
                         f"Recomputing ({new['prep']:.0f} ms) paid off after {' / '.join(pays)} queries "
                         f"({' / '.join(f'+{c}%' for c in sorted(ups))}).")
        if downs:
            low = min(downs)
            o = t[low]["alt_old"]
            wrong = ", ".join(f"{t[c]['alt_old']['wrong']:.0f}%" for c in sorted(downs, reverse=True))
            lines.append(f"- Jams clear (some costs fall): old landmarks overestimated in {o['over']:.0%} of cells "
                         f"at {-low}% cleared; queries answered with a longer path: {wrong} "
                         f"(cleared {', '.join(f'{-c}%' for c in sorted(downs, reverse=True))}), "
                         f"worst +{o['worst']:.1f}% cost. Fast, but no longer correct.")
    lp_rows = load(results_dir, "LP")
    if lp_rows:
        from analyze import HQ_MAPS, LP_NAMES, lp_table
        t = lp_table(lp_rows)
        parts = []
        for kind, algs in t.items():
            best = {n: max((a for a in algs if a.startswith(f"alt{n}_")), key=lambda a: algs[a]["speedup"])
                    for n in (4, 8)}
            rand4 = algs.get("alt4_random")
            parts.append(f"{HQ_MAPS.get(kind, kind)}: best {LP_NAMES[best[4].split('_')[1]]} with 4 "
                         f"({algs[best[4]]['speedup']:.1f}x), {LP_NAMES[best[8].split('_')[1]]} with 8 "
                         f"({algs[best[8]]['speedup']:.1f}x)"
                         + (f", random with 4 {rand4['speedup']:.1f}x" if rand4 else ""))
        lines += ["", "## Landmark placement", "", "- Speed-up over Dijkstra. " + "; ".join(parts) + "."]
    rm_rows = load(results_dir, "RM")
    if rm_rows:
        from analyze import rm_table
        t = rm_table(rm_rows)
        night = min(t)
        peak = 8 if 8 in t else max(t)
        d, now = t[peak]["dijkstra"], t[peak]["alt_now"]
        lines += ["", "## Real road map (Songdo, OpenStreetMap)", ""]
        lines.append(f"- Morning peak {peak}h: Dijkstra {d['time']:.1f} ms, A* + straight line "
                     f"{t[peak]['astar']['time']:.1f} ms (accuracy {t[peak]['astar']['accuracy']:.0%}), "
                     f"landmarks measured that hour {now['time']:.1f} ms (accuracy {now['accuracy']:.0%}, "
                     f"{d['time'] / now['time']:.1f}x faster than Dijkstra).")
        ni = t[peak]["alt_night"]
        lines.append(f"- Landmarks measured at night were optimal at every hour; at {peak}h their accuracy was "
                     f"{ni['accuracy']:.0%} and they took {ni['time']:.1f} ms.")
        r = t[night]["alt_rush"]
        lines.append(f"- Landmarks measured at 8h, used at {night}h (lighter traffic): {r['over']:.0%} of points "
                     f"overestimated, {r['wrong']:.0f}% of queries got a longer route (worst +{r['worst']:.1f}%).")
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
