"""Run the experiments and save raw measurements as CSV.

    python3 src/bench.py                  # every experiment
    python3 src/bench.py --exp A C        # only some
    python3 src/bench.py --quick          # tiny smoke test, writes results/quick/

Experiments (see EXPERIMENTS below):
    A B C D E   BFS vs Dijkstra vs A*: size, walls, map kind, cost, endpoints
    T           A* tie-breaking rules
    W           weighted A* (h multiplied by 1, 1.5, 2, 3, 5)
    H           heapq (C) vs a plain-Python binary heap
    BI          bidirectional Dijkstra vs Dijkstra and A*
    noise       same map many times: how much does the clock wobble?

Output (one file per experiment):
    results/<exp>.csv        one row = one run of one algorithm on one map
    results/<exp>.env.json   machine, Python version, git commit, settings

Measurement rules (kept the same for every experiment):
  - each condition uses N_MAPS maps (seeds BASE_SEED .. BASE_SEED + N_MAPS - 1)
  - each map is searched REPEATS times by every algorithm of the experiment
  - the algorithm order rotates every repetition, so no algorithm always
    runs first (cache effects, CPU frequency changes get spread out)
  - only the search call is timed (perf_counter_ns); map generation and the
    reference values below are computed outside the timed part
  - the garbage collector is paused during the timed call
  - every run is checked against the optimal cost (rules in check());
    a wrong answer stops the run with an error

Reference values stored with every row (computed once per map, not timed):
  open_cells    open cells in the map
  reachable     open cells reachable from start
  manhattan     Manhattan distance start -> goal (= A*'s h at the start)
  optimal_cost  optimal path cost (from Dijkstra), empty if no path
"""

import argparse
import csv
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bidirectional import bidirectional_dijkstra  # noqa: E402
from grid import manhattan  # noqa: E402
from maps import component, make_map, open_cell_count  # noqa: E402
from search import astar, bfs, dijkstra  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE_SEED = 2026
N_MAPS = 10
REPEATS = 5
NOISE_REPEATS = 30

# label -> (function, rule). Rules for check():
#   "exact"     must return the optimal cost
#   "any"       must find a path when one exists, never cheaper than optimal (BFS)
#   float w     must stay within w * optimal (weighted A*)
MAIN = {
    "bfs": (bfs, "any"),
    "dijkstra": (dijkstra, "exact"),
    "astar": (astar, "exact"),
}
TIE = {f"astar_{tb}": (partial(astar, tie_break=tb), "exact") for tb in ("high_g", "low_g", "fifo")}
WEIGHTS = (1, 1.5, 2, 3, 5)
WEIGHTED = {"dijkstra": (dijkstra, "exact")}
WEIGHTED.update({f"astar_w{w:g}": (partial(astar, weight=w), "exact" if w == 1 else float(w))
                 for w in WEIGHTS})
HEAP = {
    "dijkstra_heapq": (dijkstra, "exact"),
    "dijkstra_python": (partial(dijkstra, heap_impl="python"), "exact"),
    "astar_heapq": (astar, "exact"),
    "astar_python": (partial(astar, heap_impl="python"), "exact"),
}
BIDIR = {
    "dijkstra": (dijkstra, "exact"),
    "astar": (astar, "exact"),
    "bidir": (bidirectional_dijkstra, "exact"),
}

# Baseline condition: change one thing at a time away from this.
BASE = dict(kind="random", size=200, wall_ratio=0.2, weighted=False, endpoints="fixed")


def cond(**changes):
    c = dict(BASE)
    c.update(changes)
    # parameters that only some map kinds take
    if c["kind"] in ("maze", "city"):
        c.pop("wall_ratio", None)
    if c["kind"] == "terrain":
        c.pop("wall_ratio", None)
        c.pop("weighted", None)
    return c


# name -> (title, algorithm set, conditions)
EXPERIMENTS = {
    "A": ("map size", MAIN, [cond(size=n) for n in (50, 100, 200, 400)]),
    "B": ("wall ratio", MAIN, [cond(wall_ratio=p) for p in (0.0, 0.1, 0.2, 0.3)]),
    "C": ("map kind (unit cost)", MAIN, [
        cond(),
        cond(kind="maze"),
        cond(kind="maze", loop_ratio=0.1),
        cond(kind="nopath"),
        cond(kind="uwall", wall_ratio=0.0),
        cond(kind="city", weighted=False),
    ]),
    "D": ("cost (same walls, unit vs 1..9)", MAIN, [
        cond(weighted=False),
        cond(weighted=True),
        cond(kind="city", weighted=False),
        cond(kind="city", weighted=True),
        cond(kind="terrain"),
    ]),
    "E": ("endpoints (fixed corners vs random)", MAIN, [
        cond(endpoints="fixed"),
        cond(endpoints="random"),
        cond(weighted=True, endpoints="fixed"),
        cond(weighted=True, endpoints="random"),
    ]),
    "T": ("A* tie-breaking", TIE, [
        cond(wall_ratio=0.0),
        cond(wall_ratio=0.1),
        cond(),
        cond(kind="city", weighted=False),
        cond(weighted=True),
    ]),
    "W": ("weighted A*", WEIGHTED, [
        cond(),
        cond(weighted=True),
        cond(kind="terrain"),
        cond(kind="uwall", wall_ratio=0.0),
        cond(kind="city", weighted=True),
    ]),
    "H": ("heapq vs plain-Python heap", HEAP, [
        cond(),
        cond(weighted=True),
        cond(size=400),
    ]),
    "BI": ("bidirectional Dijkstra", BIDIR, [
        cond(),
        cond(weighted=True),
        cond(endpoints="random"),
        cond(weighted=True, endpoints="random"),
        cond(kind="maze"),
        cond(kind="maze", loop_ratio=0.1),
        cond(kind="nopath"),
        cond(kind="uwall", wall_ratio=0.0),
        cond(kind="terrain"),
        cond(kind="city", weighted=True),
    ]),
}

FIELDS = [
    "experiment", "condition", "kind", "size", "params", "endpoints",
    "map_index", "seed", "attempts", "start", "goal",
    "open_cells", "reachable", "manhattan", "optimal_cost",
    "rep", "order", "algorithm",
    "time_ms", "found", "cost", "steps", "expanded", "stale", "max_frontier",
]


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def environment():
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "codespaces": os.environ.get("CODESPACES") == "true",
        "github_actions": os.environ.get("GITHUB_ACTIONS") == "true",
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": _git("status", "--porcelain") not in ("", "unknown"),
    }


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def timed_run(fn, grid, start, goal):
    gc.collect()
    gc.disable()
    try:
        t0 = time.perf_counter_ns()
        result = fn(grid, start, goal)
        t1 = time.perf_counter_ns()
    finally:
        gc.enable()
    return result, (t1 - t0) / 1e6


def condition_label(c):
    parts = [c["kind"], str(c["size"])]
    for key in ("wall_ratio", "loop_ratio"):
        if key in c:
            parts.append(f"{key.split('_')[0]}{c[key]}")
    if c["kind"] == "terrain":
        parts.append("weighted")
    elif "weighted" in c:
        parts.append("weighted" if c["weighted"] else "unit")
    parts.append(c["endpoints"])
    return "-".join(parts)


def build_map(c, seed):
    params = {k: v for k, v in c.items() if k not in ("kind", "size", "endpoints")}
    return make_map(c["kind"], c["size"], seed, endpoints=c["endpoints"], **params), params


def check(name, result, optimal, case, rule="exact"):
    where = f"{case.kind} seed={case.seed} {name}"
    if optimal is None:
        if result.found:
            raise AssertionError(f"{where}: found a path where none exists")
        return
    if not result.found:
        raise AssertionError(f"{where}: missed an existing path")
    if result.cost < optimal:
        raise AssertionError(f"{where}: cost {result.cost} below optimal {optimal}")
    if rule == "exact" and result.cost != optimal:
        raise AssertionError(f"{where}: cost {result.cost} != optimal {optimal}")
    if isinstance(rule, float) and result.cost > rule * optimal:
        raise AssertionError(f"{where}: cost {result.cost} above {rule} x optimal {optimal}")


def map_info(exp, c, case, params, grid):
    ref = dijkstra(grid, case.start, case.goal)              # not timed
    optimal = ref.cost if ref.found else None
    info = dict(
        experiment=exp, condition=condition_label(c), kind=case.kind, size=c["size"],
        params=json.dumps(params, sort_keys=True), endpoints=c["endpoints"],
        seed=case.seed, attempts=case.attempts,
        start=f"{case.start[0]} {case.start[1]}", goal=f"{case.goal[0]} {case.goal[1]}",
        open_cells=open_cell_count(grid), reachable=len(component(grid, case.start)),
        manhattan=manhattan(grid, grid.index(case.start), grid.index(case.goal)),
        optimal_cost="" if optimal is None else optimal,
    )
    return info, optimal


def run_map(algos, case, info, optimal, repeats, writer, times, expanded):
    names = list(algos)
    for rep in range(repeats):
        k = rep % len(names)
        order = names[k:] + names[:k]
        for pos, name in enumerate(order):
            fn, rule = algos[name]
            result, ms = timed_run(fn, case.grid, case.start, case.goal)
            check(name, result, optimal, case, rule)
            writer.writerow(dict(
                info, rep=rep, order=pos, algorithm=name, time_ms=f"{ms:.4f}",
                found=int(result.found),
                cost="" if result.cost is None else result.cost,
                steps="" if result.path is None else len(result.path) - 1,
                expanded=result.expanded, stale=result.stale,
                max_frontier=result.max_frontier,
            ))
            times[name].append(ms)
            expanded[name].append(result.expanded)
    return repeats * len(names)


def run_condition(exp, algos, c, n_maps, repeats, writer, log=print):
    times = {a: [] for a in algos}
    expanded = {a: [] for a in algos}
    rows = 0
    for m in range(n_maps):
        case, params = build_map(c, BASE_SEED + m)
        info, optimal = map_info(exp, c, case, params, case.grid)
        info["map_index"] = m
        rows += run_map(algos, case, info, optimal, repeats, writer, times, expanded)
    log(f"  {condition_label(c)}")
    for a in algos:
        log(f"      {a:<16} {statistics.median(times[a]):9.2f} ms {statistics.median(expanded[a]):>9.0f} cells")
    return rows


def run_noise(n_repeats, writer, size=200, log=print):
    """Same map, same algorithm, many times: how much does the clock wobble?"""
    c = cond(size=size)
    case, params = build_map(c, BASE_SEED)
    info, optimal = map_info("noise", c, case, params, case.grid)
    info["map_index"] = 0
    times = {a: [] for a in MAIN}
    expanded = {a: [] for a in MAIN}
    rows = run_map(MAIN, case, info, optimal, n_repeats, writer, times, expanded)
    for a, t in times.items():
        med = statistics.median(t)
        q = statistics.quantiles(t, n=4)
        log(f"  {a:<9} median {med:8.2f} ms   IQR {q[0]:.2f}-{q[2]:.2f}   "
            f"min {min(t):.2f}  max {max(t):.2f}  spread(IQR/median) {(q[2] - q[0]) / med:.1%}")
    return rows


def warm_up():
    case, _ = build_map(cond(size=50), BASE_SEED)
    for algos in (MAIN, TIE, WEIGHTED, HEAP, BIDIR):
        for fn, _ in algos.values():
            fn(case.grid, case.start, case.goal)


def quick_version(conditions):
    out = []
    for c in conditions:
        c = dict(c)
        c["size"] = max(10, c["size"] // 8)     # keeps sizes distinct: 50,100,200,400 -> 10,12,25,50
        out.append(c)
    return out


def run(experiments, out_dir, n_maps, repeats, quick=False, log=print):
    out_dir.mkdir(parents=True, exist_ok=True)
    warm_up()
    env = environment()
    log(f"python {env['python']} | {env['cpu_model']} | {env['cpu_count']} cpus | commit {env['git_commit']}")
    total_start = time.perf_counter()
    for exp in experiments:
        t0 = time.perf_counter()
        csv_path = out_dir / f"{exp}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            if exp == "noise":
                log("[noise] same map, repeated runs")
                n = 5 if quick else NOISE_REPEATS
                rows = run_noise(n, writer, size=30 if quick else 200, log=log)
                settings = dict(repeats=n, algorithms=list(MAIN))
            else:
                title, algos, conditions = EXPERIMENTS[exp]
                if quick:
                    conditions = quick_version(conditions)
                log(f"[{exp}] {title}   (median time / median expanded cells)")
                rows = sum(run_condition(exp, algos, c, n_maps, repeats, writer, log)
                           for c in conditions)
                settings = dict(title=title, algorithms=list(algos), conditions=conditions,
                                n_maps=n_maps, repeats=repeats)
        elapsed = time.perf_counter() - t0
        meta = dict(
            experiment=exp, rows=rows, seconds=round(elapsed, 1), quick=quick,
            base_seed=BASE_SEED, settings=settings, environment=env,
            finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            command=" ".join([Path(sys.executable).name, *sys.argv]),
        )
        (out_dir / f"{exp}.env.json").write_text(json.dumps(meta, indent=2) + "\n")
        shown = csv_path.relative_to(ROOT) if csv_path.is_relative_to(ROOT) else csv_path
        log(f"  -> {shown} ({rows} rows, {elapsed:.1f} s)")
    log(f"done in {time.perf_counter() - total_start:.1f} s")


def main(argv=None):
    all_exps = [*EXPERIMENTS, "noise"]
    p = argparse.ArgumentParser(description="Run the path-finding experiments.")
    p.add_argument("--exp", nargs="+", choices=all_exps, default=all_exps)
    p.add_argument("--quick", action="store_true", help="small maps, few repeats (smoke test)")
    p.add_argument("--maps", type=int, default=None, help=f"maps per condition (default {N_MAPS})")
    p.add_argument("--repeats", type=int, default=None, help=f"repeats per map (default {REPEATS})")
    p.add_argument("--out", type=Path, default=None, help="output folder (default results/)")
    args = p.parse_args(argv)
    n_maps = args.maps or (2 if args.quick else N_MAPS)
    repeats = args.repeats or (2 if args.quick else REPEATS)
    out = args.out or (ROOT / "results" / "quick" if args.quick else ROOT / "results")
    run(args.exp, out, n_maps, repeats, quick=args.quick)


if __name__ == "__main__":
    main()
