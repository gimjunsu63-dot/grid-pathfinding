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
    O           8-directional moves (octile heuristic) and Jump Point Search
    J           traffic jams: share of jammed road cells in a city, 0% to 50%
    HQ          heuristic quality: Dijkstra, A* + Manhattan, A* + landmarks (1-16),
                and the accuracy ruler h = k x real remaining cost
    DY          dynamic traffic: jams appear or clear after the landmarks were measured;
                old landmarks vs recomputed landmarks vs Manhattan vs Dijkstra
    LP          landmark placement: farthest-point vs corners vs random, 4 and 8 landmarks
    RM          real road map (Songdo, OpenStreetMap) at different hours of the day:
                Dijkstra, A* + straight line, landmarks measured now / at night / at 8h
    RY          the same, but every trip starts at Yonsei University International Campus
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

Per algorithm and map (not timed as search time):
  prep_ms       precomputation before the search (landmark distances, ruler), 0 if none.
                For old landmarks (DY) it is the original precomputation, paid BEFORE the
                traffic changed; after the change they cost nothing.
  h_accuracy    mean over cells of h(v) / real remaining cost (4-direction maps only)
  h_over        share of those cells where h(v) is larger than the real remaining cost
                (0 for every heuristic that never overestimates)
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
from dataclasses import dataclass
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bidirectional import bidirectional_dijkstra  # noqa: E402
from grid import Grid8, manhattan  # noqa: E402
from jps import jump_point_search  # noqa: E402
from landmarks import PLACEMENTS, OracleHeuristic, accuracy_stats, landmark_heuristic, remaining_to_goal  # noqa: E402
import roads  # noqa: E402
from maps import MapCase, change_jams, component, make_map, open_cell_count  # noqa: E402
from search import astar, bfs, dijkstra  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE_SEED = 2026
N_MAPS = 10
REPEATS = 5
NOISE_REPEATS = 30

@dataclass(frozen=True)
class Spec:
    """How to run and check one algorithm.

    fn         fn(grid, start, goal), or fn(grid, start, goal, state) when prepare is set
    rule       check() rule: "exact" = optimal cost, "any" = a path no cheaper than optimal
               (BFS), float w = at most w x optimal (weighted A*),
               "exact_if_costs_rise" = optimal unless some jams cleared (old landmarks)
    needs      "uniform": only run where every open cell costs 1 (JPS)
    prepare    prepare(grid, start, goal) -> state, run once per map; its time is prep_ms
    heuristic  heuristic(grid, goal_index, state) -> h(v, t), used for h_accuracy
    """
    fn: object
    rule: object = "exact"
    needs: str = ""
    prepare: object = None
    heuristic: object = None


def as_spec(v):
    """Algorithm tables may hold plain (fn, rule[, needs]) tuples."""
    return v if isinstance(v, Spec) else Spec(v[0], v[1], v[2] if len(v) > 2 else "")


def _h_zero(grid, t, state):
    return lambda v, goal: 0


def _h_grid(grid, t, state):
    return grid.heuristic


def _h_state(grid, t, state):
    return state


MAIN = {
    "bfs": Spec(bfs, "any"),
    "dijkstra": Spec(dijkstra, heuristic=_h_zero),
    "astar": Spec(astar, heuristic=_h_grid),
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

MOVES8 = {
    "bfs": (bfs, "any"),             # fewest moves is not cheapest once diagonals cost more
    "dijkstra": (dijkstra, "exact"),
    "astar": (astar, "exact"),
    "bidir": (bidirectional_dijkstra, "exact"),
    "jps": (jump_point_search, "exact", "uniform"),
}
TRAFFIC = {
    "bfs": (bfs, "any"),
    "dijkstra": (dijkstra, "exact"),
    "astar": (astar, "exact"),
    "bidir": (bidirectional_dijkstra, "exact"),
}
JAM_RATIOS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def search_with(grid, start, goal, state):
    """A* using a prepared heuristic object (landmarks or the ruler)."""
    return astar(grid, start, goal, heuristic=state)


def prepare_landmarks(grid, start, goal, count):
    return landmark_heuristic(grid, start, count, placement="farthest")


def prepare_ruler(grid, start, goal, k):
    t = grid.index(goal)
    return OracleHeuristic(remaining_to_goal(grid, t), t, k)


LANDMARK_COUNTS = (1, 2, 4, 8, 16)
RULER_KS = (0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.2, 1.5)
HQ = {
    "dijkstra": Spec(dijkstra, heuristic=_h_zero),
    "astar": Spec(astar, heuristic=_h_grid),
}
HQ.update({f"alt{n}": Spec(search_with, prepare=partial(prepare_landmarks, count=n), heuristic=_h_state)
           for n in LANDMARK_COUNTS})
HQ.update({f"ruler{k:g}": Spec(search_with, "exact" if k <= 1 else float(k),
                               prepare=partial(prepare_ruler, k=k), heuristic=_h_state)
           for k in RULER_KS})

def prepare_placed(grid, start, goal, count, placement):
    return landmark_heuristic(grid, start, count, placement=placement)


PLACED = {"dijkstra": Spec(dijkstra, heuristic=_h_zero)}
PLACED.update({f"alt{n}_{pl}": Spec(search_with, prepare=partial(prepare_placed, count=n, placement=pl),
                                    heuristic=_h_state)
               for n in (4, 8) for pl in PLACEMENTS})


def prepare_old_landmarks(grid, start, goal, count):
    """Landmarks measured on the map as it was BEFORE the traffic changed (grid.before)."""
    return landmark_heuristic(grid.before, start, count, placement="farthest")


DYNAMIC_LANDMARKS = 8          # the best count on the jammed city in experiment HQ
DYNAMIC = {
    "dijkstra": Spec(dijkstra, heuristic=_h_zero),
    "astar": Spec(astar, heuristic=_h_grid),
    "alt_new": Spec(search_with, prepare=partial(prepare_landmarks, count=DYNAMIC_LANDMARKS),
                    heuristic=_h_state),
    "alt_old": Spec(search_with, "exact_if_costs_rise",
                    prepare=partial(prepare_old_landmarks, count=DYNAMIC_LANDMARKS), heuristic=_h_state),
}
CHANGES = (0.1, 0.3, 0.5)

ROAD_LANDMARKS = 8
ROAD_HOURS = (3, 6, 7, 8, 9, 10, 12, 15, 17, 18, 19, 21)
_ROAD_STATES = {}


def prepare_road_landmarks(grid, start, goal, hour=None):
    """Landmarks measured on the travel times of `hour` (default: the graph's own hour).

    Measured once per network and hour, then reused by every query; prep_ms is
    the time of that one measurement.
    """
    hour = grid.hour if hour is None else hour
    key = (id(grid.network), hour)
    if key not in _ROAD_STATES:
        t0 = time.perf_counter_ns()
        state = roads.road_landmarks(grid.network.graph_at(hour), ROAD_LANDMARKS)
        state.prep_ms = (time.perf_counter_ns() - t0) / 1e6
        _ROAD_STATES[key] = state
    return _ROAD_STATES[key]


ROADS = {
    "dijkstra": Spec(dijkstra, heuristic=_h_zero),
    "astar": Spec(astar, heuristic=_h_grid),                     # straight line / top speed
    "alt_now": Spec(search_with, prepare=prepare_road_landmarks, heuristic=_h_state),
    # night = free flow, the cheapest hour: theory says these stay exact at every hour
    "alt_night": Spec(search_with, prepare=partial(prepare_road_landmarks, hour=roads.NIGHT),
                      heuristic=_h_state),
    # measured in the morning peak: may overestimate when traffic is lighter
    "alt_rush": Spec(search_with, "any", prepare=partial(prepare_road_landmarks, hour=roads.RUSH),
                     heuristic=_h_state),
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


def is_uniform(c):
    return not c.get("weighted") and c["kind"] != "terrain"


def jam(p):
    """City where every road costs 1 except jammed cells (cost 9): only the jam share changes."""
    return cond(kind="city", weighted=True, local_cost=1, jam_arterials=True, jam_ratio=p)


def dyn(appear=0.0, clear=0.0):
    """Jammed city (20%) whose traffic changes after the landmarks were measured.

    appear: share of free road cells that become jammed. clear: share of jams that clear.
    """
    return dict(jam(0.2), endpoints="random", jam_up=appear, jam_down=clear)


def road(hour, start="random"):
    """Songdo road network at this hour of the day (time-of-day traffic model in roads.py).

    start="yonsei": every trip starts at Yonsei University International Campus.
    """
    return dict(kind="road", size=0, endpoints=start, hour=hour)


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
    "O": ("8-directional moves and Jump Point Search", MOVES8, [
        cond(moves=8, wall_ratio=0.0),
        cond(moves=8),
        cond(moves=8, kind="maze"),
        cond(moves=8, kind="nopath"),
        cond(moves=8, kind="uwall", wall_ratio=0.0),
        cond(moves=8, kind="city", weighted=False),
        cond(moves=8, weighted=True),
    ]),
    "J": ("traffic jams in a city", TRAFFIC, [jam(p) for p in JAM_RATIOS]),
    # Landmarks are computed once per map and serve any query, so start and goal are random
    # (fixed corners would put the first landmark right on the start or the goal).
    "HQ": ("heuristic quality: ruler, Manhattan, landmarks", HQ, [
        cond(weighted=True, endpoints="random"),
        cond(kind="maze", endpoints="random"),
        dict(jam(0.2), endpoints="random"),
    ]),
    "DY": ("dynamic traffic: old vs recomputed landmarks", DYNAMIC,
           [dyn()] + [dyn(appear=p) for p in CHANGES] + [dyn(clear=p) for p in CHANGES]),
    "LP": ("landmark placement", PLACED, [
        cond(weighted=True, endpoints="random"),
        cond(kind="maze", endpoints="random"),
        dict(jam(0.2), endpoints="random"),
    ]),
    "RM": ("real road map (Songdo, OpenStreetMap) by hour of day", ROADS, [road(h) for h in ROAD_HOURS]),
    "RY": ("real road map, every trip from Yonsei University International Campus", ROADS,
           [road(h, start="yonsei") for h in ROAD_HOURS]),
}
# DY answers each query once per map, so it uses more maps to count wrong answers.
# RM: one map, so "maps" are start/goal pairs (30 per hour, the same pairs at every hour).
# RY: same, with the start fixed at Yonsei (30 goals).
MAP_FACTOR = {"DY": 3, "RM": 3, "RY": 3}

FIELDS = [
    "experiment", "condition", "kind", "size", "params", "endpoints",
    "map_index", "seed", "attempts", "start", "goal",
    "open_cells", "reachable", "manhattan", "h_start", "optimal_cost",
    "rep", "order", "algorithm",
    "time_ms", "found", "cost", "steps", "expanded", "stale", "max_frontier", "scanned",
    "prep_ms", "h_accuracy", "h_over",
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
    if c["kind"] == "road":
        where = "songdo-center" if c.get("quick") else "songdo"
        frm = "-from-yonsei" if c.get("endpoints") == "yonsei" else ""
        return f"road-{where}{frm}-{c['hour']:02d}h"
    parts = [c["kind"], str(c["size"])]
    if c.get("moves") == 8:
        parts.append("8dir")
    for key in ("wall_ratio", "loop_ratio", "jam_ratio"):
        if key in c:
            parts.append(f"{key.split('_')[0]}{c[key]}")
    if "jam_up" in c:
        parts.append(f"up{c['jam_up']}-down{c['jam_down']}")
    if c["kind"] == "terrain":
        parts.append("weighted")
    elif "weighted" in c:
        parts.append("weighted" if c["weighted"] else "unit")
    parts.append(c["endpoints"])
    return "-".join(parts)


def build_road(c, seed):
    net = roads.load_network(quick=c.get("quick", False))
    graph = net.graph_at(c["hour"])
    fixed = roads.nearest_node(graph, roads.YONSEI) if c.get("endpoints") == "yonsei" else None
    start, goal = roads.random_query(graph, seed, start=fixed)
    params = dict(hour=c["hour"], network=net.name)
    return MapCase("road", graph, start, goal, seed, 1, dict(params)), params


def build_map(c, seed):
    if c["kind"] == "road":
        return build_road(c, seed)
    skip = ("kind", "size", "endpoints", "moves", "jam_up", "jam_down")
    params = {k: v for k, v in c.items() if k not in skip}
    case = make_map(c["kind"], c["size"], seed, endpoints=c["endpoints"], **params)
    if c.get("moves") == 8:
        case.grid = Grid8.from_grid(case.grid)
        params = dict(params, moves=8)
    if "jam_up" in c:                       # dynamic traffic: search the changed map,
        before = case.grid                  # keep the old one for old landmarks
        case.grid = change_jams(before, seed, appear=c["jam_up"], clear=c["jam_down"])
        case.grid.before = before
        params = dict(params, jam_up=c["jam_up"], jam_down=c["jam_down"])
        case.params = dict(case.params, jam_up=c["jam_up"], jam_down=c["jam_down"])
    return case, params


def check(name, result, optimal, case, rule="exact"):
    where = f"{case.kind} seed={case.seed} {name}"
    if optimal is None:
        if result.found:
            raise AssertionError(f"{where}: found a path where none exists")
        return
    if not result.found:
        raise AssertionError(f"{where}: missed an existing path")
    if rule == "exact_if_costs_rise":       # theory: old landmarks stay safe while costs only rise
        rule = "any" if case.params.get("jam_down") else "exact"
    if result.cost < optimal:
        raise AssertionError(f"{where}: cost {result.cost} below optimal {optimal}")
    if rule == "exact" and result.cost != optimal:
        raise AssertionError(f"{where}: cost {result.cost} != optimal {optimal}")
    if isinstance(rule, float) and result.cost > rule * optimal:
        raise AssertionError(f"{where}: cost {result.cost} above {rule} x optimal {optimal}")


def map_info(exp, c, case, params, grid):
    ref = dijkstra(grid, case.start, case.goal)              # not timed
    optimal = ref.cost if ref.found else None
    s, t = grid.index(case.start), grid.index(case.goal)
    if case.kind == "road":           # every node reaches every other; "manhattan" = straight line in m
        open_cells = reachable = grid.n
        straight = round(grid.straight_m(s, t))
    else:
        open_cells, reachable = open_cell_count(grid), len(component(grid, case.start))
        straight = manhattan(grid, s, t)
    info = dict(
        experiment=exp, condition=condition_label(c), kind=case.kind, size=c["size"],
        params=json.dumps(params, sort_keys=True), endpoints=c["endpoints"],
        seed=case.seed, attempts=case.attempts,
        start=" ".join(map(str, case.start)) if isinstance(case.start, tuple) else str(case.start),
        goal=" ".join(map(str, case.goal)) if isinstance(case.goal, tuple) else str(case.goal),
        open_cells=open_cells, reachable=reachable, manhattan=straight,
        h_start=grid.heuristic(s, t),
        optimal_cost="" if optimal is None else optimal,
    )
    return info, optimal


def prepare_all(specs, case):
    """Run every prepare step once (timed as prep_ms) and measure heuristic accuracy."""
    grid = case.grid
    calls, prep, acc, over = {}, {}, {}, {}
    for name, sp in specs.items():
        if sp.prepare is None:
            calls[name] = sp.fn
            continue
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter_ns()
            state = sp.prepare(grid, case.start, case.goal)
            prep[name] = getattr(state, "prep_ms", None) or (time.perf_counter_ns() - t0) / 1e6
        finally:
            gc.enable()
        calls[name] = partial(sp.fn, state=state)
        calls[name].state = state
    exact_remaining = getattr(grid, "remaining_to", None)      # road graphs
    if (grid.moves == 4 or exact_remaining) and any(sp.heuristic for sp in specs.values()):
        t = grid.index(case.goal)
        rem = exact_remaining(t) if exact_remaining else remaining_to_goal(grid, t)   # not timed
        for name, sp in specs.items():
            if sp.heuristic:
                h = sp.heuristic(grid, t, getattr(calls[name], "state", None))
                acc[name], over[name] = accuracy_stats(h, t, rem)
    return calls, prep, acc, over


def run_map(algos, case, info, optimal, repeats, writer, times, expanded):
    specs = {name: as_spec(v) for name, v in algos.items()}
    calls, prep, acc, over = prepare_all(specs, case)
    names = list(specs)
    for rep in range(repeats):
        k = rep % len(names)
        order = names[k:] + names[:k]
        for pos, name in enumerate(order):
            rule = specs[name].rule
            result, ms = timed_run(calls[name], case.grid, case.start, case.goal)
            check(name, result, optimal, case, rule)
            writer.writerow(dict(
                info, rep=rep, order=pos, algorithm=name, time_ms=f"{ms:.4f}",
                found=int(result.found),
                cost="" if result.cost is None else result.cost,
                steps="" if result.path is None else len(result.path) - 1,
                expanded=result.expanded, stale=result.stale,
                max_frontier=result.max_frontier, scanned=result.scanned,
                prep_ms=f"{prep[name]:.4f}" if name in prep else 0,
                h_accuracy="" if acc.get(name) is None else f"{acc[name]:.5f}",
                h_over="" if over.get(name) is None else f"{over[name]:.5f}",
            ))
            times[name].append(ms)
            expanded[name].append(result.expanded)
    return repeats * len(names)


def run_condition(exp, algos, c, n_maps, repeats, writer, log=print):
    algos = {k: v for k, v in algos.items() if not (as_spec(v).needs == "uniform" and not is_uniform(c))}
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
    case8, _ = build_map(cond(size=50, moves=8), BASE_SEED)
    case_dy, _ = build_map(dict(dyn(appear=0.1, clear=0.1), size=50), BASE_SEED)
    tables = [(MAIN, case), (TIE, case), (WEIGHTED, case), (HEAP, case), (BIDIR, case),
              (TRAFFIC, case), (MOVES8, case8), (HQ, case), (PLACED, case), (DYNAMIC, case_dy)]
    if roads.data_available():
        tables.append((ROADS, build_map(dict(road(roads.RUSH), quick=True), BASE_SEED)[0]))
    for algos, c in tables:
        calls, _, _, _ = prepare_all({n: as_spec(v) for n, v in algos.items()}, c)
        for fn in calls.values():
            fn(c.grid, c.start, c.goal)


def quick_version(conditions):
    out = []
    for c in conditions:
        c = dict(c)
        c["size"] = max(10, c["size"] // 8)     # keeps sizes distinct: 50,100,200,400 -> 10,12,25,50
        if c["kind"] == "road":
            c["quick"] = True                   # a small central part of Songdo
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
            elif exp in ("RM", "RY") and not roads.data_available():
                log(f"[{exp}] skipped: data/songdo_roads.json not found (run python3 tools/fetch_osm.py)")
                rows, settings = 0, dict(skipped="no road data")
            else:
                title, algos, conditions = EXPERIMENTS[exp]
                if quick:
                    conditions = quick_version(conditions)
                log(f"[{exp}] {title}   (median time / median expanded cells)")
                maps = n_maps * MAP_FACTOR.get(exp, 1)
                rows = sum(run_condition(exp, algos, c, maps, repeats, writer, log)
                           for c in conditions)
                settings = dict(title=title, algorithms=list(algos), conditions=conditions,
                                n_maps=maps, repeats=repeats)
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
