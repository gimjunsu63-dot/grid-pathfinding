"""BFS, Dijkstra and A* on a Grid, with the same inputs and outputs.

All three:
  - take (grid, start, goal) with start/goal as (row, col)
  - stop when the goal is POPPED from the frontier (not when it is pushed),
    so "expanded" means the same thing for every algorithm
  - return a SearchResult

Dijkstra and A* are the same function (best_first). Dijkstra is simply A*
with h = 0. Sharing the code means any speed difference between them comes
from the heuristic, not from different implementation habits.

Counters (definitions are fixed here and reused in the report):
  expanded      cells popped and processed (stale heap entries NOT counted)
  stale         heap entries popped and skipped because a better g was
                already found (lazy deletion; always 0 for BFS)
  max_frontier  largest size the queue/heap reached, stale entries included
                (that is the memory actually held)
"""

import heapq
from collections import deque
from dataclasses import dataclass

import binheap
from grid import manhattan, path_cost, reconstruct_path


@dataclass
class SearchResult:
    path: list | None       # list of (r, c) from start to goal, or None
    cost: int | None        # total entry cost of path, or None
    expanded: int = 0
    stale: int = 0
    max_frontier: int = 0

    @property
    def found(self):
        return self.path is not None


def _trivial(grid, start, goal):
    """Handle cases every algorithm must answer the same way.

    Returns a SearchResult if the answer is immediate, else None.
    """
    if not grid.is_open(start) or not grid.is_open(goal):
        return SearchResult(path=None, cost=None)
    if start == goal:
        return SearchResult(path=[start], cost=0, expanded=1, max_frontier=1)
    return None


# ---------------------------------------------------------------------------
# BFS
# ---------------------------------------------------------------------------


def bfs(grid, start, goal):
    """Breadth-first search. Optimal only when every entry cost is equal.

    Ignores cell costs while searching (finds the path with fewest moves),
    but reports the real cost of the path it found.
    """
    quick = _trivial(grid, start, goal)
    if quick is not None:
        return quick

    s = grid.index(start)
    g = grid.index(goal)
    parent = {s: s}          # doubles as the visited set
    queue = deque([s])
    expanded = 0
    max_frontier = 1

    while queue:
        cur = queue.popleft()
        expanded += 1
        if cur == g:
            path = reconstruct_path(grid, parent, s, g)
            return SearchResult(path, path_cost(grid, path), expanded, 0, max_frontier)
        for nxt, _ in grid.neighbors(cur):
            if nxt not in parent:    # mark visited on push, so each cell enters once
                parent[nxt] = cur
                queue.append(nxt)
        if len(queue) > max_frontier:
            max_frontier = len(queue)

    return SearchResult(None, None, expanded, 0, max_frontier)


# ---------------------------------------------------------------------------
# Dijkstra / A*  (one shared implementation)
# ---------------------------------------------------------------------------

TIE_BREAKS = ("high_g", "low_g", "fifo")

# (push, pop) pairs. "heapq" is the standard module (core in C),
# "python" is binheap.py, a plain-Python binary heap with the same behavior.
HEAPS = {
    "heapq": (heapq.heappush, heapq.heappop),
    "python": (binheap.push, binheap.pop),
}


def best_first(grid, start, goal, use_heuristic=True, weight=1.0, tie_break="high_g",
               heap_impl="heapq"):
    """Pop the cell with smallest f = g + weight * h.

    use_heuristic=False -> h = 0 -> Dijkstra.
    weight > 1          -> weighted A* (faster, may lose optimality).
    tie_break decides the order among equal f:
        "high_g" prefer larger g (cells closer to the goal)  <- default
        "low_g"  prefer smaller g
        "fifo"   insertion order
    heap_impl picks the priority queue: "heapq" (default) or "python".
    Neither has decrease-key, so a cell may be pushed more than once;
    outdated entries are skipped when popped (lazy deletion).
    """
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"tie_break must be one of {TIE_BREAKS}")
    if heap_impl not in HEAPS:
        raise ValueError(f"heap_impl must be one of {tuple(HEAPS)}")
    push, pop = HEAPS[heap_impl]
    quick = _trivial(grid, start, goal)
    if quick is not None:
        return quick

    s = grid.index(start)
    t = grid.index(goal)

    if use_heuristic:
        def h(idx):
            return weight * manhattan(grid, idx, t)
    else:
        def h(idx):
            return 0

    # second key of the heap entry implements the tie-break
    if tie_break == "high_g":
        def tie(gval):
            return -gval
    elif tie_break == "low_g":
        def tie(gval):
            return gval
    else:
        def tie(gval):
            return 0

    dist = {s: 0}
    parent = {s: s}
    closed = set()
    counter = 0          # last key: keeps heap order stable and avoids comparing more fields
    heap = [(h(s), tie(0), counter, 0, s)]
    expanded = 0
    stale = 0
    max_frontier = 1

    while heap:
        _, _, _, gcur, cur = pop(heap)
        if cur in closed or gcur > dist[cur]:
            stale += 1
            continue
        closed.add(cur)
        expanded += 1
        if cur == t:
            path = reconstruct_path(grid, parent, s, t)
            return SearchResult(path, gcur, expanded, stale, max_frontier)
        for nxt, step in grid.neighbors(cur):
            if nxt in closed:
                continue
            ng = gcur + step
            if ng < dist.get(nxt, float("inf")):
                dist[nxt] = ng
                parent[nxt] = cur
                counter += 1
                push(heap, (ng + h(nxt), tie(ng), counter, ng, nxt))
        if len(heap) > max_frontier:
            max_frontier = len(heap)

    return SearchResult(None, None, expanded, stale, max_frontier)


def dijkstra(grid, start, goal, tie_break="high_g", heap_impl="heapq"):
    return best_first(grid, start, goal, use_heuristic=False, tie_break=tie_break,
                      heap_impl=heap_impl)


def astar(grid, start, goal, weight=1.0, tie_break="high_g", heap_impl="heapq"):
    return best_first(grid, start, goal, use_heuristic=True, weight=weight,
                      tie_break=tie_break, heap_impl=heap_impl)


ALGORITHMS = {
    "bfs": bfs,
    "dijkstra": dijkstra,
    "astar": astar,
}
