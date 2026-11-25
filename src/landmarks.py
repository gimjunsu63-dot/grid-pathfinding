"""Landmark heuristic (ALT: A*, Landmarks, Triangle inequality) and the accuracy ruler.

Landmark idea
-------------
Pick a few cells as landmarks and, before any search, run Dijkstra from each
landmark L to get d(L -> x) for every cell x. For a query with goal t, the
triangle inequality gives two lower bounds on the remaining cost d(v -> t):

    d(L -> t) <= d(L -> v) + d(v -> t)   =>   d(v -> t) >= d(L -> t) - d(L -> v)
    d(v -> L) <= d(v -> t) + d(t -> L)   =>   d(v -> t) >= d(v -> L) - d(t -> L)

h(v) = the largest of these bounds over all landmarks (and 0). Every bound is
<= the real remaining cost, so A* with this h still returns optimal paths.

Our grids charge the cost of the cell you ENTER, so d(a -> b) != d(b -> a).
Reversing a path swaps which end cell is paid for, which gives the identity

    d(b -> a) = d(a -> b) + cost[a] - cost[b]      (a, b in the same open area)

so one Dijkstra from L also gives d(x -> L) = d(L -> x) + cost[L] - cost[x],
and the second bound becomes  d(L -> v) - d(L -> t) - cost[v] + cost[t].

The ruler (not a search method)
-------------------------------
OracleHeuristic uses the REAL remaining cost d(v -> t) times k. Knowing it
requires one Dijkstra from the goal first, which already solves the problem,
so it is never a competitor. It is a ruler: with k = 0.5 every cell's estimate
is exactly 50% accurate, which lets us measure "how much search does an
estimate of accuracy k save?" and place Manhattan and landmarks on that curve.
k <= 1 keeps A* optimal; k > 1 overestimates and may lose optimality.

Accuracy
--------
accuracy(h) = mean over reachable cells v (v != goal) of h(v) / d(v -> t).
1.0 is perfect, 0.0 is Dijkstra (no information).

Old landmarks after the map changes (dynamic-environment experiment)
--------------------------------------------------------------------
Landmark distances measured on the OLD map give h_old(v) <= d_old(v -> t).
  - Costs only went up (new jams): d_old(v -> t) <= d_new(v -> t), so h_old is
    still a lower bound, and h_old(u) <= c_old(u, v) + h_old(v) <= c_new(u, v) + h_old(v)
    keeps it consistent. A* stays optimal, the estimate just gets less accurate.
  - Some costs went down (jams cleared): d_new can be smaller than d_old, so
    h_old may overestimate and A* may return a longer path.
"""

import heapq
import random

from grid import WALL

INF = float("inf")


# ---------------------------------------------------------------------------
# distances
# ---------------------------------------------------------------------------


def distances_from(grid, src):
    """Dijkstra from flat index src to every cell: list of d(src -> x), INF if unreachable."""
    n = grid.height * grid.width
    dist = [INF] * n
    if grid.costs[src] == WALL:
        return dist
    dist[src] = 0
    heap = [(0, src)]
    while heap:
        d, cur = heapq.heappop(heap)
        if d > dist[cur]:
            continue
        for nxt, step in grid.neighbors(cur):
            nd = d + step
            if nd < dist[nxt]:
                dist[nxt] = nd
                heapq.heappush(heap, (nd, nxt))
    return dist


def remaining_to_goal(grid, goal):
    """d(v -> goal) for every cell v (INF if v cannot reach the goal). 4-direction grids."""
    costs = grid.costs
    d_from_goal = distances_from(grid, goal)
    ct = costs[goal]
    return [d + ct - costs[v] if d < INF else INF for v, d in enumerate(d_from_goal)]


# ---------------------------------------------------------------------------
# choosing landmarks
# ---------------------------------------------------------------------------

PLACEMENTS = ("farthest", "corners", "random")


def _nearest_open(grid, r, c, area):
    """Open cell of `area` closest (Manhattan) to (r, c)."""
    best, best_d = None, INF
    for idx in area:
        rr, cc = divmod(idx, grid.width)
        d = abs(rr - r) + abs(cc - c)
        if d < best_d:
            best, best_d = idx, d
    return best


def choose_landmarks(grid, count, area, placement="farthest", seed=0):
    """Return (landmarks, distance arrays). `area` = cells of the start's open area.

    farthest  first: the cell farthest from the area's first cell, then each next
              landmark is the cell farthest from all chosen ones (greedy). The
              first k landmarks of a larger set are the same as a set of size k.
    corners   cells nearest the 4 corners, then the 4 edge midpoints, then farthest
    random    seeded random cells of the area
    """
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}")
    cells = sorted(area)
    count = min(count, len(cells))
    marks, dists = [], []

    def add(idx):
        marks.append(idx)
        dists.append(distances_from(grid, idx))

    if placement == "random":
        rng = random.Random(f"{seed}:landmarks")
        for idx in rng.sample(cells, count):
            add(idx)
        return marks, dists

    if placement == "corners":
        h, w = grid.height - 1, grid.width - 1
        spots = [(0, 0), (0, w), (h, 0), (h, w), (0, w // 2), (h // 2, 0), (h // 2, w), (h, w // 2)]
        for r, c in spots[:count]:
            idx = _nearest_open(grid, r, c, cells)
            if idx not in marks:
                add(idx)

    # farthest-point greedy (also fills the remaining "corners" slots)
    if not marks:
        first_dist = distances_from(grid, cells[0])
        add(max(cells, key=lambda i: first_dist[i]))
    nearest = [min(d[i] for d in dists) for i in range(len(dists[0]))]
    while len(marks) < count:
        nxt = max(cells, key=lambda i: nearest[i] if nearest[i] < INF else -1)
        add(nxt)
        new = dists[-1]
        nearest = [min(a, b) for a, b in zip(nearest, new)]
    return marks, dists


# ---------------------------------------------------------------------------
# heuristics
# ---------------------------------------------------------------------------


class LandmarkHeuristic:
    """h(v, t) = max over landmarks of the two triangle-inequality bounds (see top)."""

    def __init__(self, grid, landmarks, dists):
        self.costs = grid.costs
        self.landmarks = list(landmarks)
        self.dists = list(dists)
        self._t = None
        self._per_t = []

    def __len__(self):
        return len(self.landmarks)

    def __call__(self, v, t):
        if t != self._t:                                   # cache the goal's side once per query
            self._t = t
            self._per_t = [(dl, dl[t]) for dl in self.dists]
            self._ct = self.costs[t]
        cv_minus_ct = self.costs[v] - self._ct
        best = 0
        for dl, at in self._per_t:
            dv = dl[v]
            a = at - dv                    # d(L->t) - d(L->v)
            if a > best:
                best = a
            b = dv - at - cv_minus_ct      # d(v->L) - d(t->L)
            if b > best:
                best = b
        return best


class OracleHeuristic:
    """k x the real remaining cost. A ruler for accuracy, not a search method."""

    def __init__(self, remaining, goal, k):
        self.remaining = remaining
        self.goal = goal
        self.k = k

    def __call__(self, v, t):
        if t != self.goal:
            raise ValueError("OracleHeuristic was built for another goal")
        return self.k * self.remaining[v]


def manhattan_heuristic(grid):
    return grid.heuristic


def zero_heuristic(v, t):
    return 0


def accuracy_stats(h, goal, remaining):
    """Over cells that can reach the goal (goal excluded):
    (mean of h(v) / d(v -> goal), share of cells where h(v) > d(v -> goal)).

    The second number is 0 for every admissible heuristic. It becomes positive
    for the ruler with k > 1 and for landmarks measured before jams cleared.
    """
    total, over, n = 0.0, 0, 0
    for v, rem in enumerate(remaining):
        if rem == INF or rem == 0 or v == goal:
            continue
        est = h(v, goal)
        total += est / rem
        if est > rem + 1e-9:
            over += 1
        n += 1
    return (total / n, over / n) if n else (None, None)


def accuracy(h, goal, remaining):
    """Mean of h(v) / d(v -> goal) over cells that can reach the goal (goal excluded)."""
    return accuracy_stats(h, goal, remaining)[0]


def landmark_heuristic(grid, start, count, placement="farthest", seed=0):
    """Build a LandmarkHeuristic for the open area that contains `start` (precomputation)."""
    from maps import component              # local import: maps imports grid, not landmarks
    area = component(grid, start)
    marks, dists = choose_landmarks(grid, count, area, placement, seed)
    return LandmarkHeuristic(grid, marks, dists)
