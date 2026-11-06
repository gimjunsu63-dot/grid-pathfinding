"""Map generators for the experiments. Standard library only.

Every generator is deterministic: the same arguments and seed always give the
same map. Walls and cell costs come from two separate random streams, so a
unit-cost map and a weighted map with the same seed have exactly the same
walls. That lets experiment D change the costs and nothing else.

Kinds (see make_map):
    random   random obstacles with probability wall_ratio
    maze     recursive-backtracker maze, built with an explicit stack
             (no recursion limit), optional extra openings (loop_ratio)
    nopath   random map whose goal is sealed off by walls
    uwall    open map with a U-shaped wall facing the start (an A* trap)
    terrain  clustered costs 1..9 from value noise, no walls
    city     roads between building blocks; arterial roads are cheap,
             local roads cost more, some road cells are closed

Endpoints are fixed at opposite corners unless noted. pick_random_endpoints()
replaces them with two random cells of the same connected component.

Run this file to print a small sample of every kind:
    python3 src/maps.py
"""

import random
from collections import deque
from dataclasses import dataclass, field, replace

from grid import WALL, Grid, manhattan


@dataclass
class MapCase:
    kind: str
    grid: Grid
    start: tuple
    goal: tuple
    seed: int
    attempts: int = 1          # maps drawn before one was accepted (connectivity rule)
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rng(seed, stream):
    # str seeds are hashed with SHA-512 by the random module, so this is
    # stable across runs and machines (not affected by PYTHONHASHSEED).
    return random.Random(f"{seed}:{stream}")


def _weighted_cost(cost_rng):
    return 1 + int(cost_rng.random() * 9)      # 1..9


def component(grid, start):
    """Set of flat indices reachable from start (4-directional)."""
    if not grid.is_open(start):
        return set()
    s = grid.index(start)
    seen = {s}
    queue = deque([s])
    while queue:
        cur = queue.popleft()
        for nxt, _ in grid.neighbors(cur):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def is_connected(grid, start, goal):
    return grid.is_open(start) and grid.is_open(goal) and grid.index(goal) in component(grid, start)


def open_cell_count(grid):
    return sum(1 for c in grid.costs if c != WALL)


# ---------------------------------------------------------------------------
# random obstacles
# ---------------------------------------------------------------------------


def random_map(height, width, seed, wall_ratio=0.2, weighted=False,
               require_path=True, max_attempts=1000):
    """Each cell is a wall with probability wall_ratio.

    Connectivity rule: if start and goal are not connected, draw a new wall
    layout (same seed, next draws of the stream) and count the attempt.
    Start and goal cells are always kept open.
    """
    wall_rng = _rng(seed, "walls")
    cost_rng = _rng(seed, "costs")
    n = height * width
    costs = [_weighted_cost(cost_rng) if weighted else 1 for _ in range(n)]
    start, goal = (0, 0), (height - 1, width - 1)
    s_idx, g_idx = 0, n - 1

    for attempt in range(1, max_attempts + 1):
        cells = [WALL if wall_rng.random() < wall_ratio else c for c in costs]
        cells[s_idx] = costs[s_idx]
        cells[g_idx] = costs[g_idx]
        grid = Grid(height, width, cells)
        if not require_path or is_connected(grid, start, goal):
            return MapCase("random", grid, start, goal, seed, attempt,
                           dict(wall_ratio=wall_ratio, weighted=weighted))
    raise RuntimeError(f"no connected map after {max_attempts} attempts")


# ---------------------------------------------------------------------------
# maze
# ---------------------------------------------------------------------------


def maze_map(height, width, seed, loop_ratio=0.0, weighted=False):
    """Recursive backtracker, written with an explicit stack.

    Maze cells sit at even (row, col); the cells between them are walls that
    get carved. With loop_ratio = 0 the maze is perfect (exactly one path
    between any two cells). loop_ratio > 0 opens extra walls to make loops.
    If a side length is even, the last row/column stays wall, so the goal is
    the bottom-right-most maze cell.
    """
    rng = _rng(seed, "maze")
    cost_rng = _rng(seed, "costs")
    rows = (height + 1) // 2
    cols = (width + 1) // 2
    cells = [WALL] * (height * width)

    def carve(r, c):
        cells[r * width + c] = _weighted_cost(cost_rng) if weighted else 1

    visited = [[False] * cols for _ in range(rows)]
    visited[0][0] = True
    carve(0, 0)
    stack = [(0, 0)]
    while stack:
        i, j = stack[-1]
        options = []
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols and not visited[ni][nj]:
                options.append((ni, nj))
        if not options:
            stack.pop()
            continue
        ni, nj = options[int(rng.random() * len(options))]
        visited[ni][nj] = True
        carve(i + ni, j + nj)          # wall cell between (2i,2j) and (2ni,2nj)
        carve(2 * ni, 2 * nj)
        stack.append((ni, nj))

    if loop_ratio > 0:
        for r in range(height):
            for c in range(width):
                between_h = r % 2 == 0 and c % 2 == 1 and c + 1 < width
                between_v = r % 2 == 1 and c % 2 == 0 and r + 1 < height
                if (between_h or between_v) and cells[r * width + c] == WALL:
                    if rng.random() < loop_ratio:
                        carve(r, c)

    goal = (2 * (rows - 1), 2 * (cols - 1))
    return MapCase("maze", Grid(height, width, cells), (0, 0), goal, seed, 1,
                   dict(loop_ratio=loop_ratio, weighted=weighted))


# ---------------------------------------------------------------------------
# no path
# ---------------------------------------------------------------------------


def nopath_map(height, width, seed, wall_ratio=0.2, weighted=False):
    """A connected random map, then the goal corner is sealed off.

    Every algorithm has to search the whole component of the start before
    it can say "no path", which is the worst case for all three.
    """
    base = random_map(height, width, seed, wall_ratio, weighted, require_path=True)
    cells = list(base.grid.costs)
    gr, gc = base.goal
    for r, c in ((gr - 1, gc), (gr, gc - 1)):
        if 0 <= r < height and 0 <= c < width:
            cells[r * width + c] = WALL
    grid = Grid(height, width, cells)
    return MapCase("nopath", grid, base.start, base.goal, seed, base.attempts,
                   dict(wall_ratio=wall_ratio, weighted=weighted))


# ---------------------------------------------------------------------------
# U-shaped wall
# ---------------------------------------------------------------------------


def uwall_map(height, width, seed, wall_ratio=0.0, weighted=False):
    """Open map with a U-shaped wall whose opening faces the start.

    The straight line from start to goal runs into the cup, so A* (which
    follows the heuristic) fills the inside of the cup before going around.
    Optional light random obstacles via wall_ratio.
    """
    wall_rng = _rng(seed, "walls")
    cost_rng = _rng(seed, "costs")
    cells = []
    for _ in range(height * width):
        if wall_ratio > 0 and wall_rng.random() < wall_ratio:
            cells.append(WALL)
        else:
            cells.append(_weighted_cost(cost_rng) if weighted else 1)

    top, bottom = height // 4, height - 1 - height // 4
    right = width * 6 // 10
    left = right - width * 3 // 10
    for r in range(top, bottom + 1):
        cells[r * width + right] = WALL
    for c in range(left, right + 1):
        cells[top * width + c] = WALL
        cells[bottom * width + c] = WALL

    start = (height // 2, width // 10)
    goal = (height // 2, width - 1 - width // 10)
    for r, c in (start, goal):
        if cells[r * width + c] == WALL:
            cells[r * width + c] = 1
    grid = Grid(height, width, cells)
    if not is_connected(grid, start, goal):
        raise RuntimeError("uwall map is not connected; lower wall_ratio")
    return MapCase("uwall", grid, start, goal, seed, 1,
                   dict(wall_ratio=wall_ratio, weighted=weighted))


# ---------------------------------------------------------------------------
# terrain (clustered costs)
# ---------------------------------------------------------------------------


def _value_noise(height, width, rng, scale):
    """Smooth random field in [0, 1): random values on a coarse lattice,
    interpolated with a smoothstep curve."""
    gh = height // scale + 2
    gw = width // scale + 2
    lattice = [[rng.random() for _ in range(gw)] for _ in range(gh)]
    out = []
    for r in range(height):
        y = r / scale
        i = int(y)
        ty = y - i
        ty = ty * ty * (3 - 2 * ty)
        row0, row1 = lattice[i], lattice[i + 1]
        for c in range(width):
            x = c / scale
            j = int(x)
            tx = x - j
            tx = tx * tx * (3 - 2 * tx)
            top = row0[j] + (row0[j + 1] - row0[j]) * tx
            bot = row1[j] + (row1[j + 1] - row1[j]) * tx
            out.append(top + (bot - top) * ty)
    return out


def terrain_map(height, width, seed, scale=None):
    """Costs 1..9 that come in patches (hills, mud, flat ground).

    Two octaves of value noise, stretched to the full 1..9 range.
    No walls, so the map is always connected.
    """
    rng = _rng(seed, "terrain")
    if scale is None:
        scale = max(4, min(height, width) // 8)
    coarse = _value_noise(height, width, rng, scale)
    fine = _value_noise(height, width, rng, max(2, scale // 2))
    field_ = [a + 0.5 * b for a, b in zip(coarse, fine)]
    lo, hi = min(field_), max(field_)
    span = (hi - lo) or 1.0
    cells = [min(9, 1 + int((v - lo) / span * 9)) for v in field_]
    return MapCase("terrain", Grid(height, width, cells), (0, 0), (height - 1, width - 1),
                   seed, 1, dict(scale=scale, weighted=True))


# ---------------------------------------------------------------------------
# city blocks
# ---------------------------------------------------------------------------


def city_map(height, width, seed, block=6, arterial_every=4, weighted=True,
             closure_ratio=0.05, jam_ratio=0.1, max_attempts=1000):
    """Road grid between building blocks.

    Roads run every (block + 1) cells plus along the bottom and right edges.
    Buildings are walls. With weighted=True:
        arterial road (every arterial_every-th road)   cost 1
        local road                                     cost 3
        traffic jam (jam_ratio of local road cells)    cost 9
    With weighted=False every road cell costs 1.
    closure_ratio of road cells are closed (walls); if that disconnects start
    and goal, closures are drawn again and the attempt is counted.
    """
    step = block + 1
    cost_rng = _rng(seed, "costs")
    close_rng = _rng(seed, "closures")

    def is_road_line(k, last):
        return k % step == 0 or k == last

    def is_arterial(k, last):
        return k % step == 0 and (k // step) % arterial_every == 0

    roads = []
    base = []
    for r in range(height):
        road_r = is_road_line(r, height - 1)
        for c in range(width):
            road_c = is_road_line(c, width - 1)
            if not (road_r or road_c):
                base.append(WALL)
                continue
            roads.append(r * width + c)
            if not weighted:
                base.append(1)
            elif (road_r and is_arterial(r, height - 1)) or (road_c and is_arterial(c, width - 1)):
                base.append(1)
            else:
                base.append(9 if cost_rng.random() < jam_ratio else 3)

    start, goal = (0, 0), (height - 1, width - 1)
    keep = {0, height * width - 1}
    for attempt in range(1, max_attempts + 1):
        cells = list(base)
        for idx in roads:
            if idx not in keep and close_rng.random() < closure_ratio:
                cells[idx] = WALL
        grid = Grid(height, width, cells)
        if is_connected(grid, start, goal):
            return MapCase("city", grid, start, goal, seed, attempt,
                           dict(block=block, weighted=weighted, closure_ratio=closure_ratio))
    raise RuntimeError(f"no connected city map after {max_attempts} attempts")


# ---------------------------------------------------------------------------
# endpoints and dispatch
# ---------------------------------------------------------------------------


def pick_random_endpoints(case, min_distance=None, max_attempts=10000):
    """Replace start/goal with two random cells of the start's component.

    The pair must be at least min_distance apart (Manhattan), default
    (height + width) / 4, so trivial neighbors are not picked.
    """
    grid = case.grid
    rng = _rng(case.seed, "endpoints")
    if min_distance is None:
        min_distance = (grid.height + grid.width) // 4
    cells = sorted(component(grid, case.start))     # sorted: deterministic order
    if len(cells) < 2:
        raise RuntimeError("component too small for random endpoints")
    for _ in range(max_attempts):
        a = cells[int(rng.random() * len(cells))]
        b = cells[int(rng.random() * len(cells))]
        if manhattan(grid, a, b) >= min_distance:
            params = dict(case.params, endpoints="random")
            return replace(case, start=grid.pos(a), goal=grid.pos(b), params=params)
    raise RuntimeError("could not find endpoints far enough apart")


MAP_KINDS = {
    "random": random_map,
    "maze": maze_map,
    "nopath": nopath_map,
    "uwall": uwall_map,
    "terrain": terrain_map,
    "city": city_map,
}


def make_map(kind, size, seed, endpoints="fixed", **params):
    """Square size x size map of the given kind."""
    if kind not in MAP_KINDS:
        raise ValueError(f"unknown map kind {kind!r}; choose from {sorted(MAP_KINDS)}")
    case = MAP_KINDS[kind](size, size, seed, **params)
    if endpoints == "random":
        if kind == "nopath":
            raise ValueError("nopath maps keep fixed endpoints")
        case = pick_random_endpoints(case)
    elif endpoints != "fixed":
        raise ValueError("endpoints must be 'fixed' or 'random'")
    return case


if __name__ == "__main__":
    samples = [
        ("random", dict(wall_ratio=0.25)),
        ("random", dict(wall_ratio=0.25, weighted=True)),
        ("maze", {}),
        ("maze", dict(loop_ratio=0.15)),
        ("nopath", {}),
        ("uwall", {}),
        ("terrain", {}),
        ("city", dict(block=4)),
    ]
    for kind, params in samples:
        case = make_map(kind, 21, seed=1, **params)
        label = ", ".join(f"{k}={v}" for k, v in params.items()) or "default"
        print(f"--- {kind} ({label}) start={case.start} goal={case.goal} attempts={case.attempts}")
        print(case.grid.render(start=case.start, goal=case.goal))
        print()
