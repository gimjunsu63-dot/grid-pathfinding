"""Jump Point Search (JPS) on an 8-directional grid with uniform costs.

Idea: on an open grid with equal costs, many paths are "symmetric" (the same
length, just moves in a different order). A* expands all of them. JPS skips
along straight and diagonal lines without putting the cells on the heap, and
stops only at "jump points": cells where the best path might have to turn
(next to a wall corner), or the goal. Only jump points go on the heap.

Rules used here (the no-corner-cutting variant, matching Grid8):
  - a diagonal step needs both cells it squeezes past to be open
  - moving straight, stop at a cell if a wall just behind it on one side
    opens up beside it (a "forced" turn)
  - moving diagonally, stop at a cell if a straight scan from it in either
    of its two straight directions finds a jump point
Search order is the same as A* (f = g + octile, ties to larger g).

Requirements: a Grid8 whose open cells all cost 1 (JPS is only correct with
uniform costs). Costs are in Grid8 units (straight 10, diagonal 14), so the
returned cost equals Dijkstra's on the same Grid8.

Counters: expanded = jump points popped and processed (same meaning as A*'s
expanded cells), stale as in search.py, scanned = cells looked at while
jumping (the work JPS does instead of using the heap).
"""

from grid import DIAGONAL, STRAIGHT, WALL
from search import HEAPS, SearchResult, _trivial

ALL_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


def _line_cost(r0, c0, r1, c1):
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    return DIAGONAL * min(dr, dc) + STRAIGHT * (max(dr, dc) - min(dr, dc))


def _sign(x):
    return (x > 0) - (x < 0)


def jump_point_search(grid, start, goal, heap_impl="heapq", trace=None):
    """trace: optional dict; if given, fills trace["scanned"] (cells passed while jumping)
    and trace["jump_points"] (cells expanded), for the figures. Leave it None when timing."""
    if getattr(grid, "moves", 4) != 8:
        raise ValueError("JPS needs an 8-directional grid (Grid8)")
    if any(c not in (WALL, 1) for c in grid.costs):
        raise ValueError("JPS needs uniform costs (every open cell cost 1)")
    quick = _trivial(grid, start, goal)
    if quick is not None:
        return quick

    push, pop = HEAPS[heap_impl]
    if trace is not None:
        trace["scanned"], trace["jump_points"] = [], []
    H, W = grid.height, grid.width
    costs = grid.costs
    gr, gc = goal
    scanned = 0

    def is_open(r, c):
        return 0 <= r < H and 0 <= c < W and costs[r * W + c] != WALL

    def step_ok(r, c, dr, dc):
        """Can we move from (r, c) one step in direction (dr, dc)?"""
        if not is_open(r + dr, c + dc):
            return False
        if dr and dc:
            return is_open(r + dr, c) and is_open(r, c + dc)
        return True

    def jump(r, c, dr, dc):
        """Walk from (r, c) in direction (dr, dc); return the first jump point or None."""
        nonlocal scanned
        while step_ok(r, c, dr, dc):
            r += dr
            c += dc
            scanned += 1
            if trace is not None:
                trace["scanned"].append((r, c))
            if r == gr and c == gc:
                return r, c
            if dr and dc:
                if jump(r, c, dr, 0) is not None or jump(r, c, 0, dc) is not None:
                    return r, c
            elif dc:                                   # moving horizontally
                if (is_open(r - 1, c) and not is_open(r - 1, c - dc)) or \
                        (is_open(r + 1, c) and not is_open(r + 1, c - dc)):
                    return r, c
            else:                                      # moving vertically
                if (is_open(r, c - 1) and not is_open(r - dr, c - 1)) or \
                        (is_open(r, c + 1) and not is_open(r - dr, c + 1)):
                    return r, c
        return None

    def directions(r, c, parent):
        """Directions worth scanning from (r, c), given where we came from."""
        if parent is None:
            return ALL_DIRS
        pr, pc = parent
        dr, dc = _sign(r - pr), _sign(c - pc)
        out = []
        if dr and dc:
            if is_open(r + dr, c):
                out.append((dr, 0))
            if is_open(r, c + dc):
                out.append((0, dc))
            if is_open(r + dr, c) and is_open(r, c + dc):
                out.append((dr, dc))
        elif dc:
            if is_open(r, c + dc):
                out.append((0, dc))
                if is_open(r - 1, c):
                    out.append((-1, dc))
                if is_open(r + 1, c):
                    out.append((1, dc))
            if is_open(r - 1, c):
                out.append((-1, 0))
            if is_open(r + 1, c):
                out.append((1, 0))
        else:
            if is_open(r + dr, c):
                out.append((dr, 0))
                if is_open(r, c - 1):
                    out.append((dr, -1))
                if is_open(r, c + 1):
                    out.append((dr, 1))
            if is_open(r, c - 1):
                out.append((0, -1))
            if is_open(r, c + 1):
                out.append((0, 1))
        return out

    def h(r, c):
        return _line_cost(r, c, gr, gc)              # octile distance to the goal

    dist = {start: 0}
    parent = {start: None}
    closed = set()
    counter = 0
    heap = [(h(*start), 0, counter, 0, start)]
    expanded = 0
    stale = 0
    max_frontier = 1

    while heap:
        _, _, _, g, cur = pop(heap)
        if cur in closed or g > dist[cur]:
            stale += 1
            continue
        closed.add(cur)
        expanded += 1
        if trace is not None:
            trace["jump_points"].append(cur)
        if cur == goal:
            return SearchResult(_expand_path(parent, goal), g, expanded, stale, max_frontier,
                                scanned=scanned)
        r, c = cur
        for dr, dc in directions(r, c, parent[cur]):
            jp = jump(r, c, dr, dc)
            if jp is None or jp in closed:
                continue
            ng = g + _line_cost(r, c, *jp)
            if ng < dist.get(jp, float("inf")):
                dist[jp] = ng
                parent[jp] = cur
                counter += 1
                push(heap, (ng + h(*jp), -ng, counter, ng, jp))
        if len(heap) > max_frontier:
            max_frontier = len(heap)

    return SearchResult(None, None, expanded, stale, max_frontier, scanned=scanned)


def _expand_path(parent, goal):
    """Jump points -> every cell in between (they lie on straight or diagonal lines)."""
    points = []
    cur = goal
    while cur is not None:
        points.append(cur)
        cur = parent[cur]
    points.reverse()
    path = [points[0]]
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        dr, dc = _sign(r1 - r0), _sign(c1 - c0)
        r, c = r0, c0
        while (r, c) != (r1, c1):
            r += dr
            c += dc
            path.append((r, c))
    return path
