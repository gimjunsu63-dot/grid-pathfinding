"""Bidirectional Dijkstra: search from the start and from the goal at once.

Same inputs and SearchResult as the other algorithms. Works for unit and
weighted maps.

Costs are paid when ENTERING a cell, so the two directions pay differently:
    forward  (from start)  moving cur -> nxt   costs cost[nxt]
    backward (from goal)   moving cur -> nxt   means nxt -> cur going forward,
                                               which costs cost[cur]
With that, dist_f[x] + dist_b[x] is the cost of the best known path through x.

Rules:
  - each step expands one cell from the side whose heap is smaller
  - every relaxation that reaches a cell already seen by the other side is a
    candidate path; the best one is kept (best)
  - stop when top_f + top_b >= best: no unexplored path can be cheaper
    (lazy heaps may hold stale tops, which only makes the test stricter)
  - if either side runs out of cells, the search is over; with no candidate
    there is no path. A sealed-off goal is found after very few expansions.

Counters follow search.py: expanded = cells popped and processed on both
sides, stale = outdated heap entries skipped, max_frontier = largest total
size of both heaps.
"""

from grid import path_cost
from search import HEAPS, SearchResult, _trivial


def bidirectional_dijkstra(grid, start, goal, heap_impl="heapq"):
    quick = _trivial(grid, start, goal)
    if quick is not None:
        return quick
    push, pop = HEAPS[heap_impl]
    costs = grid.costs
    s = grid.index(start)
    t = grid.index(goal)

    dist = ({s: 0}, {t: 0})
    parent = ({s: s}, {t: t})
    closed = (set(), set())
    heaps = ([(0, 0, s)], [(0, 0, t)])
    counter = 0
    best = float("inf")
    meet = None
    expanded = 0
    stale = 0
    max_frontier = 2

    while heaps[0] and heaps[1]:
        if heaps[0][0][0] + heaps[1][0][0] >= best:
            break
        side = 0 if len(heaps[0]) <= len(heaps[1]) else 1
        other = 1 - side
        g, _, cur = pop(heaps[side])
        if cur in closed[side] or g > dist[side][cur]:
            stale += 1
            continue
        closed[side].add(cur)
        expanded += 1
        my_dist, my_parent, my_heap = dist[side], parent[side], heaps[side]
        their_dist = dist[other]
        for nxt, entry_cost in grid.neighbors(cur):
            if nxt in closed[side]:
                continue
            ng = g + (entry_cost if side == 0 else costs[cur])
            if ng < my_dist.get(nxt, float("inf")):
                my_dist[nxt] = ng
                my_parent[nxt] = cur
                counter += 1
                push(my_heap, (ng, counter, nxt))
            if nxt in their_dist:
                total = my_dist[nxt] + their_dist[nxt]
                if total < best:
                    best = total
                    meet = nxt
        size = len(heaps[0]) + len(heaps[1])
        if size > max_frontier:
            max_frontier = size

    if meet is None:
        return SearchResult(None, None, expanded, stale, max_frontier)

    # start ... meet from the forward parents, meet ... goal from the backward ones
    front = []
    cur = meet
    while cur != s:
        front.append(cur)
        cur = parent[0][cur]
    front.append(s)
    front.reverse()
    back = []
    cur = meet
    while cur != t:
        cur = parent[1][cur]
        back.append(cur)
    path = [grid.pos(i) for i in front + back]
    return SearchResult(path, path_cost(grid, path), expanded, stale, max_frontier)
