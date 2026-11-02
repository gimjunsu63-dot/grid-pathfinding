"""Grid map shared by all three search algorithms.

A map is an H x W grid stored as one flat list. Each cell holds its entry cost:
    0      -> obstacle (cannot enter)
    1..9   -> cost paid when moving INTO this cell

The start cell's own cost is never paid. Movement is 4-directional.

Positions are (row, col) tuples at the public API, and flat indices
(row * width + col) inside the algorithms, because ints are cheaper to hash
and compare than tuples.
"""

WALL = 0


class Grid:
    def __init__(self, height, width, costs):
        if len(costs) != height * width:
            raise ValueError("costs length must be height * width")
        self.height = height
        self.width = width
        self.costs = costs  # flat list, 0 = wall

    # ----- construction -------------------------------------------------

    @classmethod
    def from_strings(cls, rows):
        """Build a grid from text rows.

        '#' = wall, '.' = cost 1, '1'..'9' = that cost.
        Handy for hand-written test cases.
        """
        height = len(rows)
        width = len(rows[0])
        costs = []
        for row in rows:
            if len(row) != width:
                raise ValueError("all rows must have the same length")
            for ch in row:
                if ch == "#":
                    costs.append(WALL)
                elif ch == ".":
                    costs.append(1)
                elif ch in "123456789":
                    costs.append(int(ch))
                else:
                    raise ValueError(f"unknown cell character: {ch!r}")
        return cls(height, width, costs)

    # ----- index helpers -------------------------------------------------

    def index(self, pos):
        r, c = pos
        return r * self.width + c

    def pos(self, idx):
        return divmod(idx, self.width)

    def in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def is_open(self, pos):
        return self.in_bounds(pos) and self.costs[self.index(pos)] != WALL

    # ----- neighbors (the one function every algorithm shares) -----------

    def neighbors(self, idx):
        """Yield (neighbor_index, entry_cost) for open 4-neighbors of idx.

        Order is fixed (up, down, left, right) so every algorithm sees
        neighbors in the same order.
        """
        w = self.width
        costs = self.costs
        r, c = divmod(idx, w)
        if r > 0:
            n = idx - w
            if costs[n]:
                yield n, costs[n]
        if r < self.height - 1:
            n = idx + w
            if costs[n]:
                yield n, costs[n]
        if c > 0:
            n = idx - 1
            if costs[n]:
                yield n, costs[n]
        if c < w - 1:
            n = idx + 1
            if costs[n]:
                yield n, costs[n]

    # ----- display -------------------------------------------------------

    def render(self, path=None, start=None, goal=None):
        """Return the grid as text. Path cells are drawn as '*'."""
        on_path = set(path or [])
        lines = []
        for r in range(self.height):
            chars = []
            for c in range(self.width):
                p = (r, c)
                cost = self.costs[self.index(p)]
                if p == start:
                    chars.append("S")
                elif p == goal:
                    chars.append("G")
                elif cost == WALL:
                    chars.append("#")
                elif p in on_path:
                    chars.append("*")
                else:
                    chars.append("." if cost == 1 else str(cost))
            lines.append("".join(chars))
        return "\n".join(lines)


# ----- helpers used by algorithms and tests --------------------------------


def manhattan(grid, a_idx, b_idx):
    """Manhattan distance between two flat indices.

    Admissible for 4-directional moves with every entry cost >= 1,
    so A* with this heuristic still returns optimal paths on weighted maps.
    """
    ar, ac = divmod(a_idx, grid.width)
    br, bc = divmod(b_idx, grid.width)
    return abs(ar - br) + abs(ac - bc)


def reconstruct_path(grid, parent, start_idx, goal_idx):
    """Follow parent links back from goal to start. Returns list of (r, c)."""
    path = []
    cur = goal_idx
    while cur != start_idx:
        path.append(grid.pos(cur))
        cur = parent[cur]
    path.append(grid.pos(start_idx))
    path.reverse()
    return path


def path_cost(grid, path):
    """Total cost of a path: sum of entry costs, start cell excluded."""
    return sum(grid.costs[grid.index(p)] for p in path[1:])


def is_valid_path(grid, path, start, goal):
    """Check that path starts/ends correctly, moves 1 step at a time, avoids walls."""
    if not path or path[0] != start or path[-1] != goal:
        return False
    for p in path:
        if not grid.is_open(p):
            return False
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return False
    return True
