"""Grid map shared by all three search algorithms.

A map is an H x W grid stored as one flat list. Each cell holds its entry cost:
    0      -> obstacle (cannot enter)
    1..9   -> cost paid when moving INTO this cell

The start cell's own cost is never paid.

Two kinds of grid share this interface:
    Grid    4-directional moves; a move into cell v costs cost[v]
    Grid8   8-directional moves (see Grid8); costs are scaled by 10 so that
            straight and diagonal steps stay integers (10 and 14)
Every algorithm only uses neighbors(), move_cost() and heuristic(), so the
same search code runs on both.

Positions are (row, col) tuples at the public API, and flat indices
(row * width + col) inside the algorithms, because ints are cheaper to hash
and compare than tuples.
"""

WALL = 0


class Grid:
    moves = 4

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

    def move_cost(self, a_idx, b_idx):
        """Cost of one move from a to b (b must be a neighbor of a)."""
        return self.costs[b_idx]

    def is_move(self, a, b):
        """True if b is reachable from a in one move. a, b are (row, col)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    def heuristic(self, a_idx, b_idx):
        """Lower bound on the cost from a to b, used by A*."""
        return manhattan(self, a_idx, b_idx)

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


# ----- 8-directional grid ---------------------------------------------------

STRAIGHT = 10        # a straight step into a cell of cost c costs 10 * c
DIAGONAL = 14        # a diagonal step costs 14 * c (14 ~ 10 * sqrt(2))


class Grid8(Grid):
    """8-directional grid. Same cells and costs as Grid, more moves.

    Rules:
      - straight step into v costs STRAIGHT * cost[v], diagonal DIAGONAL * cost[v]
      - no corner cutting: a diagonal step is allowed only when both cells it
        squeezes past are open, so the path never touches a wall corner
        (this also means 8-dir and 4-dir have the same connected areas)
    Path costs are therefore 10x the 4-direction units; divide by 10 to compare.
    """

    moves = 8

    @classmethod
    def from_grid(cls, grid):
        return cls(grid.height, grid.width, grid.costs)

    def neighbors(self, idx):
        """Yield (neighbor_index, step_cost): up, down, left, right, then diagonals."""
        w = self.width
        costs = self.costs
        r, c = divmod(idx, w)
        up = r > 0 and costs[idx - w]
        down = r < self.height - 1 and costs[idx + w]
        left = c > 0 and costs[idx - 1]
        right = c < w - 1 and costs[idx + 1]
        if up:
            yield idx - w, STRAIGHT * up
        if down:
            yield idx + w, STRAIGHT * down
        if left:
            yield idx - 1, STRAIGHT * left
        if right:
            yield idx + 1, STRAIGHT * right
        if up and left and costs[idx - w - 1]:
            yield idx - w - 1, DIAGONAL * costs[idx - w - 1]
        if up and right and costs[idx - w + 1]:
            yield idx - w + 1, DIAGONAL * costs[idx - w + 1]
        if down and left and costs[idx + w - 1]:
            yield idx + w - 1, DIAGONAL * costs[idx + w - 1]
        if down and right and costs[idx + w + 1]:
            yield idx + w + 1, DIAGONAL * costs[idx + w + 1]

    def move_cost(self, a_idx, b_idx):
        ar, ac = divmod(a_idx, self.width)
        br, bc = divmod(b_idx, self.width)
        step = DIAGONAL if ar != br and ac != bc else STRAIGHT
        return step * self.costs[b_idx]

    def is_move(self, a, b):
        dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
        if (dr, dc) in ((0, 1), (1, 0)):
            return True
        if (dr, dc) == (1, 1):          # diagonal: both squeezed cells must be open
            return self.is_open((a[0], b[1])) and self.is_open((b[0], a[1]))
        return False

    def heuristic(self, a_idx, b_idx):
        return octile(self, a_idx, b_idx)


# ----- helpers used by algorithms and tests --------------------------------


def manhattan(grid, a_idx, b_idx):
    """Manhattan distance between two flat indices.

    Admissible for 4-directional moves with every entry cost >= 1,
    so A* with this heuristic still returns optimal paths on weighted maps.
    """
    ar, ac = divmod(a_idx, grid.width)
    br, bc = divmod(b_idx, grid.width)
    return abs(ar - br) + abs(ac - bc)


def octile(grid, a_idx, b_idx):
    """Octile distance between two flat indices, in Grid8 units (straight 10, diagonal 14).

    The cheapest way to cover the row difference dr and the column difference dc
    on an empty 8-direction grid with every cost 1: take min(dr, dc) diagonal steps
    (each 14), then max(dr, dc) - min(dr, dc) straight steps (each 10).
    This never exceeds the real cost (every cost is >= 1), so A* stays optimal.

    Example: dr = 3, dc = 5 -> 3 diagonal + 2 straight = 3 * 14 + 2 * 10 = 62.
    """
    # TODO: write this function yourself. Instructions: docs/todo_octile.md
    # Check it with: python3 -m unittest tests.test_moves8 -k Octile -v
    raise NotImplementedError("octile() is not written yet - see docs/todo_octile.md")


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
    """Total cost of a path: sum of move costs, start cell excluded."""
    idx = [grid.index(p) for p in path]
    return sum(grid.move_cost(a, b) for a, b in zip(idx, idx[1:]))


def is_valid_path(grid, path, start, goal):
    """Check that path starts/ends correctly, moves 1 step at a time, avoids walls."""
    if not path or path[0] != start or path[-1] != goal:
        return False
    for p in path:
        if not grid.is_open(p):
            return False
    for a, b in zip(path, path[1:]):
        if not grid.is_move(a, b):
            return False
    return True
