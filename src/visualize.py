"""Draw what each algorithm explored, as PNG images. Standard library only.

    python3 src/visualize.py            # writes figures/maps/*.png, figures/maps8/*.png + figures/README.md

For every map kind, one PNG with three panels side by side:
    BFS | Dijkstra | A*
Colors:
    dark gray        wall
    light -> darker  open cell, cost 1 -> 9
    tinted           cell the algorithm expanded
    strong color     the path it returned
    red / purple     start / goal

How the explored cells are recorded without touching search.py:
every algorithm calls grid.neighbors(cell) exactly once for each cell it
expands (except the goal, where it stops). RecordingGrid logs those calls,
so the picture shows the real code's behavior, in the real order.
"""

import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench import timed_run  # noqa: E402
from grid import WALL, Grid, Grid8  # noqa: E402
from jps import jump_point_search  # noqa: E402
from maps import make_map  # noqa: E402
from search import ALGORITHMS  # noqa: E402
from svgchart import ALGO_COLORS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"

# One color per algorithm, the same as in every chart (svgchart.ALGO_COLORS).
def _rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _shade(rgb, amount):
    return tuple(round(c * (1 - amount)) for c in rgb)


COLORS = {name: {"explored": _rgb(ALGO_COLORS[name]), "path": _shade(_rgb(ALGO_COLORS[name]), 0.45)}
          for name in ("bfs", "dijkstra", "astar", "jps")}
JUMP_POINT_COLOR = (20, 20, 20)
WALL_COLOR = (45, 45, 45)
START_COLOR = (220, 20, 60)
GOAL_COLOR = (120, 40, 200)
BACKGROUND = (255, 255, 255)
EXPLORED_ALPHA = 0.45
GAP = 10


class RecordingGrid(Grid):
    """A Grid that remembers which cells had their neighbors requested."""

    def __init__(self, grid):
        super().__init__(grid.height, grid.width, grid.costs)
        self.order = []

    def neighbors(self, idx):
        self.order.append(idx)
        return super().neighbors(idx)


class RecordingGrid8(Grid8):
    """The same for 8-directional grids."""

    def __init__(self, grid):
        super().__init__(grid.height, grid.width, grid.costs)
        self.order = []

    def neighbors(self, idx):
        self.order.append(idx)
        return super().neighbors(idx)


def explored_cells(name, grid, start, goal):
    """Run one algorithm and return (result, expanded cells in order)."""
    rec = RecordingGrid8(grid) if grid.moves == 8 else RecordingGrid(grid)
    result = ALGORITHMS[name](rec, start, goal)
    order = list(rec.order)
    if result.found and start != goal:
        order.append(grid.index(goal))          # expanded, but neighbors not asked
    if len(order) != result.expanded and start != goal:
        raise AssertionError(f"{name}: recorded {len(order)} cells, reported {result.expanded}")
    return result, order


# ---------------------------------------------------------------------------
# tiny PNG writer
# ---------------------------------------------------------------------------


class Canvas:
    def __init__(self, width, height, color=BACKGROUND):
        self.width = width
        self.height = height
        self.pixels = bytearray(bytes(color) * (width * height))

    def rect(self, x, y, w, h, color):
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + w), min(self.height, y + h)
        if x0 >= x1 or y0 >= y1:
            return
        row = bytes(color) * (x1 - x0)
        for yy in range(y0, y1):
            start = (yy * self.width + x0) * 3
            self.pixels[start:start + len(row)] = row

    def save_png(self, path):
        raw = bytearray()
        stride = self.width * 3
        for y in range(self.height):
            raw.append(0)                              # filter type: none
            raw += self.pixels[y * stride:(y + 1) * stride]

        def chunk(tag, data):
            body = tag + data
            return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

        header = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)  # 8-bit RGB
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + \
            chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")
        Path(path).write_bytes(png)


def _mix(base, color, alpha):
    return tuple(round(b + (c - b) * alpha) for b, c in zip(base, color))


def _cost_color(cost):
    # cost 1 -> near white, cost 9 -> warm gray
    t = (cost - 1) / 8
    return _mix((248, 248, 248), (150, 135, 120), t)


def draw_panel(canvas, ox, oy, px, grid, start, goal, expanded, path, colors):
    explored = set(expanded)
    on_path = {grid.index(p) for p in (path or [])}
    for idx, cost in enumerate(grid.costs):
        r, c = divmod(idx, grid.width)
        if cost == WALL:
            color = WALL_COLOR
        else:
            color = _cost_color(cost)
            if idx in explored:
                color = _mix(color, colors["explored"], EXPLORED_ALPHA)
            if idx in on_path:
                color = colors["path"]
        canvas.rect(ox + c * px, oy + r * px, px, px, color)
    # endpoints drawn a bit larger so they stay visible on big maps
    m = max(px + 4, 8)
    for (r, c), color in ((start, START_COLOR), (goal, GOAL_COLOR)):
        cx, cy = ox + c * px + px // 2, oy + r * px + px // 2
        canvas.rect(cx - m // 2 - 1, cy - m // 2 - 1, m + 2, m + 2, (255, 255, 255))
        canvas.rect(cx - m // 2, cy - m // 2, m, m, color)


def draw_case(case, path_png, panel_px=360):
    """Three panels (bfs, dijkstra, astar) for one map. Returns stats per algorithm."""
    grid = case.grid
    px = max(1, panel_px // max(grid.width, grid.height))
    pw, ph = grid.width * px, grid.height * px
    canvas = Canvas(3 * pw + 4 * GAP, ph + 2 * GAP)
    stats = {}
    for k, name in enumerate(ALGORITHMS):
        result, order = explored_cells(name, grid, case.start, case.goal)
        _, ms = timed_run(ALGORITHMS[name], grid, case.start, case.goal)
        draw_panel(canvas, GAP + k * (pw + GAP), GAP, px, grid, case.start, case.goal,
                   order, result.path, COLORS[name])
        stats[name] = dict(
            found=result.found, cost=result.cost,
            steps=len(result.path) - 1 if result.found else None,
            expanded=result.expanded, stale=result.stale,
            max_frontier=result.max_frontier, time_ms=ms,
        )
    canvas.save_png(path_png)
    return stats


def draw_case8(case, path_png, panel_px=360):
    """8 directions: Dijkstra | A* | JPS. JPS shows scanned cells (tint) and jump points (dark)."""
    grid = Grid8.from_grid(case.grid)
    px = max(1, panel_px // max(grid.width, grid.height))
    pw, ph = grid.width * px, grid.height * px
    canvas = Canvas(3 * pw + 4 * GAP, ph + 2 * GAP)
    stats = {}
    for k, name in enumerate(("dijkstra", "astar")):
        result, order = explored_cells(name, grid, case.start, case.goal)
        _, ms = timed_run(ALGORITHMS[name], grid, case.start, case.goal)
        draw_panel(canvas, GAP + k * (pw + GAP), GAP, px, grid, case.start, case.goal,
                   order, result.path, COLORS[name])
        stats[name] = dict(found=result.found, cost=result.cost, expanded=result.expanded,
                           scanned=0, time_ms=ms)
    trace = {}
    result = jump_point_search(grid, case.start, case.goal, trace=trace)
    _, ms = timed_run(jump_point_search, grid, case.start, case.goal)
    ox = GAP + 2 * (pw + GAP)
    scanned = [grid.index(p) for p in trace["scanned"]]
    draw_panel(canvas, ox, GAP, px, grid, case.start, case.goal, scanned, result.path, COLORS["jps"])
    inset = max(1, px // 4)
    for r, c in trace["jump_points"]:
        canvas.rect(ox + c * px + inset, GAP + r * px + inset, px - 2 * inset, px - 2 * inset, JUMP_POINT_COLOR)
    stats["jps"] = dict(found=result.found, cost=result.cost, expanded=result.expanded,
                        scanned=result.scanned, time_ms=ms)
    canvas.save_png(path_png)
    return stats


# ---------------------------------------------------------------------------
# gallery
# ---------------------------------------------------------------------------

GALLERY = [
    ("random", "Random obstacles 20%, unit cost", dict(wall_ratio=0.2)),
    ("random-weighted", "Random obstacles 20%, cost 1-9 (same walls as above)",
     dict(kind="random", wall_ratio=0.2, weighted=True)),
    ("maze", "Perfect maze (exactly one path)", {}),
    ("maze-loops", "Maze with loops (10% extra openings)", dict(kind="maze", loop_ratio=0.1)),
    ("nopath", "No path (goal sealed off)", {}),
    ("uwall", "U-shaped wall (A* trap)", {}),
    ("terrain", "Terrain (cost 1-9 in patches)", {}),
    ("city", "City blocks (arterial 1, local road 3, jam 9)", dict(block=6)),
]

GALLERY8 = [
    ("random", "8 directions: random obstacles 20%", dict(wall_ratio=0.2)),
    ("uwall", "8 directions: U-shaped wall", {}),
    ("city", "8 directions: city blocks, every road cost 1", dict(block=6, weighted=False)),
    ("maze-loops", "8 directions: maze with loops", dict(kind="maze", loop_ratio=0.1)),
]

LABELS = {"bfs": "BFS", "dijkstra": "Dijkstra", "astar": "A*", "jps": "JPS"}


def make_figures(out_dir=OUT_DIR, size=60, seed=2026, log=print):
    maps_dir = out_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Explored cells",
        "",
        f"Generated by `make figures`. Map size {size} x {size}, seed {seed}.",
        "",
        "Each figure shows, left to right: **BFS | Dijkstra | A***.",
        "",
        "| Color | Meaning |",
        "| --- | --- |",
        "| dark gray | wall |",
        "| white -> warm gray | open cell, cost 1 -> 9 |",
        "| light blue / aqua / orange | cells expanded by BFS / Dijkstra / A* |",
        "| dark blue / aqua / orange | path returned |",
        "| red / purple | start / goal |",
        "",
        "Times are a single run made while drawing, for reference only. "
        "The numbers used for comparison are the experiment results in `results/`.",
        "",
    ]
    for name, title, params in GALLERY:
        params = dict(params)
        kind = params.pop("kind", name)
        case = make_map(kind, size, seed, **params)
        png = maps_dir / f"{name}.png"
        stats = draw_case(case, png)
        lines += [f"## {title}", "", f"![{name}](maps/{png.name})", "",
                  "| | path cost | moves | expanded | max frontier | time (ms) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for alg, s in stats.items():
            cost = s["cost"] if s["found"] else "no path"
            steps = s["steps"] if s["found"] else "-"
            lines.append(f"| {LABELS[alg]} | {cost} | {steps} | {s['expanded']} | "
                         f"{s['max_frontier']} | {s['time_ms']:.2f} |")
        lines.append("")
        log(f"  {png.relative_to(out_dir.parent) if png.is_relative_to(out_dir.parent) else png}")
    maps8 = out_dir / "maps8"
    maps8.mkdir(parents=True, exist_ok=True)
    lines += [
        "# 8 directions and Jump Point Search", "",
        "Left to right: **Dijkstra | A* (octile distance) | JPS**. Costs are in 8-direction units "
        "(straight 10, diagonal 14).", "",
        "In the JPS panel the magenta tint marks cells JPS only *looked at* while jumping, "
        "and the black squares are the jump points it actually put on the heap.", "",
    ]
    for name, title, params in GALLERY8:
        params = dict(params)
        kind = params.pop("kind", name)
        case = make_map(kind, size, seed, **params)
        png = maps8 / f"{name}.png"
        stats = draw_case8(case, png)
        lines += [f"## {title}", "", f"![{name}](maps8/{png.name})", "",
                  "| | path cost | expanded (JPS: jump points) | scanned while jumping | time (ms) |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for alg, st in stats.items():
            cost = st["cost"] if st["found"] else "no path"
            lines.append(f"| {LABELS[alg]} | {cost} | {st['expanded']} | {st['scanned'] or '-'} | {st['time_ms']:.2f} |")
        lines.append("")
        log(f"  {png.relative_to(out_dir.parent) if png.is_relative_to(out_dir.parent) else png}")
    (out_dir / "README.md").write_text("\n".join(lines))
    log(f"  {out_dir / 'README.md'}")


if __name__ == "__main__":
    make_figures()
