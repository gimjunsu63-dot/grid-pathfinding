"""Small demo: run BFS, Dijkstra and A* on two hand-made maps.

Run: make run
"""

from grid import Grid
from search import ALGORITHMS

UNIT_MAP = [
    "..........",
    ".####.....",
    "....#..##.",
    ".##.#...#.",
    "..#.###.#.",
    "..#.....#.",
    "..#####.#.",
    "..........",
]

# Fewest moves goes straight through the 9s (6 moves, cost 46);
# the cheap path goes around the wall (10 moves, cost 10).
WEIGHTED_MAP = [
    ".99999.",
    ".#####.",
    ".......",
]


def run_case(title, rows, start, goal):
    grid = Grid.from_strings(rows)
    print(f"=== {title} ({grid.height}x{grid.width}, start={start}, goal={goal}) ===")
    print(f"{'algorithm':<10}{'cost':>6}{'steps':>7}{'expanded':>10}{'stale':>7}{'max_frontier':>14}")
    results = {}
    for name, fn in ALGORITHMS.items():
        r = fn(grid, start, goal)
        results[name] = r
        steps = len(r.path) - 1 if r.found else "-"
        cost = r.cost if r.found else "-"
        print(f"{name:<10}{cost:>6}{steps:>7}{r.expanded:>10}{r.stale:>7}{r.max_frontier:>14}")
    if results["astar"].found:
        for name in ("bfs", "astar"):
            print(f"\n{name} path:")
            print(grid.render(results[name].path, start, goal))
    else:
        print("\nno path")
    print()


if __name__ == "__main__":
    run_case("unit cost", UNIT_MAP, (0, 0), (7, 9))
    run_case("weighted cost", WEIGHTED_MAP, (0, 0), (0, 6))
    run_case("no path", ["..#..", "..#..", "..#.."], (0, 0), (2, 4))
