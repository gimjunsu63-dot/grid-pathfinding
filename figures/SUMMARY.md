# Summary

Computed from `results/*.csv` by `make summary`. Times are medians (ms). Details and charts: [RESULTS.md](RESULTS.md).

## Who is fastest where (experiments A-E)

Only algorithms that returned the optimal path can win a condition; BFS is excluded where costs differ, because its path is more expensive.

| condition | fastest (optimal path) | BFS | Dijkstra | A* | A* speed-up over Dijkstra | A* cells vs Dijkstra | BFS path extra cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random 20%, 50x50 | **A*** | 2.4 | 5.3 | 0.8 | 6.36x | 12% | +0% |
| random 20%, 100x100 | **A*** | 5.8 | 12.5 | 2.3 | 5.41x | 11% | +0% |
| random 20% | **A*** | 24.2 | 56.9 | 6.9 | 8.19x | 8% | +0% |
| random 20%, 400x400 | **A*** | 105.6 | 259.1 | 30.1 | 8.59x | 8% | +0% |
| random 0% | **A*** | 30.3 | 68.6 | 1.1 | 60.19x | 1% | +0% |
| random 10% | **A*** | 27.0 | 61.8 | 5.9 | 10.42x | 6% | +0% |
| random 30% | **A*** | 20.5 | 46.1 | 8.1 | 5.73x | 10% | +0% |
| maze | **BFS** | 7.3 | 12.3 | 14.9 | 0.83x | 99% | +0% |
| maze + loops | **BFS** | 14.2 | 31.9 | 31.6 | 1.01x | 69% | +0% |
| no path | **BFS** | 23.5 | 53.7 | 90.7 | 0.59x | 100% | - |
| U wall | **A*** | 29.7 | 67.7 | 23.2 | 2.92x | 30% | +0% |
| city | **A*** | 6.7 | 14.4 | 4.4 | 3.28x | 15% | +0% |
| random 20%, cost 1-9 | **Dijkstra** | 23.5 | 57.0 | 72.9 | 0.78x | 100% | +42% |
| city, cost 1-9 | **Dijkstra** | 6.8 | 15.1 | 17.3 | 0.87x | 92% | +41% |
| terrain | **Dijkstra** | 30.0 | 74.5 | 92.2 | 0.81x | 100% | +50% |
| random 20%, random ends | **A*** | 19.4 | 45.4 | 2.4 | 18.61x | 3% | +0% |
| random 20%, cost 1-9, random ends | **A*** | 20.1 | 48.3 | 43.8 | 1.10x | 71% | +29% |

- Fastest overall: BFS in 3, Dijkstra in 3, A* in 11 of 17 conditions.
- A* is slower than Dijkstra in 5 of 17 conditions.
- Best case for A*: random 0% (60.2x faster than Dijkstra, 1% of its cells).
- Worst case for A*: no path (1.7x slower than Dijkstra, 100% of its cells).
- BFS ignores costs: where costs differ it is still the quickest to answer, but its path costs up to +50% more (terrain).

## Key numbers

- Growth with map size (log-log slope of time, 1 = proportional): BFS 0.92, Dijkstra 0.95, A* 0.86.
- Time per expanded cell (microseconds): BFS 0.75, Dijkstra 1.72, A* 2.61. A* pays 3.5x BFS per cell, so it must cut cells to win.
- Tie-breaking: up to 100x more cells when ties go to smaller g (random 0%).
- Weighted A* with weight 2: random 20%: 2% of Dijkstra's cells, +12.7% cost; random 20%, cost 1-9: 100% of Dijkstra's cells, +0.1% cost; terrain: 95% of Dijkstra's cells, +1.9% cost; U wall: 27% of Dijkstra's cells, +0.0% cost; city, cost 1-9: 29% of Dijkstra's cells, +2.8% cost.
- Plain-Python heap instead of heapq: 1.65x to 2.25x slower, with exactly the same cells expanded.
- Bidirectional Dijkstra: from 0.0% of Dijkstra's cells (no path) to 100% (maze).
- 8 directions: BFS's path costs up to +0.1% even when every cost is 1 (a diagonal step costs more than a straight one). JPS takes 19% of A*'s cells off the heap (median) but is faster than A* in 3 of 6 maps, because it scans the cells between jump points.
- Traffic jams: with no jams A* is 5.0x faster than Dijkstra; from jams 10% it is slower. At jams 50%, the fastest optimal search is Bidirectional.

## Heuristic quality (main experiment)

| map | Manhattan accuracy | Manhattan cells vs Dijkstra | best landmark count | landmark accuracy | landmark cells vs Dijkstra | landmark speed-up over Dijkstra | queries to pay off precompute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random 20%, cost 1-9 | 26% | 74% | 16 | 97% | 2.9% | 9.2x | 18 |
| maze | 9% | 99% | 4 | 99% | 30.6% | 2.1x | 12 |
| city, jams 20% | 36% | 59% | 8 | 92% | 6.6% | 7.7x | 11 |

- Landmarks need precomputation (one Dijkstra per landmark), so they pay off only when the same map answers several queries; see the last column.

## Dynamic traffic (old landmarks vs recomputed)

- Jams appear (costs only rise): old landmarks were optimal in every run; their accuracy fell 91%, 72%, 51%, 39% (no change -> +50%), and at +50% they took 12.1 ms vs Dijkstra 9.8 ms and recomputed landmarks 1.6 ms. Recomputing (130 ms) paid off after 39 / 16 / 12 queries (+10% / +30% / +50%).
- Jams clear (some costs fall): old landmarks overestimated in 97% of cells at 50% cleared; queries answered with a longer path: 47%, 83%, 87% (cleared 10%, 30%, 50%), worst +18.5% cost. Fast, but no longer correct.

## Landmark placement

- Speed-up over Dijkstra. random 20%, cost 1-9: best corners with 4 (6.4x), corners with 8 (8.3x), random with 4 2.8x; maze: best farthest with 4 (1.8x), corners with 8 (1.5x), random with 4 1.6x; city, jams 20%: best farthest with 4 (5.8x), farthest with 8 (7.5x), random with 4 2.5x.
- Measurement noise (IQR / median, same map 30 runs): BFS 14.9%, Dijkstra 10.6%, A* 25.9%.
