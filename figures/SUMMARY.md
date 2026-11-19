# Summary

Computed from `results/*.csv` by `make summary`. Times are medians (ms). Details and charts: [RESULTS.md](RESULTS.md).

## Who is fastest where (experiments A-E)

Only algorithms that returned the optimal path can win a condition; BFS is excluded where costs differ, because its path is more expensive.

| condition | fastest (optimal path) | BFS | Dijkstra | A* | A* speed-up over Dijkstra | A* cells vs Dijkstra | BFS path extra cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random 20%, 50x50 | **A*** | 2.4 | 5.5 | 1.0 | 5.42x | 12% | +0% |
| random 20%, 100x100 | **A*** | 7.3 | 16.7 | 2.8 | 5.90x | 11% | +0% |
| random 20% | **A*** | 26.5 | 64.0 | 8.5 | 7.56x | 8% | +0% |
| random 20%, 400x400 | **A*** | 115.3 | 289.1 | 32.5 | 8.90x | 8% | +0% |
| random 0% | **A*** | 34.7 | 77.2 | 1.2 | 66.30x | 1% | +0% |
| random 10% | **A*** | 31.0 | 68.1 | 6.3 | 10.73x | 6% | +0% |
| random 30% | **A*** | 23.0 | 53.7 | 8.1 | 6.60x | 10% | +0% |
| maze | **BFS** | 9.3 | 16.0 | 18.9 | 0.85x | 99% | +0% |
| maze + loops | **BFS** | 14.5 | 38.1 | 32.1 | 1.19x | 69% | +0% |
| no path | **BFS** | 27.3 | 60.0 | 102.2 | 0.59x | 100% | - |
| U wall | **A*** | 36.1 | 75.9 | 24.7 | 3.08x | 30% | +0% |
| city | **A*** | 7.3 | 15.4 | 4.4 | 3.46x | 15% | +0% |
| random 20%, cost 1-9 | **Dijkstra** | 25.6 | 65.4 | 81.3 | 0.80x | 100% | +42% |
| city, cost 1-9 | **Dijkstra** | 7.0 | 17.1 | 19.4 | 0.89x | 92% | +41% |
| terrain | **Dijkstra** | 34.1 | 84.7 | 106.2 | 0.80x | 100% | +50% |
| random 20%, random ends | **A*** | 20.1 | 49.6 | 2.1 | 23.18x | 3% | +0% |
| random 20%, cost 1-9, random ends | **A*** | 20.0 | 54.0 | 47.9 | 1.13x | 71% | +29% |

- Fastest overall: BFS in 3, Dijkstra in 3, A* in 11 of 17 conditions.
- A* is slower than Dijkstra in 5 of 17 conditions.
- Best case for A*: random 0% (66.3x faster than Dijkstra, 1% of its cells).
- Worst case for A*: no path (1.7x slower than Dijkstra, 100% of its cells).
- BFS ignores costs: where costs differ it is still the quickest to answer, but its path costs up to +50% more (terrain).

## Key numbers

- Growth with map size (log-log slope of time, 1 = proportional): BFS 0.93, Dijkstra 0.96, A* 0.83.
- Time per expanded cell (microseconds): BFS 0.84, Dijkstra 1.97, A* 2.76. A* pays 3.3x BFS per cell, so it must cut cells to win.
- Tie-breaking: up to 100x more cells when ties go to smaller g (random 0%).
- Weighted A* with weight 2: random 20%: 2% of Dijkstra's cells, +12.7% cost; random 20%, cost 1-9: 100% of Dijkstra's cells, +0.1% cost; terrain: 95% of Dijkstra's cells, +1.9% cost; U wall: 27% of Dijkstra's cells, +0.0% cost; city, cost 1-9: 29% of Dijkstra's cells, +2.8% cost.
- Plain-Python heap instead of heapq: 1.57x to 2.19x slower, with exactly the same cells expanded.
- Bidirectional Dijkstra: from 0.0% of Dijkstra's cells (no path) to 100% (maze).
- 8 directions: BFS's path costs up to +0.1% even when every cost is 1 (a diagonal step costs more than a straight one). JPS takes 19% of A*'s cells off the heap (median) but is faster than A* in 3 of 6 maps, because it scans the cells between jump points.
- Traffic jams: with no jams A* is 4.1x faster than Dijkstra; from jams 20% it is slower. At jams 50%, the fastest optimal search is Bidirectional.

## Heuristic quality (main experiment)

| map | Manhattan accuracy | Manhattan cells vs Dijkstra | best landmark count | landmark accuracy | landmark cells vs Dijkstra | landmark speed-up over Dijkstra | queries to pay off precompute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random 20%, cost 1-9 | 26% | 74% | 8 | 95% | 5.6% | 9.1x | 9 |
| maze | 9% | 99% | 4 | 99% | 30.6% | 2.0x | 13 |
| city, jams 20% | 36% | 59% | 8 | 92% | 6.6% | 8.4x | 10 |

- Landmarks need precomputation (one Dijkstra per landmark), so they pay off only when the same map answers several queries; see the last column.
- Measurement noise (IQR / median, same map 30 runs): BFS 36.6%, Dijkstra 12.7%, A* 42.3%.
