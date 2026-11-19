# Summary

Computed from `results/*.csv` by `make summary`. Times are medians (ms). Details and charts: [RESULTS.md](RESULTS.md).

## Who is fastest where (experiments A-E)

Only algorithms that returned the optimal path can win a condition; BFS is excluded where costs differ, because its path is more expensive.

| condition | fastest (optimal path) | BFS | Dijkstra | A* | A* speed-up over Dijkstra | A* cells vs Dijkstra | BFS path extra cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random 20%, 50x50 | **A*** | 1.5 | 3.1 | 0.7 | 4.14x | 12% | +0% |
| random 20%, 100x100 | **A*** | 7.8 | 14.2 | 2.9 | 4.82x | 11% | +0% |
| random 20% | **A*** | 25.6 | 59.3 | 7.0 | 8.49x | 8% | +0% |
| random 20%, 400x400 | **A*** | 106.5 | 263.1 | 30.3 | 8.70x | 8% | +0% |
| random 0% | **A*** | 31.0 | 71.3 | 1.2 | 60.65x | 1% | +0% |
| random 10% | **A*** | 27.0 | 64.2 | 6.1 | 10.48x | 6% | +0% |
| random 30% | **A*** | 20.5 | 47.8 | 7.8 | 6.16x | 10% | +0% |
| maze | **BFS** | 7.5 | 14.8 | 16.8 | 0.88x | 99% | +0% |
| maze + loops | **BFS** | 14.5 | 34.0 | 29.9 | 1.13x | 69% | +0% |
| no path | **BFS** | 24.0 | 55.8 | 95.0 | 0.59x | 100% | - |
| U wall | **A*** | 33.2 | 72.2 | 24.1 | 2.99x | 30% | +0% |
| city | **A*** | 6.8 | 14.7 | 2.9 | 5.00x | 15% | +0% |
| random 20%, cost 1-9 | **Dijkstra** | 24.1 | 61.2 | 75.6 | 0.81x | 100% | +42% |
| city, cost 1-9 | **Dijkstra** | 6.9 | 15.7 | 18.0 | 0.87x | 92% | +41% |
| terrain | **Dijkstra** | 30.7 | 77.2 | 95.2 | 0.81x | 100% | +50% |
| random 20%, random ends | **A*** | 19.3 | 43.7 | 2.1 | 21.28x | 3% | +0% |
| random 20%, cost 1-9, random ends | **A*** | 19.7 | 51.0 | 42.9 | 1.19x | 71% | +29% |

- Fastest overall: BFS in 3, Dijkstra in 3, A* in 11 of 17 conditions.
- A* is slower than Dijkstra in 5 of 17 conditions.
- Best case for A*: random 0% (60.6x faster than Dijkstra, 1% of its cells).
- Worst case for A*: no path (1.7x slower than Dijkstra, 100% of its cells).
- BFS ignores costs: where costs differ it is still the quickest to answer, but its path costs up to +50% more (terrain).

## Key numbers

- Growth with map size (log-log slope of time, 1 = proportional): BFS 1.01, Dijkstra 1.06, A* 0.86.
- Time per expanded cell (microseconds): BFS 0.76, Dijkstra 1.78, A* 2.66. A* pays 3.5x BFS per cell, so it must cut cells to win.
- Tie-breaking: up to 100x more cells when ties go to smaller g (random 0%).
- Weighted A* with weight 2: random 20%: 2% of Dijkstra's cells, +12.7% cost; random 20%, cost 1-9: 100% of Dijkstra's cells, +0.1% cost; terrain: 95% of Dijkstra's cells, +1.9% cost; U wall: 27% of Dijkstra's cells, +0.0% cost; city, cost 1-9: 29% of Dijkstra's cells, +2.8% cost.
- Plain-Python heap instead of heapq: 1.61x to 2.19x slower, with exactly the same cells expanded.
- Bidirectional Dijkstra: from 0.0% of Dijkstra's cells (no path) to 100% (maze).
- 8 directions: BFS's path costs up to +0.1% even when every cost is 1 (a diagonal step costs more than a straight one). JPS takes 19% of A*'s cells off the heap (median) but is faster than A* in 3 of 6 maps, because it scans the cells between jump points.
- Traffic jams: with no jams A* is 4.8x faster than Dijkstra; from jams 10% it is slower. At jams 50%, the fastest optimal search is Bidirectional.
- Measurement noise (IQR / median, same map 30 runs): BFS 14.1%, Dijkstra 8.0%, A* 10.5%.
