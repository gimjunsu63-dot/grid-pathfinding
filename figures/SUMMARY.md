# Summary

Computed from `results/*.csv` by `make summary`. Times are medians (ms). Details and charts: [RESULTS.md](RESULTS.md).

## Who is fastest where (experiments A-E)

Only algorithms that returned the optimal path can win a condition; BFS is excluded where costs differ, because its path is more expensive.

| condition | fastest (optimal path) | BFS | Dijkstra | A* | A* speed-up over Dijkstra | A* cells vs Dijkstra | BFS path extra cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random 20%, 50x50 | **A*** | 1.5 | 3.1 | 0.8 | 4.20x | 12% | +0% |
| random 20%, 100x100 | **A*** | 5.9 | 12.7 | 2.3 | 5.63x | 11% | +0% |
| random 20% | **A*** | 25.8 | 55.6 | 7.0 | 8.01x | 8% | +0% |
| random 20%, 400x400 | **A*** | 114.0 | 263.2 | 29.3 | 8.98x | 8% | +0% |
| random 0% | **A*** | 32.1 | 69.0 | 1.1 | 60.63x | 1% | +0% |
| random 10% | **A*** | 29.9 | 61.8 | 5.9 | 10.42x | 6% | +0% |
| random 30% | **A*** | 21.5 | 45.8 | 7.4 | 6.18x | 10% | +0% |
| maze | **BFS** | 8.4 | 13.3 | 15.5 | 0.86x | 99% | +0% |
| maze + loops | **BFS** | 14.2 | 32.2 | 28.0 | 1.15x | 69% | +0% |
| no path | **BFS** | 26.3 | 53.8 | 88.5 | 0.61x | 100% | - |
| U wall | **A*** | 29.1 | 66.4 | 22.4 | 2.96x | 30% | +0% |
| city | **A*** | 7.1 | 14.7 | 3.0 | 4.98x | 15% | +0% |
| random 20%, cost 1-9 | **Dijkstra** | 25.0 | 56.6 | 68.6 | 0.82x | 100% | +42% |
| city, cost 1-9 | **Dijkstra** | 6.8 | 15.2 | 17.3 | 0.88x | 92% | +41% |
| terrain | **Dijkstra** | 29.6 | 71.3 | 91.9 | 0.78x | 100% | +50% |
| random 20%, random ends | **A*** | 19.3 | 43.1 | 2.4 | 18.13x | 3% | +0% |
| random 20%, cost 1-9, random ends | **A*** | 18.7 | 47.4 | 41.8 | 1.13x | 71% | +29% |

- Fastest overall: BFS in 3, Dijkstra in 3, A* in 11 of 17 conditions.
- A* is slower than Dijkstra in 5 of 17 conditions.
- Best case for A*: random 0% (60.6x faster than Dijkstra, 1% of its cells).
- Worst case for A*: no path (1.6x slower than Dijkstra, 100% of its cells).
- BFS ignores costs: where costs differ it is still the quickest to answer, but its path costs up to +50% more (terrain).

## Key numbers

- Growth with map size (log-log slope of time, 1 = proportional): BFS 1.04, Dijkstra 1.06, A* 0.87.
- Time per expanded cell (microseconds): BFS 0.75, Dijkstra 1.69, A* 2.56. A* pays 3.4x BFS per cell, so it must cut cells to win.
- Tie-breaking: up to 100x more cells when ties go to smaller g (random 0%).
- Weighted A* with weight 2: random 20%: 2% of Dijkstra's cells, +12.7% cost; random 20%, cost 1-9: 100% of Dijkstra's cells, +0.1% cost; terrain: 95% of Dijkstra's cells, +1.9% cost; U wall: 27% of Dijkstra's cells, +0.0% cost; city, cost 1-9: 29% of Dijkstra's cells, +2.8% cost.
- Plain-Python heap instead of heapq: 1.67x to 2.16x slower, with exactly the same cells expanded.
- Bidirectional Dijkstra: from 0.0% of Dijkstra's cells (no path) to 100% (maze).
- Measurement noise (IQR / median, same map 30 runs): BFS 8.7%, Dijkstra 11.3%, A* 3.7%.
