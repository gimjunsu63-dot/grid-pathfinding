# Summary

Computed from `results/*.csv` by `make summary`. Times are medians (ms). Details and charts: [RESULTS.md](RESULTS.md).

## Who is fastest where (experiments A-E)

Only algorithms that returned the optimal path can win a condition; BFS is excluded where costs differ, because its path is more expensive.

| condition | fastest (optimal path) | BFS | Dijkstra | A* | A* speed-up over Dijkstra | A* cells vs Dijkstra | BFS path extra cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random 20%, 50x50 | **A*** | 1.5 | 3.1 | 0.7 | 4.39x | 12% | +0% |
| random 20%, 100x100 | **A*** | 5.7 | 12.5 | 2.4 | 5.29x | 11% | +0% |
| random 20% | **A*** | 24.0 | 58.1 | 7.0 | 8.25x | 8% | +0% |
| random 20%, 400x400 | **A*** | 106.1 | 259.2 | 29.4 | 8.83x | 8% | +0% |
| random 0% | **A*** | 30.3 | 70.2 | 1.2 | 60.25x | 1% | +0% |
| random 10% | **A*** | 27.4 | 62.9 | 6.3 | 10.01x | 6% | +0% |
| random 30% | **A*** | 20.0 | 47.8 | 7.8 | 6.16x | 10% | +0% |
| maze | **BFS** | 8.9 | 14.9 | 15.5 | 0.96x | 99% | +0% |
| maze + loops | **BFS** | 14.2 | 32.2 | 29.4 | 1.10x | 69% | +0% |
| no path | **BFS** | 23.2 | 52.3 | 90.9 | 0.58x | 100% | - |
| U wall | **A*** | 29.8 | 69.1 | 24.2 | 2.86x | 30% | +0% |
| city | **A*** | 6.8 | 14.8 | 2.9 | 5.05x | 15% | +0% |
| random 20%, cost 1-9 | **Dijkstra** | 23.8 | 59.2 | 72.9 | 0.81x | 100% | +42% |
| city, cost 1-9 | **Dijkstra** | 6.8 | 15.5 | 17.7 | 0.88x | 92% | +41% |
| terrain | **Dijkstra** | 30.1 | 76.6 | 95.1 | 0.81x | 100% | +50% |
| random 20%, random ends | **A*** | 19.5 | 43.5 | 2.0 | 21.39x | 3% | +0% |
| random 20%, cost 1-9, random ends | **A*** | 19.2 | 50.8 | 40.3 | 1.26x | 71% | +29% |

- Fastest overall: BFS in 3, Dijkstra in 3, A* in 11 of 17 conditions.
- A* is slower than Dijkstra in 5 of 17 conditions.
- Best case for A*: random 0% (60.3x faster than Dijkstra, 1% of its cells).
- Worst case for A*: no path (1.7x slower than Dijkstra, 100% of its cells).
- BFS ignores costs: where costs differ it is still the quickest to answer, but its path costs up to +50% more (terrain).

## Key numbers

- Growth with map size (log-log slope of time, 1 = proportional): BFS 1.03, Dijkstra 1.07, A* 0.89.
- Time per expanded cell (microseconds): BFS 0.75, Dijkstra 1.74, A* 2.64. A* pays 3.5x BFS per cell, so it must cut cells to win.
- Tie-breaking: up to 100x more cells when ties go to smaller g (random 0%).
- Weighted A* with weight 2: random 20%: 2% of Dijkstra's cells, +12.7% cost; random 20%, cost 1-9: 100% of Dijkstra's cells, +0.1% cost; terrain: 95% of Dijkstra's cells, +1.9% cost; U wall: 27% of Dijkstra's cells, +0.0% cost; city, cost 1-9: 29% of Dijkstra's cells, +2.8% cost.
- Plain-Python heap instead of heapq: 1.66x to 2.18x slower, with exactly the same cells expanded.
- Bidirectional Dijkstra: from 0.0% of Dijkstra's cells (no path) to 100% (maze).
- 8 directions: BFS's path costs up to +0.1% even when every cost is 1 (a diagonal step costs more than a straight one). JPS takes 19% of A*'s cells off the heap (median) but is faster than A* in 3 of 6 maps, because it scans the cells between jump points.
- Traffic jams: with no jams A* is 5.0x faster than Dijkstra; from jams 10% it is slower. At jams 50%, the fastest optimal search is Bidirectional.

## Heuristic quality (main experiment)

| map | Manhattan accuracy | Manhattan cells vs Dijkstra | best landmark count | landmark accuracy | landmark cells vs Dijkstra | landmark speed-up over Dijkstra | queries to pay off precompute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random 20%, cost 1-9 | 26% | 74% | 16 | 97% | 2.9% | 10.1x | 17 |
| maze | 9% | 99% | 4 | 99% | 30.6% | 2.3x | 12 |
| city, jams 20% | 36% | 59% | 8 | 92% | 6.6% | 8.4x | 11 |

- Landmarks need precomputation (one Dijkstra per landmark), so they pay off only when the same map answers several queries; see the last column.

## Dynamic traffic (old landmarks vs recomputed)

- Jams appear (costs only rise): old landmarks were optimal in every run; their accuracy fell 91%, 72%, 51%, 39% (no change -> +50%), and at +50% they took 12.2 ms vs Dijkstra 10.3 ms and recomputed landmarks 1.5 ms. Recomputing (128 ms) paid off after 39 / 16 / 12 queries (+10% / +30% / +50%).
- Jams clear (some costs fall): old landmarks overestimated in 97% of cells at 50% cleared; queries answered with a longer path: 47%, 83%, 87% (cleared 10%, 30%, 50%), worst +18.5% cost. Fast, but no longer correct.

## Landmark placement

- Speed-up over Dijkstra. random 20%, cost 1-9: best corners with 4 (6.2x), corners with 8 (8.8x), random with 4 3.1x; maze: best corners with 4 (1.7x), corners with 8 (1.5x), random with 4 1.6x; city, jams 20%: best farthest with 4 (5.9x), farthest with 8 (7.5x), random with 4 2.6x.

## Real road map (Songdo, OpenStreetMap)

- Morning peak 8h: Dijkstra 2.6 ms, A* + straight line 3.1 ms (accuracy 20%), landmarks measured that hour 0.5 ms (accuracy 86%, 5.0x faster than Dijkstra).
- Landmarks measured at night were optimal at every hour; at 8h their accuracy was 54% and they took 2.5 ms.
- Landmarks measured at 8h, used at 3h (lighter traffic): 96% of points overestimated, 70% of queries got a longer route (worst +18.9%).

## Trips from Yonsei University (real road map)

- 8h, 30 trips from the campus: Dijkstra 2.8 ms, A* + straight line 3.1 ms, landmarks measured that hour 0.7 ms (4.0x faster than Dijkstra).
- Landmarks measured at 8h: used at 3h, 33% of trips got a longer route (worst +19.4%); used at 18h, 0% of trips got a longer route. Landmarks measured at night: optimal at every hour.

## Landmarks per time band (real road map)

- 8h, traffic = model: landmarks measured now 0.60 ms, time band 1.40 ms, night 2.93 ms, Dijkstra 2.74 ms.
- Traffic up to 10% off the model, longer routes over all hours: measured now 7.5%, night 0.6%, time band 3.3%, time band with margin 0.0%.

## Measurement noise

- IQR / median, same map 30 runs: BFS 17.0%, Dijkstra 8.9%, A* 19.5%.
