"""Tests for the real road map: graph building, traffic model, heuristics. Run: make test

Most tests use a small made-up street grid in the same format as
data/songdo_roads.json, so they do not depend on the downloaded map.
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import bench  # noqa: E402
import roads  # noqa: E402
from search import astar, dijkstra  # noqa: E402


def street_grid(n=14, seed=1):
    """n x n streets near Songdo with mixed road types, some one-way streets and a dead-end one-way."""
    rng = random.Random(seed)
    lat0, lon0, step = 37.388, 126.632, 0.0012
    ids, nodes = {}, []
    for r in range(n):
        for c in range(n):
            ids[r, c] = len(nodes) + 1
            nodes.append([ids[r, c], lat0 + r * step + rng.uniform(-2e-4, 2e-4),
                          lon0 + c * step + rng.uniform(-2e-4, 2e-4)])
    ways, kinds = [], ["primary", "secondary", "tertiary", "residential", "trunk"]
    for r in range(n):
        ways.append(dict(id=len(ways) + 1, highway="primary" if r % 5 == 0 else rng.choice(kinds),
                         oneway="yes" if r % 4 == 3 else "", junction="", maxspeed="",
                         nodes=[ids[r, c] for c in range(n)]))
    for c in range(n):
        ways.append(dict(id=len(ways) + 1, highway=rng.choice(kinds), oneway="-1" if c % 6 == 5 else "",
                         junction="", maxspeed="50" if c == 0 else "", nodes=[ids[r, c] for r in range(n)]))
    # a one-way spur that leads out of the grid and never comes back: not strongly connected
    nodes += [[9001, lat0 - step, lon0], [9002, lat0 - 2 * step, lon0]]
    ways.append(dict(id=999, highway="residential", oneway="yes", junction="", maxspeed="",
                     nodes=[ids[0, 0], 9001, 9002]))
    return {"nodes": nodes, "ways": ways}


class TestNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = roads.network_from_osm(street_grid())

    def test_maxspeed_parsing(self):
        self.assertEqual(roads.parse_maxspeed("60"), 60)
        self.assertEqual(roads.parse_maxspeed("60 km/h"), 60)
        self.assertEqual(roads.parse_maxspeed("30 mph"), 48)
        self.assertIsNone(roads.parse_maxspeed(""))

    def test_one_way_streets(self):
        data = {"nodes": [[1, 37.39, 126.64], [2, 37.39, 126.641], [3, 37.391, 126.641]],
                "ways": [dict(id=1, highway="primary", oneway="yes", junction="", maxspeed="", nodes=[1, 2]),
                         dict(id=2, highway="primary", oneway="-1", junction="", maxspeed="", nodes=[3, 2]),
                         dict(id=3, highway="primary", oneway="", junction="", maxspeed="", nodes=[3, 1])]}
        net = roads.network_from_osm(data)
        pairs = set(zip(net.eu, net.ev))
        self.assertEqual(net.n, 3)                       # 1 -> 2 -> 3 -> 1 is a cycle, all kept
        self.assertEqual(len(pairs), 4)                  # 1->2, 2->3, 3->1, 1->3

    def test_only_strongly_connected_part_is_kept(self):
        self.assertEqual(self.net.n, 14 * 14)            # the one-way spur is dropped
        g = self.net.graph_at(roads.NIGHT)
        self.assertTrue(all(d < roads.INF for d in g.distances(0)))
        self.assertTrue(all(d < roads.INF for d in g.distances(0, reverse=True)))

    def test_no_hour_is_faster_than_the_night(self):
        night = self.net.costs_at(roads.NIGHT)
        self.assertEqual(roads.rush(roads.NIGHT), 0)
        for hour in range(24):
            self.assertTrue(all(c >= n for c, n in zip(self.net.costs_at(hour), night)))
        self.assertGreater(sum(self.net.costs_at(roads.RUSH)), 1.3 * sum(night))

    def test_commute_direction(self):
        """Into the centre is slower in the morning, out of it in the evening; nothing beats the night."""
        self.assertAlmostEqual(roads.tide(8), 1.0, places=2)
        self.assertLess(roads.tide(18.5), -0.99)
        self.assertLess(abs(roads.tide(12)), 0.01)
        inward = [i for i, c in enumerate(self.net.toward) if c > 0.7]
        outward = [i for i, c in enumerate(self.net.toward) if c < -0.7]
        self.assertTrue(inward and outward)
        for hour, slower, faster in ((8, inward, outward), (18, outward, inward)):
            m = self.net.multipliers(hour)
            self.assertGreater(sum(m[i] for i in slower) / len(slower), sum(m[i] for i in faster) / len(faster))
            self.assertTrue(all(x >= 1.0 for x in m))

    def test_straight_line_never_overestimates_and_is_consistent(self):
        g = self.net.graph_at(roads.RUSH)
        for seed in range(5):
            _, t = roads.random_query(g, seed, min_m=300)
            rem = g.remaining_to(t)
            for v in range(g.n):
                self.assertLessEqual(g.heuristic(v, t), rem[v])
                for u, c in g.neighbors(v):
                    self.assertLessEqual(g.heuristic(v, t), c + g.heuristic(u, t))

    def test_night_landmarks_stay_safe_at_every_hour(self):
        h = roads.road_landmarks(self.net.graph_at(roads.NIGHT), 4)
        for hour in bench.ROAD_HOURS:
            g = self.net.graph_at(hour)
            for seed in range(3):
                s, t = roads.random_query(g, seed, min_m=300)
                rem = g.remaining_to(t)
                for v in range(g.n):
                    self.assertLessEqual(h(v, t), rem[v])
                    for u, c in g.neighbors(v):
                        self.assertLessEqual(h(v, t), c + h(u, t))
                self.assertEqual(astar(g, s, t, heuristic=h).cost, dijkstra(g, s, t).cost)

    def test_rush_hour_landmarks_overestimate_at_night(self):
        h = roads.road_landmarks(self.net.graph_at(roads.RUSH), 4)
        g = self.net.graph_at(roads.NIGHT)
        _, t = roads.random_query(g, 0, min_m=300)
        rem = g.remaining_to(t)
        self.assertTrue(any(h(v, t) > rem[v] for v in range(g.n)))

    def test_searches_agree_with_dijkstra(self):
        for hour in (roads.NIGHT, roads.RUSH, 18):
            g = self.net.graph_at(hour)
            h = roads.road_landmarks(g, 4)
            for seed in range(5):
                s, t = roads.random_query(g, seed, min_m=300)
                best = dijkstra(g, s, t).cost
                self.assertEqual(astar(g, s, t).cost, best)
                self.assertEqual(astar(g, s, t, heuristic=h).cost, best)


class TestFixedStart(unittest.TestCase):
    def setUp(self):
        self.g = roads.network_from_osm(street_grid()).graph_at(roads.RUSH)

    def test_nearest_node(self):
        v = 7
        self.assertEqual(roads.nearest_node(self.g, (self.g.lat[v], self.g.lon[v])), v)

    def test_trips_from_one_place(self):
        start = roads.nearest_node(self.g, (self.g.lat[0], self.g.lon[0]))
        goals = set()
        for seed in range(10):
            s, t = roads.random_query(self.g, seed, min_m=300, start=start)
            self.assertEqual(s, start)
            self.assertGreaterEqual(self.g.straight_m(s, t), 300)
            goals.add(t)
        self.assertGreater(len(goals), 1)
        self.assertEqual(roads.random_query(self.g, 3, min_m=300, start=start),
                         roads.random_query(self.g, 3, min_m=300, start=start))


@unittest.skipUnless(roads.data_available(), "data/songdo_roads.json not downloaded yet")
class TestSongdoData(unittest.TestCase):
    def test_songdo_map_loads(self):
        net = roads.load_network(quick=True)
        self.assertGreater(net.n, 50)
        g = net.graph_at(roads.RUSH)
        s, t = roads.random_query(g, 2026)
        self.assertEqual(astar(g, s, t).cost, dijkstra(g, s, t).cost)

    def test_yonsei_is_on_the_map(self):
        g = roads.load_network().graph_at(roads.RUSH)
        v = roads.nearest_node(g, roads.YONSEI)
        self.assertLess(roads.haversine(*roads.YONSEI, g.lat[v], g.lon[v]), 50)
        c = roads.nearest_node(g, roads.CENTER)
        self.assertEqual(astar(g, v, c).cost, dijkstra(g, v, c).cost)


if __name__ == "__main__":
    unittest.main()
