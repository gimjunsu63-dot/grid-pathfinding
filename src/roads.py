"""Real road network (OpenStreetMap, Songdo) and a time-of-day traffic model.

Graph
-----
Nodes are road points from data/songdo_roads.json, renumbered 0..n-1. Only the
largest strongly connected part is kept, so every node can reach every other
node with one-way streets respected. An edge is one road segment between two
consecutive points of a way. Its cost is the travel time in tenths of a second
(an integer, rounded up), so the searches see the same kind of numbers as on
the grids.

RoadGraph has the methods the searches call on a Grid (index, pos, is_open,
neighbors, move_cost, is_move, heuristic), so Dijkstra and A* run unchanged.

Heuristics
----------
straight line   great-circle distance / fastest free-flow speed on the map. No
                path is shorter than the straight line and no road is faster, so
                it never overestimates: the road-map version of Manhattan.
landmarks       ALT on a directed graph (one-way streets): one Dijkstra from each
                landmark L gives d(L->x), one on the reversed graph gives d(x->L):
                    d(v->t) >= d(L->t) - d(L->v)    and    d(v->t) >= d(v->L) - d(t->L)

Time-of-day model (a model, not measured traffic)
-------------------------------------------------
travel time = free-flow time x (1 + SLOWDOWN[group] x rush(hour) x bottleneck[edge]
                                   x wobble[edge, hour] x direction[edge, hour])

  rush(hour)    0 at night (23h-5h), peaks at 8h and 18-19h, 0.35 at midday
  SLOWDOWN      extra time at the worst hour, as a share of free flow
  bottleneck    seeded random per segment, 0.4-1.6 (some roads are worse every day)
  wobble        seeded random per segment and hour, 0.85-1.15
  direction     commute direction: 1 + 0.6 x toward[edge] x tide(hour), 0.4-1.6
                  toward  cosine between the segment's direction and the direction to the
                          centre of Songdo (Central Park, business district): +1 heading in,
                          -1 heading out
                  tide    +1 in the morning peak, -1 in the evening peak, about 0 at midday,
                          so roads into the centre are slower in the morning and roads out
                          of it in the evening (traffic toward Songdo jams in the morning:
                          see the report for the news source)

Every factor is >= 0, so no hour is faster than free flow at night. Landmarks
measured at night are therefore lower bounds at every hour (see landmarks.py
for why rising costs keep old landmarks safe).

Time bands (experiment RB)
--------------------------
The same argument works for any set of hours: take, for every segment, its
cheapest travel time over the hours of a band (morning 7-9h, day 10-16h,
evening 17-20h, night and shoulders 21-6h). No hour of the band is cheaper, so landmarks
measured on these costs never overestimate at any hour of the band, and they
are much closer to rush-hour traffic than night landmarks are.

Real traffic is not the forecast. actual_costs() moves every segment by up to
+-FORECAST_ERROR around the model (seeded per day). Band costs times
(1 - FORECAST_ERROR) are still below every such day, so landmarks measured on
them stay exact; landmarks measured on the forecast itself can overestimate.
"""

import heapq
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "songdo_roads.json"
INF = float("inf")
EARTH_RADIUS_M = 6_371_000.0

SPEED_KMH = {
    "motorway": 100, "trunk": 80, "primary": 60, "secondary": 50, "tertiary": 40,
    "unclassified": 30, "residential": 30, "living_street": 20,
    "motorway_link": 60, "trunk_link": 50, "primary_link": 40, "secondary_link": 40, "tertiary_link": 30,
}
GROUP = {
    "motorway": "highway", "trunk": "highway", "motorway_link": "highway", "trunk_link": "highway",
    "primary": "arterial", "primary_link": "arterial",
    "secondary": "collector", "tertiary": "collector", "secondary_link": "collector", "tertiary_link": "collector",
}
SLOWDOWN = {"highway": 1.5, "arterial": 1.2, "collector": 0.8, "local": 0.3}
GROUPS = ("highway", "arterial", "collector", "local")
CENTER = (37.3926, 126.6393)                  # Songdo Central Park (business district)
# Yonsei University International Campus (85 Songdogwahak-ro): the campus corner on Songdogwahak-ro,
# across from Triple Street. Found by laying the road data over a Google Maps screenshot of the campus.
YONSEI = (37.38018, 126.66759)
DIRECTIONAL = 0.6                             # how much the commute direction matters
NIGHT = 3                                     # an hour with free-flow traffic
RUSH = 8                                      # the morning peak

# small area around Songdo Central Park for quick runs and tests
QUICK_BBOX = (37.384, 126.628, 37.400, 126.648)
# hours with similar traffic go together: a light hour in a band would pull its cheapest costs
# down to free flow (a first try with 6-9h for the morning gave landmarks no better than night ones)
BANDS = {"morning": (7, 8, 9), "day": (10, 11, 12, 13, 14, 15, 16), "evening": (17, 18, 19, 20),
         "night": (21, 22, 23, 0, 1, 2, 3, 4, 5, 6)}
FORECAST_ERROR = 0.1                          # real traffic within +-10% of the model


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def parse_maxspeed(text):
    """'60', '60 km/h', '30 mph' -> km/h, or None."""
    digits = "".join(ch for ch in str(text).split(";")[0] if ch.isdigit())
    if not digits:
        return None
    v = int(digits)
    return round(v * 1.609) if "mph" in str(text) else v


def _morning(hour):
    return math.exp(-((hour - 8.0) / 1.1) ** 2 / 2)


def _evening(hour):
    return math.exp(-((hour - 18.5) / 1.4) ** 2 / 2)


def rush(hour):
    """How busy the roads are, 0 (night) to 1 (worst hour)."""
    if hour < 5 or hour >= 23:
        return 0.0
    midday = 0.35 if 10 <= hour <= 16 else 0.0
    return max(_morning(hour), _evening(hour), midday)


def band_of(hour):
    """Name of the time band that holds this hour."""
    return next(name for name, hours in BANDS.items() if hour in hours)


def tide(hour):
    """Which way commuters go: +1 into the centre (morning), -1 out of it (evening)."""
    return _morning(hour) - _evening(hour)


# ---------------------------------------------------------------------------
# network: points, segments, free-flow speeds
# ---------------------------------------------------------------------------


class RoadNetwork:
    """Everything that does not depend on the hour. graph_at(hour) gives a searchable graph."""

    def __init__(self, lat, lon, eu, ev, length, group, speed, name="songdo", seed=2026):
        self.lat, self.lon = lat, lon
        self.eu, self.ev = eu, ev                  # segment i goes eu[i] -> ev[i]
        self.length = length                      # meters
        self.group = group                        # road group of the segment
        self.speed = speed                        # free-flow speed, m/s
        self.name = name
        self.n = len(lat)
        self.vmax = max(speed)
        rng = random.Random(f"{seed}:bottleneck")
        self.bottleneck = [rng.uniform(0.4, 1.6) for _ in eu]
        self.toward = [self._toward_center(u, v) for u, v in zip(eu, ev)]
        self.seed = seed
        self._graphs = {}

    def __len__(self):
        return self.n

    def _toward_center(self, u, v):
        """Cosine between segment u->v and the direction from its midpoint to CENTER (0 near the centre)."""
        k = math.cos(math.radians(CENTER[0]))
        dx, dy = (self.lon[v] - self.lon[u]) * k, self.lat[v] - self.lat[u]
        mx, my = (self.lon[u] + self.lon[v]) / 2, (self.lat[u] + self.lat[v]) / 2
        cx, cy = (CENTER[1] - mx) * k, CENTER[0] - my
        nd, nc = math.hypot(dx, dy), math.hypot(cx, cy)
        if nd == 0 or nc * 111_000 < 200:            # within 200 m of the centre: no direction
            return 0.0
        return (dx * cx + dy * cy) / (nd * nc)

    def free_flow_seconds(self):
        return [L / s for L, s in zip(self.length, self.speed)]

    def multipliers(self, hour):
        r = rush(hour)
        if r == 0:
            return [1.0] * len(self.eu)
        rng = random.Random(f"{self.seed}:wobble:{hour}")
        td = tide(hour)
        return [1.0 + SLOWDOWN[g] * r * b * rng.uniform(0.85, 1.15) * (1 + DIRECTIONAL * c * td)
                for g, b, c in zip(self.group, self.bottleneck, self.toward)]

    def costs_at(self, hour):
        """Travel time of every segment at this hour, in tenths of a second (integer, >= 1)."""
        return [max(1, math.ceil(10 * t * m - 1e-9))
                for t, m in zip(self.free_flow_seconds(), self.multipliers(hour))]

    def graph_at(self, hour):
        if hour not in self._graphs:
            self._graphs[hour] = RoadGraph(self, hour, self.costs_at(hour))
        return self._graphs[hour]

    def graph_from(self, costs, hour=None):
        """A searchable graph with these segment costs (band costs, one day's traffic)."""
        return RoadGraph(self, hour, costs)

    def band_costs(self, band, margin=0.0):
        """Cheapest cost of every segment over the hours of a band, times (1 - margin), at least 1.

        No hour of the band is cheaper, and with margin = e no day whose traffic is at most
        e lighter than the model is cheaper either: landmarks measured on these costs are
        lower bounds there.
        """
        costs = [min(c) for c in zip(*(self.costs_at(h) for h in BANDS[band]))]
        if margin:
            costs = [max(1, math.floor(c * (1 - margin))) for c in costs]
        return costs

    def actual_costs(self, hour, error, seed):
        """One day's traffic at this hour: every segment off the model by up to +-error (seeded)."""
        rng = random.Random(f"{self.seed}:day:{seed}:{hour}")
        return [max(1, math.ceil(c * rng.uniform(1 - error, 1 + error))) for c in self.costs_at(hour)]


def _largest_scc(n, edges):
    """Nodes of the largest strongly connected component (iterative Kosaraju)."""
    out = [[] for _ in range(n)]
    inc = [[] for _ in range(n)]
    for a, b in edges:
        out[a].append(b)
        inc[b].append(a)
    seen = [False] * n
    order = []
    for s in range(n):
        if seen[s]:
            continue
        seen[s] = True
        stack = [(s, 0)]
        while stack:
            v, i = stack.pop()
            if i < len(out[v]):
                stack.append((v, i + 1))
                w = out[v][i]
                if not seen[w]:
                    seen[w] = True
                    stack.append((w, 0))
            else:
                order.append(v)
    comp = [-1] * n
    best, best_size, label = -1, 0, 0
    for s in reversed(order):
        if comp[s] != -1:
            continue
        size, stack = 0, [s]
        comp[s] = label
        while stack:
            v = stack.pop()
            size += 1
            for w in inc[v]:
                if comp[w] == -1:
                    comp[w] = label
                    stack.append(w)
        if size > best_size:
            best, best_size = label, size
        label += 1
    return {v for v in range(n) if comp[v] == best}


def network_from_osm(data, bbox=None, name="songdo", seed=2026):
    """Build a RoadNetwork from the dict saved by tools/fetch_osm.py."""
    coords = {i: (lat, lon) for i, lat, lon in data["nodes"]}
    if bbox:
        s, w, n_, e = bbox
        coords = {i: c for i, c in coords.items() if s <= c[0] <= n_ and w <= c[1] <= e}
    raw = []                                    # (osm_a, osm_b, length, highway, speed)
    for way in data["ways"]:
        hw = way.get("highway", "")
        if hw not in SPEED_KMH:
            continue
        kmh = parse_maxspeed(way.get("maxspeed", "")) or SPEED_KMH[hw]
        ow = str(way.get("oneway", "")).lower()
        roundabout = way.get("junction", "") in ("roundabout", "circular")
        backward_only = ow == "-1"
        one_way = backward_only or ow in ("yes", "1", "true") or (
            (hw == "motorway" or roundabout) and ow != "no")
        pts = way["nodes"]
        for a, b in zip(pts, pts[1:]):
            if a not in coords or b not in coords or a == b:
                continue
            L = haversine(*coords[a], *coords[b])
            if L <= 0:
                continue
            if backward_only:
                raw.append((b, a, L, hw, kmh))
            else:
                raw.append((a, b, L, hw, kmh))
                if not one_way:
                    raw.append((b, a, L, hw, kmh))
    ids = sorted({x for r in raw for x in r[:2]})
    pos = {osm: k for k, osm in enumerate(ids)}
    keep = _largest_scc(len(ids), [(pos[a], pos[b]) for a, b, *_ in raw])
    new = {}
    for k in sorted(keep):
        new[k] = len(new)
    lat, lon = [0.0] * len(new), [0.0] * len(new)
    for osm, k in pos.items():
        if k in new:
            lat[new[k]], lon[new[k]] = coords[osm]
    eu, ev, length, group, speed = [], [], [], [], []
    for a, b, L, hw, kmh in raw:
        ka, kb = pos[a], pos[b]
        if ka in new and kb in new:
            eu.append(new[ka])
            ev.append(new[kb])
            length.append(L)
            group.append(GROUP.get(hw, "local"))
            speed.append(kmh / 3.6)
    return RoadNetwork(lat, lon, eu, ev, length, group, speed, name=name, seed=seed)


_CACHE = {}


def load_network(path=DATA, quick=False):
    """The Songdo network (cached per process). quick=True: a small central area."""
    key = (str(path), quick)
    if key not in _CACHE:
        data = json.loads(Path(path).read_text())
        _CACHE[key] = network_from_osm(data, bbox=QUICK_BBOX if quick else None,
                                       name="songdo-center" if quick else "songdo")
    return _CACHE[key]


def data_available(path=DATA):
    return Path(path).exists()


# ---------------------------------------------------------------------------
# graph at one hour
# ---------------------------------------------------------------------------


class RoadGraph:
    """Directed graph with integer edge costs. Same search interface as Grid."""

    moves = 0                                  # not a 4- or 8-direction grid

    def __init__(self, network, hour, costs):
        self.network = network
        self.hour = hour
        self.n = network.n
        self.lat, self.lon = network.lat, network.lon
        adj = [dict() for _ in range(self.n)]
        for u, v, c in zip(network.eu, network.ev, costs):
            if c < adj[u].get(v, INF):           # two ways over the same pair: keep the faster
                adj[u][v] = c
        self.adj = [list(d.items()) for d in adj]
        self.edges = sum(len(a) for a in self.adj)
        radj = [[] for _ in range(self.n)]
        for u, lst in enumerate(self.adj):
            for v, c in lst:
                radj[v].append((u, c))
        self.radj = radj
        self.tenths_per_meter = 10.0 / network.vmax       # straight-line heuristic scale
        self._rlat = [math.radians(x) for x in self.lat]
        self._rlon = [math.radians(x) for x in self.lon]
        self._cos = [math.cos(x) for x in self._rlat]

    # --- the interface the searches use ---------------------------------
    def index(self, pos):
        return pos

    def pos(self, idx):
        return idx

    def is_open(self, pos):
        return 0 <= pos < self.n

    def neighbors(self, idx):
        return self.adj[idx]

    def move_cost(self, a, b):
        return min(c for v, c in self.adj[a] if v == b)

    def is_move(self, a, b):
        return any(v == b for v, _ in self.adj[a])

    def straight_m(self, a, b):
        la, lb = self._rlat[a], self._rlat[b]
        dp, dl = lb - la, self._rlon[b] - self._rlon[a]
        x = math.sin(dp / 2) ** 2 + self._cos[a] * self._cos[b] * math.sin(dl / 2) ** 2
        return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(x)))

    def heuristic(self, a, b):
        """Straight line at the fastest free-flow speed, in tenths of a second (rounded down)."""
        return int(self.straight_m(a, b) * self.tenths_per_meter)

    # --- helpers for experiments -----------------------------------------
    def distances(self, src, reverse=False):
        """Dijkstra from src: d(src -> x), or d(x -> src) with reverse=True."""
        adj = self.radj if reverse else self.adj
        dist = [INF] * self.n
        dist[src] = 0
        heap = [(0, src)]
        while heap:
            d, v = heapq.heappop(heap)
            if d > dist[v]:
                continue
            for w, c in adj[v]:
                nd = d + c
                if nd < dist[w]:
                    dist[w] = nd
                    heapq.heappush(heap, (nd, w))
        return dist

    def remaining_to(self, goal):
        """Real remaining cost d(v -> goal) for every node (used for accuracy)."""
        return self.distances(goal, reverse=True)


# ---------------------------------------------------------------------------
# landmarks on a directed graph
# ---------------------------------------------------------------------------


class DirectedLandmarkHeuristic:
    """h(v, t) = max over landmarks of d(L->t) - d(L->v) and d(v->L) - d(t->L), and 0."""

    def __init__(self, landmarks, fwd, bwd, hour=None):
        self.landmarks = list(landmarks)
        self.fwd, self.bwd = fwd, bwd
        self.hour = hour                     # hour whose travel times were measured
        self._t = None
        self._per_t = []

    def __len__(self):
        return len(self.landmarks)

    def __call__(self, v, t):
        if t != self._t:
            self._t = t
            self._per_t = [(f, b, f[t], b[t]) for f, b in zip(self.fwd, self.bwd)]
        best = 0
        for f, b, ft, bt in self._per_t:
            x = ft - f[v]
            if x > best:
                best = x
            y = b[v] - bt
            if y > best:
                best = y
        return best


def road_landmarks(graph, count):
    """Farthest-point landmarks: each next one is the node farthest (both ways) from those chosen."""
    marks, fwd, bwd = [], [], []
    d0 = graph.distances(0)
    nxt = max(range(graph.n), key=lambda x: d0[x])
    nearest = None
    while len(marks) < count:
        marks.append(nxt)
        f, b = graph.distances(nxt), graph.distances(nxt, reverse=True)
        fwd.append(f)
        bwd.append(b)
        both = [x + y for x, y in zip(f, b)]
        nearest = both if nearest is None else [min(p, q) for p, q in zip(nearest, both)]
        nxt = max(range(graph.n), key=lambda x: nearest[x])
    return DirectedLandmarkHeuristic(marks, fwd, bwd, hour=graph.hour)


def nearest_node(graph, place):
    """The road point closest to (lat, lon) in a straight line."""
    lat, lon = place
    return min(range(graph.n), key=lambda v: haversine(lat, lon, graph.lat[v], graph.lon[v]))


def random_query(graph, seed, min_m=None, start=None):
    """A seeded random (start, goal) pair at least min_m meters apart in a straight line.

    With start given, only the goal is random (trips from one place).
    """
    rng = random.Random(f"{seed}:road-query")
    if min_m is None:
        min_m = 1500 if graph.n > 2000 else 300
    for _ in range(10_000):
        s = rng.randrange(graph.n) if start is None else start
        t = rng.randrange(graph.n)
        if s != t and graph.straight_m(s, t) >= min_m:
            return s, t
    raise RuntimeError("no start/goal pair far enough apart")
