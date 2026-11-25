"""Download the drivable road network of Songdo (Incheon) from OpenStreetMap, once.

    python3 tools/fetch_osm.py          # writes data/songdo_roads.json

The experiments read the saved file, not the live map, so every run (and every
reader of this repository) uses exactly the same roads. Running this again later
gives a slightly different file, because OpenStreetMap keeps changing.

Map data (c) OpenStreetMap contributors, available under the Open Database
License (ODbL 1.0): https://www.openstreetmap.org/copyright
Standard library only.
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "songdo_roads.json"

BBOX = (37.355, 126.615, 37.415, 126.700)      # south, west, north, east (Songdo International City)
HIGHWAYS = ("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
            "living_street", "motorway_link", "trunk_link", "primary_link", "secondary_link",
            "tertiary_link")
ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
LICENSE = "Map data (c) OpenStreetMap contributors, ODbL 1.0, https://www.openstreetmap.org/copyright"


def overpass_query(bbox=BBOX):
    s, w, n, e = bbox
    pattern = "|".join(HIGHWAYS)
    return (f'[out:json][timeout:180];'
            f'way["highway"~"^({pattern})$"]({s},{w},{n},{e});'
            f'(._;>;);out body;')


def download(query):
    body = urllib.parse.urlencode({"data": query}).encode()
    errors = []
    for url in ENDPOINTS:
        print(f"asking {url} ...")
        try:
            req = urllib.request.Request(url, data=body,
                                         headers={"User-Agent": "grid-pathfinding course project (Yonsei)"})
            with urllib.request.urlopen(req, timeout=240) as resp:
                return json.load(resp), url
        except Exception as exc:                  # try the next mirror
            print(f"  failed: {exc}")
            errors.append(f"{url}: {exc}")
    sys.exit("could not download the map from any server:\n  " + "\n  ".join(errors))


def compact(raw, url, query):
    """Keep what the experiments need: road ways (type, direction, speed) and node coordinates."""
    coords, ways = {}, []
    for el in raw.get("elements", []):
        if el["type"] == "node":
            coords[el["id"]] = (round(el["lat"], 7), round(el["lon"], 7))
        elif el["type"] == "way":
            tags = el.get("tags", {})
            ways.append({"id": el["id"], "highway": tags.get("highway", ""),
                         "oneway": tags.get("oneway", ""), "junction": tags.get("junction", ""),
                         "maxspeed": tags.get("maxspeed", ""), "nodes": el["nodes"]})
    used = sorted({n for w in ways for n in w["nodes"] if n in coords})
    return {
        "source": "OpenStreetMap via the Overpass API",
        "license": LICENSE,
        "endpoint": url,
        "osm_base": raw.get("osm3s", {}).get("timestamp_osm_base", ""),
        "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bbox_south_west_north_east": list(BBOX),
        "query": query,
        "nodes": [[n, *coords[n]] for n in used],
        "ways": ways,
    }


def main():
    query = overpass_query()
    raw, url = download(query)
    data = compact(raw, url, query)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, separators=(",", ":"))
        f.write("\n")
    size = OUT.stat().st_size / 1e6
    print(f"saved {OUT.relative_to(ROOT)}: {len(data['ways'])} roads, {len(data['nodes'])} points, "
          f"{size:.1f} MB, map as of {data['osm_base']}")


if __name__ == "__main__":
    main()
