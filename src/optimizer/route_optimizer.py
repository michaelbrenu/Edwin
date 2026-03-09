"""
src/optimizer/route_optimizer.py
=================================
Waste-collection route optimiser for Accra, Ghana.

Algorithm
---------
1.  Filter bins by collection priority (high-priority first, then fill level).
2.  Partition bins into truck loads (MAX_BINS_PER_TRUCK).
3.  For each truck's load, apply Nearest-Neighbour TSP heuristic
    starting from the central depot.
4.  Return an ordered route with cumulative distance estimates.

Distance metric
---------------
Haversine formula — great-circle distance between GPS coordinates.
For urban Accra the straight-line distance underestimates road
distance by ~20-30 %; a road-network factor of 1.3 is applied.

Scalability note
----------------
At 1 000+ bins, replace the O(n²) nearest-neighbour loop with Google
OR-Tools VRP solver (pip install ortools).  The interface is identical;
only `_nn_route()` needs to be swapped out.  OR-Tools handles
time-windows, vehicle capacities and multi-depot scenarios natively.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    DEPOT_LOCATION,
    MAX_BINS_PER_TRUCK,
    PRIORITY_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Empirical road-distance inflation factor for Accra road network
ROAD_FACTOR = 1.30
EARTH_RADIUS_KM = 6_371.0


# ── Haversine ─────────────────────────────────────────────────────────────────
def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Compute the great-circle distance (km) between two GPS points.

    Parameters
    ----------
    lat1, lng1 : float  — first point in decimal degrees
    lat2, lng2 : float  — second point in decimal degrees

    Returns
    -------
    float  — straight-line distance in kilometres
    """
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ     = math.radians(lat2 - lat1)
    Δλ     = math.radians(lng2 - lng1)

    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(EARTH_RADIUS_KM * c, 4)


def road_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Estimated road distance (km) = Haversine × road factor."""
    return round(haversine(lat1, lng1, lat2, lng2) * ROAD_FACTOR, 4)


# ── Hotspot detection ─────────────────────────────────────────────────────────
def identify_hotspots(bins: list[dict]) -> dict:
    """
    Identify waste hotspots: zones where mean fill level exceeds the
    priority threshold or where a specific waste type dominates.

    Returns
    -------
    dict keyed by zone name with aggregated metrics.
    """
    from collections import defaultdict

    zones: dict[str, list] = defaultdict(list)
    for b in bins:
        zones[b["location_name"]].append(b)

    hotspots = {}
    for zone, records in zones.items():
        fills      = [r["fill_level"] for r in records]
        types      = [r["waste_type"] for r in records]
        mean_fill  = round(sum(fills) / len(fills), 3)
        dominant   = max(set(types), key=types.count)
        high_count = sum(1 for f in fills if f >= PRIORITY_THRESHOLD)

        hotspots[zone] = {
            "total_bins":       len(records),
            "mean_fill_level":  mean_fill,
            "dominant_type":    dominant,
            "high_priority_bins": high_count,
            "is_hotspot":       mean_fill >= PRIORITY_THRESHOLD,
        }
        if hotspots[zone]["is_hotspot"]:
            logger.warning("HOTSPOT detected: %s (mean fill %.0f%%)", zone, mean_fill * 100)

    return hotspots


# ── Nearest-Neighbour TSP ─────────────────────────────────────────────────────
def _nn_route(depot: dict, stops: list[dict]) -> tuple[list[dict], float]:
    """
    Nearest-neighbour greedy TSP heuristic.

    Starting from the depot, repeatedly visit the closest unvisited bin.
    Returns (ordered_stops, total_road_km).
    """
    unvisited  = stops.copy()
    route      = []
    current    = depot
    total_dist = 0.0

    while unvisited:
        nearest   = min(
            unvisited,
            key=lambda b: road_distance(
                current["lat"], current["lng"], b["latitude"], b["longitude"]
            ),
        )
        dist       = road_distance(
            current["lat"], current["lng"],
            nearest["latitude"], nearest["longitude"],
        )
        total_dist += dist
        route.append(nearest)
        current    = {"lat": nearest["latitude"], "lng": nearest["longitude"]}
        unvisited.remove(nearest)

    # Return to depot
    total_dist += road_distance(
        current["lat"], current["lng"],
        depot["lat"], depot["lng"],
    )
    return route, round(total_dist, 3)


# ── Main optimiser entry point ────────────────────────────────────────────────
def optimise_routes(bins: list[dict]) -> list[dict]:
    """
    Optimise collection routes for all bins.

    Steps
    -----
    1. Sort by priority: high-fill bins first.
    2. Partition into truck loads of MAX_BINS_PER_TRUCK.
    3. Apply nearest-neighbour TSP per truck.
    4. Annotate each bin record with route_sequence and truck_id.

    Parameters
    ----------
    bins : list[dict]
        Raw bin records (from mock_data_generator or classifier pipeline).

    Returns
    -------
    list[dict]  — same records, annotated with routing metadata.
    """
    depot = DEPOT_LOCATION

    # 1. Sort: high priority first, then descending fill level
    priority_order = {"high": 0, "normal": 1}
    sorted_bins = sorted(
        bins,
        key=lambda b: (priority_order.get(b.get("collection_priority", "normal"), 1),
                       -b.get("fill_level", 0)),
    )

    # 2. Partition into truck loads
    chunks = [
        sorted_bins[i: i + MAX_BINS_PER_TRUCK]
        for i in range(0, len(sorted_bins), MAX_BINS_PER_TRUCK)
    ]

    annotated: list[dict] = []
    route_summary: list[dict] = []

    for truck_idx, chunk in enumerate(chunks):
        truck_id = f"TRUCK-{truck_idx + 1:02d}"
        ordered, total_km = _nn_route(
            {"lat": depot["lat"], "lng": depot["lng"]}, chunk
        )

        for seq, bin_record in enumerate(ordered, start=1):
            bin_record["route_sequence"] = seq
            bin_record["truck_id"]       = truck_id
            annotated.append(bin_record)

        route_summary.append({
            "truck_id":   truck_id,
            "num_bins":   len(chunk),
            "total_km":   total_km,
            "stops":      [b["bin_id"] for b in ordered],
        })
        logger.info("%-12s | %2d bins | %.2f km", truck_id, len(chunk), total_km)

    logger.info(
        "Route optimisation complete: %d trucks, %d total bins.",
        len(chunks), len(annotated),
    )
    return annotated, route_summary


# ── CLI smoke-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Load mock dataset
    data_path = Path(__file__).parent.parent.parent / "data" / "exports" / "mock_bins.json"
    if not data_path.exists():
        print("Run mock_data_generator.py first.")
    else:
        with open(data_path) as f:
            bins = json.load(f)

        hotspots = identify_hotspots(bins)
        print("\n── Hotspot Report ──────────────────────────────")
        for zone, info in hotspots.items():
            flag = " *** HOTSPOT ***" if info["is_hotspot"] else ""
            print(f"  {zone:30s} fill={info['mean_fill_level']:.0%}  "
                  f"dominant={info['dominant_type']:8s}{flag}")

        annotated, summary = optimise_routes(bins)
        print("\n── Route Summary ────────────────────────────────")
        for r in summary:
            print(f"  {r['truck_id']}  {r['num_bins']} bins  {r['total_km']:.1f} km")
