"""
src/data/mock_data_generator.py
================================
Generates a realistic mock dataset of waste-bin records for Accra, Ghana.

Output artefacts
----------------
data/exports/mock_bins.json   — raw bin records (used by pipeline)
data/exports/mock_bins.csv    — same data, tabular (for quick inspection)
data/raw/                     — synthetic solid-colour PNG "images" that
                                act as stand-ins until real photos are
                                captured in the field.

Accra locations modelled
------------------------
| Zone            | Lat      | Lng      | Character               |
|-----------------|----------|----------|-------------------------|
| Makola Market   | 5.5474   | -0.2044  | High organic / plastic  |
| Kwame Nkrumah   | 5.5688   | -0.2263  | Mixed (Circle)          |
| East Legon      | 5.6037   | -0.1614  | Affluent – more plastic |
| Osu             | 5.5545   | -0.1786  | Night-market organic    |
| Labadi Beach    | 5.5524   | -0.1429  | Coastal plastic heavy   |
| Adabraka        | 5.5581   | -0.2157  | Dense residential mixed |
| Kaneshie        | 5.5574   | -0.2386  | Market + metal scrap    |
| Madina          | 5.6716   | -0.1664  | Peri-urban organic      |
| Tema Industrial | 5.6698   | -0.0166  | Metal / industrial      |
| Accra Central   | 5.5502   | -0.2174  | Government district     |

Scalability note
----------------
For 1 000+ bins, replace the in-memory list comprehension with a
chunked generator that writes records in batches to avoid RAM pressure.
"""

from __future__ import annotations

import json
import csv
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DIRS, WASTE_LABELS

logger = logging.getLogger(__name__)

# ── Seed for reproducibility ──────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Accra zone definitions ────────────────────────────────────────────────────
ACCRA_ZONES = [
    {
        "name": "Makola Market",
        "lat": 5.5474, "lng": -0.2044,
        "waste_weights": {"plastic": 0.38, "organic": 0.48, "metal": 0.09, "electronic": 0.05},
        "bins": 8,
    },
    {
        "name": "Kwame Nkrumah Circle",
        "lat": 5.5688, "lng": -0.2263,
        "waste_weights": {"plastic": 0.33, "organic": 0.33, "metal": 0.24, "electronic": 0.10},
        "bins": 6,
    },
    {
        "name": "East Legon",
        "lat": 5.6037, "lng": -0.1614,
        "waste_weights": {"plastic": 0.50, "organic": 0.25, "metal": 0.10, "electronic": 0.15},
        "bins": 5,
    },
    {
        "name": "Osu",
        "lat": 5.5545, "lng": -0.1786,
        "waste_weights": {"plastic": 0.28, "organic": 0.57, "metal": 0.09, "electronic": 0.06},
        "bins": 6,
    },
    {
        "name": "Labadi Beach",
        "lat": 5.5524, "lng": -0.1429,
        "waste_weights": {"plastic": 0.62, "organic": 0.24, "metal": 0.09, "electronic": 0.05},
        "bins": 4,
    },
    {
        "name": "Adabraka",
        "lat": 5.5581, "lng": -0.2157,
        "waste_weights": {"plastic": 0.33, "organic": 0.38, "metal": 0.22, "electronic": 0.07},
        "bins": 7,
    },
    {
        "name": "Kaneshie",
        "lat": 5.5574, "lng": -0.2386,
        "waste_weights": {"plastic": 0.22, "organic": 0.30, "metal": 0.33, "electronic": 0.15},
        "bins": 5,
    },
    {
        "name": "Madina",
        "lat": 5.6716, "lng": -0.1664,
        "waste_weights": {"plastic": 0.28, "organic": 0.52, "metal": 0.13, "electronic": 0.07},
        "bins": 5,
    },
    {
        "name": "Tema Industrial",
        "lat": 5.6698, "lng": -0.0166,
        "waste_weights": {"plastic": 0.15, "organic": 0.10, "metal": 0.50, "electronic": 0.25},
        "bins": 4,
    },
]

# Colour palette for synthetic images (R,G,B) per waste type
COLOUR_MAP = {
    "plastic":    (30, 144, 255),    # dodger blue
    "organic":    (34, 139, 34),     # forest green
    "metal":      (169, 169, 169),   # dark grey
    "electronic": (124, 58, 237),    # violet
}


def _jitter(coord: float, sigma: float = 0.003) -> float:
    """Add small Gaussian noise to a coordinate to spread bins within a zone."""
    return round(coord + np.random.normal(0, sigma), 6)


def _synthetic_image(waste_type: str, bin_id: str, size: int = 128) -> Path:
    """
    Create a solid-colour PNG image as a stand-in for a real waste photo.
    The colour encodes the dominant waste type for quick visual QA.
    """
    colour = COLOUR_MAP.get(waste_type, (200, 200, 200))
    # Add slight noise so each image is unique (avoids identical file hashes)
    noise  = np.random.randint(-20, 20, (size, size, 3), dtype=np.int16)
    base   = np.full((size, size, 3), colour, dtype=np.int16)
    pixel  = np.clip(base + noise, 0, 255).astype(np.uint8)
    img    = Image.fromarray(pixel, "RGB")
    path   = DIRS["raw"] / f"{bin_id}.png"
    img.save(path)
    return path


def generate_mock_dataset(num_bins: int | None = None) -> list[dict]:
    """
    Build and persist the mock bin dataset.

    Parameters
    ----------
    num_bins : int | None
        Override total bin count (defaults to sum of zone['bins']).

    Returns
    -------
    list[dict]  — one record per bin, matching POWERBI_CSV_COLUMNS schema.
    """
    records    = []
    base_time  = datetime(2024, 6, 1, 6, 0, 0)   # collection day 06:00 AM
    bin_counter = 1

    for zone in ACCRA_ZONES:
        zone_bins = zone["bins"] if num_bins is None else max(1, num_bins // len(ACCRA_ZONES))
        weights   = zone["waste_weights"]
        types     = list(weights.keys())
        probs     = list(weights.values())

        for _ in range(zone_bins):
            bin_id    = f"BIN-{bin_counter:04d}"
            waste_type = random.choices(types, weights=probs, k=1)[0]
            fill_level = round(random.uniform(0.30, 1.00), 2)   # 30 – 100 %
            confidence = round(random.uniform(0.55, 0.97), 3)

            # Synthetic image
            img_path   = _synthetic_image(waste_type, bin_id)

            record = {
                "bin_id":             bin_id,
                "location_name":      zone["name"],
                "latitude":           _jitter(zone["lat"]),
                "longitude":          _jitter(zone["lng"]),
                "waste_type":         waste_type,
                "confidence":         confidence,
                "fill_level":         fill_level,
                "collection_priority": "high" if fill_level >= 0.75 else "normal",
                "route_sequence":     0,          # filled by route optimiser
                "truck_id":           "",          # filled by route optimiser
                "timestamp":          (base_time + timedelta(minutes=bin_counter * 3)).isoformat(),
                "image_path":         str(img_path),
            }
            records.append(record)
            bin_counter += 1

    # ── Persist JSON ────────────────────────────────────────────────────────
    json_path = DIRS["exports"] / "mock_bins.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
    logger.info("Saved %d bin records -> %s", len(records), json_path)

    # ── Persist CSV ─────────────────────────────────────────────────────────
    csv_path  = DIRS["exports"] / "mock_bins.csv"
    if records:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    logger.info("Saved CSV -> %s", csv_path)

    return records


def load_mock_dataset() -> list[dict]:
    """Load the pre-generated dataset from disk (avoids regenerating)."""
    json_path = DIRS["exports"] / "mock_bins.json"
    if not json_path.exists():
        logger.info("No cached dataset found — generating …")
        return generate_mock_dataset()
    with open(json_path, encoding="utf-8") as fh:
        return json.load(fh)


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    bins = generate_mock_dataset()
    print(f"\n[OK] Generated {len(bins)} bin records across {len(ACCRA_ZONES)} Accra zones.")
    print(f"   JSON   -> {DIRS['exports'] / 'mock_bins.json'}")
    print(f"   CSV    -> {DIRS['exports'] / 'mock_bins.csv'}")
    print(f"   Images -> {DIRS['raw']}  ({len(bins)} synthetic PNGs)")
