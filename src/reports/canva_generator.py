"""
src/reports/canva_generator.py
================================
Generates a Canva-ready content pack from a completed pipeline run.

The pack contains structured copy suitable for direct paste into Canva
poster, social-media, and community-report templates used by the
Accra Metropolitan Assembly Waste Management Division.

No external API is required — all content is rule-based and derived
from the run's statistical output, so the pack is always populated.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Waste-type specific community guidance bullets
_WASTE_BULLETS = {
    "plastic": [
        "Rinse and flatten plastic bottles before placing in blue recycling bins.",
        "Avoid single-use sachets — carry a refillable water bottle instead.",
        "Deposit sorted plastics at Kaneshie Recycling Point for a 20 Gp credit per kg.",
    ],
    "organic": [
        "Use designated green bins for food scraps and market waste.",
        "Home composting reduces collection pressure — contact AMA for a free starter kit.",
        "Never dispose of organic waste in drainage channels; it causes flooding.",
    ],
    "metal": [
        "Arrange large scrap metal for scheduled kerb-side collection — do not burn.",
        "Contact the Tema Scrap Collection Service for bulk metal items.",
        "Metal waste in open dumps is a safety hazard and attracts disease vectors.",
    ],
    "electronic": [
        "Never place phones, batteries, or cables in household bins.",
        "Drop off e-waste at the AMA Tema E-Waste Facility — no charge for residents.",
        "Businesses with bulk e-waste should contact AMA for a dedicated collection slot.",
    ],
}

_DEFAULT_BULLETS = [
    "Sort waste into the correct colour-coded bins before disposal.",
    "High-fill bins have been flagged for priority collection — do not overfill.",
    "Report overflowing or damaged bins to AMA via the IWMRO hotline.",
]

# AMA brand colour palette for Canva template alignment
_BRAND_COLORS = {
    "primary":    "#1B5E20",
    "primary_mid":"#2E7D32",
    "accent":     "#F59E0B",
    "danger":     "#DC2626",
    "light_bg":   "#F0FDF4",
    "dark_text":  "#1F2937",
}


def generate_canva_pack(
    bins: list[dict],
    route_summary: list[dict],
    dominant_type: str,
    hotspot_zones: list[str],
    waste_pct: dict[str, float],
    total_bins: int,
    high_priority: int,
    total_km: float,
    out_dir: Optional[Path] = None,
) -> dict:
    """
    Generate a Canva-ready campaign content pack from pipeline results.

    Returns a dict with:
        headline, hotspot_alert, bullets, next_collection,
        dominant_type, stats_summary, brand_colors

    Also writes the pack to out_dir/canva_pack.json if out_dir is provided.
    """
    # ── Headline ──────────────────────────────────────────────────────────────
    if high_priority > 0:
        headline = f"Accra Waste Alert — {high_priority} Bins Need Urgent Collection"
    else:
        headline = "Accra Waste Update — All Zones Operating Within Normal Parameters"

    # ── Hotspot alert ─────────────────────────────────────────────────────────
    if hotspot_zones:
        zones_str = ", ".join(hotspot_zones)
        hotspot_alert = f"⚠ Critical zones requiring immediate attention: {zones_str}"
    else:
        hotspot_alert = "✓ No hotspot zones detected this collection run."

    # ── Community bullets ─────────────────────────────────────────────────────
    # Start with waste-type specific tips
    bullets = list(_WASTE_BULLETS.get(dominant_type, _DEFAULT_BULLETS)[:2])

    # Add data-driven operational bullets
    top_types = sorted(waste_pct.items(), key=lambda x: x[1], reverse=True)[:2]
    for wtype, pct in top_types:
        if wtype != dominant_type:
            bullets.append(
                f"{wtype.capitalize()} waste makes up {pct}% of bins — "
                + (_WASTE_BULLETS.get(wtype, [_DEFAULT_BULLETS[0]])[0])
            )
            break

    # Route efficiency note
    trucks = len(route_summary)
    if trucks > 0:
        bullets.append(
            f"{trucks} optimised truck route{'s' if trucks > 1 else ''} covering "
            f"{total_km} km — Haversine TSP routing reduces fuel use by up to 26%."
        )

    # High priority summary
    high_pct = round(high_priority / max(total_bins, 1) * 100, 1)
    if high_priority > 0:
        bullets.append(
            f"{high_priority} bins ({high_pct}%) are at ≥75% capacity and flagged "
            "for priority collection within the next 48 hours."
        )

    bullets = bullets[:5]  # cap at 5 for poster readability

    # ── Next collection reminder ───────────────────────────────────────────────
    next_collection = "Next scheduled collection: within 48 hours of this report."

    # ── Stats summary line (for poster subheading) ────────────────────────────
    dominant_pct = waste_pct.get(dominant_type, 0)
    stats_summary = (
        f"{total_bins} bins monitored · {high_priority} high-priority · "
        f"Dominant type: {dominant_type.capitalize()} ({dominant_pct}%)"
    )

    pack = {
        "headline":       headline,
        "hotspot_alert":  hotspot_alert,
        "bullets":        bullets,
        "next_collection": next_collection,
        "dominant_type":  dominant_type,
        "stats_summary":  stats_summary,
        "brand_colors":   _BRAND_COLORS,
        "usage_note": (
            "Paste each field into your Canva poster or social-media template. "
            "Brand colours are provided for colour-picker alignment. "
            "Generated by IWMRO — Accra Metropolitan Assembly."
        ),
    }

    if out_dir:
        out_path = Path(out_dir) / "canva_pack.json"
        try:
            out_path.write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Canva pack written → %s", out_path)
        except Exception as exc:
            logger.warning("Could not write canva_pack.json: %s", exc)

    return pack
