"""
src/api/pipeline.py
====================
Master orchestration pipeline for IWMRO.

Data flow
---------
input_image(s)
    │
    ▼  [1] privacy_filter.py  ─→  PII-free image (data/processed/)
    │
    ▼  [2] waste_classifier.py ─→  JSON metadata per bin
    │
    ▼  [3] route_optimizer.py  ─→  Routed bin list + hotspot map
    │
    ▼  [4] powerbi_exporter    ─→  CSV / JSON for Power BI import
    │
    ▼  [5] report_generator    ─→  .txt / .html / .json campaign report
    │
    ▼  data/exports/  (all artefacts)

Usage
-----
python -m src.api.pipeline                  # uses mock dataset
python -m src.api.pipeline --image path.jpg # single real image
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import DIRS, POWERBI_CSV_COLUMNS

logger = logging.getLogger(__name__)


# ── Step helpers ──────────────────────────────────────────────────────────────

def _step_privacy(image_paths: list[Path]) -> list[Path]:
    """
    Step 1 — Apply privacy filter to all input images.
    Returns list of processed image paths.
    """
    from src.privacy.privacy_filter import filter_batch
    results = filter_batch(image_paths)
    processed = [p for p, _ in results]
    logger.info("[Step 1] Privacy filter: %d images processed.", len(processed))
    return processed


def _step_classify(processed_paths: list[Path], bin_records: list[dict]) -> list[dict]:
    """
    Step 2 — Run CLIP classifier on each processed image and merge
    classification results into the corresponding bin record.
    """
    path_map = {p.stem: p for p in processed_paths}

    # Only load the model if at least one processed image matches a bin record
    matchable = [r for r in bin_records if path_map.get(r["bin_id"]) and path_map[r["bin_id"]].exists()]
    if not matchable:
        logger.info("[Step 2] No images match bin IDs — keeping pre-assigned labels.")
        return bin_records

    try:
        from models.waste_classifier import WasteClassifier
        clf = WasteClassifier()
    except Exception as exc:
        logger.warning("[Step 2] CLIP model could not load (%s) — keeping pre-assigned labels.", exc)
        return bin_records

    for record in matchable:
        bin_id = record["bin_id"]
        img_p  = path_map[bin_id]
        try:
            result = clf.classify(img_p)
            record["waste_type"]  = result["label"]
            record["confidence"]  = result["confidence"]
            record["uncertain"]   = result.get("uncertain", False)
            record["all_scores"]  = result.get("all_scores", {})
        except Exception as exc:
            logger.warning("CLIP classify failed for %s: %s — keeping mock label.", bin_id, exc)

    logger.info("[Step 2] Classification complete for %d bins.", len(bin_records))
    return bin_records


def _step_optimise(bin_records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Step 3 — Route optimisation + hotspot detection."""
    from src.optimizer.route_optimizer import optimise_routes, identify_hotspots
    hotspots = identify_hotspots(bin_records)
    annotated, route_summary = optimise_routes(bin_records)
    logger.info("[Step 3] Route optimisation: %d bins across %d trucks.",
                len(annotated), len(route_summary))
    return annotated, route_summary, hotspots


def _step_export_powerbi(annotated: list[dict], hotspots: dict) -> dict[str, Path]:
    """
    Step 4 — Export Power BI–ready artefacts.

    Outputs
    -------
    powerbi_bins.csv   — bin-level detail (matches POWERBI_CSV_COLUMNS)
    powerbi_hotspots.csv  — zone-level hotspot summary
    powerbi_bins.json     — JSON mirror for API consumers
    """
    exports: dict[str, Path] = {}

    # ── Bin-level CSV ─────────────────────────────────────────────────────
    csv_path = DIRS["exports"] / "powerbi_bins.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=POWERBI_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(annotated)
    exports["bins_csv"] = csv_path
    logger.info("[Step 4] Power BI bins CSV -> %s", csv_path)

    # ── Hotspot CSV ────────────────────────────────────────────────────────
    hotspot_path = DIRS["exports"] / "powerbi_hotspots.csv"
    hotspot_rows = [
        {
            "zone":                z,
            "total_bins":          info["total_bins"],
            "mean_fill_level":     info["mean_fill_level"],
            "dominant_type":       info["dominant_type"],
            "high_priority_bins":  info["high_priority_bins"],
            "is_hotspot":          info["is_hotspot"],
        }
        for z, info in hotspots.items()
    ]
    with open(hotspot_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(hotspot_rows[0].keys()) if hotspot_rows else [])
        if hotspot_rows:
            writer.writeheader()
            writer.writerows(hotspot_rows)
    exports["hotspots_csv"] = hotspot_path

    # ── JSON mirror ────────────────────────────────────────────────────────
    json_path = DIRS["exports"] / "powerbi_bins.json"
    json_path.write_text(json.dumps(annotated, indent=2), encoding="utf-8")
    exports["bins_json"] = json_path

    return exports


def _step_report(annotated: list[dict], route_summary: list[dict]) -> dict[str, Path]:
    """Step 5 — Generate text + HTML + JSON campaign reports."""
    from src.reports.report_generator import generate_reports
    outputs = generate_reports(annotated, route_summary)
    logger.info("[Step 5] Reports generated: %s", list(outputs.keys()))
    return outputs


def _step_sheets(annotated: list[dict], run_id: str) -> int:
    """
    Step 6 — Push bin records to Google Sheets (optional).
    Returns the number of rows appended, or 0 if skipped.
    """
    from src.integrations.sheets_logger import log_bins_to_sheet
    return log_bins_to_sheet(annotated, run_id=run_id)


# ── Pipeline manifest ─────────────────────────────────────────────────────────
def _write_manifest(artefacts: dict, elapsed: float, route_summary: list = None) -> Path:
    """Write a pipeline_manifest.json describing all output files."""
    manifest = {
        "run_timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "artefacts": {k: str(v) for k, v in artefacts.items()},
        "route_summary": route_summary or [],
    }
    path = DIRS["exports"] / "pipeline_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


# ── Entry point ───────────────────────────────────────────────────────────────
def run_pipeline(
    image_paths: Optional[list[Path]] = None,
    use_mock: bool = True,
    skip_classifier: bool = False,
    progress_cb=None,
    cancel_event=None,
) -> dict:
    """
    Execute the full IWMRO pipeline.

    Parameters
    ----------
    image_paths    : list of real image paths (None → use mock dataset)
    use_mock       : if True and image_paths is None, generate mock data
    skip_classifier: if True, use existing waste_type labels (faster demo)
    progress_cb    : optional callable(dict) — receives structured progress
                     events for real-time streaming to the web dashboard.

    Returns
    -------
    dict of all output artefacts.
    """
    def emit(event: dict):
        if progress_cb:
            try:
                progress_cb(event)
            except Exception:
                pass

    def check_cancel(at_step: int):
        if cancel_event and cancel_event.is_set():
            emit({"type": "pipeline_stopped", "at_step": at_step})
            raise StopIteration(f"Pipeline cancelled after step {at_step}")

    t0 = time.perf_counter()
    logger.info("=" * 60)
    logger.info("  IWMRO Pipeline  -  %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    # ── Load or generate bin records ──────────────────────────────────────
    from src.data.mock_data_generator import load_mock_dataset, generate_mock_dataset
    if use_mock and not image_paths:
        bin_records = load_mock_dataset()
    else:
        bin_records = generate_mock_dataset()

    image_paths = image_paths or [Path(r["image_path"]) for r in bin_records if r.get("image_path")]
    emit({"type": "init", "total_bins": len(bin_records), "total_steps": 5})

    # ── Step 1: Privacy filter ────────────────────────────────────────────
    emit({
        "type": "step_start", "step": 1, "name": "Privacy Filter",
        "desc": f"Scanning {len(image_paths)} images for faces and licence plates...",
    })
    processed_paths = _step_privacy(image_paths)
    emit({
        "type": "step_done", "step": 1,
        "result": f"{len(processed_paths)} images scanned · PII regions anonymised",
    })
    check_cancel(1)

    # ── Step 2: Classify ──────────────────────────────────────────────────
    if not skip_classifier:
        # Check upfront how many images can actually be classified
        from pathlib import Path as _Path
        proc_map    = {p.stem: p for p in processed_paths}
        clip_targets = sum(
            1 for r in bin_records
            if proc_map.get(r["bin_id"]) and proc_map[r["bin_id"]].exists()
        )
        clip_desc = (
            f"Running CLIP zero-shot classification on {clip_targets} images..."
            if clip_targets
            else "No matching images found — using pre-assigned waste labels."
        )
        emit({
            "type": "step_start", "step": 2, "name": "Waste Classifier",
            "desc": clip_desc,
        })
        bin_records = _step_classify(processed_paths, bin_records)
        emit({
            "type": "step_done", "step": 2,
            "result": f"{len(bin_records)} bins classified  (plastic / organic / metal)",
        })
    else:
        logger.info("[Step 2] Skipping classifier - using existing labels.")
        emit({
            "type": "step_skip", "step": 2, "name": "Waste Classifier",
            "result": "Using pre-assigned labels — install torch + transformers to enable CLIP",
        })
    check_cancel(2)

    # ── Step 3: Optimise ──────────────────────────────────────────────────
    emit({
        "type": "step_start", "step": 3, "name": "Route Optimiser",
        "desc": f"Running Haversine TSP + hotspot detection across {len(bin_records)} bins...",
    })
    annotated, route_summary, hotspots = _step_optimise(bin_records)
    hot_zones  = [z for z, info in hotspots.items() if info["is_hotspot"]]
    trucks     = len(set(b["truck_id"] for b in annotated if b.get("truck_id")))
    total_km   = round(sum(r.get("total_km", 0) for r in route_summary), 1)
    result_parts = [f"{trucks} trucks optimised", f"{total_km} km total route"]
    if hot_zones:
        result_parts.append(f"Hotspot: {', '.join(hot_zones)}")
    emit({
        "type": "step_done", "step": 3,
        "result": "  ·  ".join(result_parts),
        "hotspots": hot_zones,
    })
    check_cancel(3)

    # ── Step 4: Export for Power BI ───────────────────────────────────────
    emit({
        "type": "step_start", "step": 4, "name": "Power BI Export",
        "desc": "Writing optimised bin data to CSV and JSON for Power BI import...",
    })
    export_paths = _step_export_powerbi(annotated, hotspots)
    emit({
        "type": "step_done", "step": 4,
        "result": f"powerbi_bins.csv ({len(annotated)} rows)  ·  powerbi_hotspots.csv",
    })
    check_cancel(4)

    # ── Step 5: Generate reports ──────────────────────────────────────────
    emit({
        "type": "step_start", "step": 5, "name": "Campaign Reports",
        "desc": "Generating community education reports (text, HTML, JSON)...",
    })
    report_paths = _step_report(annotated, route_summary)
    emit({
        "type": "step_done", "step": 5,
        "result": "campaign_report.txt  ·  .html  ·  campaign_payload.json",
    })

    # ── Step 6: Google Sheets sync (optional) ────────────────────────────
    run_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    from config import SHEETS_CREDS_FILE
    if SHEETS_CREDS_FILE.exists():
        emit({
            "type": "step_start", "step": 6, "name": "Sheets Sync",
            "desc": f"Pushing {len(annotated)} bin records to Google Sheets...",
        })
        rows_pushed = _step_sheets(annotated, run_id=run_id)
        emit({
            "type": "step_done", "step": 6,
            "result": f"{rows_pushed} rows appended to IWMRO_Data_Log",
        })
    else:
        emit({
            "type": "step_skip", "step": 6, "name": "Sheets Sync",
            "result": "creds.json not found — add service-account key to enable",
        })

    # ── Fallback AI insights (always generated, no API key needed) ────────
    try:
        from collections import Counter as _Counter
        from src.utils.insights_fallback import generate_fallback_insights
        _type_counts  = dict(_Counter(b.get("waste_type", "unknown") for b in annotated))
        _dominant     = max(_type_counts, key=_type_counts.get) if _type_counts else "plastic"
        _total        = len(annotated)
        _waste_pct    = {wt: round(cnt / max(_total, 1) * 100, 1) for wt, cnt in _type_counts.items()}
        _fallback     = generate_fallback_insights(
            bins=annotated, route_summary=route_summary, dominant_type=_dominant,
            hotspot_zones=hot_zones, waste_pct=_waste_pct, total_km=total_km,
        )
        fi_path = DIRS["exports"] / "fallback_insights.json"
        fi_path.write_text(json.dumps(_fallback, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Fallback insights -> %s", fi_path)
    except Exception as _exc:
        logger.warning("Could not generate fallback insights: %s", _exc)

    # ── Manifest ──────────────────────────────────────────────────────────
    all_artefacts = {**export_paths, **report_paths}
    manifest_path = _write_manifest(all_artefacts, time.perf_counter() - t0, route_summary)
    all_artefacts["manifest"] = manifest_path

    elapsed = time.perf_counter() - t0
    emit({"type": "pipeline_done", "elapsed": round(elapsed, 1), "hotspots": hot_zones})

    logger.info("=" * 60)
    logger.info("  Pipeline complete in %.1f s", elapsed)
    logger.info("  All outputs -> %s", DIRS["exports"])
    logger.info("=" * 60)

    return all_artefacts


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Run the IWMRO pipeline.")
    parser.add_argument("--image",  type=Path, help="Path to a single image.")
    parser.add_argument("--images", type=Path, help="Directory of images.")
    parser.add_argument("--skip-classifier", action="store_true",
                        help="Skip CLIP inference (use mock labels).")
    args = parser.parse_args()

    images = None
    if args.image:
        images = [args.image]
    elif args.images:
        images = list(args.images.glob("*.jpg")) + list(args.images.glob("*.png"))

    artefacts = run_pipeline(
        image_paths=images,
        use_mock=(images is None),
        skip_classifier=args.skip_classifier,
    )

    print("\n── Output artefacts ──────────────────────────────────────")
    for name, path in artefacts.items():
        print(f"  {name:20s}  {path}")
