"""
app.py — IWMRO Web Dashboard
==============================
FastAPI application that wraps the full IWMRO pipeline behind a
browser-based interface with interactive map, charts, and downloads.

Start:
    python app.py
    # or for auto-reload during development:
    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import queue as sync_queue
import shutil
import sys
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Load .env so OPENAI_API_KEY persists across restarts
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

from config import DIRS

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("iwmro.web")

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="IWMRO Dashboard", version="1.0.0")

(ROOT / "static").mkdir(exist_ok=True)
(ROOT / "templates").mkdir(exist_ok=True)

app.mount("/static",  StaticFiles(directory=ROOT / "static"),  name="static")
app.mount("/exports", StaticFiles(directory=DIRS["exports"]),  name="exports")

templates = Jinja2Templates(directory=ROOT / "templates")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _clip_available() -> bool:
    try:
        import torch          # noqa: F401
        import transformers   # noqa: F401
        return True
    except ImportError:
        return False


def _openai_available() -> bool:
    try:
        import openai         # noqa: F401
        return bool(os.environ.get("OPENAI_API_KEY"))
    except ImportError:
        return False


def _load_results() -> dict | None:
    bins_path    = DIRS["exports"] / "powerbi_bins.json"
    hotspot_path = DIRS["exports"] / "powerbi_hotspots.csv"
    manifest_path = DIRS["exports"] / "pipeline_manifest.json"

    if not bins_path.exists():
        return None

    bins = json.loads(bins_path.read_text(encoding="utf-8"))

    # Zone analysis from hotspots CSV
    hotspots = []
    if hotspot_path.exists():
        with open(hotspot_path, encoding="utf-8") as f:
            hotspots = list(csv.DictReader(f))

    # Manifest (route_summary + timing)
    manifest = {}
    route_summary = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        route_summary = manifest.get("route_summary", [])

    # If route_summary missing (old manifest), rebuild from bins
    if not route_summary:
        truck_map: dict[str, dict] = {}
        for b in bins:
            tid = b.get("truck_id", "")
            if not tid:
                continue
            if tid not in truck_map:
                truck_map[tid] = {"truck_id": tid, "num_bins": 0, "total_km": 0.0}
            truck_map[tid]["num_bins"] += 1
        route_summary = sorted(truck_map.values(), key=lambda x: x["truck_id"])

    # Aggregate stats
    total_bins    = len(bins)
    high_priority = sum(1 for b in bins if b.get("collection_priority") == "high")
    type_counts   = dict(Counter(b.get("waste_type", "unknown") for b in bins))
    trucks        = len(set(b.get("truck_id", "") for b in bins if b.get("truck_id")))
    total_km      = round(sum(r.get("total_km", 0) for r in route_summary), 1)
    hotspot_zones = [z["zone"] for z in hotspots if z.get("is_hotspot") == "True"]

    # Waste % for chart labels
    waste_pct = {
        wt: round(cnt / max(total_bins, 1) * 100, 1)
        for wt, cnt in type_counts.items()
    }

    # Campaign report text
    campaign_text = ""
    campaign_path = DIRS["exports"] / "campaign_report.txt"
    if campaign_path.exists():
        campaign_text = campaign_path.read_text(encoding="utf-8")

    # Dominant waste type for logic callout
    dominant_type = max(type_counts, key=type_counts.get) if type_counts else "plastic"

    return {
        "bins":           bins,
        "bins_json":      json.dumps(bins),
        "hotspots":       hotspots,
        "route_summary":  route_summary,
        "total_bins":     total_bins,
        "high_priority":  high_priority,
        "high_pct":       round(high_priority / max(total_bins, 1) * 100, 1),
        "trucks":         trucks,
        "total_km":       total_km,
        "type_counts":    type_counts,
        "waste_pct":      waste_pct,
        "hotspot_zones":  hotspot_zones,
        "run_timestamp":  manifest.get("run_timestamp", ""),
        "elapsed_s":      manifest.get("elapsed_seconds", 0),
        "clip_available": _clip_available(),
        "campaign_text":  campaign_text,
        "dominant_type":  dominant_type,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    has_results = (DIRS["exports"] / "powerbi_bins.json").exists()
    return templates.TemplateResponse("index.html", {
        "request":        request,
        "clip_available": _clip_available(),
        "has_results":    has_results,
        "openai_available": _openai_available(),
    })


@app.post("/set-api-key")
async def set_api_key(key: str = Form(...)):
    """Save the OpenAI API key to .env and update the running process."""
    key = key.strip()
    env_path = ROOT / ".env"

    # Read existing .env lines, replace or append OPENAI_API_KEY
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    found = False
    for i, line in enumerate(lines):
        if line.startswith("OPENAI_API_KEY="):
            lines[i] = f"OPENAI_API_KEY={key}"
            found = True
            break
    if not found:
        lines.append(f"OPENAI_API_KEY={key}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Apply immediately to the running process
    os.environ["OPENAI_API_KEY"] = key
    logger.info("OpenAI API key updated.")
    return RedirectResponse("/?key_saved=1", status_code=303)


@app.post("/run")
async def run(
    request: Request,
    demo: str = Form("1"),
    use_clip: str = Form("0"),
    images: List[UploadFile] = File(default=[]),
):
    skip_classifier = (use_clip != "1") or not _clip_available()

    uploaded = []
    if images and images[0].filename:
        for img in images:
            dest = DIRS["raw"] / img.filename
            with open(dest, "wb") as fh:
                shutil.copyfileobj(img.file, fh)
            uploaded.append(dest)
        logger.info("Received %d uploaded image(s).", len(uploaded))

    from src.api.pipeline import run_pipeline
    run_pipeline(
        image_paths=uploaded if uploaded else None,
        use_mock=True,
        skip_classifier=skip_classifier,
    )
    return RedirectResponse("/results", status_code=303)


@app.post("/run-stream")
async def run_stream(
    request: Request,
    demo: str = Form("1"),
    use_clip: str = Form("0"),
    images: List[UploadFile] = File(default=[]),
):
    """
    Run the full pipeline and stream real-time progress events back to the
    browser via Server-Sent Events (text/event-stream).

    The client reads the stream using fetch() + ReadableStream and updates
    the loading overlay step-by-step as events arrive.
    """
    # Save any uploaded images
    uploaded = []
    if images and images[0].filename:
        for img in images:
            dest = DIRS["raw"] / img.filename
            with open(dest, "wb") as fh:
                shutil.copyfileobj(img.file, fh)
            uploaded.append(dest)

    skip_classifier = (use_clip != "1") or not _clip_available()

    # Thread-safe event queue: pipeline thread pushes, async generator pulls
    q: sync_queue.Queue = sync_queue.Queue()

    def progress_cb(event: dict):
        q.put(json.dumps(event))

    def run_in_thread():
        try:
            from src.api.pipeline import run_pipeline
            run_pipeline(
                image_paths=uploaded if uploaded else None,
                use_mock=True,
                skip_classifier=skip_classifier,
                progress_cb=progress_cb,
            )
        except Exception as exc:
            q.put(json.dumps({"type": "error", "msg": str(exc)}))
        finally:
            q.put(None)  # sentinel — stop the generator

    threading.Thread(target=run_in_thread, daemon=True).start()

    loop = asyncio.get_running_loop()

    async def event_generator():
        while True:
            # Await queue item without blocking the event loop
            raw = await loop.run_in_executor(None, q.get)
            if raw is None:
                return
            yield f"data: {raw}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/results", response_class=HTMLResponse)
async def results(request: Request):
    data = _load_results()
    if not data:
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse("results.html", {
        "request": request,
        "openai_available": _openai_available(),
        **data,
    })


@app.get("/download/{filename}")
async def download(filename: str):
    path = DIRS["exports"] / filename
    if not path.exists() or not path.is_file():
        return HTMLResponse("File not found.", status_code=404)
    return FileResponse(path, filename=filename)


@app.get("/generate-report")
async def generate_report():
    """Stream an AI-generated collection insights report via SSE."""
    data = _load_results()
    if not data:
        return HTMLResponse("No results available.", status_code=404)
    if not _openai_available():
        return HTMLResponse("OPENAI_API_KEY not set.", status_code=503)

    waste_lines  = "\n".join(f"  - {wt.capitalize()}: {pct}%" for wt, pct in data["waste_pct"].items())
    route_lines  = "\n".join(
        f"  - {r['truck_id']}: {r['num_bins']} bins, {round(r.get('total_km', 0), 1)} km"
        for r in data["route_summary"]
    )
    hotspots     = ", ".join(data["hotspot_zones"]) if data["hotspot_zones"] else "None detected"
    ts           = data["run_timestamp"][:19].replace("T", " ") if data["run_timestamp"] else "unknown"

    dominant_type = data.get("dominant_type", "plastic")
    dominant_pct  = data["waste_pct"].get(dominant_type, 0)

    community_tips_map = {
        "plastic":    "Rinse and flatten bottles before disposal. Avoid single-use sachets — use refillable containers. Deposit plastics at the Kaneshie Recycling Point for a 20 Gp credit.",
        "organic":    "Use designated green bins for food scraps. Compost at home — contact AMA for a free starter kit. Never dump food waste in drainage channels.",
        "metal":      "Contact the Tema Scrap Collection Service for bulky metal items. Do not burn metal waste — toxic fumes are a serious health hazard. Arrange large items for scheduled kerb-side collection.",
        "electronic": "Never place e-waste in household bins or open dumps. Drop off phones, batteries and cables at the Tema E-Waste Facility. Businesses should contact AMA for bulk e-waste collection.",
    }
    community_tip = community_tips_map.get(dominant_type, "Sort waste into correct bins before disposal.")

    prompt = f"""You are an expert waste management analyst writing an official Community Campaign Report for the Accra Metropolitan Assembly (AMA), Ghana.

Analyse the real-time collection data below and produce a complete, professional campaign document.

COLLECTION RUN: {ts}
BINS MONITORED: {data['total_bins']} across Accra zones
HIGH-PRIORITY BINS: {data['high_priority']} ({data['high_pct']}%)
TRUCKS DISPATCHED: {data['trucks']}
TOTAL ROUTE DISTANCE: {data['total_km']} km
DOMINANT WASTE TYPE: {dominant_type.capitalize()} ({dominant_pct}% of bins)

WASTE COMPOSITION:
{waste_lines}

HOTSPOT ZONES (requiring urgent collection): {hotspots}

TRUCK ROUTES:
{route_lines}

COMMUNITY GUIDANCE NOTE FOR {dominant_type.upper()} WASTE:
{community_tip}

Write the report using EXACTLY these five markdown headings (no others):
## Executive Summary
## Key Findings
## Hotspot & Priority Alert
## Operational Recommendations
## Community Action Points

Rules:
- Reference actual numbers from the data throughout.
- "Hotspot & Priority Alert" must name the specific hotspot zones and explain urgency.
- "Community Action Points" must give 3 specific, practical tips for Accra residents about {dominant_type} waste — incorporate the guidance note above.
- Keep each section to 2–4 sentences. Professional language suitable for AMA management and public distribution."""

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def event_generator():
        try:
            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                max_tokens=900,
                temperature=0.5,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'text': delta})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Start ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 54)
    print("  IWMRO Web Dashboard")
    print("  http://localhost:8000")
    print("=" * 54 + "\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
