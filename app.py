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
from pydantic import BaseModel
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

from config import (
    DIRS,
    BASELINE_DISTANCE_FACTOR, FUEL_EFFICIENCY_KM_PER_L, FUEL_PRICE_GHS_PER_L,
    AVG_SPEED_KMH, RECYCLING_RATE_PCT, AVG_BIN_WEIGHT_KG, RECYCLING_VALUE_GHS_PER_KG,
    CLIP_MODEL_ID, POWERBI_EMBED_URL as _POWERBI_EMBED_URL_DEFAULT,
    FAILURE_MODES, RBAC_ROLES, DATA_RETENTION_DAYS, BLUR_STRENGTH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("iwmro.web")

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="IWMRO Dashboard", version="1.0.0")

(ROOT / "static").mkdir(exist_ok=True)
(ROOT / "templates").mkdir(exist_ok=True)

app.mount("/static",  StaticFiles(directory=ROOT / "static"),  name="static")
app.mount("/exports", StaticFiles(directory=DIRS["exports"]),  name="exports")

templates = Jinja2Templates(directory=ROOT / "templates")

# ── Pipeline cancellation flag ────────────────────────────────────────────────
_pipeline_cancel: threading.Event = threading.Event()


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

    # ── Business metrics ──────────────────────────────────────────────────────
    biz = _compute_business_metrics(total_km, total_bins, trucks)

    # ── Model performance ─────────────────────────────────────────────────────
    model_perf = _compute_model_performance(bins)

    # ── Sample bins (3 high + 3 normal, up to 6) ──────────────────────────────
    high_bins   = [b for b in bins if b.get("collection_priority") == "high"][:3]
    normal_bins = [b for b in bins if b.get("collection_priority") != "high"][:3]
    sample_bins = (high_bins + normal_bins)[:6]

    # ── Canva content pack ────────────────────────────────────────────────────
    from src.reports.canva_generator import generate_canva_pack
    canva_pack = generate_canva_pack(
        bins=bins, route_summary=route_summary, dominant_type=dominant_type,
        hotspot_zones=hotspot_zones, waste_pct=waste_pct, total_bins=total_bins,
        high_priority=high_priority, total_km=total_km, out_dir=DIRS["exports"],
    )

    # ── Fallback AI insights ──────────────────────────────────────────────────
    fallback_insights = None
    fallback_path = DIRS["exports"] / "fallback_insights.json"
    if fallback_path.exists():
        try:
            fallback_insights = json.loads(fallback_path.read_text(encoding="utf-8"))
        except Exception:
            pass

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
        # business metrics
        **biz,
        # model evaluation
        "model_perf":     model_perf,
        "sample_bins":    sample_bins,
        # tool integrations
        "canva_pack":     canva_pack,
        "fallback_insights": fallback_insights,
        # config constants for templates
        "failure_modes":  FAILURE_MODES,
        "rbac_roles":     RBAC_ROLES,
        "data_retention_days": DATA_RETENTION_DAYS,
        "clip_model_id":  CLIP_MODEL_ID,
        "powerbi_embed_url": os.environ.get("POWERBI_EMBED_URL", _POWERBI_EMBED_URL_DEFAULT),
        "assumptions": {
            "baseline_factor":  BASELINE_DISTANCE_FACTOR,
            "fuel_efficiency":  FUEL_EFFICIENCY_KM_PER_L,
            "fuel_price":       FUEL_PRICE_GHS_PER_L,
            "avg_speed":        AVG_SPEED_KMH,
            "blur_strength":    BLUR_STRENGTH,
        },
    }


def _compute_business_metrics(total_km: float, total_bins: int, trucks: int) -> dict:
    """Calculate route optimisation value vs an unoptimised baseline."""
    baseline_km   = round(total_km * BASELINE_DISTANCE_FACTOR, 1)
    saved_km      = round(baseline_km - total_km, 1)
    litres_saved  = round(saved_km / max(FUEL_EFFICIENCY_KM_PER_L, 0.01), 1)
    cost_saved    = round(litres_saved * FUEL_PRICE_GHS_PER_L, 2)
    time_saved    = round(saved_km / max(AVG_SPEED_KMH, 1) * 60, 0)   # minutes
    recyclable    = round(total_bins * RECYCLING_RATE_PCT)
    recycle_value = round(recyclable * AVG_BIN_WEIGHT_KG * RECYCLING_VALUE_GHS_PER_KG, 2)
    return {
        "baseline_km":    baseline_km,
        "saved_km":       saved_km,
        "litres_saved":   litres_saved,
        "cost_saved_ghs": cost_saved,
        "time_saved_min": int(time_saved),
        "recyclable_bins": recyclable,
        "recycling_value_ghs": recycle_value,
    }


def _compute_model_performance(bins: list) -> dict:
    """Summarise classifier confidence across all bins."""
    confs = [float(b.get("confidence", 0)) for b in bins if b.get("confidence") is not None]
    if not confs:
        return {"avg_confidence": 0, "low_conf_count": 0, "low_conf_pct": 0}
    avg  = round(sum(confs) / len(confs) * 100, 1)
    low  = sum(1 for c in confs if c < 0.40)
    return {
        "avg_confidence":  avg,
        "low_conf_count":  low,
        "low_conf_pct":    round(low / max(len(confs), 1) * 100, 1),
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

    # Reset cancel flag for fresh run
    _pipeline_cancel.clear()

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
                cancel_event=_pipeline_cancel,
            )
        except StopIteration:
            pass  # clean cancel — pipeline_stopped event already emitted
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


@app.get("/analysis", response_class=HTMLResponse)
async def analysis(request: Request):
    data = _load_results()
    if not data:
        return HTMLResponse(
            "<html><body style='font-family:sans-serif;padding:3rem;text-align:center'>"
            "<h2 style='color:#1b5e20'>No pipeline data yet</h2>"
            "<p style='color:#6b7280;margin:1rem 0'>Run the pipeline first to generate analysis data.</p>"
            "<a href='/' style='color:#15803d;font-weight:600'>Go to home page &rarr;</a>"
            "</body></html>",
            status_code=200,
        )
    return templates.TemplateResponse("analysis.html", {
        "request": request,
        **data,
    })


@app.get("/campaign-report", response_class=HTMLResponse)
async def campaign_report(request: Request):
    """Serve a standalone printable campaign report page."""
    data = _load_results()
    if not data:
        return HTMLResponse(
            "<html><body style='font-family:sans-serif;padding:3rem;text-align:center'>"
            "<h2 style='color:#1b5e20'>No pipeline data yet</h2>"
            "<p style='color:#6b7280;margin:1rem 0'>Run the pipeline first.</p>"
            "<a href='/' style='color:#15803d;font-weight:600'>Go to home page &rarr;</a>"
            "</body></html>",
            status_code=200,
        )
    return templates.TemplateResponse("campaign_report.html", {
        "request": request,
        **data,
    })


class _PbiUrlPayload(BaseModel):
    url: str


@app.post("/set-powerbi-url")
async def set_powerbi_url(payload: _PbiUrlPayload):
    """
    Save the Power BI embed URL to .env so it persists across restarts.
    Also updates the running process via os.environ so it takes effect immediately.
    """
    url = payload.url.strip()
    # Basic sanity check — must look like a Power BI embed URL
    if url and "powerbi.com" not in url:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="URL must be from powerbi.com")

    # Update running process immediately
    os.environ["POWERBI_EMBED_URL"] = url

    # Persist to .env file
    env_path = ROOT / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    key = "POWERBI_EMBED_URL"
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}=") or line.startswith(f"{key} ="):
            lines[i] = f'{key}="{url}"'
            updated = True
            break
    if not updated:
        lines.append(f'{key}="{url}"')
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"ok": True, "url": url}


@app.post("/stop-pipeline")
async def stop_pipeline():
    """Signal the running pipeline to abort after its current step."""
    _pipeline_cancel.set()
    return {"ok": True}


@app.get("/healthz")
async def healthz():
    """Streamlit Cloud health-check endpoint."""
    return {"status": "ok"}


@app.get("/download/{filename}")
async def download(filename: str):
    path = DIRS["exports"] / filename
    if not path.exists() or not path.is_file():
        return HTMLResponse("File not found.", status_code=404)
    return FileResponse(path, filename=filename)


@app.get("/image/{folder}/{filename}")
async def serve_image(folder: str, filename: str):
    """Serve a raw or processed bin image (security: folder whitelist only)."""
    if folder not in ("raw", "processed"):
        return HTMLResponse("Invalid folder.", status_code=400)
    # Prevent path traversal
    safe_name = Path(filename).name
    if safe_name != filename or ".." in filename:
        return HTMLResponse("Invalid filename.", status_code=400)
    path = DIRS[folder] / safe_name
    if not path.exists() or not path.is_file():
        return HTMLResponse("Image not found.", status_code=404)
    return FileResponse(path)


@app.post("/validate-image")
async def validate_image(image: UploadFile = File(...)):
    """
    Quick AI check: does the uploaded image contain waste/refuse?
    Returns {is_waste, confidence, label, note}.
    Uses GPT-4o-mini vision if an API key is set; falls back to CLIP
    if already cached; otherwise accepts all valid images.
    """
    import base64
    from PIL import Image as PILImage
    import io

    img_bytes = await image.read()

    # ── Basic validity check ────────────────────────────────────────────────
    try:
        pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = pil_img.size
    except Exception as exc:
        return {"is_waste": False, "confidence": 0.0,
                "label": "invalid", "note": f"Cannot read image: {exc}"}

    # ── OpenAI GPT-4o-mini vision ────────────────────────────────────────────
    if _openai_available():
        try:
            from openai import AsyncOpenAI
            b64 = base64.b64encode(img_bytes).decode()
            mime = image.content_type or "image/jpeg"
            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"}},
                        {"type": "text",
                         "text": (
                             "You are a waste management AI. "
                             "Does this image show a waste bin, garbage, rubbish, litter, refuse, "
                             "or any form of solid waste? "
                             "Reply in this exact format:\n"
                             "VERDICT: YES or NO\n"
                             "REASON: one sentence"
                         )},
                    ],
                }],
                max_tokens=80,
                temperature=0,
            )
            reply = resp.choices[0].message.content.strip()
            is_waste = "VERDICT: YES" in reply.upper()
            reason_line = next(
                (l for l in reply.splitlines() if l.upper().startswith("REASON:")), ""
            )
            reason = reason_line.replace("REASON:", "").replace("Reason:", "").strip()
            return {
                "is_waste": is_waste,
                "confidence": 0.92 if is_waste else 0.88,
                "label": "waste detected" if is_waste else "not waste",
                "note": reason or ("Waste content confirmed by GPT-4o-mini vision." if is_waste
                                   else "No waste content detected."),
                "method": "gpt-4o-mini",
            }
        except Exception as exc:
            logger.warning("Vision validation failed: %s — falling back.", exc)

    # ── CLIP fallback (only if model is already cached in memory) ────────────
    try:
        from models.waste_classifier import WasteClassifier
        # Save to a temp path for the classifier
        tmp_path = DIRS["raw"] / f"_validate_{image.filename or 'tmp.jpg'}"
        tmp_path.write_bytes(img_bytes)
        clf = WasteClassifier()
        result = clf.classify(tmp_path)
        tmp_path.unlink(missing_ok=True)
        conf = float(result.get("confidence", 0))
        is_waste = conf > 0.25
        return {
            "is_waste": is_waste,
            "confidence": round(conf, 3),
            "label": result.get("label", "unknown"),
            "note": (
                f"CLIP classified as {result['label']} ({round(conf*100,1)}% confidence)."
                if is_waste else
                "CLIP confidence too low — image may not contain recognisable waste."
            ),
            "method": "clip",
        }
    except Exception:
        pass

    # ── Final fallback: accept valid images ──────────────────────────────────
    return {
        "is_waste": True,
        "confidence": 0.5,
        "label": "unverified",
        "note": f"Valid image ({w}×{h}px). Install torch+transformers or add an OpenAI key for AI validation.",
        "method": "basic",
    }


@app.get("/download-canva")
async def download_canva():
    """Download the Canva content pack JSON."""
    path = DIRS["exports"] / "canva_pack.json"
    if not path.exists():
        return HTMLResponse("Canva pack not yet generated. Run the pipeline first.", status_code=404)
    return FileResponse(path, filename="canva_pack.json", media_type="application/json")


@app.get("/generate-report")
async def generate_report():
    """Stream an AI-generated collection insights report via SSE."""
    data = _load_results()
    if not data:
        return HTMLResponse("No results available.", status_code=404)

    # ── Fallback: stream pre-generated insights token-by-token ────────────────
    if not _openai_available():
        fi = data.get("fallback_insights")
        if not fi:
            return HTMLResponse("OPENAI_API_KEY not set and no cached insights available.", status_code=503)

        # Combine sections into markdown text matching the GPT output format
        sections = [
            ("## Executive Summary", fi.get("executive_summary", "")),
            ("## Key Findings",      fi.get("key_findings", "")),
            ("## Hotspot & Priority Alert", fi.get("hotspot_alert", "")),
            ("## Operational Recommendations", fi.get("recommendations", "")),
            ("## Community Action Points",     fi.get("community_action", "")),
        ]
        full_text = "\n\n".join(f"{heading}\n{body}" for heading, body in sections)

        async def fallback_generator():
            import asyncio
            # Stream word by word to simulate live generation
            words = full_text.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'text': chunk})}\n\n"
                if i % 8 == 0:
                    await asyncio.sleep(0.02)
            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(
            fallback_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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
