# IWMRO — Planning Document
## Integrated Waste Management and Recycling Optimizer
**Accra Metropolitan Assembly (AMA) · Master's Capstone Project**

---

## 1. Project Overview

IWMRO is an AI-powered waste management dashboard built for the Accra Metropolitan Assembly, Ghana. It processes field imagery from waste bins across 9 Accra zones, classifies waste types, detects collection hotspots, optimises truck routes, and generates community campaign reports — all in a single automated pipeline run accessible through a web browser.

---

## 2. System Workflow

### 2.1 High-Level Pipeline

```
Field Images / Mock Data
        │
        ▼
┌─────────────────────┐
│  Step 1: Capture    │  Raw JPEG/PNG images from field bins
│  & Privacy Filter   │  → Detect & blur faces and licence plates (OpenCV)
└─────────┬───────────┘
          │  anonymised images
          ▼
┌─────────────────────┐
│  Step 2: Classify   │  CLIP zero-shot AI assigns waste type
│  & Score Fill Level │  (Plastic / Organic / Metal / Electronic)
└─────────┬───────────┘
          │  waste labels · fill levels · priority flags
          ▼
┌─────────────────────┐
│  Step 3: Detect     │  Flag high-fill-level zones as hotspots
│  Hotspots & Route   │  Assign bins to trucks via Haversine TSP
│  Trucks             │  Nearest-neighbour route optimisation
└─────────┬───────────┘
          │  hotspot zones · truck assignments · route km
          ▼
┌─────────────────────┐
│  Step 4: Build      │  Write powerbi_bins.csv, hotspots.csv,
│  Dashboard Exports  │  pipeline_manifest.json
└─────────┬───────────┘
          │  structured export files
          ▼
┌─────────────────────┐
│  Step 5: Generate   │  Identify dominant waste type
│  Campaign Report    │  Write zone-specific community guidance
│                     │  Compile AMA campaign report (TXT)
└─────────────────────┘
          │
          ▼
    Results Dashboard
    (FastAPI + Jinja2)
```

### 2.2 Results Dashboard Flow

```
/results page loads
      │
      ├── KPI Cards (bins · priority · trucks · km)
      ├── Leaflet Map (interactive, zone-filterable)
      ├── Sidebar (Waste Breakdown donut · Zone Analysis)
      ├── Charts Row (Fill Distribution · Priority by Zone · Truck Load)
      │
      ├── [AI Mode ON]  ──► Community Campaign Report (AI-generated)
      │     │                 Auto-triggers GPT-4o mini via SSE stream
      │     │                 5 sections: Executive Summary · Key Findings ·
      │     │                 Hotspot Alert · Recommendations · Community Action
      │     └──► Download PDF button (appears after generation)
      │
      ├── [AI Mode OFF] ──► Static Community Campaign Report
      │     │                 Hardcoded tips by dominant waste type
      │     │                 Hotspot alerts · KPI strip · waste breakdown
      │     └──► Download PDF button (always visible)
      │
      └── Truck Route Summary Table · Pipeline Info footer
```

### 2.3 AI Report Generation (SSE Streaming)

```
Browser                      FastAPI                     OpenAI
  │                             │                           │
  │── GET /generate-report ────►│                           │
  │                             │── chat.completions ──────►│
  │                             │   model: gpt-4o-mini      │
  │                             │   stream: True            │
  │◄── data: {"text": "..." } ──│◄── chunk ─────────────────│
  │◄── data: {"text": "..." } ──│◄── chunk ─────────────────│
  │   (renders in real-time)    │                           │
  │◄── data: {"done": true} ────│                           │
  │                             │                           │
  │  [PDF button appears]       │                           │
```

### 2.4 Session Modes

| Mode | Condition | Results Page Behaviour |
|------|-----------|------------------------|
| **AI Report Mode** | `OPENAI_API_KEY` set | AI generates full campaign report on page load; static report hidden |
| **Standard Mode** | No API key | Static hardcoded campaign report shown; AI card shows disabled state |

---

## 3. Tech Stack

### 3.1 Backend

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Web Framework | **FastAPI** (Python) | REST API + SSE streaming endpoints |
| ASGI Server | **Uvicorn** | Production-grade async server |
| Templating | **Jinja2** | Server-side HTML rendering |
| AI Classification | **CLIP** (`clip-vit-base-patch32`) | Zero-shot waste type classification from images |
| ML Runtime | **PyTorch** + **TorchVision** | CLIP model inference |
| Computer Vision | **OpenCV** (`opencv-python`) | Haar cascade face & licence plate detection + Gaussian blur |
| Route Optimisation | **Haversine + Nearest-Neighbour TSP** | Truck route assignment across Accra zones |
| Data Processing | **NumPy**, **Pandas**, **SciPy** | Spatial calculations, aggregation |
| AI Report Generation | **OpenAI API** (`gpt-4o-mini`) | Streaming community campaign report generation |
| Google Sheets | **gspread** + **oauth2client** | Live data logging and analytics read target |
| Config & Secrets | **python-dotenv** | `.env` file management; API key persistence |
| Reporting | **Jinja2**, **Matplotlib**, **Seaborn** | HTML report templates, hotspot heat-map snapshots |

### 3.2 Frontend

| Layer | Technology | Purpose |
|-------|-----------|---------|
| CSS Framework | **Tailwind CSS** (pre-built static) | Utility-first styling |
| Typography | **Inter** (Google Fonts) | Dashboard font |
| Maps | **Leaflet.js** v1.9.4 | Interactive Accra waste bin map (OpenStreetMap tiles) |
| Charts | **Chart.js** v4.4.0 | Donut, bar, stacked bar, dual-axis bar charts |
| Streaming | **Fetch API + ReadableStream** | SSE consumption for pipeline progress and AI report |
| PDF Export | **Browser Print Window** | Formatted print-to-PDF for both report modes |

### 3.3 Data & Exports

| Format | File | Consumer |
|--------|------|----------|
| JSON | `powerbi_bins.json` | Results dashboard map + charts |
| CSV | `powerbi_bins.csv` | Power BI / Excel download |
| CSV | `powerbi_hotspots.csv` | Zone analysis panel |
| JSON | `pipeline_manifest.json` | Route summary + pipeline timing |
| TXT | `campaign_report.txt` | Static campaign report copy |

### 3.4 Project Structure

```
iwmro/
├── app.py                    # FastAPI application (routes, SSE, AI report)
├── config.py                 # Central configuration (thresholds, paths, labels)
├── main.py                   # CLI entrypoint
├── requirements.txt          # Python dependencies
├── .env                      # Runtime secrets (gitignored)
├── creds.json                # Google service account (gitignored)
│
├── src/
│   ├── api/
│   │   └── pipeline.py       # Full pipeline orchestrator with progress callbacks
│   ├── data/
│   │   └── mock_data_generator.py   # Synthetic 50-bin Accra dataset
│   ├── privacy/
│   │   └── privacy_filter.py        # OpenCV face/plate blur
│   ├── optimizer/
│   │   └── route_optimizer.py       # Haversine TSP route optimiser
│   ├── reports/
│   │   └── report_generator.py      # Campaign report writer
│   └── integrations/
│       ├── sheets_logger.py         # Google Sheets write (pipeline results)
│       └── sheets_loader.py         # Google Sheets read (analytics)
│
├── templates/
│   ├── index.html            # Home page (pipeline launcher, API key config)
│   └── results.html          # Results dashboard (map, charts, reports)
│
├── static/
│   ├── app.css               # Custom dashboard CSS
│   ├── tailwind.css          # Pre-built Tailwind output
│   └── tailwind.src.css      # Tailwind source
│
├── data/
│   ├── raw/                  # Uploaded field images (gitignored)
│   ├── processed/            # Anonymised images (gitignored)
│   └── exports/              # Pipeline output files (gitignored)
│
├── models/                   # CLIP model cache (gitignored)
└── tests/
    └── test_pipeline.py      # Pipeline unit tests
```

---

## 4. Key API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Home page (pipeline launcher + API key config) |
| `POST` | `/run` | Run pipeline synchronously → redirect to `/results` |
| `POST` | `/run-stream` | Run pipeline with SSE progress events |
| `GET` | `/results` | Results dashboard |
| `GET` | `/generate-report` | Stream AI-generated campaign report (SSE) |
| `POST` | `/set-api-key` | Save OpenAI API key to `.env` |
| `GET` | `/download/{filename}` | Download export file |

---

## 5. Configuration Parameters (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CLIP_MODEL_ID` | `openai/clip-vit-base-patch32` | Hugging Face model |
| `WASTE_LABELS` | Plastic / Organic / Metal / Electronic | Zero-shot class labels |
| `CONFIDENCE_THRESHOLD` | `0.40` | Below this → flagged "uncertain" |
| `BLUR_STRENGTH` | `25` | Gaussian kernel size for privacy blur |
| `DEPOT_LOCATION` | Accra Central Depot (5.5502, -0.2174) | Truck dispatch origin |
| `MAX_BINS_PER_TRUCK` | `20` | Capacity constraint |
| `PRIORITY_THRESHOLD` | `0.75` | Fill level above which bin is high-priority |
| `OPTIMISER_MODE` | `nearest_neighbour` | Route algorithm (alt: `ortools` for 1000+ bins) |
| `ACCRA_LAT_RANGE` | `(5.45, 5.75)` | Geographic bounds for validation |
| `ACCRA_LNG_RANGE` | `(-0.35, 0.05)` | Geographic bounds for validation |

---

## 6. Scalability Notes

- **< 1,000 bins**: Nearest-neighbour TSP (`O(n²)`) — current default
- **≥ 1,000 bins**: Switch `OPTIMISER_MODE = "ortools"` to use Google OR-Tools VRP solver
- **AI classification**: CLIP runs on CPU by default; GPU acceleration available with CUDA-enabled PyTorch
- **Google Sheets**: Pipeline results auto-logged to `IWMRO_Data_Log`; analytics read from `IWMRO_Global_Data`

---

## 7. Setup & Running

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Configure OpenAI API key via the web UI
#    → http://localhost:8000 → AI Report Configuration panel
#    Or set manually in .env:
echo "OPENAI_API_KEY=sk-..." >> .env

# 3. (Optional) Add Google Sheets service account
#    Place creds.json in project root

# 4. Start the server
python app.py
# → http://localhost:8000

# 5. Run with auto-reload (development)
uvicorn app:app --reload --port 8000
```

---

*Document generated for IWMRO v1.0 — Accra Metropolitan Assembly Waste Management Division*
