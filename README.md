# IWMRO — Integrated Waste Management and Recycling Optimizer
### Master's Capstone Project | Accra, Ghana

---

## Project Overview

IWMRO is a multi-tool AI prototype that automates the full lifecycle of urban waste monitoring for Accra, Ghana — from image classification at the bin to community campaign generation.

```
input_image ──► [Privacy Filter] ──► [Waste Classifier] ──► JSON metadata
                                                                   │
                                         ┌─────────────────────────┘
                                         ▼
                              [Route Optimizer] ──► CSV for Power BI
                                         │
                                         ▼
                              [Report Generator] ──► Canva / Campaign
```

---

## Directory Structure

```
iwmro/
├── config.py                          # Central configuration (all tunables)
├── main.py                            # Top-level entry point
├── requirements.txt
│
├── models/
│   └── waste_classifier.py            # CLIP zero-shot classifier
│
├── src/
│   ├── api/
│   │   └── pipeline.py                # Master orchestrator
│   ├── data/
│   │   └── mock_data_generator.py     # Mock Accra bin dataset
│   ├── optimizer/
│   │   └── route_optimizer.py         # Haversine TSP + hotspot detection
│   ├── privacy/
│   │   └── privacy_filter.py          # Face + licence-plate blurring
│   └── reports/
│       └── report_generator.py        # Text/HTML/JSON campaign reports
│
├── data/
│   ├── raw/          ← input images
│   ├── processed/    ← privacy-filtered images
│   └── exports/      ← all pipeline outputs (CSV, JSON, HTML)
│
└── tests/
    └── test_pipeline.py               # pytest unit tests
```

---

## Quick Start

```bash
# 1 — Install dependencies
pip install -r requirements.txt

# 2 — Run full demo (generates mock Accra data + executes pipeline)
python main.py --demo

# 3 — Process a real image
python main.py --image path/to/bin_photo.jpg

# 4 — Generate mock dataset only
python main.py --generate-data

# 5 — Run tests
pytest tests/ -v
```

---

## Module Descriptions

### `models/waste_classifier.py`
Uses **OpenAI CLIP** (ViT-B/32) via Hugging Face `transformers` for zero-shot image classification.

| Label       | Semantic Prompt        |
|-------------|------------------------|
| plastic     | `"plastic waste"`      |
| organic     | `"organic waste"`      |
| metal       | `"metal waste"`        |

CLIP computes cosine similarity between the image embedding and each text embedding; the highest-scoring label wins.  A `confidence < 0.40` threshold flags uncertain predictions for human review.

**Scalability path**: Replace with fine-tuned MobileNetV2 head on TrashNet dataset for ≥95% accuracy at 1 000+ bins/day.

---

### `src/data/mock_data_generator.py`
Generates **50 synthetic bin records** across 9 Accra zones with:
- Realistic GPS coordinates (jittered from zone centroids)
- Zone-specific waste-type probability distributions
- Solid-colour PNG stand-in images per waste type
- Timestamps starting 06:00 on 2024-06-01

---

### `src/optimizer/route_optimizer.py`

**Haversine formula** (great-circle distance):

```
a = sin²(Δφ/2) + cos(φ₁)·cos(φ₂)·sin²(Δλ/2)
d = 2R · atan2(√a, √(1−a))
```

A road-distance inflation factor of **1.30×** accounts for Accra's road network.

**Nearest-Neighbour TSP heuristic** (O(n²)):
1. Start from Accra Central Depot (5.5502°N, 0.2174°W)
2. Repeatedly visit the closest unvisited high-priority bin
3. Partition into truck loads of ≤20 bins

**Scalability path**: Swap `_nn_route()` for Google OR-Tools VRP solver (drop-in replacement) at 1 000+ bins.

---

### `src/privacy/privacy_filter.py`
Implements **Ghana Data Protection Act 2012 (Act 843)** compliance:

1. Detect faces using `haarcascade_frontalface_default.xml`
2. Detect plates using `haarcascade_russian_plate_number.xml`
3. Apply Gaussian blur (kernel size 25) + padding (10 px) to each region
4. Save anonymised image to `data/processed/`

**Scalability path**: Replace Haar cascades with YOLOv8-face for ~98% recall; use `ProcessPoolExecutor` for multi-core batch processing.

---

### `src/reports/report_generator.py`
Produces three campaign artefacts from aggregated bin data:

| Artefact                    | Purpose                              |
|-----------------------------|--------------------------------------|
| `campaign_report.txt`       | Paste into Canva community poster    |
| `campaign_report.html`      | Power BI web visual / browser view   |
| `campaign_payload.json`     | Hubtel SMS / social media API post   |

Reports include dynamic tips keyed to the dominant waste type (plastic / organic / metal) and flag hotspot zones by name.

---

### `src/api/pipeline.py`
Orchestrates all five steps in sequence and writes a `pipeline_manifest.json` listing every output file with its path.

---

## Power BI Integration

Import `data/exports/powerbi_bins.csv` into **Power BI Desktop**:

| Column               | Type    | Use                          |
|----------------------|---------|------------------------------|
| `bin_id`             | Text    | Slicer / drill-through        |
| `latitude/longitude` | Decimal | Map visual (ArcGIS / Bing)   |
| `waste_type`         | Text    | Donut chart / stacked bar     |
| `fill_level`         | Decimal | KPI card / conditional format |
| `collection_priority`| Text    | Filter pane                   |
| `route_sequence`     | Integer | Route animation               |
| `truck_id`           | Text    | Fleet utilisation visual      |

`powerbi_hotspots.csv` drives a **zone-level choropleth** (mean fill level by neighbourhood).

---

## Scalability Architecture (1 000+ Bins)

| Component          | Current (prototype)          | At Scale                                      |
|--------------------|------------------------------|-----------------------------------------------|
| Classifier         | CLIP zero-shot, single image | MobileNetV2 fine-tuned, DataLoader batch 32   |
| Route optimiser    | Nearest-neighbour O(n²)      | OR-Tools VRP with time windows                |
| Privacy filter     | Sequential Haar cascade      | YOLOv8-face + ProcessPoolExecutor             |
| Data ingestion     | In-memory list               | Pandas chunked CSV reader / PostgreSQL        |
| Reporting          | Jinja2 templates on disk     | Celery async task queue + S3 storage          |
| Deployment         | Local Python script          | Docker container on GCP Cloud Run / Azure ACI |

---

## Ethical Considerations

- **Privacy**: Haar cascade + Gaussian blur removes PII before any AI processing.
- **Bias**: CLIP was pre-trained on internet images; performance on Accra-specific waste may differ — field validation required.
- **Data sovereignty**: All processing runs locally; no images leave the field device.
- **Transparency**: Confidence scores and `uncertain` flags allow human reviewers to override AI decisions.

---

## License
MIT — for academic and non-commercial use.
