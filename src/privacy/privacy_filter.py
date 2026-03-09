"""
src/privacy/privacy_filter.py
==============================
Privacy-preserving pre-processor for waste bin images.

Purpose
-------
Before a field photo is fed into the classifier, this module:
  1. Detects human faces using Haar-cascade classifiers (OpenCV).
  2. Detects vehicle licence plates using a secondary cascade.
  3. Applies Gaussian blur to each detected region.
  4. Saves the anonymised image to data/processed/.

Ethical rationale (for capstone report)
-----------------------------------------
Waste bin cameras in urban Accra inevitably capture passers-by and
parked vehicles.  Processing raw images without anonymisation would
constitute a GDPR-adjacent data-protection violation under Ghana's
Data Protection Act 2012 (Act 843).  This filter operationalises the
principle of *data minimisation* — only the waste content of the image
is relevant; face and plate identifiers are noise to be removed.

Limitations & production hardening
------------------------------------
•  Haar cascades trade recall for speed; a YOLOv8-face model achieves
   ~98% recall vs ~85% for cascades.  Upgrade for production.
•  Plate detection uses a generic cascade; a Ghana-specific model
   trained on GH-XX-XXXX plates would improve accuracy.
•  Side-face and occluded faces are not detected — combine with a
   body-silhouette cascade for higher coverage.

Scalability note
----------------
For 1 000+ images/hour, run this module as a multiprocessing pool
(concurrent.futures.ProcessPoolExecutor) — each worker is CPU-bound
and OpenCV releases the GIL during image operations.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    BLUR_STRENGTH,
    CASCADE_FACE,
    CASCADE_PLATE,
    PRIVACY_PADDING,
    DIRS,
)

logger = logging.getLogger(__name__)


# ── Cascade loader ─────────────────────────────────────────────────────────────
def _load_cascade(name: str) -> Optional[cv2.CascadeClassifier]:
    """
    Load an OpenCV Haar cascade by filename.
    Returns None if the cascade data file is not found (graceful degradation).
    """
    path = cv2.data.haarcascades + name
    if not Path(path).exists():
        logger.warning("Cascade '%s' not found — skipping detection.", name)
        return None
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        logger.warning("Failed to load cascade '%s'.", name)
        return None
    return cascade


# Lazy-loaded cascades (module-level, initialised once)
_FACE_CASCADE:  Optional[cv2.CascadeClassifier] = None
_PLATE_CASCADE: Optional[cv2.CascadeClassifier] = None


def _get_cascades() -> tuple:
    global _FACE_CASCADE, _PLATE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE  = _load_cascade(CASCADE_FACE)
    if _PLATE_CASCADE is None:
        _PLATE_CASCADE = _load_cascade(CASCADE_PLATE)
    return _FACE_CASCADE, _PLATE_CASCADE


# ── Region detection ───────────────────────────────────────────────────────────
def _detect_regions(gray: np.ndarray, cascade: cv2.CascadeClassifier) -> list[tuple]:
    """
    Run cascade detector on a greyscale frame.

    Returns list of (x, y, w, h) bounding boxes with padding applied.
    """
    detections = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if not isinstance(detections, np.ndarray) or len(detections) == 0:
        return []

    padded = []
    h_img, w_img = gray.shape
    for (x, y, w, h) in detections:
        x1 = max(0, x - PRIVACY_PADDING)
        y1 = max(0, y - PRIVACY_PADDING)
        x2 = min(w_img, x + w + PRIVACY_PADDING)
        y2 = min(h_img, y + h + PRIVACY_PADDING)
        padded.append((x1, y1, x2 - x1, y2 - y1))
    return padded


# ── Blur application ──────────────────────────────────────────────────────────
def _blur_regions(img_bgr: np.ndarray, regions: list[tuple]) -> np.ndarray:
    """
    Apply Gaussian blur to each bounding-box region in-place.

    The blur kernel must be odd; if BLUR_STRENGTH is even, increment by 1.
    """
    k = BLUR_STRENGTH if BLUR_STRENGTH % 2 == 1 else BLUR_STRENGTH + 1
    result = img_bgr.copy()
    for (x, y, w, h) in regions:
        roi = result[y: y + h, x: x + w]
        result[y: y + h, x: x + w] = cv2.GaussianBlur(roi, (k, k), 0)
    return result


# ── Public API ────────────────────────────────────────────────────────────────
def filter_image(
    input_path: str | Path,
    output_dir: Optional[Path] = None,
) -> tuple[Path, dict]:
    """
    Anonymise a single image: detect + blur faces and licence plates.

    Parameters
    ----------
    input_path  : path to the raw image file
    output_dir  : directory to save the processed image
                  (defaults to data/processed/)

    Returns
    -------
    (processed_path, privacy_report)
        processed_path : Path to the anonymised image
        privacy_report : dict with counts of detections
    """
    input_path = Path(input_path)
    output_dir = output_dir or DIRS["processed"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load image ────────────────────────────────────────────────────────
    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {input_path}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    face_cascade, plate_cascade = _get_cascades()

    # ── Detect ────────────────────────────────────────────────────────────
    face_regions  = _detect_regions(gray, face_cascade)  if face_cascade  else []
    plate_regions = _detect_regions(gray, plate_cascade) if plate_cascade else []
    all_regions   = face_regions + plate_regions

    # ── Blur ─────────────────────────────────────────────────────────────
    anonymised = _blur_regions(img_bgr, all_regions) if all_regions else img_bgr

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = output_dir / input_path.name
    cv2.imwrite(str(out_path), anonymised)

    privacy_report = {
        "source":          str(input_path),
        "processed":       str(out_path),
        "faces_blurred":   len(face_regions),
        "plates_blurred":  len(plate_regions),
        "regions_total":   len(all_regions),
        "anonymised":      bool(all_regions),
    }

    if all_regions:
        logger.info(
            "Blurred %d face(s) + %d plate(s) in '%s'",
            len(face_regions), len(plate_regions), input_path.name,
        )
    else:
        logger.debug("No PII detected in '%s' — copied as-is.", input_path.name)
        # Still copy to processed/ so downstream pipeline has a consistent path
        if not out_path.exists():
            shutil.copy2(input_path, out_path)

    return out_path, privacy_report


def filter_batch(
    input_paths: list[str | Path],
    output_dir: Optional[Path] = None,
) -> list[tuple[Path, dict]]:
    """
    Apply privacy filter to a list of images.

    Returns list of (processed_path, privacy_report) tuples.

    Scalability: swap the for-loop with ProcessPoolExecutor for
    multi-core parallelism on field servers with 1 000+ images.
    """
    results = []
    for path in input_paths:
        try:
            results.append(filter_image(path, output_dir))
        except Exception as exc:
            logger.error("Privacy filter failed for %s: %s", path, exc)
            results.append((Path(path), {"error": str(exc)}))
    return results


def get_pil_image(processed_path: Path) -> Image.Image:
    """Helper: load a processed image as a PIL Image for the classifier."""
    return Image.open(processed_path).convert("RGB")


# ── CLI smoke-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    raw_dir = DIRS["raw"]
    images  = list(raw_dir.glob("*.png")) + list(raw_dir.glob("*.jpg"))
    if not images:
        print("No images in data/raw/ — run mock_data_generator.py first.")
        sys.exit(0)

    print(f"Filtering {len(images)} images …")
    results = filter_batch(images)
    blurred = sum(1 for _, r in results if r.get("anonymised"))
    print(f"Done.  {blurred}/{len(results)} images had PII regions blurred.")
    print(f"Processed images saved to: {DIRS['processed']}")
