"""
models/waste_classifier.py
==========================
Waste image classifier using OpenAI CLIP (zero-shot) via Hugging Face.

Architecture
------------
Input image  →  CLIP ViT-B/32 image encoder  →  cosine similarity
                                         against CLIP text encodings
                                         of ["plastic waste",
                                             "organic waste",
                                             "metal waste"]
             →  softmax probabilities  →  top-1 label + confidence

Why CLIP?
---------
CLIP requires **no fine-tuning data**; it classifies by semantic
similarity between image and text, making it ideal for a research
prototype where labelled waste images are scarce.

Scalability note
----------------
For production with 1 000+ bins, replace the zero-shot approach with
a fine-tuned MobileNetV2 head trained on labelled waste images
(e.g., TrashNet dataset).  The `WasteClassifier` interface stays
identical — only the backend changes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLIP_MODEL_ID, WASTE_LABELS, CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


class WasteClassifier:
    """
    Zero-shot waste image classifier backed by CLIP.

    Parameters
    ----------
    model_id : str
        Hugging Face model card identifier (default from config.py).
    device : str | None
        'cuda', 'cpu', or None (auto-detect).

    Example
    -------
    >>> clf = WasteClassifier()
    >>> result = clf.classify("data/raw/sample.jpg")
    >>> print(result)
    {
        "label": "plastic",
        "raw_label": "plastic waste",
        "confidence": 0.82,
        "all_scores": {"plastic": 0.82, "organic": 0.11, "metal": 0.07},
        "uncertain": False
    }
    """

    def __init__(
        self,
        model_id: str = CLIP_MODEL_ID,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading CLIP model '%s' on %s …", model_id, self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model     = CLIPModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.labels    = WASTE_LABELS
        logger.info("CLIP model ready.  Labels: %s", self.labels)

    # ------------------------------------------------------------------ #
    def classify(self, image_path: str | Path) -> dict:
        """
        Classify a single image and return structured metadata.

        Parameters
        ----------
        image_path : str | Path
            Path to the input image (JPEG / PNG / WEBP …).

        Returns
        -------
        dict with keys: label, raw_label, confidence, all_scores, uncertain
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        return self._run_inference(image, str(image_path))

    # ------------------------------------------------------------------ #
    def classify_pil(self, image: Image.Image, source_name: str = "pil_image") -> dict:
        """Classify a PIL image directly (used by the pipeline after privacy filtering)."""
        return self._run_inference(image.convert("RGB"), source_name)

    # ------------------------------------------------------------------ #
    def _run_inference(self, image: Image.Image, name: str) -> dict:
        inputs = self.processor(
            text=self.labels,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits  = outputs.logits_per_image          # shape [1, num_labels]
            probs   = logits.softmax(dim=1).squeeze()   # shape [num_labels]

        scores = {
            label.split()[0]: round(float(p), 4)
            for label, p in zip(self.labels, probs)
        }
        top_label    = max(scores, key=scores.get)
        top_conf     = scores[top_label]
        raw_label    = next(l for l in self.labels if l.startswith(top_label))

        result = {
            "source":      name,
            "label":       top_label,
            "raw_label":   raw_label,
            "confidence":  top_conf,
            "all_scores":  scores,
            "uncertain":   top_conf < CONFIDENCE_THRESHOLD,
        }
        logger.debug("Classified '%s' → %s (%.1f%%)", name, top_label, top_conf * 100)
        return result

    # ------------------------------------------------------------------ #
    def batch_classify(self, image_paths: list[str | Path]) -> list[dict]:
        """
        Classify multiple images and return a list of result dicts.

        Scalability: For 1 000+ bins this method can be extended with
        DataLoader batching (batch_size=32) to saturate GPU throughput.
        """
        results = []
        for path in image_paths:
            try:
                results.append(self.classify(path))
            except Exception as exc:
                logger.warning("Skipping %s — %s", path, exc)
                results.append({"source": str(path), "label": "error", "confidence": 0.0})
        return results
