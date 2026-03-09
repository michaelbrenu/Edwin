"""
tests/test_pipeline.py
=======================
Unit tests for IWMRO core modules.

Run with:
    pytest tests/ -v

No GPU or internet connection required — all AI calls are mocked.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_image_path(tmp_path) -> Path:
    """Create a tiny 64×64 synthetic RGB PNG for testing."""
    img  = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8), "RGB")
    path = tmp_path / "test_bin.png"
    img.save(path)
    return path


@pytest.fixture
def sample_bins() -> list[dict]:
    return [
        {
            "bin_id": "BIN-0001", "location_name": "Makola Market",
            "latitude": 5.5474, "longitude": -0.2044,
            "waste_type": "plastic", "confidence": 0.82,
            "fill_level": 0.90, "collection_priority": "high",
            "route_sequence": 0, "truck_id": "",
            "timestamp": "2024-06-01T06:00:00",
            "image_path": "",
        },
        {
            "bin_id": "BIN-0002", "location_name": "East Legon",
            "latitude": 5.6037, "longitude": -0.1614,
            "waste_type": "organic", "confidence": 0.75,
            "fill_level": 0.55, "collection_priority": "normal",
            "route_sequence": 0, "truck_id": "",
            "timestamp": "2024-06-01T06:03:00",
            "image_path": "",
        },
        {
            "bin_id": "BIN-0003", "location_name": "Tema Industrial",
            "latitude": 5.6698, "longitude": -0.0166,
            "waste_type": "metal", "confidence": 0.91,
            "fill_level": 0.80, "collection_priority": "high",
            "route_sequence": 0, "truck_id": "",
            "timestamp": "2024-06-01T06:06:00",
            "image_path": "",
        },
    ]


# ── Haversine tests ───────────────────────────────────────────────────────────

class TestHaversine:
    def test_same_point_is_zero(self):
        from src.optimizer.route_optimizer import haversine
        assert haversine(5.55, -0.20, 5.55, -0.20) == 0.0

    def test_known_distance_accra_tema(self):
        """Accra Central ↔ Tema is ~25 km by road; Haversine ≈ 19 km."""
        from src.optimizer.route_optimizer import haversine
        dist = haversine(5.5502, -0.2174, 5.6698, -0.0166)
        assert 15.0 < dist < 25.0, f"Unexpected distance: {dist}"

    def test_symmetry(self):
        from src.optimizer.route_optimizer import haversine
        d1 = haversine(5.55, -0.20, 5.60, -0.16)
        d2 = haversine(5.60, -0.16, 5.55, -0.20)
        assert abs(d1 - d2) < 1e-6

    def test_positive(self):
        from src.optimizer.route_optimizer import haversine
        assert haversine(5.54, -0.20, 5.67, -0.01) > 0


# ── Hotspot detection tests ───────────────────────────────────────────────────

class TestHotspots:
    def test_high_fill_zone_is_hotspot(self, sample_bins):
        from src.optimizer.route_optimizer import identify_hotspots
        # Makola has fill 0.90 — should be a hotspot
        hotspots = identify_hotspots(sample_bins)
        assert hotspots["Makola Market"]["is_hotspot"] is True

    def test_normal_fill_zone_not_hotspot(self, sample_bins):
        from src.optimizer.route_optimizer import identify_hotspots
        hotspots = identify_hotspots(sample_bins)
        assert hotspots["East Legon"]["is_hotspot"] is False

    def test_dominant_type_correct(self, sample_bins):
        from src.optimizer.route_optimizer import identify_hotspots
        hotspots = identify_hotspots(sample_bins)
        assert hotspots["Makola Market"]["dominant_type"] == "plastic"

    def test_returns_all_zones(self, sample_bins):
        from src.optimizer.route_optimizer import identify_hotspots
        hotspots = identify_hotspots(sample_bins)
        zones = {b["location_name"] for b in sample_bins}
        assert set(hotspots.keys()) == zones


# ── Route optimiser tests ─────────────────────────────────────────────────────

class TestRouteOptimiser:
    def test_all_bins_present_in_output(self, sample_bins):
        from src.optimizer.route_optimizer import optimise_routes
        annotated, _ = optimise_routes(sample_bins)
        assert len(annotated) == len(sample_bins)

    def test_route_sequences_assigned(self, sample_bins):
        from src.optimizer.route_optimizer import optimise_routes
        annotated, _ = optimise_routes(sample_bins)
        sequences = [b["route_sequence"] for b in annotated]
        assert all(s > 0 for s in sequences)

    def test_truck_ids_assigned(self, sample_bins):
        from src.optimizer.route_optimizer import optimise_routes
        annotated, _ = optimise_routes(sample_bins)
        assert all(b["truck_id"].startswith("TRUCK-") for b in annotated)

    def test_route_summary_has_distance(self, sample_bins):
        from src.optimizer.route_optimizer import optimise_routes
        _, summary = optimise_routes(sample_bins)
        assert all(r["total_km"] > 0 for r in summary)

    def test_high_priority_bins_appear_first(self, sample_bins):
        """High-priority bins should appear in route_sequence 1 or 2."""
        from src.optimizer.route_optimizer import optimise_routes
        annotated, _ = optimise_routes(sample_bins)
        sorted_annotated = sorted(annotated, key=lambda b: b["route_sequence"])
        first = sorted_annotated[0]
        assert first["collection_priority"] == "high"


# ── Privacy filter tests ──────────────────────────────────────────────────────

class TestPrivacyFilter:
    def test_output_file_created(self, sample_image_path, tmp_path):
        from src.privacy.privacy_filter import filter_image
        out_path, report = filter_image(sample_image_path, output_dir=tmp_path)
        assert out_path.exists()

    def test_report_has_required_keys(self, sample_image_path, tmp_path):
        from src.privacy.privacy_filter import filter_image
        _, report = filter_image(sample_image_path, output_dir=tmp_path)
        for key in ("source", "processed", "faces_blurred", "plates_blurred"):
            assert key in report, f"Missing key: {key}"

    def test_blur_regions_does_not_crash_on_empty(self):
        from src.privacy.privacy_filter import _blur_regions
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = _blur_regions(img, [])
        assert result.shape == img.shape

    def test_blur_modifies_region(self):
        from src.privacy.privacy_filter import _blur_regions
        img = np.ones((100, 100, 3), dtype=np.uint8) * 200
        img[20:50, 20:50] = 0   # dark region
        result = _blur_regions(img, [(20, 20, 30, 30)])
        # Blurred region should not be identical to original zeroed region
        original_roi = img[20:50, 20:50]
        blurred_roi  = result[20:50, 20:50]
        assert not np.array_equal(original_roi, blurred_roi)

    def test_nonexistent_image_raises(self, tmp_path):
        from src.privacy.privacy_filter import filter_image
        with pytest.raises(ValueError):
            filter_image(tmp_path / "ghost.png", output_dir=tmp_path)


# ── Mock data generator tests ─────────────────────────────────────────────────

class TestMockDataGenerator:
    def test_generates_expected_count(self, tmp_path, monkeypatch):
        """The generator should create one record per bin across all zones."""
        import src.data.mock_data_generator as mdg

        # Redirect DIRS to tmp_path
        monkeypatch.setattr(mdg, "DIRS", {
            "raw":     tmp_path / "raw",
            "exports": tmp_path / "exports",
        })
        (tmp_path / "raw").mkdir()
        (tmp_path / "exports").mkdir()

        records = mdg.generate_mock_dataset()
        total_expected = sum(z["bins"] for z in mdg.ACCRA_ZONES)
        assert len(records) == total_expected

    def test_records_have_required_fields(self, tmp_path, monkeypatch):
        import src.data.mock_data_generator as mdg
        monkeypatch.setattr(mdg, "DIRS", {
            "raw":     tmp_path / "raw",
            "exports": tmp_path / "exports",
        })
        (tmp_path / "raw").mkdir()
        (tmp_path / "exports").mkdir()

        records = mdg.generate_mock_dataset()
        required = {"bin_id", "location_name", "latitude", "longitude",
                    "waste_type", "fill_level", "collection_priority"}
        for r in records:
            assert required.issubset(r.keys())

    def test_coordinates_within_accra_bounds(self, tmp_path, monkeypatch):
        import src.data.mock_data_generator as mdg
        from config import ACCRA_LAT_RANGE, ACCRA_LNG_RANGE
        monkeypatch.setattr(mdg, "DIRS", {
            "raw":     tmp_path / "raw",
            "exports": tmp_path / "exports",
        })
        (tmp_path / "raw").mkdir()
        (tmp_path / "exports").mkdir()

        records = mdg.generate_mock_dataset()
        for r in records:
            assert ACCRA_LAT_RANGE[0] <= r["latitude"]  <= ACCRA_LAT_RANGE[1], r
            assert ACCRA_LNG_RANGE[0] <= r["longitude"] <= ACCRA_LNG_RANGE[1], r


# ── Classifier tests (mocked) ──────────────────────────────────────────────────

class TestWasteClassifier:
    @patch("models.waste_classifier.CLIPModel.from_pretrained")
    @patch("models.waste_classifier.CLIPProcessor.from_pretrained")
    def test_classify_returns_expected_keys(self, mock_proc, mock_model, sample_image_path):
        """Mock CLIP to avoid downloading weights in CI."""
        import torch
        from models.waste_classifier import WasteClassifier

        # Mock processor
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_proc.return_value.return_value = mock_inputs

        # Mock model output
        fake_logits = torch.tensor([[0.6, 0.3, 0.1]])
        mock_out    = MagicMock()
        mock_out.logits_per_image = fake_logits
        mock_model.return_value.return_value = mock_out
        mock_model.return_value.to.return_value = mock_model.return_value

        clf    = WasteClassifier()
        result = clf.classify(sample_image_path)

        assert "label"      in result
        assert "confidence" in result
        assert "all_scores" in result
        assert "uncertain"  in result
        assert result["label"] in ("plastic", "organic", "metal")
