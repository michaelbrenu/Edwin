"""
main.py — IWMRO top-level entry point.

Quick start
-----------
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate mock dataset + run full pipeline (no real images needed)
python main.py --demo

# 3. Run pipeline on a real image
python main.py --image path/to/bin_photo.jpg

# 4. Generate mock dataset only
python main.py --generate-data

# 5. Run privacy filter only
python main.py --filter-only
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("iwmro.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("iwmro")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IWMRO — Integrated Waste Management and Recycling Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--demo",          action="store_true",
                        help="Run full pipeline using mock Accra dataset.")
    parser.add_argument("--generate-data", action="store_true",
                        help="Generate mock dataset only (no classifier).")
    parser.add_argument("--filter-only",   action="store_true",
                        help="Apply privacy filter to data/raw/ images only.")
    parser.add_argument("--image",         type=Path,
                        help="Path to a single image for end-to-end processing.")
    parser.add_argument("--images-dir",    type=Path,
                        help="Directory of images for batch processing.")
    parser.add_argument("--skip-classifier", action="store_true",
                        help="Use existing waste labels; skip CLIP inference.")
    args = parser.parse_args()

    if args.generate_data:
        from src.data.mock_data_generator import generate_mock_dataset
        bins = generate_mock_dataset()
        print(f"\n[OK] {len(bins)} bin records generated.")
        return

    if args.filter_only:
        from config import DIRS
        from src.privacy.privacy_filter import filter_batch
        raw_images = list(DIRS["raw"].glob("*.png")) + list(DIRS["raw"].glob("*.jpg"))
        if not raw_images:
            print("No images in data/raw/. Run --generate-data first.")
            return
        results = filter_batch(raw_images)
        blurred = sum(1 for _, r in results if r.get("anonymised"))
        print(f"[OK] {blurred}/{len(results)} images anonymised -> data/processed/")
        return

    images = None
    if args.image:
        images = [args.image]
    elif args.images_dir:
        images = (
            list(args.images_dir.glob("*.jpg")) +
            list(args.images_dir.glob("*.png"))
        )
        if not images:
            print(f"No JPEG/PNG images found in {args.images_dir}")
            return

    # Default: demo mode if no specific flag given
    if not args.demo and not images:
        print("No mode specified. Running --demo …\n")
        args.demo = True

    from src.api.pipeline import run_pipeline
    artefacts = run_pipeline(
        image_paths=images,
        use_mock=(images is None),
        skip_classifier=args.skip_classifier,
    )

    print("\n" + "=" * 58)
    print("  IWMRO Pipeline - Output Artefacts")
    print("=" * 58)
    for name, path in artefacts.items():
        print(f"  {name:22s}  {path}")
    print("=" * 58)
    print("\nOpen data/exports/campaign_report.html in a browser for")
    print("the formatted community report, or import powerbi_bins.csv")
    print("into Power BI Desktop for interactive visualisation.\n")


if __name__ == "__main__":
    main()
