from src.agent import run_oracle
from src.normalization import normalize_features
from src.quantification import extract_features
from src.segmentation import BioSegmentor
from src.ingestion import load_bio_image
import sys
import os
import argparse
import logging
import numpy as np
import pandas as pd
import asyncio
from pathlib import Path

# Add src to path so we can import modules if running directly
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BioOracle-Main")


def main():
    parser = argparse.ArgumentParser(description="Bio-Oracle Pipeline CLI")
    parser.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument("--output", type=str, default="output",
                        help="Directory to save results")
    parser.add_argument("--test", action="store_true",
                        help="Run a test with dummy data")
    parser.add_argument("--ask", type=str,
                        help="Ask the Bio-Oracle Agent a question about the processed data.")

    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    # --- Agent Mode ---
    if args.ask:
        data_path = out_dir / "cell_features.parquet"
        if not data_path.exists():
            print(
                f"Error: No data found at {data_path}. Run the pipeline on an image first.")
            return

        print(f"Bio-Oracle is thinking about: '{args.ask}'...")
        try:
            response = asyncio.run(run_oracle(args.ask, str(data_path)))
            print("\n>> BIO-ORACLE SAYS:")
            print(response)
        except Exception as e:
            logger.error(f"Failed to run agent: {e}")
        return

    # --- Test Mode ---
    if args.test:
        logger.info("Running test mode with synthetic data...")
        # Create dummy image: (3, 256, 256) -> RGB
        dummy_img = np.random.randint(0, 255, (3, 256, 256), dtype=np.uint8)

        # Test Segmentation
        img_for_segmentation = np.transpose(dummy_img, (1, 2, 0))  # H,W,C
        segmentor = BioSegmentor(use_gpu=True)
        masks, _ = segmentor.segment(img_for_segmentation, channels=[1, 2])

        # Test Quantification (Need non-empty masks for real test, but here we likely get 0 cells with random data)
        # Let's manually inject a "cell" for feature test
        masks[100:150, 100:150] = 1

        features = extract_features(
            dummy_img, masks, channel_names=['R', 'G', 'B'])
        if not features.empty:
            logger.info(f"Extracted features for {len(features)} cells.")
            # Test Normalization
            norm_features = normalize_features(features)
            logger.info("Normalization successful.")

            # Save dummy data for agent test
            norm_features.to_parquet(out_dir / "cell_features.parquet")

        logger.info(
            "Test complete. System seems operational. You can now use --ask 'Summarize this dataset'")
        return

    if not args.image:
        print("Please provide an image path using --image or run with --test")
        return

    # 1. Ingestion
    img_data, metadata = load_bio_image(args.image)
    if img_data is None:
        logger.error("Failed to load image.")
        return

    # 2. Segmentation
    # Assumption: img_data is (C, Y, X). Cellpose expects (Y, X, C).
    if img_data.shape[0] < 10:  # Heuristic: Channels first
        img_for_seg = np.transpose(img_data, (1, 2, 0))
    else:
        img_for_seg = img_data

    segmentor = BioSegmentor()
    # Use metadata channel info if available, otherwise default
    masks, _ = segmentor.segment(img_for_seg, channels=[2, 3])

    # 3. Scientific Quantification
    logger.info("Starting quantification...")
    channel_names = metadata.get('channel_names')
    features = extract_features(img_data, masks, channel_names=channel_names)

    # 4. Normalization
    if not features.empty:
        features_norm = normalize_features(features, method='robust_zscore')

        # 5. Save Results
        np.save(out_dir / "masks.npy", masks)
        features_norm.to_parquet(out_dir / "cell_features.parquet")
        features_norm.to_csv(out_dir / "cell_features.csv", index=False)
        logger.info(f"Results saved to {out_dir}")
        print("\nProcessing complete. You can now query the data with:")
        print(
            f"python -m src.main --ask \"Are there any outlier cells with high intensity?\"")
    else:
        logger.warning(
            "No cells detected, skipping feature extraction and saving empty artifacts.")
        np.save(out_dir / "masks.npy", masks)


if __name__ == "__main__":
    main()
