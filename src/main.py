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


def process_single_image(image_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingestion
    img_data, metadata = load_bio_image(image_path)
    if img_data is None:
        logger.error(f"Failed to load image: {image_path}")
        return False

    # 2. Segmentation
    if img_data.shape[0] < 10:  # Heuristic: Channels first
        img_for_seg = np.transpose(img_data, (1, 2, 0))
    else:
        img_for_seg = img_data

    segmentor = BioSegmentor()
    masks, _ = segmentor.segment(img_for_seg, channels=[2, 3])

    # 3. Scientific Quantification
    logger.info(f"Starting quantification for {image_path}...")
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
        return True
    else:
        logger.warning(
            f"No cells detected in {image_path}, skipping feature extraction.")
        np.save(out_dir / "masks.npy", masks)
        return False


def main():
    parser = argparse.ArgumentParser(description="Bio-Oracle Pipeline CLI")
    parser.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument("--output", type=str, default="output",
                        help="Directory to save results")
    parser.add_argument("--test", action="store_true",
                        help="Run a test with dummy data")
    parser.add_argument("--ask", type=str,
                        help="Ask the Bio-Oracle Agent a question about the processed data.")
    parser.add_argument("--batch-process", type=str,
                        help="Path to a directory for batch processing all images within.")

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

    # --- Batch Process Mode ---
    if args.batch_process:
        input_dir = Path(args.batch_process)
        if not input_dir.exists() or not input_dir.is_dir():
            logger.error(f"Batch directory not found: {input_dir}")
            return

        image_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.czi'}
        images = sorted([f for f in input_dir.iterdir()
                        if f.suffix.lower() in image_extensions])

        if not images:
            logger.warning(f"No images found in {input_dir}")
            return

        logger.info(f"Starting batch process for {len(images)} images...")
        for img_path in images:
            logger.info(f"Processing: {img_path.name}")
            process_single_image(str(img_path), out_dir / img_path.stem)
        return

    # --- Test Mode ---
    if args.test:
        logger.info("Running test mode with synthetic data...")
        dummy_img = np.random.randint(0, 255, (3, 256, 256), dtype=np.uint8)
        img_for_segmentation = np.transpose(dummy_img, (1, 2, 0))  # H,W,C
        segmentor = BioSegmentor(use_gpu=True)
        masks, _ = segmentor.segment(img_for_segmentation, channels=[1, 2])
        masks[100:150, 100:150] = 1

        features = extract_features(
            dummy_img, masks, channel_names=['R', 'G', 'B'])
        if not features.empty:
            logger.info(f"Extracted features for {len(features)} cells.")
            norm_features = normalize_features(features)
            logger.info("Normalization successful.")
            norm_features.to_parquet(out_dir / "cell_features.parquet")

        logger.info(
            "Test complete. System seems operational. You can now use --ask 'Summarize this dataset'")
        return

    if args.image:
        success = process_single_image(args.image, out_dir)
        if success:
            print("\nProcessing complete. You can now query the data with:")
            print(
                f"python -m src.main --ask \"Are there any outlier cells with high intensity?\"")
    else:
        print("Please provide an image path using --image, --batch-process, or run with --test")


if __name__ == "__main__":
    main()
