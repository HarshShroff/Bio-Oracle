import os
import re
import numpy as np
import tifffile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Preprocessor")


def preprocess_bbbc021(raw_dir_path: str, output_dir_path: str):
    """
    Scans raw BBBC021 directory, groups single-channel Tiffs by Well/Site,
    and saves them as multi-channel OME-TIFFs (C, Y, X).
    """
    raw_path = Path(raw_dir_path)
    # The zip usually extracts into a subdirectory, so let's try to find it if raw_path is just 'data/raw'
    # But usually user passes 'data/raw/Week1_22123'

    out_path = Path(output_dir_path)
    out_path.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        logger.error(f"Raw directory not found: {raw_path}")
        return

    # regex to parse: Week1_150607_B02_s1_w1...
    # Groups: 1=Prefix, 2=Well, 3=Site, 4=Channel(w1/w2/w4)
    # Filenames look like: Week1_150607_B02_s1_w107447158-AC76-4844-8431-E6A954BD1174.tif
    # We want robust matching.
    pattern = re.compile(r"(Week1_150607)_([A-P]\d{2})_(s\d)_(w\d)")

    # Dictionary to hold groupings: { "Prefix_Well_Site": { "w1": path, "w2": path, "w4": path } }
    groups = {}

    logger.info(f"Scanning {raw_path}...")

    files = list(raw_path.glob("*.tif"))
    if not files:
        logger.warning(
            f"No .tif files found in {raw_path}. Did extraction work?")
        return

    for f in files:
        match = pattern.search(f.name)
        if match:
            prefix, well, site, channel = match.groups()
            key = f"{prefix}_{well}_{site}"

            if key not in groups:
                groups[key] = {}
            groups[key][channel] = f

    logger.info(f"Found {len(groups)} unique fields of view.")

    # Process each group
    for key, channels in groups.items():
        # Check if we have all 3 channels
        if 'w1' in channels and 'w2' in channels and 'w4' in channels:
            # Check if output already exists
            outfile = out_path / f"{key}.tif"
            if outfile.exists():
                # logger.info(f"Skipping {key}, already exists.")
                continue

            logger.info(f"Processing {key}...")

            # Load images
            # w1 = Nuclei (Blue), w2 = Tubulin (Green), w4 = Actin (Red)
            # Stack order: Channel 0=w1, Channel 1=w2, Channel 2=w4

            try:
                img_w1 = tifffile.imread(channels['w1'])
                img_w2 = tifffile.imread(channels['w2'])
                img_w4 = tifffile.imread(channels['w4'])

                # Check shapes
                if img_w1.shape != img_w2.shape or img_w1.shape != img_w4.shape:
                    logger.warning(
                        f"Shape mismatch for {key}: {img_w1.shape}, {img_w2.shape}, {img_w4.shape}")
                    continue

                # Stack: Shape (3, Y, X)
                stack = np.array([img_w1, img_w2, img_w4])

                # Save
                tifffile.imwrite(outfile, stack, photometric='minisblack')
                # logger.info(f"Saved {outfile}")
            except Exception as e:
                logger.error(f"Failed to process {key}: {e}")

        else:
            # logger.warning(f"Skipping {key}: Incomplete channels {channels.keys()}")
            pass

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    # Point to the standard location
    raw_data_loc = "data/raw/Week1_22123"
    preprocess_bbbc021(raw_data_loc, "data/processed")
