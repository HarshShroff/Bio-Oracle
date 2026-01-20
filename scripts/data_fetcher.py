import os
import requests
import zipfile
import logging
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataFetcher")

BBBC021_METADATA_URL = "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_image.csv"
BBBC021_WEEK1_ZIP_URL = "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22123.zip"


def download_file(url: str, dest_path: Path):
    if dest_path.exists():
        logger.info(f"File already exists: {dest_path}")
        return

    logger.info(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as f, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
        logger.info("Download complete.")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        # Clean up partial file
        if dest_path.exists():
            dest_path.unlink()


def main():
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download Metadata
    metadata_path = data_dir / "BBBC021_metadata.csv"
    download_file(BBBC021_METADATA_URL, metadata_path)

    # 2. Download Week 1 Zip
    zip_path = data_dir / "Week1_22123.zip"
    download_file(BBBC021_WEEK1_ZIP_URL, zip_path)

    # 3. Unzip if needed
    extract_to = data_dir
    # Check if directory already exists (heuristic)
    week1_dir = data_dir / "Week1_22123"

    if zip_path.exists() and not week1_dir.exists():
        logger.info(f"Extracting {zip_path}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            logger.info(f"Extracted to {extract_to}")

            # Clean up zip to save space? Optional. Let's keep it for now.
            # zip_path.unlink()
        except zipfile.BadZipFile:
            logger.error("Error: The downloaded file is not a valid zip file.")


if __name__ == "__main__":
    main()
