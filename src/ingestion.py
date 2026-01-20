import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from aicsimageio import AICSImage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_bio_image(file_path: str) -> Tuple[Optional[np.ndarray], dict]:
    """
    Loads a microscopy image (CZI, TIFF, etc.) and returns the image data and metadata.

    Args:
        file_path (str): Absolute path to the image file.

    Returns:
        Tuple[Optional[np.ndarray], dict]: 
            - Image data as a numpy array. 
              Expected shape for standard processing: (Channels, Height, Width) or (Z, C, H, W).
              Returns None if loading fails.
            - Metadata dictionary.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None, {}

    try:
        logger.info(f"Loading image from {file_path}")
        img = AICSImage(file_path)

        # AICSImageIO standardizes dimensions to "TCZYX"
        # We generally want to extract the relevant spatial data.
        # For this pipeline, assuming we want the raw data primarily.

        # Let's try to get a standard (C, Y, X) or (Z, C, Y, X) format
        data = img.data  # This is usually TCZYX

        # Check dimensions
        logger.info(f"Original shape (TCZYX): {data.shape}")

        # Squeeze out T if it's 1 (Time)
        if data.shape[0] == 1:
            data = data.squeeze(0)

        # Squeeze out Z if it's 1 (Z-stack)
        if data.ndim == 4 and data.shape[0] == 1:  # (Z, C, Y, X) where Z=1
            data = data.squeeze(0)  # -> (C, Y, X)
        elif data.ndim == 4 and data.shape[1] == 1:  # (C, Z, Y, X)
            if data.shape[0] == 3:  # Likely C, Z, Y, X
                data = data.squeeze(1)  # -> (C, Y, X)

        # Metadata
        metadata = {
            "shape": data.shape,
            "dims": img.dims,
            "channel_names": img.channel_names,
            "physical_pixel_sizes": img.physical_pixel_sizes
        }

        logger.info(f"Loaded image shape: {data.shape}")
        return data, metadata

    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return None, {}


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizes image intensity to 0-1 range per channel.
    Useful for visualization or certain pre-processing steps.
    """
    if image is None:
        return None

    image = image.astype(np.float32)

    # Normalize per channel
    # Assuming standard Layout (..., C, Y, X) or (C, Y, X)
    # This is a naive normalization (min-max).
    # Scientific rigor might require more specific normalization (e.g. to control).

    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val > 0:
        return (image - min_val) / (max_val - min_val)
    return image
