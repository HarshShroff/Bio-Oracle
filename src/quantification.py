import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_circularity(area, perimeter):
    """
    Calculates circularity: 4 * pi * Area / Perimeter^2
    1.0 is a perfect circle.
    """
    if perimeter == 0:
        return 0
    return (4 * np.pi * area) / (perimeter ** 2)


def extract_features(image: np.ndarray, masks: np.ndarray, channel_names: list = None) -> pd.DataFrame:
    """
    Extracts morphometric and intensity features from segmented cells.

    Args:
        image (np.ndarray): Multi-channel image array (C, Y, X).
        masks (np.ndarray): Label mask array (Y, X).
        channel_names (list, optional): List of channel names. Defaults to ['Ch1', 'Ch2', ...].

    Returns:
        pd.DataFrame: DataFrame containing features for each cell.
    """
    logger.info("Extracting phenotypic features...")

    if masks.max() == 0:
        logger.warning("No cells found in mask. Returning empty DataFrame.")
        return pd.DataFrame()

    # Ensure image matches mask dimensions
    # Image: (C, Y, X), Mask: (Y, X)
    if image.shape[1:] != masks.shape:
        # Try to handle potential mismatches or transposes
        if image.shape[:2] == masks.shape:
            # Image is (Y, X, C) or (Y, X)
            if image.ndim == 3:
                image = np.transpose(image, (2, 0, 1))  # to (C, Y, X)
            else:
                image = image[np.newaxis, ...]  # to (1, Y, X)

    n_channels = image.shape[0]
    if channel_names is None:
        channel_names = [f"Channel_{i+1}" for i in range(n_channels)]

    # Define properties to extract from shape (using mask only)
    shape_props = ['label', 'area', 'perimeter',
                   'solidity', 'eccentricity', 'centroid']

    # Extract shape features
    try:
        # We use the first channel for intensity-neutral shape props, or just pass the mask
        # But regionprops needs an intensity image to calculate intensity features.
        # We will do shape first, then intensity per channel.

        # 1. Shape Features
        # regionprops_table is efficient
        shape_data = regionprops_table(masks, properties=shape_props)
        df = pd.DataFrame(shape_data)

        # Calculate derived shape metrics
        df['circularity'] = df.apply(lambda row: calculate_circularity(
            row['area'], row['perimeter']), axis=1)

        # 2. Intensity Features per Channel
        intensity_props = ['mean_intensity', 'max_intensity', 'min_intensity']

        for ch_idx in range(n_channels):
            ch_name = channel_names[ch_idx] if ch_idx < len(
                channel_names) else f"Ch{ch_idx}"

            # Extract intensity features for this channel
            ch_image = image[ch_idx]
            ch_data = regionprops_table(
                masks, intensity_image=ch_image, properties=intensity_props)
            ch_df = pd.DataFrame(ch_data)

            # Rename columns to include channel name
            ch_df.columns = [f"{col}_{ch_name}" for col in ch_df.columns]

            # Concatenate to main df (they sort by label by default so indices align if no missing labels)
            # regionprops_table returns arrays indexed by found objects.
            # We assume the order is consistent (it is sorted by label).
            df = pd.concat([df, ch_df], axis=1)

        logger.info(
            f"Feature extraction complete. Extracted {len(df)} cells with {len(df.columns)} features.")
        return df

    except Exception as e:
        logger.error(f"Error during feature extraction: {e}")
        raise e
