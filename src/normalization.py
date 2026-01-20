import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def robust_zscore(series: pd.Series) -> pd.Series:
    """
    Calculates Robust Z-score: (x - median) / (MAD * 1.4826)
    MAD = median(|x - median|)
    """
    median = series.median()
    mad = (series - median).abs().median()

    if mad == 0:
        return series - median  # Fallback to centering if no variation

    return (series - median) / (mad * 1.4826)


def normalize_features(df: pd.DataFrame, method: str = 'robust_zscore', exclude_cols: list = None) -> pd.DataFrame:
    """
    Normalizes feature columns in the DataFrame.

    Args:
        df (pd.DataFrame): Input features.
        method (str): 'zscore' or 'robust_zscore'.
        exclude_cols (list, optional): Columns to exclude (e.g., label, centroid). 
                                     If None, defaults to identifying typical metadata.

    Returns:
        pd.DataFrame: DataFrame with normalized columns (preserves original as well? No, usually replaces or adds suffix).
                      Let's add suffix for clarity: _norm
    """
    if df.empty:
        return df

    logger.info(f"Normalizing features using {method}...")

    if exclude_cols is None:
        exclude_cols = ['label', 'centroid-0', 'centroid-1',
                        'centroid-2', 'file_name', 'well_id']

    # Identify numeric columns that are not excluded
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    target_cols = [c for c in numeric_cols if c not in exclude_cols]

    result_df = df.copy()

    for col in target_cols:
        if method == 'exact_zscore':
            mean = df[col].mean()
            std = df[col].std()
            if std != 0:
                result_df[f"{col}_norm"] = (df[col] - mean) / std
            else:
                result_df[f"{col}_norm"] = 0
        elif method == 'robust_zscore':
            result_df[f"{col}_norm"] = robust_zscore(df[col])
        else:
            logger.warning(
                f"Unknown normalization method: {method}. Skipping.")

    logger.info("Normalization complete.")
    return result_df
