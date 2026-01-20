import os
import logging
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data Structures & Dependencies ---


class OracleDeps:
    """Dependencies injected into the agent at runtime."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self._df = None

    @property
    def df(self) -> pd.DataFrame:
        """Lazy load the dataframe."""
        if self._df is None:
            if os.path.exists(self.data_path):
                try:
                    self._df = pd.read_parquet(self.data_path)
                    logger.info(
                        f"Loaded data from {self.data_path}: {self._df.shape}")
                except Exception as e:
                    logger.error(f"Failed to load data: {e}")
                    self._df = pd.DataFrame()
            else:
                logger.warning(f"Data file not found at {self.data_path}")
                self._df = pd.DataFrame()
        return self._df

# --- Agent Definition ---


# We use Gemini 2.5 Pro for speed/cost effectiveness in this loop
model = GeminiModel('gemini-2.5-pro')

bio_oracle = Agent(
    model,
    deps_type=OracleDeps,
    system_prompt=(
        "You are Bio-Oracle, an expert Translational Scientist specializing in High-Content Screening (HCS). "
        "Your goal is to interpret cellular phenotypic data to identify drug effects, toxicity, or biological states. "
        "You have access to a dataset of single-cell measurements (morphology and intensity). "
        "Always rely on the tools provided to query the data. Do not hallucinate values. "
        "When analyzing, consider 'Z-scored' features: values > 2 or < -2 are significantly different from the population mean. "
        "Use scientific terminology (e.g., 'nuclear fragmentation', 'hypertrophy', 'cytotoxicity')."
    )
)

# --- Tools ---


@bio_oracle.tool
def get_dataset_summary(ctx: RunContext[OracleDeps]) -> str:
    """
    Returns a high-level summary of the loaded dataset, including column names and cell count.
    Use this to understand what features are available.
    """
    df = ctx.deps.df
    if df.empty:
        return "The dataset is empty or could not be loaded."

    summary = f"Dataset contains {len(df)} single cells.\n"
    summary += f"Available Features: {', '.join(df.columns.tolist())}\n"
    return summary


@bio_oracle.tool
def get_feature_stats(ctx: RunContext[OracleDeps], feature_name: str) -> str:
    """
    Calculates statistics (Mean, Median, Std, Min, Max) for a specific feature.
    Useful for establishing baseline or detecting shifts.

    Args:
        feature_name: The exact name of the column to analyze (e.g., 'area', 'circularity_norm').
    """
    df = ctx.deps.df
    if df.empty:
        return "No data available."

    if feature_name not in df.columns:
        # fuzzy match could be added here, but strict for now
        return f"Feature '{feature_name}' not found. Available: {list(df.columns)}"

    stats = df[feature_name].describe()
    return stats.to_string()


@bio_oracle.tool
def identify_outliers(ctx: RunContext[OracleDeps], feature_name: str, threshold: float = 3.0) -> str:
    """
    Identifies cells that are outliers for a given feature (absolute value > threshold).
    Returns the count and the mean value of these outliers.
    """
    df = ctx.deps.df
    if df.empty:
        return "No data available."

    if feature_name not in df.columns:
        return f"Feature '{feature_name}' not found."

    # Assuming features might be normalized, but we handle raw too
    # We look for value > threshold or < -threshold
    outliers = df[df[feature_name].abs() > threshold]

    return (
        f"Found {len(outliers)} outliers for '{feature_name}' (threshold +/- {threshold}). "
        f"Mean value of outliers: {outliers[feature_name].mean():.2f}"
    )


async def run_oracle(query: str, data_path: str):
    """
    Entry point to run the agent.
    """
    if not os.getenv('GEMINI_API_KEY'):
        logger.error(
            "GEMINI_API_KEY is not set. Please set it in your environment.")
        return "Error: GEMINI_API_KEY not set."

    logger.info(f"Bio-Oracle received query: {query}")
    deps = OracleDeps(data_path)

    try:
        result = await bio_oracle.run(query, deps=deps)
        # hasattr check to be safe
        if hasattr(result, 'data'):
            return result.data
        else:
            return str(result)
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        return f"Agent encountered an error: {e}"
