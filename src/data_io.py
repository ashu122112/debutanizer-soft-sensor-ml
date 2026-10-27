"""
src/data_io.py
Debutanizer Soft Sensor - Data Loading and Validation Module.

This module loads the normalized industrial debutanizer column dataset,
validates that it meets expected dimensional and numerical constraints (2394 rows,
values in [0, 1]), and saves an exploratory data summary to results/data_summary.md.
"""

import os
import pandas as pd
import numpy as np


def load_training_data(filepath: str = "data/debutanizer_final_training_data.csv") -> pd.DataFrame:
    """
    Load and validate the normalized debutanizer training dataset.
    
    Parameters
    ----------
    filepath : str
        Path to the normalized CSV file.
        
    Returns
    -------
    pd.DataFrame
        The validated DataFrame with 2394 rows and 8 columns (u1-u7 inputs and C4 target).
        
    Raises
    ------
    ValueError
        If the file row count is not 2394, or values fall outside [0, 1], or missing values exist.
    """
    # Plain-English comment: Read CSV using pandas without altering column names
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training data file not found at: {filepath}")
        
    df = pd.read_csv(filepath)
    
    # Check row count
    expected_rows = 2394
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, but found {len(df)} rows.")
        
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        raise ValueError(f"Found {missing_count} missing values in training data.")
        
    # Check that values are within [0, 1] with a tiny tolerance for floating-point noise
    val_min = df.min().min()
    val_max = df.max().max()
    if val_min < -1e-6 or val_max > 1.0 + 1e-6:
        raise ValueError(f"Data values out of normalized range [0, 1]: min={val_min}, max={val_max}")
        
    return df


def load_real_units_data(filepath: str = "data/debutanizer_data_estimated_real_units.csv") -> pd.DataFrame:
    """
    Load the optional illustration dataset rescaled to ASSUMED real units.
    Never used for model training; only used for physical illustration.
    
    Parameters
    ----------
    filepath : str
        Path to the assumed real units CSV file.
        
    Returns
    -------
    pd.DataFrame
        DataFrame in assumed physical units.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Estimated real units file not found at: {filepath}")
    return pd.read_csv(filepath)


def generate_data_summary(df: pd.DataFrame, output_path: str = "results/data_summary.md") -> str:
    """
    Generate a markdown summary of the dataset and save it to results/data_summary.md.
    
    Parameters
    ----------
    df : pd.DataFrame
        The loaded training dataset.
    output_path : str
        Output file path for the summary.
        
    Returns
    -------
    str
        The summary content.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Calculate statistics
    n_rows, n_cols = df.shape
    dtypes = df.dtypes.to_dict()
    missing = df.isnull().sum().to_dict()
    n_duplicates = df.duplicated().sum()
    stats = df.describe().round(4)
    
    # Build markdown report
    summary_md = []
    summary_md.append("# Debutanizer Soft Sensor - Stage 1 Data Summary\n")
    summary_md.append("## Overview")
    summary_md.append(f"- **Total Rows:** {n_rows} (Confirmed exactly 2394 rows in chronological time order)")
    summary_md.append(f"- **Total Columns:** {n_cols} (7 process input sensors `u1` to `u7` plus target `C4`)")
    summary_md.append(f"- **Missing Values:** {sum(missing.values())} missing entries")
    summary_md.append(f"- **Duplicate Rows:** {n_duplicates} duplicate rows")
    summary_md.append(f"- **Value Range:** Min = {df.min().min():.4f}, Max = {df.max().max():.4f} (Confirmed strictly bounded within [0, 1])")
    summary_md.append("- **Proprietary Notice:** The original refinery units, sensor calibrations, and physical variable definitions were not publicly disclosed by the original authors (Fortuna et al.) to preserve commercial confidentiality.\n")
    
    summary_md.append("## Column Data Types & Missing Counts")
    summary_md.append("| Column | Data Type | Missing Count | Min | Max |")
    summary_md.append("|---|---|---|---|---|")
    for col in df.columns:
        summary_md.append(f"| `{col}` | {dtypes[col]} | {missing[col]} | {df[col].min():.4f} | {df[col].max():.4f} |")
    summary_md.append("")
    
    summary_md.append("## Descriptive Statistics (describe())")
    # Format describe() as a markdown table manually without extra dependencies
    header = "| Metric | " + " | ".join(stats.columns) + " |"
    divider = "|:---|" + "|".join([":---:" for _ in stats.columns]) + "|"
    summary_md.append(header)
    summary_md.append(divider)
    for idx, row in stats.iterrows():
        row_str = f"| **{idx}** | " + " | ".join([f"{v:.4f}" for v in row.values]) + " |"
        summary_md.append(row_str)
    summary_md.append("")
    
    summary_md.append("## 3-Line Viva Explanation (Stage 1)")
    summary_md.append("> 1. The dataset contains 2,394 sequential operational time samples of a refinery debutanizer column with zero missing values.")
    summary_md.append("> 2. All 7 process variables (u1-u7) and the bottom butane target (C4) are normalized between 0 and 1 by the plant owners for commercial confidentiality.")
    summary_md.append("> 3. Preserving strict chronological ordering without random shuffling is crucial because process measurements are time-dependent and have dynamic memory.")
    
    content = "\n".join(summary_md)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    return content


if __name__ == "__main__":
    print("Loading normalized training data...")
    data = load_training_data()
    print(f"Data successfully loaded with shape: {data.shape}")
    summary = generate_data_summary(data)
    print("Saved data summary to results/data_summary.md")
