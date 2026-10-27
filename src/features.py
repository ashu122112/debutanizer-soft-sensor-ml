"""
src/features.py
Debutanizer Soft Sensor - Feature Engineering Module (Stage 5).

Constructs two strictly separated feature sets:
1. Causal Variant:
   - Uses ONLY the 7 process variables (u1-u7) and their past values (lags 0..max_lag).
   - Trailing 5-sample mean and std (using only past values up to time t).
   - One-step change (delta: x(t) - x(t-1)).
   - Strictly NO past or present C4 values in the feature matrix!
   - Tested for max_lag = 4 and max_lag = 16.
   - Conceptual meaning: Works continuously in real time even when the lab is offline.

2. Autoregressive Variant:
   - Contains all causal process features above PLUS past C4 laboratory values
     at lags 4, 5, and 6 ONLY (i.e., C4(t-4), C4(t-5), C4(t-6)).
   - Strictly FORBIDS lags 0, 1, 2, 3 of C4 (which would represent unmeasured future lab results).
   - Conceptual meaning: Leverages recent historical lab assays to correct model drift.
"""

import pandas as pd
import numpy as np


def build_causal_features(df: pd.DataFrame, max_lag: int = 4) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build causal feature set using only process variables u1-u7.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset containing u1-u7 and target C4.
    max_lag : int
        Maximum lag order for process variables (4 or 16).
        
    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) after dropping rows lacking complete lag/rolling history.
    """
    input_cols = [c for c in df.columns if c != "C4" and c.lower() != "y"]
    features = {}
    
    # 1. Process variables and their lags: 0, 1, ..., max_lag
    for col in input_cols:
        for lag in range(max_lag + 1):
            feat_name = f"{col}" if lag == 0 else f"{col}_lag{lag}"
            features[feat_name] = df[col].shift(lag)
            
        # 2. One-step change (delta: value at t minus value at t-1)
        features[f"{col}_diff1"] = df[col] - df[col].shift(1)
        
        # 3. Trailing 5-sample rolling mean and rolling std
        # rolling(5) over past values up to t (requires min_periods=5)
        features[f"{col}_roll_mean5"] = df[col].rolling(window=5, min_periods=5).mean()
        features[f"{col}_roll_std5"] = df[col].rolling(window=5, min_periods=5).std()
        
    X = pd.DataFrame(features, index=df.index)
    y = df["C4"].copy()
    
    # Drop rows with incomplete history (NaNs due to lags and rolling windows)
    # Do NOT impute or fill; drop cleanly as required
    valid_idx = X.dropna().index
    X_clean = X.loc[valid_idx].copy()
    y_clean = y.loc[valid_idx].copy()
    
    # Validation safety check: Assert no C4 column exists in causal feature matrix
    validate_feature_rules(X_clean, is_causal=True)
    
    return X_clean, y_clean


def build_autoregressive_features(df: pd.DataFrame, max_lag: int = 4) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build autoregressive feature set combining causal process features with
    valid past laboratory C4 values at lags 4, 5, and 6 only.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset containing u1-u7 and target C4.
    max_lag : int
        Maximum lag order for process variables (4 or 16).
        
    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) after dropping rows lacking complete lag/rolling history.
    """
    # Start with all causal process features
    input_cols = [c for c in df.columns if c != "C4" and c.lower() != "y"]
    features = {}
    
    for col in input_cols:
        for lag in range(max_lag + 1):
            feat_name = f"{col}" if lag == 0 else f"{col}_lag{lag}"
            features[feat_name] = df[col].shift(lag)
            
        features[f"{col}_diff1"] = df[col] - df[col].shift(1)
        features[f"{col}_roll_mean5"] = df[col].rolling(window=5, min_periods=5).mean()
        features[f"{col}_roll_std5"] = df[col].rolling(window=5, min_periods=5).std()
        
    # Add past C4 values at lags 4, 5, and 6 ONLY
    for lag in [4, 5, 6]:
        features[f"C4_lag{lag}"] = df["C4"].shift(lag)
        
    X = pd.DataFrame(features, index=df.index)
    y = df["C4"].copy()
    
    # Drop initial rows lacking full history
    valid_idx = X.dropna().index
    X_clean = X.loc[valid_idx].copy()
    y_clean = y.loc[valid_idx].copy()
    
    # Validation safety check: Assert autoregressive features contain only lags >= 4
    validate_feature_rules(X_clean, is_causal=False)
    
    return X_clean, y_clean


def validate_feature_rules(X: pd.DataFrame, is_causal: bool):
    """
    Validation check that fails if causal features contain any C4 column,
    or if autoregressive features contain C4 lags below 4.
    
    Raises
    ------
    AssertionError
        If any leakage rule is violated.
    """
    cols = list(X.columns)
    c4_cols = [c for c in cols if "c4" in c.lower()]
    
    if is_causal:
        assert len(c4_cols) == 0, f"Causal feature violation! Found C4 features in causal set: {c4_cols}"
    else:
        # Must only have C4_lag4, C4_lag5, C4_lag6 (never lag 0..3)
        for c in c4_cols:
            if c in ["C4", "C4_lag0", "C4_lag1", "C4_lag2", "C4_lag3"]:
                raise AssertionError(f"Autoregressive leakage violation! Found unallowed near-term C4 feature: {c}")


def build_steady_state_row(candidate_u: dict, feature_columns: list, max_lag: int = 4) -> pd.DataFrame:
    """
    Construct a single steady-state input row for Genetic Algorithm optimization.
    At steady state:
    - All lags equal the candidate sensor value: u_i(t) = u_i(t-k) = candidate_u_i.
    - Trailing rolling mean equals the candidate value.
    - Trailing rolling std equals 0.0 (no fluctuations).
    - One-step change equals 0.0 (no movement).
    
    Parameters
    ----------
    candidate_u : dict
        Dictionary of { 'u1': val1, ..., 'u7': val7 }.
    feature_columns : list
        The exact ordered column names from the causal training feature matrix.
    max_lag : int
        Maximum lag order used during training.
        
    Returns
    -------
    pd.DataFrame
        Single-row DataFrame matching feature_columns exactly.
    """
    row = {}
    for col, val in candidate_u.items():
        for lag in range(max_lag + 1):
            name = f"{col}" if lag == 0 else f"{col}_lag{lag}"
            row[name] = float(val)
        row[f"{col}_diff1"] = 0.0
        row[f"{col}_roll_mean5"] = float(val)
        row[f"{col}_roll_std5"] = 0.0
        
    df_row = pd.DataFrame([row])
    
    # Reorder and verify identical columns and order
    missing = [c for c in feature_columns if c not in df_row.columns]
    extra = [c for c in df_row.columns if c not in feature_columns]
    if missing or extra:
        raise ValueError(f"Steady state row mismatch! Missing: {missing}, Extra: {extra}")
        
    return df_row[feature_columns]


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data_io import load_training_data
    df = load_training_data()
    
    Xc_4, yc_4 = build_causal_features(df, max_lag=4)
    print(f"Causal Lag-4: X shape = {Xc_4.shape}, y shape = {yc_4.shape}")
    
    Xc_16, yc_16 = build_causal_features(df, max_lag=16)
    print(f"Causal Lag-16: X shape = {Xc_16.shape}, y shape = {yc_16.shape}")
    
    Xa_4, ya_4 = build_autoregressive_features(df, max_lag=4)
    print(f"Autoregressive Lag-4: X shape = {Xa_4.shape}, y shape = {ya_4.shape}")
    
    Xa_16, ya_16 = build_autoregressive_features(df, max_lag=16)
    print(f"Autoregressive Lag-16: X shape = {Xa_16.shape}, y shape = {ya_16.shape}")
    
    # Test steady state row builder
    cand = {f"u{i}": 0.5 for i in range(1, 8)}
    ss_row = build_steady_state_row(cand, list(Xc_4.columns), max_lag=4)
    print("Steady state row constructed successfully, shape:", ss_row.shape)
