"""
src/evaluate.py
Debutanizer Soft Sensor - Evaluation and Splitting Module (Stage 3 & 4).

CRITICAL RULE: This file is written once and never modified.
It enforces strict chronological data splitting (60% train, 20% validation, 20% test),
ensures scalers are fitted ONLY on the training block, computes standard regression
metrics (R2, RMSE, MAE, bias), and logs results incrementally to results/model_results.csv.

Why random splitting is strictly forbidden:
Process measurements are recorded sequentially in time and exhibit dynamic memory.
A random shuffle split allows future or neighbouring time points into the training set,
letting the model "peek" at the immediate future, which unrealistically inflates performance
metrics and causes catastrophic failure when deployed in real industrial operation.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def chronological_split(X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.60, val_ratio: float = 0.20):
    """
    Split time-series data chronologically without shuffling.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix in time order.
    y : pd.Series
        Target series in time order.
    train_ratio : float
        Proportion of samples for the training block (default 0.60).
    val_ratio : float
        Proportion of samples for the validation block (default 0.20).
        Remaining proportion (1 - train_ratio - val_ratio) goes to test.
        
    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    # Slice strictly in time order (first 60%, next 20%, last 20%)
    X_train = X.iloc[:train_end].copy()
    y_train = y.iloc[:train_end].copy()
    
    X_val = X.iloc[train_end:val_end].copy()
    y_val = y.iloc[train_end:val_end].copy()
    
    X_test = X.iloc[val_end:].copy()
    y_test = y.iloc[val_end:].copy()
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fit StandardScaler on the training block only, and apply to validation and test.
    
    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Unscaled feature sets.
        
    Returns
    -------
    tuple
        (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    # Fit strictly on training data to prevent data leakage from validation or test
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate evaluation metrics: R2, RMSE, MAE, and bias (mean of prediction minus actual).
    
    Parameters
    ----------
    y_true : np.ndarray
        True observed target values.
    y_pred : np.ndarray
        Model predicted target values.
        
    Returns
    -------
    dict
        Dictionary containing r2, rmse, mae, and bias.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    
    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "bias": bias
    }


def log_model_result(
    model_name: str,
    feature_variant: str,
    best_params: dict,
    train_metrics: dict,
    val_metrics: dict,
    test_metrics: dict,
    results_path: str = "results/model_results.csv"
):
    """
    Append one evaluation record to results/model_results.csv without overwriting existing rows.
    
    Parameters
    ----------
    model_name : str
        Name of the model (e.g., 'LinearRegression', 'RandomForest', 'Persistence').
    feature_variant : str
        Variant description (e.g., 'causal_lag4', 'causal_lag16', 'autoregressive_lag4').
    best_params : dict
        Selected hyperparameters on validation set.
    train_metrics, val_metrics, test_metrics : dict
        Calculated metric dictionaries.
    results_path : str
        Target CSV file path.
    """
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    row_data = {
        "model_name": model_name,
        "feature_variant": feature_variant,
        "best_params": str(best_params),
        "train_r2": train_metrics.get("r2", np.nan),
        "train_rmse": train_metrics.get("rmse", np.nan),
        "train_mae": train_metrics.get("mae", np.nan),
        "train_bias": train_metrics.get("bias", np.nan),
        "val_r2": val_metrics.get("r2", np.nan),
        "val_rmse": val_metrics.get("rmse", np.nan),
        "val_mae": val_metrics.get("mae", np.nan),
        "val_bias": val_metrics.get("bias", np.nan),
        "test_r2": test_metrics.get("r2", np.nan),
        "test_rmse": test_metrics.get("rmse", np.nan),
        "test_mae": test_metrics.get("mae", np.nan),
        "test_bias": test_metrics.get("bias", np.nan),
    }
    
    df_row = pd.DataFrame([row_data])
    
    if os.path.exists(results_path):
        df_row.to_csv(results_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(results_path, mode="w", header=True, index=False)


def evaluate_persistence_baseline(
    raw_c4: pd.Series,
    valid_indices: pd.Index,
    feature_variant_label: str = "persistence_lag4",
    results_path: str = "results/model_results.csv"
):
    """
    Evaluate the non-ML persistence reference baseline (Stage 4).
    Predicts C4 at time t as the C4 value 4 steps earlier (y_{t-4}), representing
    the latest available laboratory analysis before the current sample.
    
    Parameters
    ----------
    raw_c4 : pd.Series
        Full original C4 time series from debutanizer_final_training_data.csv.
    valid_indices : pd.Index
        The row indices corresponding to the feature dataset after lag dropping.
    feature_variant_label : str
        Label to distinguish lag-4 or lag-16 alignment.
    results_path : str
        CSV path to log results.
        
    Returns
    -------
    dict
        Dictionary of baseline metrics across splits.
    """
    # Create lag-4 series on the original continuous index
    c4_lag4 = raw_c4.shift(4)
    
    # Subselect the valid rows that match the feature dataset
    aligned_y_true = raw_c4.loc[valid_indices]
    aligned_y_pred = c4_lag4.loc[valid_indices]
    
    # Split chronologically with exact same 60/20/20 proportion
    n = len(aligned_y_true)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    
    y_train_true, y_train_pred = aligned_y_true.iloc[:train_end], aligned_y_pred.iloc[:train_end]
    y_val_true, y_val_pred = aligned_y_true.iloc[train_end:val_end], aligned_y_pred.iloc[train_end:val_end]
    y_test_true, y_test_pred = aligned_y_true.iloc[val_end:], aligned_y_pred.iloc[val_end:]
    
    train_metrics = calculate_metrics(y_train_true, y_train_pred)
    val_metrics = calculate_metrics(y_val_true, y_val_pred)
    test_metrics = calculate_metrics(y_test_true, y_test_pred)
    
    log_model_result(
        model_name="Persistence_Baseline",
        feature_variant=feature_variant_label,
        best_params={"persistence_lag": 4},
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        results_path=results_path
    )
    
    return {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics
    }
