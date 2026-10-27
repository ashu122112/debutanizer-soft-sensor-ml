"""
src/explain.py
Debutanizer Soft Sensor - Model Explainability Module (Stage 8).

This module provides:
1. Built-in feature importance (for tree models).
2. Permutation importance on the validation block (model-agnostic).
3. SHAP analysis (TreeExplainer for trees, LinearExplainer for linear models).
4. Grouped importance by source process variable (adding up all lag/rolling
   features derived from the same sensor u1..u7, and C4 lags for autoregressive).

Key warning: These importance scores show what the model USES statistically,
not what physically CAUSES C4 changes. Causality requires controlled experiments
and domain engineering knowledge beyond a data-driven model.

Near-duplicate inputs (e.g. u6 and u7 which correlate at r=0.99) will split
their combined credit arbitrarily across individual features. Always read grouped
importances alongside individual ones for such cases.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import shap


# ─────────────────────────────────────────────────────────────────────────────
# 1. Built-in Feature Importance (trees only)
# ─────────────────────────────────────────────────────────────────────────────

def get_builtin_importance(model, feature_names: list) -> pd.Series:
    """
    Extract built-in feature importance from a tree-based model.

    Works for RandomForest and XGBoost which expose feature_importances_.
    Linear models do not have this attribute; returns None for them.

    Parameters
    ----------
    model : fitted estimator
    feature_names : list of str

    Returns
    -------
    pd.Series or None
        Importances indexed by feature name, sorted descending.
    """
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names)
        return imp.sort_values(ascending=False)
    return None


def plot_builtin_importance(imp: pd.Series, model_name: str, variant: str,
                            top_n: int = 30, save_dir: str = "figures"):
    """Save a horizontal bar chart of built-in feature importances (top N)."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.28)))
    imp_top = imp.iloc[:top_n]
    imp_top[::-1].plot(kind="barh", ax=ax, color="#1f77b4")
    ax.set_title(f"Built-in Feature Importance\n{model_name} | {variant}", fontweight="bold")
    ax.set_xlabel("Mean Impurity Decrease / Feature Gain")
    plt.tight_layout()
    fname = os.path.join(save_dir, f"builtin_importance_{model_name}_{variant}.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    return fname


# ─────────────────────────────────────────────────────────────────────────────
# 2. Permutation Importance (validation block, model-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def get_permutation_importance(model, X_val: pd.DataFrame, y_val: pd.Series,
                               n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
    """
    Compute permutation importance on the validation block.

    For each feature, shuffle it n_repeats times, measure the drop in R²,
    and average across repetitions. Large drop = high importance.

    Uses the VALIDATION block (not test) to avoid contaminating test set
    with any model-selection decisions.

    Parameters
    ----------
    model : fitted estimator
    X_val : pd.DataFrame  (already scaled)
    y_val : pd.Series
    n_repeats : int
    random_state : int

    Returns
    -------
    pd.DataFrame
        Columns: [feature, mean_importance, std_importance], sorted descending.
    """
    result = permutation_importance(
        model, X_val, y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="r2"
    )
    df_perm = pd.DataFrame({
        "feature": X_val.columns,
        "mean_importance": result.importances_mean,
        "std_importance": result.importances_std,
    }).sort_values("mean_importance", ascending=False).reset_index(drop=True)
    return df_perm


def plot_permutation_importance(df_perm: pd.DataFrame, model_name: str, variant: str,
                                top_n: int = 30, save_dir: str = "figures"):
    """Save a horizontal bar chart with error bars for permutation importance."""
    os.makedirs(save_dir, exist_ok=True)
    top = df_perm.head(top_n).iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.28)))
    ax.barh(top["feature"], top["mean_importance"], xerr=top["std_importance"],
            color="#2ca02c", alpha=0.8, capsize=3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Permutation Importance (Validation Block)\n{model_name} | {variant}", fontweight="bold")
    ax.set_xlabel("Mean R² Drop When Feature Shuffled")
    plt.tight_layout()
    fname = os.path.join(save_dir, f"permutation_importance_{model_name}_{variant}.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    return fname


# ─────────────────────────────────────────────────────────────────────────────
# 3. SHAP Analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values(model, X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Compute SHAP values using the appropriate explainer.

    For tree models (RandomForest, XGBoost): uses TreeExplainer with training
    block as background.
    For LinearRegression: uses LinearExplainer with training block as background.
    For SVR: falls back to KernelExplainer on a small random subsample (slow).

    Parameters
    ----------
    model : fitted estimator
    X_train : pd.DataFrame  (training block, scaled; used as SHAP background)
    X_test  : pd.DataFrame  (test block to explain)

    Returns
    -------
    shap.Explanation or np.ndarray
        SHAP values for each test sample.
    explainer object
    """
    if isinstance(model, (RandomForestRegressor, xgb.XGBRegressor)):
        # TreeExplainer is fast and exact for tree ensembles
        explainer = shap.TreeExplainer(model, data=X_train.values,
                                       feature_names=list(X_train.columns))
        shap_values = explainer(X_test.values)
    elif isinstance(model, LinearRegression):
        # LinearExplainer is exact for linear models
        explainer = shap.LinearExplainer(model, X_train.values,
                                         feature_names=list(X_train.columns))
        shap_values = explainer(X_test.values)
    else:
        # KernelExplainer (model-agnostic): use small random background for speed
        rng = np.random.default_rng(42)
        bg_idx = rng.choice(len(X_train), size=min(100, len(X_train)), replace=False)
        bg = X_train.values[bg_idx]
        explainer = shap.KernelExplainer(model.predict, bg,
                                         feature_names=list(X_train.columns))
        # Explain a sample of 200 test points to keep it tractable
        test_sample = X_test.values[:min(200, len(X_test))]
        shap_values = explainer.shap_values(test_sample)
    return shap_values, explainer


def plot_shap_summary(shap_values, X_test: pd.DataFrame, model_name: str,
                      variant: str, save_dir: str = "figures"):
    """Save SHAP beeswarm summary plot and bar plot."""
    os.makedirs(save_dir, exist_ok=True)

    # Beeswarm summary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=25,
                      plot_type="dot", color_bar_label="Feature Value")
    plt.title(f"SHAP Summary (Beeswarm)\n{model_name} | {variant}", fontweight="bold", pad=12)
    plt.tight_layout()
    fname_bee = os.path.join(save_dir, f"shap_summary_{model_name}_{variant}.png")
    plt.savefig(fname_bee, dpi=200, bbox_inches="tight")
    plt.close()

    # Bar plot (mean |SHAP|)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=25,
                      plot_type="bar")
    plt.title(f"SHAP Feature Importance (Mean |SHAP|)\n{model_name} | {variant}", fontweight="bold", pad=12)
    plt.tight_layout()
    fname_bar = os.path.join(save_dir, f"shap_bar_{model_name}_{variant}.png")
    plt.savefig(fname_bar, dpi=200, bbox_inches="tight")
    plt.close()

    return fname_bee, fname_bar


# ─────────────────────────────────────────────────────────────────────────────
# 4. Grouped Importance by Source Variable
# ─────────────────────────────────────────────────────────────────────────────

def group_importance_by_source(imp_series: pd.Series) -> pd.Series:
    """
    Sum all per-feature importances that originate from the same source variable.

    Example: u6, u6_lag1, u6_lag2, u6_roll_mean5, u6_roll_std5, u6_diff1
    are all grouped under 'u6'.

    This prevents the near-duplicate u6/u7 pair (r=0.99) from appearing
    misleadingly small just because their joint credit is split.

    Parameters
    ----------
    imp_series : pd.Series
        Feature-level importances indexed by feature name.

    Returns
    -------
    pd.Series
        Grouped importance per source variable, sorted descending.
    """
    source_vars = ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "C4"]
    group_totals = {}
    for src in source_vars:
        # Match features that start with the source variable name followed by
        # end-of-string, underscore, or nothing (prevents u1 matching u10 etc.)
        matching = [f for f in imp_series.index
                    if f == src or f.startswith(src + "_")]
        if matching:
            group_totals[src] = imp_series[matching].sum()
    return pd.Series(group_totals).sort_values(ascending=False)


def plot_grouped_importance(grouped: pd.Series, model_name: str, variant: str,
                            save_dir: str = "figures"):
    """Save a bar chart of grouped importances by source sensor."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10.colors[:len(grouped)]
    grouped.plot(kind="bar", ax=ax, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_title(f"Grouped Feature Importance by Source Sensor\n{model_name} | {variant}",
                 fontweight="bold")
    ax.set_xlabel("Source Process Variable (sensor)")
    ax.set_ylabel("Summed Importance (all lag/rolling features)")
    ax.set_xticklabels(grouped.index, rotation=30)
    plt.tight_layout()
    fname = os.path.join(save_dir, f"grouped_importance_{model_name}_{variant}.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    return fname


# ─────────────────────────────────────────────────────────────────────────────
# 5. Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_explain(best_model, model_name: str, variant: str,
                X_train_s: pd.DataFrame, y_train: pd.Series,
                X_val_s: pd.DataFrame, y_val: pd.Series,
                X_test_s: pd.DataFrame, y_test: pd.Series,
                save_dir: str = "figures"):
    """
    Run full explainability pipeline for the best model.

    Parameters
    ----------
    best_model : fitted estimator
    model_name : str
    variant : str
    X_train_s, y_train : training block (scaled X)
    X_val_s, y_val     : validation block (scaled X)
    X_test_s, y_test   : test block (scaled X)
    save_dir : str

    Returns
    -------
    dict with all computed importance objects and saved figure paths.
    """
    results = {}

    # 1. Built-in importance
    print(f"  Computing built-in importance for {model_name}...")
    builtin = get_builtin_importance(best_model, list(X_train_s.columns))
    if builtin is not None:
        p = plot_builtin_importance(builtin, model_name, variant, save_dir=save_dir)
        results["builtin_importance"] = builtin
        results["builtin_plot"] = p

        grouped_builtin = group_importance_by_source(builtin)
        p2 = plot_grouped_importance(grouped_builtin, model_name, variant + "_builtin",
                                     save_dir=save_dir)
        results["grouped_builtin"] = grouped_builtin
        results["grouped_builtin_plot"] = p2
        print(f"    Top grouped (built-in): {grouped_builtin.head(4).to_dict()}")

    # 2. Permutation importance on validation block
    print(f"  Computing permutation importance for {model_name}...")
    df_perm = get_permutation_importance(best_model, X_val_s, y_val)
    p3 = plot_permutation_importance(df_perm, model_name, variant, save_dir=save_dir)
    results["permutation_importance"] = df_perm
    results["permutation_plot"] = p3

    perm_series = df_perm.set_index("feature")["mean_importance"]
    grouped_perm = group_importance_by_source(perm_series)
    p4 = plot_grouped_importance(grouped_perm, model_name, variant + "_perm",
                                  save_dir=save_dir)
    results["grouped_perm"] = grouped_perm
    results["grouped_perm_plot"] = p4
    print(f"    Top grouped (permutation): {grouped_perm.head(4).to_dict()}")

    # 3. SHAP analysis
    print(f"  Computing SHAP values for {model_name} (this may take a moment)...")
    try:
        shap_vals, explainer = compute_shap_values(best_model, X_train_s, X_test_s)
        p5, p6 = plot_shap_summary(shap_vals, X_test_s, model_name, variant,
                                   save_dir=save_dir)
        results["shap_values"] = shap_vals
        results["shap_summary_plot"] = p5
        results["shap_bar_plot"] = p6
    except Exception as e:
        print(f"  WARNING: SHAP failed for {model_name}: {e}")
        results["shap_error"] = str(e)

    return results


def explain_best_model_from_experiments(all_experiments: dict, save_dir: str = "figures"):
    """
    Identify the best model from results/model_results.csv and run full explanation.
    Focuses on the best causal and best autoregressive models separately.
    """
    import csv
    results_path = "results/model_results.csv"
    if not os.path.exists(results_path):
        raise FileNotFoundError("Run src/models.py first to generate results/model_results.csv")

    df_res = pd.read_csv(results_path)

    explanation_results = {}
    for variant_filter in ["causal", "autoregressive"]:
        # Find best (lowest test RMSE) among ML models in this variant family
        subset = df_res[df_res["feature_variant"].str.startswith(variant_filter) &
                        ~df_res["model_name"].str.startswith("Persistence")]
        if subset.empty:
            continue
        best_row = subset.sort_values("test_rmse").iloc[0]
        best_model_name = best_row["model_name"]
        best_variant = best_row["feature_variant"]

        print(f"\n{'='*60}")
        print(f"Explaining best {variant_filter} model: {best_model_name} on {best_variant}")
        print(f"Test RMSE={best_row['test_rmse']:.4f}, Test R²={best_row['test_r2']:.4f}")
        print(f"{'='*60}")

        exp_data = all_experiments[best_variant]
        model = exp_data["models"][best_model_name]["model"]
        X_train_s = exp_data["X_train_s"]
        y_train = exp_data["y_train"]
        X_test_s = exp_data["X_test_s"]
        y_test = exp_data["y_test"]

        # We also need X_val_s and y_val for permutation importance
        from src.evaluate import chronological_split, scale_features
        from src.features import build_causal_features, build_autoregressive_features
        from src.data_io import load_training_data
        df_raw = load_training_data()
        max_lag = 16 if "lag16" in best_variant else 4
        if "autoregressive" in best_variant:
            X_full, y_full = build_autoregressive_features(df_raw, max_lag=max_lag)
        else:
            X_full, y_full = build_causal_features(df_raw, max_lag=max_lag)
        _, X_val_orig, _, _, y_val_orig, _ = chronological_split(X_full, y_full)
        _, X_val_s_local, _, _ = scale_features(X_full.iloc[:int(len(X_full)*0.6)],
                                                 X_val_orig, X_full.iloc[int(len(X_full)*0.8):])

        result = run_explain(
            best_model=model,
            model_name=best_model_name,
            variant=best_variant,
            X_train_s=X_train_s,
            y_train=y_train,
            X_val_s=X_val_s_local,
            y_val=y_val_orig,
            X_test_s=X_test_s,
            y_test=pd.Series(y_test),
            save_dir=save_dir,
        )
        explanation_results[best_variant] = result

        # Print distillation interpretation in cautious language
        if "grouped_perm" in result:
            g = result["grouped_perm"]
            print(f"\n  Cautious distillation interpretation for {best_model_name}:")
            print(f"  The model gives highest weight to: {', '.join(g.head(3).index.tolist())}.")
            print(f"  In basic distillation terms: bottom temperatures (u6, u7) reflect")
            print(f"  reboiler heat duty and separation extent; reflux flow (u3) controls")
            print(f"  the internal liquid/vapour ratio; top pressure (u2) shifts the vapour")
            print(f"  equilibrium. These are associations the model has learned — not proven")
            print(f"  causal mechanisms. Actual causes require rigorous process experimentation.")

    return explanation_results
