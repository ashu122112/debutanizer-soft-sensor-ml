# Debutanizer Soft Sensor — ML Part

A machine-learning soft sensor for an industrial refinery debutanizer column.
Given 7 process instrument readings sampled every few minutes, the models
estimate the C4 (butane) concentration in the column bottom product — a
measurement that normally arrives hours later from the laboratory.

> **Original units and variable definitions are not published.**  
> All data values are normalised to [0, 1] for commercial confidentiality (Fortuna et al.).  
> The file `debutanizer_data_estimated_real_units.csv` uses **assumed** scale factors only
> and must **never** be used for model training.

---

## Problem Statement

Lab assays of C4 arrive with a significant delay (several sampling intervals).
A soft sensor that predicts C4 in real time from existing instrumentation lets
operators make corrective adjustments immediately instead of waiting for lab
results, reducing off-spec product and energy waste.

---

## Data

| File | Description |
|---|---|
| `data/debutanizer_final_training_data.csv` | 2394 rows, normalised [0, 1], 7 inputs + C4 target |
| `data/debutanizer_data_estimated_real_units.csv` | Same data in **assumed** units. Do not train on this. |

Columns: `u1` (top temperature), `u2` (top pressure), `u3` (reflux flow), `u4` (flow to next process), `u5` (6th tray temperature), `u6` (bottom temperature 1), `u7` (bottom temperature 2), `C4` (target: bottom butane concentration). These names are the commonly reported mapping in the literature; they are not confirmed by the original data source, and the identity of u7 in particular is disputed. Units are unknown because the data is normalised to [0, 1].

---

## Feature Variants

Two fundamentally different modelling paradigms are tested:

| Variant | Description |
|---|---|
| **Causal** | Only the 7 process variables (u1-u7) and their lags, rolling statistics, and one-step changes. **No C4 past values.** Works even when the lab is offline. |
| **Autoregressive** | As above, **plus** past C4 laboratory assays at lags 4, 5, 6 only. Lags 0-3 are forbidden (they would not be available in real operation). Requires a recent lab result. |

Both are tested with lag windows of 4 and 16 samples (long process delays motivate the wider window).

---

## Models

All models use a **strict chronological split** (first 60% train, next 20% validation, last 20% test). Random splitting is explicitly forbidden because it allows neighbouring time steps into the training set, unrealistically inflating performance metrics.

| Model | Notes |
|---|---|
| Persistence Baseline | C4(t) ≈ C4(t-4); the reference every model must beat |
| Linear Regression | No tuning; shows how much linear structure exists |
| Random Forest | Grid: n_estimators {100, 300}, max_depth {4, 8, None}, min_samples_leaf {1, 5} |
| XGBoost | Grid: n_estimators {100, 300}, max_depth {3, 4}, learning_rate {0.05, 0.1}, subsample 0.8 |
| SVR (rbf) | Grid: C {0.1, 1, 10}, epsilon {0.01, 0.05}, gamma {"scale", 0.01} |

Hyperparameter selection is based strictly on validation RMSE. The test set is never touched.

---

## Results

The persistence baseline predicts C4(t) from the lab value C4(t-4), so it needs lab data. It is the fair reference for the autoregressive models. For the causal models (sensors only, no lab values), the fair reference is R² = 0, i.e. predicting the mean. Negative R² means worse than the mean.

### Causal Variants (process variables only, no lab feedback)

| Model | Val RMSE | Test RMSE | Test MAE | Test R² | Test Bias |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Persistence_Baseline** | 0.0579 | 0.0589 | 0.0436 | 0.9152 | +0.0010 |
| LinearRegression | 0.1927 | 0.1909 | 0.1552 | 0.1094 | +0.0482 |
| RandomForest | 0.1620 | 0.2228 | 0.1785 | -0.2136 | +0.0635 |
| XGBoost | 0.1361 | 0.2193 | 0.1707 | -0.1758 | +0.0686 |
| SVR | 0.1469 | 0.1962 | 0.1558 | 0.0586 | +0.0478 |
| LinearRegression (lag16) | 0.1095 | 0.1538 | 0.1228 | 0.4238 | +0.0918 |
| RandomForest (lag16) | 0.1098 | 0.1546 | 0.1182 | 0.4179 | +0.0454 |
| XGBoost (lag16) | 0.0992 | 0.1501 | 0.1174 | 0.4515 | +0.0561 |
| SVR (lag16) | 0.1351 | 0.1898 | 0.1488 | 0.1231 | +0.0902 |

> ⚠️ **None of the causal ML models beat the persistence baseline.** The process sensors alone
> carry approximately 10-45% of the C4 variance in the test period. This is an honest result
> and should not be hidden.

### Autoregressive Variants (process variables + delayed lab assays)

| Model | Val RMSE | Test RMSE | Test MAE | Test R² | Test Bias |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Persistence_Baseline** | 0.0579 | 0.0589 | 0.0436 | 0.9152 | +0.0010 |
| LinearRegression (lag4) | 0.0204 | 0.0220 | 0.0173 | 0.9882 | +0.0115 |
| RandomForest (lag4) | 0.0428 | 0.0617 | 0.0499 | 0.9070 | +0.0251 |
| XGBoost (lag4) | 0.0467 | 0.0674 | 0.0510 | 0.8888 | +0.0254 |
| SVR (lag4) | 0.1163 | 0.1036 | 0.0750 | 0.7375 | +0.0411 |
| **LinearRegression (lag16)** | **0.0164** | **0.0195** | **0.0157** | **0.9907** | +0.0091 |
| RandomForest (lag16) | 0.0521 | 0.0544 | 0.0355 | 0.9280 | +0.0002 |
| XGBoost (lag16) | 0.0339 | 0.0584 | 0.0388 | 0.9168 | +0.0018 |

> **Linear Regression on the autoregressive feature set achieves the lowest test RMSE (0.0195)**,
> outperforming all tree models. This is not surprising: once the dominant predictor (recent C4 assay)
> is included as a feature, the relationship becomes nearly linear and tree-based non-linearities
> add modest incremental value. The result is reported honestly.

---

## Explainability (Stage 8)

- **Built-in importance** and **permutation importance** (on validation block) show which features the model relies on most.
- **SHAP** (TreeExplainer for RF/XGBoost, LinearExplainer for LR) quantifies each feature's directional contribution.
- **Grouped importance** sums all lag/rolling features from the same source sensor to correct for the near-duplicate (u6, u7 correlation r=0.99).

> These are **statistical associations**, not causal mechanisms. In basic distillation terms:
> bottom temperatures (u6, u7) reflect reboiler heat duty; reflux flow (u3) controls internal
> L/V ratio; top temperature (u1) is linked to overhead product purity. Whether these are the
> true physical drivers requires controlled process experiments.

---

## Optimisation — Genetic Algorithm (Stage 9)

A DEAP genetic algorithm uses the **best causal model** (XGBoost, lag-16) as a fast surrogate
and searches for the operating point minimising predicted C4.

| GA Setting | Value |
|---|---|
| Population | 60 |
| Generations | 80 |
| Crossover | 0.7 |
| Mutation | 0.2 |
| Selection | Tournament (size 3) |
| Seeds | 5 (seeds 42-46) |
| Bounds | 5th–95th percentile of training block |

See `results/optimisation_summary.md` for full results, the optimal operating point,
and comparison with the historically lowest-C4 operating conditions.

> ⚠️ **This result is indicative only.** The causal model achieves Test R² ≈ 0.45,
> meaning ~55% of variance is unexplained. The Aspen Plus simulation teammate must validate
> the recommended point before any plant trial.

---

## How to Run

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate      # mac/linux
# venv\Scripts\activate       # windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Stage 1 — load and validate data
PYTHONPATH=. python src/data_io.py

# 4. Stage 5 — verify feature engineering
PYTHONPATH=. python src/features.py

# 5. Stage 6 & 7 — run full model benchmark (takes ~10 min)
PYTHONPATH=. python -m src.models

# 6. Stage 8 — run explainability
PYTHONPATH=. python -m src.explain

# 7. Stage 9 — genetic algorithm optimisation (~5 min)
PYTHONPATH=. python -m src.optimise

# 8. Run all tests
PYTHONPATH=. python -m pytest tests/ -v

# 9. Launch Jupyter for interactive notebooks
jupyter notebook
```

---

## Project Structure

```
debutanizer-ml/
├── data/
│   ├── debutanizer_final_training_data.csv   ← normalised [0, 1]
│   └── debutanizer_data_estimated_real_units.csv  ← assumed units, DO NOT train
├── notebooks/
│   ├── 01_explore.ipynb
│   ├── 02_models.ipynb
│   └── 03_explain_optimise.ipynb
├── src/
│   ├── data_io.py      ← load + validate + data summary
│   ├── features.py     ← lag features, rolling stats, steady-state row
│   ├── evaluate.py     ← chronological split, scaler, metrics, result logging
│   ├── models.py       ← all 4 models + grid search + comparison plots
│   ├── explain.py      ← importance, permutation, SHAP, grouped importance
│   └── optimise.py     ← DEAP genetic algorithm, stability analysis
├── tests/
│   └── test_pipeline.py
├── figures/            ← all generated plots
├── results/
│   ├── data_summary.md
│   ├── model_results.csv
│   ├── model_comparison.md
│   ├── optimisation_results.csv
│   ├── optimisation_summary.md
│   └── optimum_for_aspen.csv
├── docs/
│   └── viva_cheatsheet.md
├── requirements.txt
└── README.md
```

---

## Limitations

1. **Normalised data only.** All values are in [0, 1]. Physical engineering units are not recoverable without the original calibration data from the plant owner.
2. **Undocumented variables.** Column names u1–u7 map to known process instrument types (temperatures, pressure, flows) but exact sensor locations, calibration ranges, and measurement frequencies are not publicly disclosed.
3. **Single chronological test period.** The 20% held-out test block represents one contiguous operational window. Model performance on other periods or after maintenance may differ.
4. **Residual drift.** The rolling-mean residual plot shows a systematic error trend toward the end of the test period, consistent with feed composition drift or sensor fouling. A deployed sensor would need periodic recalibration.
5. **Causal model limitation.** If deployed without lab feedback (causal mode), Test R² ≈ 0.45 at best. This may not be sufficient for real-time control decisions without additional validation.
6. **Optimiser indicative only.** The GA optimum is bounded by the accuracy of the causal surrogate model and must be validated through Aspen Plus simulation and plant trials before any operational change.
