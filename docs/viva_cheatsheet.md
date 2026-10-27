# Debutanizer Soft Sensor — Viva Cheatsheet

> Plain-English reference for oral examination preparation.  
> Read this alongside the results tables in `README.md` and `results/model_comparison.md`.

---

## 1. What is a Debutanizer Column?

A debutanizer is a distillation column in a petroleum refinery that separates butane (C4) and lighter components from heavier naphtha products. The key quality metric is the **C4 concentration in the bottom product**. Too much C4 means product is off-spec; too little wastes feed. Lab analysis takes hours, so a real-time prediction model (soft sensor) is extremely valuable for process control.

---

## 2. The Four ML Models — Plain English

### Linear Regression
A model that finds the best straight-line relationship between the input features and C4. It minimises the sum of squared errors between predicted and actual C4 values. The coefficients directly show the direction and strength of each feature's linear association. No randomness, no hyperparameters. Fast and fully interpretable. Becomes the unexpected winner in the autoregressive setting because once recent C4 is included as a feature, the relationship is nearly linear.

### Random Forest
A collection of decision trees (an ensemble), each built on a random subsample of the training data and a random subset of features. Individual trees overfit; averaging hundreds of trees cancels out their errors, giving a stable, low-variance prediction. The key tuning parameters are how many trees to grow (`n_estimators`), how deep each tree can grow (`max_depth`), and the minimum number of samples at each leaf (`min_samples_leaf` — larger values prevent overfitting). Feature importance comes from how much each feature reduces the impurity (spread of C4 values) across all splits in all trees.

### XGBoost
Gradient boosting: trees are added one at a time, each one trained to correct the errors of all the previous trees combined. Unlike Random Forest which parallelises independent trees, XGBoost is sequential and iterative — it literally "boosts" performance by learning from mistakes. `learning_rate` controls how aggressively each new tree corrects errors (smaller is safer, needs more trees). `max_depth` keeps individual trees shallow to prevent overfitting. Subsample adds randomness for robustness. Usually the strongest performer on tabular regression benchmarks.

### SVR (Support Vector Regression, rbf kernel)
Finds a tube of width ±epsilon around the predictions such that data points inside the tube incur no penalty. Only points outside the tube (support vectors) influence the model boundary. The `rbf` (radial basis function) kernel maps inputs into a high-dimensional space, allowing SVR to learn non-linear relationships. `C` balances margin width versus penalty for violations; `gamma` controls how local the kernel's influence is. SVR scales poorly in feature count and can underperform tree methods on large lag-feature matrices.

---

## 3. Metric Definitions

| Metric | Formula | Plain English |
|---|---|---|
| **R²** | 1 − SS_res / SS_tot | Fraction of C4 variance explained by the model. R²=1 is perfect; R²=0 means the model is no better than guessing the mean; negative R² means the model is worse than the mean. |
| **RMSE** | √(mean((ŷ − y)²)) | Root mean squared error. In the same units as C4 (normalised 0-1 here). Penalises large errors heavily. Lower is better. |
| **MAE** | mean(|ŷ − y|) | Mean absolute error. More robust to outliers than RMSE. Easier to explain: "on average the model is off by X normalised units." |
| **Bias** | mean(ŷ − y) | Systematic offset. Positive = model over-predicts on average. Zero bias is desirable; sustained bias means the model consistently errs in one direction. |

---

## 4. Why Is the Split Chronological?

Process measurements are recorded in time order and have **temporal autocorrelation** — the C4 value at time t is strongly correlated with C4 at t-1, t-2, etc. If we randomly shuffle and split, training samples from time t+1 land in the training set while time t lands in the test set. The model effectively "sees the future" during training, and the test score becomes artificially inflated by this information leakage. A chronological split (first 60% train, next 20% val, last 20% test) is the only way to honestly measure how well the model would perform in real-time deployment.

---

## 5. Why Is Persistence the Reference Baseline?

The persistence baseline predicts: **C4(t) ≈ C4(t−4)**, i.e., the most recent lab assay available 4 steps ago. This is exactly what an operator would do manually if no soft sensor existed — "repeat the last measurement until the next one arrives." Any ML model that cannot beat this simple heuristic is adding no value over current practice. In the causal setting, the persistence baseline achieves Test R² ≈ 0.92, which the causal ML models cannot beat. This is reported honestly — it is not a failure of ML; it means the process variables alone are insufficient predictors of C4 without lab feedback.

---

## 6. Why SHAP and Importance ≠ Causality

SHAP values and feature importance scores measure **how much the model's prediction changes when a feature value changes**, averaged over the training data. They answer: "what does the model use?" — not "what physically causes C4 to change?"

For example, if SHAP shows that u6 (bottom temperature) has the highest importance, this means the model has learned to rely heavily on u6 for its predictions. It does **not** prove that adjusting u6 would directly reduce C4. There may be confounding variables, measurement delays, and feedback loops that the data-driven model cannot disentangle. Causality requires controlled experiments (e.g., designed plant trials or Aspen Plus simulations with validated thermodynamic models).

Additionally, highly correlated features (u6 and u7, r=0.99) split their combined predictive credit arbitrarily between them. The grouped importance (summing u6 and u7's contributions together) is more informative than looking at either individually.

---

## 7. The Two Feature Variants — Key Exam Points

| | **Causal** | **Autoregressive** |
|---|---|---|
| **Inputs** | u1-u7 lags, rolling stats, delta | All causal features + C4 at lags 4, 5, 6 |
| **When usable** | Always (lab can be offline) | Needs a lab result from ≥4 steps ago |
| **Performance** | Test R² ≈ 0.1–0.45 | Test R² ≈ 0.74–0.99 |
| **Beats persistence?** | No | Yes (tree models and linear models) |
| **Deployable for GA?** | Yes | No (can't set past lab values) |

---

## 8. The Genetic Algorithm — Key Exam Points

- **Why GA not gradient descent?** The causal model (especially RF/XGBoost) is not differentiable with respect to inputs, so gradient-based optimisation doesn't apply. GAs are derivative-free, population-based heuristics well suited to such "black-box" objectives.
- **Steady-state assumption:** Each candidate operating point is treated as if the column has been running at those exact conditions forever (all lags equal the candidate value, rolling std = 0). This is the only sensible interpretation for a steady-state design problem.
- **Distance penalty:** Prevents the GA from recommending operating conditions outside the training data envelope, where the model cannot be trusted to extrapolate.
- **5 seeds for stability:** Running with 5 different random initialisations and reporting mean±std of the optimum shows whether the result is consistent (low std) or fragile (high std).
- **Limitations:** The result is only as reliable as the causal model. With Test R² ≈ 0.45, the optimum is indicative and needs validation via Aspen Plus and plant trials before implementation.
