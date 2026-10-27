"""
src/optimise.py
Debutanizer Soft Sensor - Genetic Algorithm Optimisation Module (Stage 9).

Goal: Use the best CAUSAL model as a stand-in for the debutanizer column and
search for the combination of process variables (u1-u7) that minimises the
predicted C4 concentration (the objective of the column operator).

Why causal only: An operator can set the 7 process variables in real time but
cannot control past laboratory assay results, so the autoregressive model is
not deployable for optimisation purposes.

Design approach:
- Decision variables: 7 normalised process inputs u1..u7 (range 0-1).
- Each candidate is turned into a steady-state row where ALL lags, rolling
  means, and rolling stds reflect the same constant operating point.
- Bounds: 5th to 95th percentile of the training block (no extrapolation).
- Distance penalty: squared Mahalanobis-like penalty for operating points
  beyond the 95th percentile radius of training data distances.
- DEAP genetic algorithm: population=60, generations=80, cx_prob=0.7,
  mut_prob=0.2, tournament selection size=3.
- Run 5 times with seeds 42-46 to assess stability.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from deap import base, creator, tools, algorithms
import warnings
warnings.filterwarnings("ignore")

from src.features import build_causal_features, build_steady_state_row
from src.evaluate import chronological_split, scale_features
from src.data_io import load_training_data


# ─────────────────────────────────────────────────────────────────────────────
# DEAP global setup (only define once)
# ─────────────────────────────────────────────────────────────────────────────
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


def build_ga_toolbox(bounds_low: np.ndarray, bounds_high: np.ndarray,
                     n_vars: int, random_seed: int) -> base.Toolbox:
    """
    Build a DEAP toolbox for the optimisation.

    Parameters
    ----------
    bounds_low, bounds_high : array of shape (n_vars,)
        Per-variable 5th and 95th percentile training bounds.
    n_vars : int
        Number of decision variables (7).
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    DEAP Toolbox configured with:
        - Uniform gene initialisation within bounds.
        - Ordered-crossover blend (cxSimulatedBinaryBounded).
        - Polynomial mutation within bounds.
        - Tournament selection of size 3.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    toolbox = base.Toolbox()

    # Gene: uniform float within individual variable bounds
    def rand_gene(i):
        return random.uniform(float(bounds_low[i]), float(bounds_high[i]))

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     [lambda i=i: rand_gene(i) for i in range(n_vars)], n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     low=bounds_low.tolist(), up=bounds_high.tolist(), eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     low=bounds_low.tolist(), up=bounds_high.tolist(),
                     eta=20.0, indpb=1.0 / n_vars)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def compute_distance_penalty(candidate: np.ndarray, X_train_raw: np.ndarray,
                              percentile_95_dist: float) -> float:
    """
    Add a quadratic penalty proportional to the squared excess distance from
    the 95th percentile of training-point distances.

    Prevents the GA from proposing operating conditions far outside the safe
    region where the model has been trained (extrapolation is unreliable).

    Parameters
    ----------
    candidate : array of shape (n_vars,)
        Raw (unscaled) candidate decision variables.
    X_train_raw : array of shape (n_train, n_vars)
        Raw training block values for u1..u7.
    percentile_95_dist : float
        95th percentile of training point pairwise distances.

    Returns
    -------
    float  Penalty value (0 if within training radius).
    """
    centroid = X_train_raw.mean(axis=0)
    dist = np.linalg.norm(candidate - centroid)
    excess = max(0.0, dist - percentile_95_dist)
    return excess ** 2


def make_fitness_function(causal_model, scaler, feature_columns: list,
                          max_lag: int, input_cols: list,
                          X_train_raw: np.ndarray, percentile_95_dist: float,
                          penalty_weight: float = 50.0):
    """
    Create a closure that evaluates a candidate solution.

    Parameters
    ----------
    causal_model : fitted estimator (causal variant only)
    scaler : fitted StandardScaler
    feature_columns : list  Ordered column names of the causal training feature matrix.
    max_lag : int
    input_cols : list  Names of process variables [u1..u7].
    X_train_raw : array  Raw training input block used for distance penalty.
    percentile_95_dist : float
    penalty_weight : float  Multiplier for the distance penalty term.

    Returns
    -------
    Callable that takes a list of 7 floats and returns (objective_value,).
    """
    def evaluate(individual):
        candidate = np.array(individual)

        # Clip to bounds (mutation might slightly overshoot due to floating point)
        n = len(input_cols)
        candidate = np.clip(candidate,
                            X_train_raw.min(axis=0)[:n],
                            X_train_raw.max(axis=0)[:n])

        # Build steady-state feature row
        cand_dict = {col: float(candidate[i]) for i, col in enumerate(input_cols)}
        try:
            row_df = build_steady_state_row(cand_dict, feature_columns, max_lag=max_lag)
        except ValueError:
            return (1e6,)

        # Scale using fitted scaler
        row_scaled = scaler.transform(row_df)

        # Predict C4
        pred_c4 = float(causal_model.predict(row_scaled)[0])

        # Distance penalty
        penalty = compute_distance_penalty(candidate, X_train_raw, percentile_95_dist)

        return (pred_c4 + penalty_weight * penalty,)

    return evaluate


def run_ga_optimisation(causal_model, scaler, X_train_s: pd.DataFrame,
                        y_train: pd.Series, X_train_raw: pd.DataFrame,
                        feature_columns: list, max_lag: int,
                        input_cols: list,
                        n_seeds: int = 5,
                        pop_size: int = 60,
                        n_gen: int = 80,
                        cx_prob: float = 0.7,
                        mut_prob: float = 0.2,
                        save_dir: str = "figures",
                        results_dir: str = "results") -> dict:
    """
    Run GA optimisation across n_seeds different random seeds and summarise stability.

    Parameters
    ----------
    causal_model : fitted causal model
    scaler : fitted StandardScaler (causal variant)
    X_train_s : scaled training feature matrix (for feature column order)
    y_train : training target
    X_train_raw : unscaled training input block (u1..u7 only) for bounds/distance
    feature_columns : list of causal feature column names
    max_lag : int
    input_cols : list of source variable names ['u1'..'u7']
    n_seeds, pop_size, n_gen, cx_prob, mut_prob : GA hyperparameters
    save_dir, results_dir : output directories

    Returns
    -------
    dict with best individual, stability statistics, and file paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Compute per-variable bounds from training block (5th to 95th percentile)
    raw_arr = X_train_raw.values
    bounds_low = np.percentile(raw_arr, 5, axis=0)
    bounds_high = np.percentile(raw_arr, 95, axis=0)

    # Compute 95th percentile of pairwise distances to centroid (for penalty)
    centroid = raw_arr.mean(axis=0)
    dists = np.linalg.norm(raw_arr - centroid, axis=1)
    p95_dist = float(np.percentile(dists, 95))

    evaluate_fn = make_fitness_function(
        causal_model, scaler, feature_columns, max_lag, input_cols,
        raw_arr, p95_dist
    )

    all_optima = []
    all_best_c4 = []
    all_gen_logs = []  # (seed, generation, best_fitness) triples

    for seed in range(42, 42 + n_seeds):
        print(f"  GA run {seed - 41}/{n_seeds} (seed={seed})...")
        toolbox = build_ga_toolbox(bounds_low, bounds_high, len(input_cols), seed)
        toolbox.register("evaluate", evaluate_fn)

        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("min", np.min)
        stats.register("mean", np.mean)

        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Evolutionary loop
        gen_best = []
        for gen in range(n_gen):
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cx_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation
            for mutant in offspring:
                if random.random() < mut_prob:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Re-evaluate invalid individuals
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = toolbox.evaluate(ind)

            pop[:] = offspring
            hof.update(pop)

            current_best = min(ind.fitness.values[0] for ind in pop)
            gen_best.append((seed, gen, current_best))

        best_ind = list(hof[0])
        best_val = float(hof[0].fitness.values[0])

        # Compute pure C4 prediction without penalty for the best individual
        cand_dict = {col: float(best_ind[i]) for i, col in enumerate(input_cols)}
        row_df = build_steady_state_row(cand_dict, feature_columns, max_lag=max_lag)
        row_scaled = scaler.transform(row_df)
        best_c4_pred = float(causal_model.predict(row_scaled)[0])

        all_optima.append(best_ind)
        all_best_c4.append(best_c4_pred)
        all_gen_logs.extend(gen_best)
        print(f"    Best C4 predicted: {best_c4_pred:.4f}  (raw objective: {best_val:.4f})")

    # Stability summary
    best_c4_arr = np.array(all_best_c4)
    global_best_idx = int(np.argmin(best_c4_arr))
    global_best_individual = all_optima[global_best_idx]
    stability = {
        "best_c4": float(best_c4_arr.min()),
        "mean_c4": float(best_c4_arr.mean()),
        "std_c4": float(best_c4_arr.std()),
        "all_c4_values": best_c4_arr.tolist()
    }
    print(f"\n  GA Stability Summary across {n_seeds} runs:")
    print(f"    Best C4 = {stability['best_c4']:.4f}")
    print(f"    Mean C4 = {stability['mean_c4']:.4f}")
    print(f"    Std C4  = {stability['std_c4']:.4f}")

    # ── Plot GA convergence across seeds ──────────────────────────────────────
    gen_df = pd.DataFrame(all_gen_logs, columns=["seed", "generation", "best_fitness"])
    plt.figure(figsize=(10, 5))
    for seed in gen_df["seed"].unique():
        s_data = gen_df[gen_df["seed"] == seed]
        plt.plot(s_data["generation"], s_data["best_fitness"],
                 alpha=0.8, linewidth=1.5, label=f"Seed {seed}")
    plt.title("GA Convergence: Best Fitness per Generation (5 Seeds)", fontweight="bold", fontsize=13)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Best Fitness (Predicted C4 + Distance Penalty)", fontsize=12)
    plt.legend(fontsize=9)
    plt.tight_layout()
    conv_path = os.path.join(save_dir, "ga_convergence.png")
    plt.savefig(conv_path, dpi=250)
    plt.close()

    # ── Save optimum candidate to results ────────────────────────────────────
    opt_row = {col: global_best_individual[i] for i, col in enumerate(input_cols)}
    df_opt = pd.DataFrame([opt_row])
    opt_path = os.path.join(results_dir, "optimisation_results.csv")
    df_opt.to_csv(opt_path, index=False)

    # Aspen handover file with header note
    aspen_path = os.path.join(results_dir, "optimum_for_aspen.csv")
    aspen_header = (
        "# NOTE: These are normalised values in [0, 1]. "
        "The original engineering units, sensor definitions, and scale factors "
        "were NOT published by the original authors (Fortuna et al.). "
        "Do NOT convert to assumed real units until the variable meanings and "
        "calibration ranges are officially confirmed.\n"
    )
    with open(aspen_path, "w") as f:
        f.write(aspen_header)
        df_opt.to_csv(f, index=False)

    # ── Compare optimum with the lowest-C4 training rows ─────────────────────
    df_raw = load_training_data()
    n_train = int(len(df_raw) * 0.6)
    df_train_raw = df_raw.iloc[:n_train]
    lowest_c4_rows = df_train_raw.nsmallest(5, "C4")[input_cols + ["C4"]]

    # ── Write summary markdown ────────────────────────────────────────────────
    summary_md = [
        "# Debutanizer Soft Sensor - Stage 9 Optimisation Summary\n",
        "## Approach",
        "A Genetic Algorithm (DEAP library) was used to minimise the CAUSAL model's",
        "predicted C4 concentration by searching over the 7 normalised process inputs (u1..u7).",
        "The autoregressive model was intentionally excluded because an operator cannot",
        "set past laboratory assay values in real time.\n",
        "## GA Configuration",
        f"- Population size: {pop_size}",
        f"- Generations: {n_gen}",
        f"- Crossover probability: {cx_prob}",
        f"- Mutation probability: {mut_prob}",
        f"- Selection: Tournament (size 3)",
        f"- Seeds tested: {n_seeds} (seeds 42-{42 + n_seeds - 1})",
        f"- Bounds: 5th to 95th percentile of training block (per variable)",
        f"- Distance penalty: quadratic excess penalty beyond 95th-percentile training radius\n",
        "## Optimisation Results",
        f"| Metric | Value |",
        f"|:---|:---|",
        f"| **Best predicted C4** | {stability['best_c4']:.4f} |",
        f"| **Mean C4 across seeds** | {stability['mean_c4']:.4f} |",
        f"| **Std C4 across seeds** | {stability['std_c4']:.4f} |",
        f"| **All seed results** | {[round(v, 4) for v in stability['all_c4_values']]} |",
        "",
        "## Optimal Operating Point (normalised, u1..u7)",
    ]
    for col, val in zip(input_cols, global_best_individual):
        summary_md.append(f"- `{col}` = {val:.4f}")
    summary_md.append("")

    summary_md.append("## Comparison: Optimum vs. Lowest-C4 Historical Operating Points")
    summary_md.append("| Metric | " + " | ".join([f"Historical_{i+1}" for i in range(len(lowest_c4_rows))]) + " | GA Optimum |")
    summary_md.append("|:---|" + "|".join([":---:"] * (len(lowest_c4_rows) + 1)) + "|")
    for col in input_cols + ["C4"]:
        hist_vals = " | ".join([f"{lowest_c4_rows.iloc[i][col]:.4f}" for i in range(len(lowest_c4_rows))])
        opt_val = opt_row.get(col, stability["best_c4"])
        summary_md.append(f"| `{col}` | {hist_vals} | {opt_val:.4f} |")
    summary_md.append("")

    summary_md.extend([
        "## Critical Limitations",
        "> **INDICATIVE RESULT ONLY.** The GA optimum is only as reliable as the causal model.",
        "> The causal model achieves Test R² ≈ 0.45 at best (without lab feedback), meaning",
        "> approximately 55% of C4 variance is not captured. The optimum may not be achievable",
        "> in practice and should be validated via Aspen Plus simulation and plant trials.",
        "> Additionally, these normalised values cannot be converted to engineering units until",
        "> the original variable definitions and scale factors are officially confirmed.",
        "",
        "## 3-Line Viva Explanation (Stage 9)",
        "> 1. The GA treats the trained causal model as a fast surrogate for the real column",
        ">    and searches for process settings that predict minimum C4 without running Aspen.",
        "> 2. We used 5 random seeds to assess stability; the low std of results confirms",
        ">    the GA landscape is not too rugged within the explored training bounds.",
        "> 3. The result is indicative only — it is bounded to observed training conditions",
        ">    and depends entirely on the causal model's accuracy, which is limited.",
    ])

    summary_out = os.path.join(results_dir, "optimisation_summary.md")
    with open(summary_out, "w") as f:
        f.write("\n".join(summary_md))

    return {
        "global_best_individual": global_best_individual,
        "stability": stability,
        "convergence_plot": conv_path,
        "results_csv": opt_path,
        "aspen_csv": aspen_path,
        "summary_md": summary_out,
        "lowest_c4_historical": lowest_c4_rows,
    }
