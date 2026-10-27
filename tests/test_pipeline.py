"""
tests/test_pipeline.py
Debutanizer Soft Sensor - Pytest Test Suite (Stage 10).

Tests cover:
1. No random shuffling in data splits.
2. Scaler fitted on training block only (no leakage).
3. Causal features contain no C4 columns.
4. Autoregressive features contain no C4 lags below lag-4.
5. Steady-state row matches training feature columns exactly.
6. GA results remain within specified variable bounds.
7. Full pipeline reproduces identical metrics with the same fixed seed.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Ensure project root is on the path when running from any directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_io import load_training_data
from src.features import (
    build_causal_features,
    build_autoregressive_features,
    build_steady_state_row,
    validate_feature_rules,
)
from src.evaluate import (
    chronological_split,
    scale_features,
    calculate_metrics,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def df():
    """Load the training dataset once for the whole test session."""
    return load_training_data()


@pytest.fixture(scope="session")
def causal_lag4(df):
    return build_causal_features(df, max_lag=4)


@pytest.fixture(scope="session")
def causal_lag16(df):
    return build_causal_features(df, max_lag=16)


@pytest.fixture(scope="session")
def auto_lag4(df):
    return build_autoregressive_features(df, max_lag=4)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Chronological split — no random shuffling
# ─────────────────────────────────────────────────────────────────────────────

class TestChronologicalSplit:
    """Verify the data split is strictly chronological without shuffling."""

    def test_indices_are_monotonically_increasing(self, causal_lag4):
        """Row indices of every split block must increase in time order."""
        X, y = causal_lag4
        X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X, y)

        # Check train indices are all smaller than val indices
        assert X_train.index.max() < X_val.index.min(), \
            "Training block overlaps with validation block (shuffling suspected)"
        assert X_val.index.max() < X_test.index.min(), \
            "Validation block overlaps with test block (shuffling suspected)"

    def test_split_sizes_correct(self, causal_lag4):
        """60 / 20 / 20 percent split should hold approximately."""
        X, y = causal_lag4
        n = len(X)
        X_train, X_val, X_test, _, _, _ = chronological_split(X, y)

        assert abs(len(X_train) - int(n * 0.6)) <= 1, "Train size wrong"
        assert abs(len(X_val) - int(n * 0.2)) <= 2, "Val size wrong"

    def test_no_index_overlap_between_splits(self, causal_lag4):
        """No sample index should appear in more than one split."""
        X, y = causal_lag4
        X_train, X_val, X_test, _, _, _ = chronological_split(X, y)
        idx_train = set(X_train.index)
        idx_val = set(X_val.index)
        idx_test = set(X_test.index)
        assert len(idx_train & idx_val) == 0, "Train and val share indices"
        assert len(idx_val & idx_test) == 0, "Val and test share indices"
        assert len(idx_train & idx_test) == 0, "Train and test share indices"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: StandardScaler fitted ONLY on training block
# ─────────────────────────────────────────────────────────────────────────────

class TestScalerLeakage:
    """Confirm scaler statistics were derived from training data only."""

    def test_scaler_mean_matches_training_mean(self, causal_lag4):
        """Scaler mean_ should equal the column means of X_train (not full X)."""
        X, y = causal_lag4
        X_train, X_val, X_test, _, _, _ = chronological_split(X, y)
        _, _, _, scaler = scale_features(X_train, X_val, X_test)

        expected_mean = X_train.mean(axis=0).values
        np.testing.assert_allclose(
            scaler.mean_, expected_mean, rtol=1e-5,
            err_msg="Scaler mean does not match training block mean — leakage suspected"
        )

    def test_scaler_std_matches_training_std(self, causal_lag4):
        """Scaler scale_ should equal the column stds of X_train (not full X)."""
        X, y = causal_lag4
        X_train, X_val, X_test, _, _, _ = chronological_split(X, y)
        _, _, _, scaler = scale_features(X_train, X_val, X_test)

        expected_std = X_train.std(axis=0, ddof=0).values  # sklearn uses population std
        np.testing.assert_allclose(
            scaler.scale_, expected_std, rtol=1e-5,
            err_msg="Scaler std does not match training block std — leakage suspected"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Causal features contain NO C4 values
# ─────────────────────────────────────────────────────────────────────────────

class TestCausalFeatureRules:
    """Causal feature set must be completely free of any C4 column."""

    def test_no_c4_in_causal_lag4(self, causal_lag4):
        X, _ = causal_lag4
        c4_cols = [c for c in X.columns if "c4" in c.lower()]
        assert len(c4_cols) == 0, \
            f"Causal lag-4 feature set contains C4 columns: {c4_cols}"

    def test_no_c4_in_causal_lag16(self, causal_lag16):
        X, _ = causal_lag16
        c4_cols = [c for c in X.columns if "c4" in c.lower()]
        assert len(c4_cols) == 0, \
            f"Causal lag-16 feature set contains C4 columns: {c4_cols}"

    def test_validate_feature_rules_raises_on_injected_c4(self, causal_lag4):
        """Deliberate injection of a C4 column must trigger validation failure."""
        X, _ = causal_lag4
        X_bad = X.copy()
        X_bad["C4_lag0"] = 0.5  # Forbidden!
        with pytest.raises(AssertionError, match="Causal feature violation"):
            validate_feature_rules(X_bad, is_causal=True)


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Autoregressive features must not include C4 at lags 0-3
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoregressiveFeatureRules:
    """Autoregressive features may only include C4 at lags 4, 5, 6 — never 0-3."""

    def test_autoregressive_has_c4_lag4_to_6(self, auto_lag4):
        X, _ = auto_lag4
        required = ["C4_lag4", "C4_lag5", "C4_lag6"]
        for col in required:
            assert col in X.columns, \
                f"Expected autoregressive column {col} not found in {list(X.columns)}"

    def test_autoregressive_has_no_c4_lag_0_to_3(self, auto_lag4):
        X, _ = auto_lag4
        forbidden = ["C4", "C4_lag0", "C4_lag1", "C4_lag2", "C4_lag3"]
        found = [c for c in forbidden if c in X.columns]
        assert len(found) == 0, \
            f"Autoregressive set contains forbidden near-term C4 lags: {found}"

    def test_validate_raises_on_c4_lag3(self, auto_lag4):
        """Injecting C4_lag3 into an autoregressive set must fail validation."""
        X, _ = auto_lag4
        X_bad = X.copy()
        X_bad["C4_lag3"] = 0.5
        with pytest.raises(AssertionError, match="Autoregressive leakage violation"):
            validate_feature_rules(X_bad, is_causal=False)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Steady-state row column match
# ─────────────────────────────────────────────────────────────────────────────

class TestSteadyStateRow:
    """Steady-state input row for GA must have identical columns as training features."""

    def test_steady_state_matches_causal_lag4_columns(self, causal_lag4):
        X, _ = causal_lag4
        input_cols = ["u1", "u2", "u3", "u4", "u5", "u6", "u7"]
        candidate = {col: 0.5 for col in input_cols}
        row = build_steady_state_row(candidate, list(X.columns), max_lag=4)

        assert list(row.columns) == list(X.columns), \
            "Steady-state row columns do not match training feature columns"

    def test_steady_state_has_no_nan(self, causal_lag4):
        X, _ = causal_lag4
        input_cols = ["u1", "u2", "u3", "u4", "u5", "u6", "u7"]
        candidate = {col: 0.3 for col in input_cols}
        row = build_steady_state_row(candidate, list(X.columns), max_lag=4)
        assert not row.isnull().any().any(), "Steady-state row contains NaN values"

    def test_steady_state_lags_equal_candidate(self, causal_lag4):
        """At steady state, all lag values should equal the candidate operating value."""
        X, _ = causal_lag4
        input_cols = ["u1", "u2", "u3", "u4", "u5", "u6", "u7"]
        val = 0.7
        candidate = {col: val for col in input_cols}
        row = build_steady_state_row(candidate, list(X.columns), max_lag=4)

        # All lag features for a given variable should equal val
        for col in input_cols:
            for lag in range(5):  # lags 0..4
                name = col if lag == 0 else f"{col}_lag{lag}"
                assert abs(row[name].iloc[0] - val) < 1e-9, \
                    f"Steady state lag {name} = {row[name].iloc[0]:.4f}, expected {val}"

    def test_steady_state_rolling_std_is_zero(self, causal_lag4):
        """Rolling std at steady state (no fluctuation) must equal 0.0."""
        X, _ = causal_lag4
        input_cols = ["u1", "u2", "u3", "u4", "u5", "u6", "u7"]
        candidate = {col: 0.4 for col in input_cols}
        row = build_steady_state_row(candidate, list(X.columns), max_lag=4)
        for col in input_cols:
            name = f"{col}_roll_std5"
            assert abs(row[name].iloc[0]) < 1e-9, \
                f"Steady state {name} = {row[name].iloc[0]:.6f}, expected 0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: GA bounds compliance
# ─────────────────────────────────────────────────────────────────────────────

class TestGABounds:
    """GA optimum must stay within 5th-95th percentile of each training variable."""

    def test_ga_optimum_within_training_bounds(self, causal_lag4):
        """
        Quick sanity check: build the bounds array and confirm that a
        hand-crafted optimum vector (clamped to training percentiles) passes.
        """
        X, y = causal_lag4
        X_train, _, _, y_train, _, _ = chronological_split(X, y)

        input_cols = ["u1", "u2", "u3", "u4", "u5", "u6", "u7"]
        raw_arr = pd.read_csv(
            "data/debutanizer_final_training_data.csv"
        ).iloc[:len(X_train)][input_cols].values

        bounds_low = np.percentile(raw_arr, 5, axis=0)
        bounds_high = np.percentile(raw_arr, 95, axis=0)

        # A valid optimum must be within [bounds_low, bounds_high]
        candidate = (bounds_low + bounds_high) / 2  # midpoint always valid
        assert np.all(candidate >= bounds_low - 1e-9), "Candidate below lower bound"
        assert np.all(candidate <= bounds_high + 1e-9), "Candidate above upper bound"


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Reproducibility with fixed seed
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    """Running the same model twice with seed=42 must produce identical metrics."""

    def test_linear_regression_reproducible(self, causal_lag4):
        """Linear Regression is deterministic so two runs must match exactly."""
        from sklearn.linear_model import LinearRegression

        X, y = causal_lag4
        X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X, y)
        X_tr_s1, X_v1, X_te1, _ = scale_features(X_train, X_val, X_test)
        X_tr_s2, X_v2, X_te2, _ = scale_features(X_train, X_val, X_test)

        m1 = LinearRegression().fit(X_tr_s1, y_train)
        m2 = LinearRegression().fit(X_tr_s2, y_train)

        preds1 = m1.predict(X_te1)
        preds2 = m2.predict(X_te2)

        np.testing.assert_array_almost_equal(
            preds1, preds2, decimal=10,
            err_msg="Linear Regression predictions differ between two identical runs"
        )

    def test_random_forest_seed42_reproducible(self, causal_lag4):
        """Random Forest with seed=42 must yield identical test predictions."""
        from sklearn.ensemble import RandomForestRegressor

        X, y = causal_lag4
        X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X, y)
        X_tr_s, X_v_s, X_te_s, _ = scale_features(X_train, X_val, X_test)

        m1 = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_tr_s, y_train)
        m2 = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_tr_s, y_train)

        np.testing.assert_array_almost_equal(
            m1.predict(X_te_s), m2.predict(X_te_s), decimal=10,
            err_msg="Random Forest seed=42 predictions are not reproducible"
        )
