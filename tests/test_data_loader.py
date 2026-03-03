"""
tests/test_data_loader.py
-------------------------
Unit tests for src/data_loader.py.

Run with:
    pytest tests/test_data_loader.py -v
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Allow imports from the project root regardless of where pytest is invoked
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import (
    load_raw_data,
    clean_data,
    engineer_features,
    get_X_y,
    load_and_prepare,
    DROP_COLS,
    REQUIRED_COLUMNS,
)


# ---------------------------------------------------------------------------
# Fixtures — small synthetic DataFrames that mimic the real dataset
# ---------------------------------------------------------------------------

def _make_minimal_df(n=10, include_nulls=False) -> pd.DataFrame:
    """Return a minimal DataFrame with all required columns."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "ID": range(n),
            "label": rng.choice(["churned", "retained"], size=n),
            "device": rng.choice(["iPhone", "Android"], size=n),
            "sessions": rng.integers(1, 100, size=n).astype(float),
            "drives": rng.integers(1, 80, size=n).astype(float),
            "activity_days": rng.integers(1, 30, size=n).astype(float),
            "driving_days": rng.integers(1, 30, size=n).astype(float),
            "driven_km_drives": rng.uniform(10, 5000, size=n),
            "total_navigations_fav1": rng.integers(0, 50, size=n).astype(float),
            "total_navigations_fav2": rng.integers(0, 50, size=n).astype(float),
            # Extra numeric column that should pass through as a feature
            "duration_minutes_drives": rng.uniform(10, 3000, size=n),
        }
    )
    if include_nulls:
        # Inject NaN labels into 3 rows
        df.loc[[2, 5, 8], "label"] = np.nan
    return df


@pytest.fixture
def sample_df():
    return _make_minimal_df()


@pytest.fixture
def sample_df_with_nulls():
    return _make_minimal_df(include_nulls=True)


# ---------------------------------------------------------------------------
# 1. load_raw_data
# ---------------------------------------------------------------------------

class TestLoadRawData:

    def test_loads_real_csv(self, tmp_path):
        """Writing a minimal CSV to disk and reading it back works."""
        df = _make_minimal_df()
        csv_path = tmp_path / "waze_dataset.csv"
        df.to_csv(csv_path, index=False)

        result = load_raw_data(str(csv_path))

        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_raw_data(str(tmp_path / "nonexistent.csv"))

    def test_raises_on_missing_required_column(self, tmp_path):
        """A CSV that drops a required column should raise ValueError."""
        df = _make_minimal_df().drop(columns=["label"])
        csv_path = tmp_path / "bad.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="label"):
            load_raw_data(str(csv_path))

    def test_all_required_columns_present_after_load(self, tmp_path):
        df = _make_minimal_df()
        csv_path = tmp_path / "waze_dataset.csv"
        df.to_csv(csv_path, index=False)

        result = load_raw_data(str(csv_path))
        for col in REQUIRED_COLUMNS:
            assert col in result.columns, f"Missing column after load: {col}"


# ---------------------------------------------------------------------------
# 2. clean_data
# ---------------------------------------------------------------------------

class TestCleanData:

    def test_removes_null_labels(self, sample_df_with_nulls):
        result = clean_data(sample_df_with_nulls)
        assert result["label"].isnull().sum() == 0

    def test_correct_row_count_after_drop(self, sample_df_with_nulls):
        # sample_df_with_nulls has 3 NaN labels
        result = clean_data(sample_df_with_nulls)
        assert len(result) == len(sample_df_with_nulls) - 3

    def test_no_rows_dropped_when_no_nulls(self, sample_df):
        result = clean_data(sample_df)
        assert len(result) == len(sample_df)

    def test_returns_copy_not_inplace(self, sample_df_with_nulls):
        """clean_data should not mutate the input."""
        original_nulls = sample_df_with_nulls["label"].isnull().sum()
        clean_data(sample_df_with_nulls)
        assert sample_df_with_nulls["label"].isnull().sum() == original_nulls

    def test_only_label_column_affects_dropping(self, sample_df):
        """Introduce NaN in a non-target column — no rows should be dropped."""
        df = sample_df.copy()
        df.loc[0, "sessions"] = np.nan
        result = clean_data(df)
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# 3. engineer_features
# ---------------------------------------------------------------------------

class TestEngineerFeatures:

    def test_new_columns_exist(self, sample_df):
        result = engineer_features(sample_df)
        expected_new_cols = [
            "km_per_drive", "sessions_per_day", "drives_per_day",
            "pct_days_active", "total_nav_fav", "is_iphone", "label_num",
        ]
        for col in expected_new_cols:
            assert col in result.columns, f"Missing engineered column: {col}"

    def test_label_num_is_binary(self, sample_df):
        result = engineer_features(sample_df)
        assert set(result["label_num"].unique()).issubset({0, 1})

    def test_label_num_churned_maps_to_1(self, sample_df):
        result = engineer_features(sample_df)
        churned_rows = result[result["label"] == "churned"]
        assert (churned_rows["label_num"] == 1).all()

    def test_label_num_retained_maps_to_0(self, sample_df):
        result = engineer_features(sample_df)
        retained_rows = result[result["label"] == "retained"]
        assert (retained_rows["label_num"] == 0).all()

    def test_is_iphone_is_binary(self, sample_df):
        result = engineer_features(sample_df)
        assert set(result["is_iphone"].unique()).issubset({0, 1})

    def test_is_iphone_correct_encoding(self, sample_df):
        result = engineer_features(sample_df)
        iphone_mask = result["device"] == "iPhone"
        assert (result.loc[iphone_mask, "is_iphone"] == 1).all()
        assert (result.loc[~iphone_mask, "is_iphone"] == 0).all()

    def test_km_per_drive_no_division_by_zero(self, sample_df):
        """Rows with drives=0 should not produce NaN or inf."""
        df = sample_df.copy()
        df.loc[0, "drives"] = 0
        result = engineer_features(df)
        assert np.isfinite(result["km_per_drive"].values).all()

    def test_pct_days_active_range(self, sample_df):
        result = engineer_features(sample_df)
        assert (result["pct_days_active"] >= 0).all()
        # activity_days is capped at 30, so pct should not greatly exceed 1
        assert (result["pct_days_active"] <= 1.0 + 1e-9).all()

    def test_total_nav_fav_addition(self, sample_df):
        result = engineer_features(sample_df)
        expected = sample_df["total_navigations_fav1"] + sample_df["total_navigations_fav2"]
        pd.testing.assert_series_equal(
            result["total_nav_fav"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_input_not_mutated(self, sample_df):
        original_cols = set(sample_df.columns)
        engineer_features(sample_df)
        assert set(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# 4. get_X_y
# ---------------------------------------------------------------------------

class TestGetXy:

    def test_drop_cols_absent_from_X(self, sample_df):
        df_feat = engineer_features(sample_df)
        X, y, feature_cols = get_X_y(df_feat)
        for col in DROP_COLS:
            assert col not in X.columns, f"Leakage: '{col}' found in X"

    def test_y_is_binary_series(self, sample_df):
        df_feat = engineer_features(sample_df)
        _, y, _ = get_X_y(df_feat)
        assert isinstance(y, pd.Series)
        assert set(y.unique()).issubset({0, 1})

    def test_feature_cols_matches_X_columns(self, sample_df):
        df_feat = engineer_features(sample_df)
        X, y, feature_cols = get_X_y(df_feat)
        assert list(X.columns) == feature_cols

    def test_X_and_y_same_length(self, sample_df):
        df_feat = engineer_features(sample_df)
        X, y, _ = get_X_y(df_feat)
        assert len(X) == len(y)

    def test_no_object_dtypes_in_X(self, sample_df):
        """Feature matrix should be fully numeric — no raw strings."""
        df_feat = engineer_features(sample_df)
        X, _, _ = get_X_y(df_feat)
        object_cols = X.select_dtypes(include="object").columns.tolist()
        assert object_cols == [], f"Object-type columns in X: {object_cols}"


# ---------------------------------------------------------------------------
# 5. load_and_prepare (integration test)
# ---------------------------------------------------------------------------

class TestLoadAndPrepare:

    def test_end_to_end_shape(self, tmp_path):
        df = _make_minimal_df(n=20)
        csv_path = tmp_path / "waze_dataset.csv"
        df.to_csv(csv_path, index=False)

        X, y, feature_cols = load_and_prepare(str(csv_path))

        assert X.shape[0] == 20
        assert len(y) == 20
        assert len(feature_cols) == X.shape[1]

    def test_unlabelled_rows_excluded(self, tmp_path):
        df = _make_minimal_df(n=20, include_nulls=True)  # 3 NaN labels
        csv_path = tmp_path / "waze_dataset.csv"
        df.to_csv(csv_path, index=False)

        X, y, _ = load_and_prepare(str(csv_path))

        assert X.shape[0] == 17  # 20 - 3 nulls
