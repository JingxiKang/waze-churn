"""
src/data_loader.py
------------------
Data loading, cleaning, and feature engineering for the Waze churn pipeline.

Usage
-----
    from src.data_loader import load_and_prepare

    X, y, feature_cols = load_and_prepare("./data/waze_dataset.csv")
"""

import pandas as pd
import numpy as np


# Columns that must be present in the raw CSV for the pipeline to work
REQUIRED_COLUMNS = [
    "ID", "label", "device",
    "sessions", "drives", "activity_days", "driving_days",
    "driven_km_drives",
    "total_navigations_fav1", "total_navigations_fav2",
]

# Columns dropped before constructing the feature matrix
DROP_COLS = ["ID", "label", "device", "label_num"]


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Read the raw Waze CSV from *filepath* and return it as a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to waze_dataset.csv (relative or absolute).

    Returns
    -------
    pd.DataFrame
        Raw DataFrame exactly as read from disk.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If any required columns are missing from the file.
    """
    df = pd.read_csv(filepath)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following required columns are missing from the dataset: {missing_cols}"
        )

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that cannot be used for supervised learning.

    Steps
    -----
    1. Drop rows where ``label`` is NaN — these have no target value and
       cannot be imputed.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame returned by :func:`load_raw_data`.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with unlabelled rows removed.
    """
    n_before = len(df)
    df_clean = df.dropna(subset=["label"]).copy()
    n_dropped = n_before - len(df_clean)

    if n_dropped > 0:
        print(f"[clean_data] Dropped {n_dropped} unlabelled rows. "
              f"Remaining: {len(df_clean)}")

    return df_clean


def engineer_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features and encode categorical columns.

    New columns added
    -----------------
    km_per_drive        : driven_km_drives / (drives + 1)
    sessions_per_day    : sessions / (activity_days + 1)
    drives_per_day      : drives / (driving_days + 1)
    pct_days_active     : activity_days / 30
    total_nav_fav       : total_navigations_fav1 + total_navigations_fav2
    is_iphone           : 1 if device == 'iPhone', else 0  (binary encode)
    label_num           : 1 if label == 'churned', else 0  (target)

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned DataFrame from :func:`clean_data`.

    Returns
    -------
    pd.DataFrame
        Copy of *df_clean* with all new columns appended.
    """
    df_feat = df_clean.copy()

    # Ratio features — +1 avoids division by zero
    df_feat["km_per_drive"] = (
        df_feat["driven_km_drives"] / (df_feat["drives"] + 1)
    )
    df_feat["sessions_per_day"] = (
        df_feat["sessions"] / (df_feat["activity_days"] + 1)
    )
    df_feat["drives_per_day"] = (
        df_feat["drives"] / (df_feat["driving_days"] + 1)
    )
    df_feat["pct_days_active"] = df_feat["activity_days"] / 30

    # Aggregate navigation favourites
    df_feat["total_nav_fav"] = (
        df_feat["total_navigations_fav1"] + df_feat["total_navigations_fav2"]
    )

    # Binary-encode device type
    df_feat["is_iphone"] = (df_feat["device"] == "iPhone").astype(int)

    # Encode target
    df_feat["label_num"] = (df_feat["label"] == "churned").astype(int)

    return df_feat


def get_X_y(df_feat: pd.DataFrame):
    """
    Split a fully-engineered DataFrame into feature matrix *X* and target *y*.

    Parameters
    ----------
    df_feat : pd.DataFrame
        DataFrame returned by :func:`engineer_features`.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (all engineered columns, minus bookkeeping columns).
    y : pd.Series
        Binary target (0 = retained, 1 = churned).
    feature_cols : list[str]
        Ordered list of column names in *X*.
    """
    feature_cols = [c for c in df_feat.columns if c not in DROP_COLS]
    X = df_feat[feature_cols]
    y = df_feat["label_num"]
    return X, y, feature_cols


def load_and_prepare(filepath: str):
    """
    Convenience wrapper: load → clean → engineer → split.

    Parameters
    ----------
    filepath : str
        Path to waze_dataset.csv.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    feature_cols : list[str]
    """
    df_raw = load_raw_data(filepath)
    df_clean = clean_data(df_raw)
    df_feat = engineer_features(df_clean)
    X, y, feature_cols = get_X_y(df_feat)

    print(f"[load_and_prepare] Ready — X: {X.shape}, "
          f"churn rate: {y.mean():.3f}")
    return X, y, feature_cols
