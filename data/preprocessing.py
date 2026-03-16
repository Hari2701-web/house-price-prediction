"""
data/preprocessing.py
────────────────────────────────────────────────────────────────
All feature-engineering and preprocessing steps live here.

Design decisions
  • We use a scikit-learn Pipeline so the exact same transforms
    are applied at training AND inference – no leakage possible.
  • StandardScaler normalises numeric features so that distance-
    based models (and regularised models) converge faster.
────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ── Feature catalogue ────────────────────────────────────────

FEATURE_COLS = [
    "MedInc",       # most predictive single feature
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

TARGET_COL = "MedHouseVal"   # median house value in $100k units


# ── Public helpers ────────────────────────────────────────────

def get_features_and_target(df: pd.DataFrame):
    """
    Split a raw DataFrame into feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame  – full dataset including target column

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2,
               random_state: int = 42):
    """
    Stratified train/test split.

    We use shuffle=True (default) so the split is random but
    reproducible thanks to random_state.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    print(f"✅ Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def build_preprocessor() -> Pipeline:
    """
    Return a sklearn Pipeline that applies StandardScaler to
    all numeric features.

    Why StandardScaler?
      • Linear Regression converges better with scaled inputs.
      • Random Forest is scale-invariant, but scaling doesn't
        hurt and keeps both models on equal footing.
    """
    preprocessor = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])
    return preprocessor


def validate_input(data: dict) -> np.ndarray:
    """
    Validate and convert a raw prediction-request dict into a
    numpy array that the model pipeline can consume.

    Parameters
    ----------
    data : dict  – keys matching FEATURE_COLS

    Returns
    -------
    np.ndarray of shape (1, n_features)

    Raises
    ------
    ValueError if any required key is missing.
    """
    missing = [col for col in FEATURE_COLS if col not in data]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    row = [float(data[col]) for col in FEATURE_COLS]
    return np.array(row).reshape(1, -1)
