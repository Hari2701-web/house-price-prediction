"""
data/data_loader.py
────────────────────────────────────────────────────────────────
Handles loading and basic inspection of the California Housing
dataset from scikit-learn.

Why a separate module?
  Real projects keep I/O concerns separate from modelling logic.
  Swapping the dataset later only requires touching this file.
────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing


def load_california_housing() -> pd.DataFrame:
    """
    Load the California Housing dataset and return it as a
    tidy pandas DataFrame with human-readable column names.

    Returns
    -------
    pd.DataFrame
        One row per census block group.
        Columns: MedInc, HouseAge, AveRooms, AveBedrms,
                 Population, AveOccup, Latitude, Longitude,
                 MedHouseVal (target).
    """
    housing = fetch_california_housing(as_frame=True)

    # Combine features + target into a single DataFrame
    df = housing.frame.copy()

    # Rename columns to be more self-documenting
    df.rename(columns={
        "MedInc":      "MedInc",          # Median income (in $10k units)
        "HouseAge":    "HouseAge",         # Median age of houses
        "AveRooms":    "AveRooms",         # Avg rooms per household
        "AveBedrms":   "AveBedrms",        # Avg bedrooms per household
        "Population":  "Population",       # Block group population
        "AveOccup":    "AveOccup",         # Avg household members
        "Latitude":    "Latitude",
        "Longitude":   "Longitude",
        "MedHouseVal": "MedHouseVal",      # TARGET – median house value ($100k)
    }, inplace=True)

    print(f"✅ Dataset loaded  → {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def basic_eda(df: pd.DataFrame) -> None:
    """
    Print a quick exploratory-data-analysis summary to stdout.
    Useful when running the notebook or calling from a script.
    """
    print("\n" + "="*55)
    print("  DATASET OVERVIEW")
    print("="*55)
    print(f"Shape          : {df.shape}")
    print(f"Missing values : {df.isnull().sum().sum()}")
    print("\n── Data Types ─────────────────────────────────────────")
    print(df.dtypes)
    print("\n── Descriptive Statistics ─────────────────────────────")
    print(df.describe().round(3))
    print("="*55 + "\n")


if __name__ == "__main__":
    df = load_california_housing()
    basic_eda(df)
