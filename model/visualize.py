"""
model/visualize.py
────────────────────────────────────────────────────────────────
Generates exploratory and model-diagnostic plots and saves them
to the model/ directory as PNG files.

Run:
    python model/visualize.py

Produces
  model/plots/01_price_distribution.png
  model/plots/02_correlation_heatmap.png
  model/plots/03_feature_importance.png
  model/plots/04_actual_vs_predicted.png
  model/plots/05_residual_plot.png
────────────────────────────────────────────────────────────────
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for servers
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader   import load_california_housing
from data.preprocessing import get_features_and_target, split_data, FEATURE_COLS


# ── Style setup ───────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PLOT_DIR = os.path.join("model", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def _save(fig, filename: str) -> None:
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 {path}")


# ── Plot 1: Price distribution ────────────────────────────────

def plot_price_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Target Variable – Median House Value", fontsize=14, fontweight="bold")

    # Raw distribution
    axes[0].hist(df["MedHouseVal"], bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Median House Value ($100k)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Raw Distribution")

    # Log-transformed (reveals shape near the tail cap at $500k)
    axes[1].hist(np.log1p(df["MedHouseVal"]), bins=50, color="#DD8452", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("log(1 + Median House Value)")
    axes[1].set_title("Log-Transformed Distribution")

    fig.tight_layout()
    _save(fig, "01_price_distribution.png")


# ── Plot 2: Correlation heatmap ───────────────────────────────

def plot_correlation_heatmap(df):
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))   # show lower triangle only

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5,
        ax=ax, cbar_kws={"shrink": 0.75}
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "02_correlation_heatmap.png")


# ── Plot 3: Feature importance (Random Forest) ────────────────

def plot_feature_importance():
    model_path = os.path.join("model", "random_forest.pkl")
    if not os.path.exists(model_path):
        print("  ⚠️  random_forest.pkl not found – run train.py first.")
        return

    with open(model_path, "rb") as f:
        rf_pipeline = pickle.load(f)

    importances = rf_pipeline.named_steps["regressor"].feature_importances_
    indices     = np.argsort(importances)[::-1]
    sorted_features    = [FEATURE_COLS[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette("viridis", len(FEATURE_COLS))
    bars   = ax.barh(sorted_features[::-1], sorted_importances[::-1],
                     color=colors, edgecolor="white")

    ax.set_xlabel("Importance Score")
    ax.set_title("Random Forest – Feature Importances", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    # Annotate bars
    for bar, val in zip(bars, sorted_importances[::-1]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    _save(fig, "03_feature_importance.png")


# ── Plot 4: Actual vs Predicted ───────────────────────────────

def plot_actual_vs_predicted():
    model_path = os.path.join("model", "best_model.pkl")
    if not os.path.exists(model_path):
        print("  ⚠️  best_model.pkl not found – run train.py first.")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = load_california_housing()
    X, y = get_features_and_target(df)
    _, X_test, _, y_test = split_data(X, y)

    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.25, s=10, color="#4C72B0")
    lims = [min(y_test.min(), y_pred.min()),
            max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Price ($100k)")
    ax.set_ylabel("Predicted Price ($100k)")
    ax.set_title("Actual vs Predicted House Prices", fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _save(fig, "04_actual_vs_predicted.png")


# ── Plot 5: Residual plot ─────────────────────────────────────

def plot_residuals():
    model_path = os.path.join("model", "best_model.pkl")
    if not os.path.exists(model_path):
        print("  ⚠️  best_model.pkl not found – run train.py first.")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = load_california_housing()
    X, y = get_features_and_target(df)
    _, X_test, _, y_test = split_data(X, y)

    y_pred    = model.predict(X_test)
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Residual Analysis", fontsize=14, fontweight="bold")

    axes[0].scatter(y_pred, residuals, alpha=0.25, s=10, color="#DD8452")
    axes[0].axhline(0, color="red", lw=1.5, ls="--")
    axes[0].set_xlabel("Predicted Price ($100k)")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Fitted Values")

    axes[1].hist(residuals, bins=60, color="#55A868", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", lw=1.5, ls="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    fig.tight_layout()
    _save(fig, "05_residual_plot.png")


# ── Entry point ───────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  HOUSE PRICE PREDICTION – VISUALIZATIONS")
    print("="*55)

    df = load_california_housing()

    print("\nGenerating plots …")
    plot_price_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_importance()
    plot_actual_vs_predicted()
    plot_residuals()

    print(f"\n✅ All plots saved to → {PLOT_DIR}/\n")


if __name__ == "__main__":
    main()
