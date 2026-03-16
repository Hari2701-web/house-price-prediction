"""
model/train.py
────────────────────────────────────────────────────────────────
Trains Linear Regression and Random Forest models, evaluates
them, then saves the best one as a pickle file.

Run directly:
    python model/train.py

What it produces
  model/linear_regression.pkl
  model/random_forest.pkl
  model/best_model.pkl          ← used by the API
  model/scaler.pkl              ← saved separately for reference
  data/feature_names.json       ← feature order used at training
────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

from data.data_loader   import load_california_housing
from data.preprocessing import (
    get_features_and_target,
    split_data,
    build_preprocessor,
    FEATURE_COLS,
)

warnings.filterwarnings("ignore")


# ── Model definitions ─────────────────────────────────────────

def get_models() -> dict:
    """
    Return a dict of {name: sklearn estimator} to be trained.

    Each model is wrapped inside a Pipeline that first runs the
    preprocessor (StandardScaler) before fitting the estimator.
    """
    models = {
        "Linear Regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("regressor",    LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("regressor",    RandomForestRegressor(
                n_estimators=200,   # 200 trees – good quality/speed trade-off
                max_depth=15,       # prevent runaway overfitting
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,          # use all CPU cores
            )),
        ]),
    }
    return models


# ── Evaluation helper ─────────────────────────────────────────

def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """
    Compute regression metrics for a fitted model.

    Returns a dict with MAE, RMSE and R² (all rounded to 4 dp).
    Also prints a formatted summary to stdout.
    """
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n  ╔══ {name} ══╗")
    print(f"  MAE  : {mae:.4f}  (mean absolute error in $100k)")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}  ({r2*100:.1f}% variance explained)")

    return {"name": name, "mae": round(mae, 4),
            "rmse": round(rmse, 4), "r2": round(r2, 4)}


# ── Persistence helpers ───────────────────────────────────────

def save_model(model, path: str) -> None:
    """Serialize a fitted model pipeline to disk with pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  💾 Saved → {path}")


def load_model(path: str):
    """Load a pickle-serialized model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Main training script ──────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  HOUSE PRICE PREDICTION – MODEL TRAINING")
    print("="*55)

    # 1. Load data
    df = load_california_housing()
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 2. Save feature order so the API knows input shape
    os.makedirs("data", exist_ok=True)
    with open("data/feature_names.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    print(f"\n✅ Feature names saved → data/feature_names.json")

    # 3. Train & evaluate all models
    models   = get_models()
    results  = []

    print("\n── Training & Evaluation ───────────────────────────────")
    for name, pipeline in models.items():
        print(f"\n  Training: {name} …")
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(name, pipeline, X_test, y_test)
        results.append((metrics, pipeline))

        # Save each model individually
        slug = name.lower().replace(" ", "_")
        save_model(pipeline, f"model/{slug}.pkl")

    # 4. Pick the best model by R² score
    best_metrics, best_model = max(results, key=lambda t: t[0]["r2"])
    print(f"\n🏆 Best model: {best_metrics['name']}  (R² = {best_metrics['r2']})")
    save_model(best_model, "model/best_model.pkl")

    # 5. Save results summary as JSON for later reference
    summary = [m for m, _ in results]
    with open("model/evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  💾 Evaluation summary → model/evaluation_results.json")

    print("\n" + "="*55)
    print("  ✅  Training complete!  Ready to serve predictions.")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
