"""Train and validate the XGBoost RSSI regressor."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from xgboost import XGBRegressor
from utils.calibration import build_uncertainty_profile
from utils.rssi_features import MODEL_FEATURES


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "geotwin-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = PROJECT_ROOT / "data/processed/train.csv"
MODEL_PATH = PROJECT_ROOT / "models/rssi_xgb.json"
PICKLE_MODEL_PATH = PROJECT_ROOT / "models/xgboost_model.pkl"
METRICS_PATH = PROJECT_ROOT / "models/metrics.json"
UNCERTAINTY_PATH = PROJECT_ROOT / "models/uncertainty.json"
PREDICTED_ACTUAL_PNG = PROJECT_ROOT / "reports/predicted_vs_actual.png"
FEATURE_IMPORTANCE_PNG = PROJECT_ROOT / "reports/feature_importance.png"

FEATURES = MODEL_FEATURES
TARGET = "rssi_dbm"

PARAM_GRID = {
    "max_depth": [4, 6, 8],
    "n_estimators": [100, 300, 500],
    "learning_rate": [0.03, 0.05, 0.1],
}


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the synthetic data used by the RSSI model."""
    dataset = pd.read_csv(path)
    required_columns = set(FEATURES + [TARGET, "num_walls_intersected"])
    missing_columns = sorted(required_columns.difference(dataset.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing_columns)}")
    if dataset.empty:
        raise ValueError("Dataset is empty.")
    if dataset[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Dataset contains missing feature or target values.")
    return dataset


def split_dataset(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create reproducible 70/15/15 splits balanced by wall-crossing count."""
    features = dataset[FEATURES]
    target = dataset[TARGET]
    strata = dataset["num_walls_intersected"]

    x_train, x_temp, y_train, y_temp, _, temp_strata = train_test_split(
        features,
        target,
        strata,
        test_size=0.30,
        random_state=42,
        stratify=strata,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=temp_strata,
    )
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Calculate the model-quality metrics reported for held-out samples."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def build_model_search() -> GridSearchCV:
    """Configure cross-validated hyperparameter selection."""
    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )
    cross_validation = KFold(n_splits=5, shuffle=True, random_state=42)
    return GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        scoring="neg_root_mean_squared_error",
        cv=cross_validation,
        # Keep cross-validation single-process so training also works in
        # constrained deployment environments without process semaphores.
        n_jobs=1,
        verbose=1,
    )


def save_predicted_vs_actual(actual: pd.Series, predicted: np.ndarray) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predicted, alpha=0.5)
    min_value = min(actual.min(), predicted.min())
    max_value = max(actual.max(), predicted.max())
    plt.plot([min_value, max_value], [min_value, max_value], color="black")
    plt.xlabel("Actual RSSI (dBm)")
    plt.ylabel("Predicted RSSI (dBm)")
    plt.title("Predicted vs Actual RSSI")
    plt.tight_layout()
    plt.savefig(PREDICTED_ACTUAL_PNG, dpi=300)
    plt.close()


def save_feature_importance(model: XGBRegressor) -> None:
    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(importance["feature"], importance["importance"])
    plt.xticks(rotation=30)
    plt.ylabel("Importance")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PNG, dpi=300)
    plt.close()


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTED_ACTUAL_PNG.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    x_train, x_validation, x_test, y_train, y_validation, y_test = split_dataset(dataset)

    print(f"Dataset shape: {dataset.shape}")
    print(f"Features: {FEATURES}")
    print(
        "Split sizes: "
        f"train={len(x_train)}, validation={len(x_validation)}, test={len(x_test)}"
    )
    print("Running grid search...")

    search = build_model_search()
    search.fit(x_train, y_train)
    best_model = search.best_estimator_

    validation_predictions = best_model.predict(x_validation)
    test_predictions = best_model.predict(x_test)
    validation = regression_metrics(y_validation, validation_predictions)
    test = regression_metrics(y_test, test_predictions)
    uncertainty = build_uncertainty_profile(
        y_validation,
        validation_predictions,
        x_validation,
    )

    best_model.save_model(MODEL_PATH)
    joblib.dump(best_model, PICKLE_MODEL_PATH)

    metrics = {
        "dataset_rows": int(len(dataset)),
        "features": FEATURES,
        "best_parameters": search.best_params_,
        "validation": validation,
        "test": test,
        "uncertainty": uncertainty,
        # Retain the original top-level test metrics for existing consumers.
        **test,
    }
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with UNCERTAINTY_PATH.open("w", encoding="utf-8") as file:
        json.dump(uncertainty, file, indent=2)

    save_predicted_vs_actual(y_test, test_predictions)
    save_feature_importance(best_model)

    print(f"Best parameters: {search.best_params_}")
    print(
        "Validation: "
        f"RMSE={validation['rmse']:.2f} dB, "
        f"MAE={validation['mae']:.2f} dB, R²={validation['r2']:.4f}"
    )
    print(
        "Test: "
        f"RMSE={test['rmse']:.2f} dB, "
        f"MAE={test['mae']:.2f} dB, R²={test['r2']:.4f}"
    )
    print(
        f"{uncertainty['confidence_level']:.0%} uncertainty interval: "
        f"±{uncertainty['global_half_width_db']:.2f} dB"
    )
    print(f"Model saved: {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")
    print(f"Uncertainty profile saved: {UNCERTAINTY_PATH}")


if __name__ == "__main__":
    main()
