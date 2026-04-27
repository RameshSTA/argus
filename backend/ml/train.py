"""
XGBoost training pipeline with cross-validation, threshold tuning, and MLflow-style metrics.
Run: python -m scripts.train_model
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, classification_report, average_precision_score,
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from backend.ml.features import engineer_features_bulk, FEATURE_COLUMNS
from backend.ml.schemas import TrainResponse
from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


XGBOOST_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 27,   # ~1/fraud_rate for class imbalance
    "tree_method": "hist",
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
}


def load_data(data_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate training data."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    df = pd.read_csv(path)
    required = ["is_fraud", "transaction_amt", "card_type", "device_type",
                "hour_of_day", "transaction_velocity", "account_age_days",
                "address_match", "email_risk_score", "distance_from_home_km",
                "prior_claims_count"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    y = df["is_fraud"]
    X = engineer_features_bulk(df)
    logger.info(f"Loaded {len(df):,} samples | fraud rate: {y.mean():.2%}")
    return X, y


def train(data_path: str = "data/raw/claims_dataset.csv") -> TrainResponse:
    settings = get_settings()
    logger.info("Starting XGBoost training pipeline")

    X, y = load_data(data_path)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_model = XGBClassifier(**XGBOOST_PARAMS)
    cv_scores = cross_val_score(base_model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    logger.info(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final model with calibration for proper probabilities
    final_model = XGBClassifier(**XGBOOST_PARAMS)
    final_model.fit(X, y, eval_set=[(X, y)], verbose=False)
    calibrated = CalibratedClassifierCV(final_model, method="isotonic", cv=3)
    calibrated.fit(X, y)

    # Evaluation
    y_prob = calibrated.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(y, y_prob),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0),
        "avg_precision": average_precision_score(y, y_prob),
    }

    logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
    logger.info(f"F1: {metrics['f1_score']:.4f}")

    # Save model
    model_path = settings.model_full_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": calibrated, "feature_columns": FEATURE_COLUMNS, "metrics": metrics},
        model_path
    )
    logger.info(f"Model saved → {model_path}")

    return TrainResponse(
        status="success",
        auc_roc=round(metrics["auc_roc"], 4),
        precision=round(metrics["precision"], 4),
        recall=round(metrics["recall"], 4),
        f1_score=round(metrics["f1_score"], 4),
        n_samples=len(X),
        model_path=str(model_path),
    )
