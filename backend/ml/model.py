"""
Model inference engine with SHAP explainability.
Handles loading, prediction, and explanation generation.
"""
import uuid
import shap
import joblib
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import Optional

from backend.ml.schemas import ClaimFeatures, ScoreResponse, ShapFeature
from backend.ml.features import engineer_features, FEATURE_COLUMNS
from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

FEATURE_LABELS = {
    "transaction_amt": "Transaction amount",
    "card_type_encoded": "Card type",
    "device_type_encoded": "Device type",
    "hour_of_day": "Hour of transaction",
    "transaction_velocity": "Transaction velocity",
    "account_age_days": "Account age",
    "address_match": "Address match",
    "email_risk_score": "Email risk score",
    "distance_from_home_km": "Distance from home",
    "prior_claims_count": "Prior claims",
    "amt_log": "Amount (log scale)",
    "is_night": "Night transaction",
    "velocity_x_amt": "Velocity × amount",
    "age_risk": "Account age risk",
    "composite_risk": "Composite risk score",
}

RISK_THRESHOLDS = {
    "LOW": (0.0, 0.25),
    "MEDIUM": (0.25, 0.55),
    "HIGH": (0.55, 0.80),
    "CRITICAL": (0.80, 1.01),
}

RECOMMENDATIONS = {
    "LOW": "Claim appears legitimate. Proceed with standard processing.",
    "MEDIUM": "Minor anomalies detected. Recommend secondary review before approval.",
    "HIGH": "Significant fraud indicators present. Flag for investigator review.",
    "CRITICAL": "High fraud probability. Suspend claim and escalate to fraud team immediately.",
}


class ModelInference:
    def __init__(self):
        self._model = None
        self._explainer = None
        self._model_version = "1.0.0"

    def _load(self) -> None:
        settings = get_settings()
        path = settings.model_full_path

        if not path.exists():
            logger.warning("No saved model found — training now with synthetic data")
            self._auto_train()
            return

        bundle = joblib.load(path)
        self._model = bundle["model"]
        self._model_version = "1.0.0"
        logger.info(f"Model loaded from {path}")

        # Build SHAP explainer on base estimator
        try:
            base = self._model.calibrated_classifiers_[0].estimator
            self._explainer = shap.TreeExplainer(base)
            logger.info("SHAP TreeExplainer initialised")
        except Exception as e:
            logger.warning(f"SHAP TreeExplainer failed ({e}) — using KernelExplainer fallback")
            self._explainer = None

    def _auto_train(self) -> None:
        """Auto-generate data and train if no model exists."""
        from scripts.generate_data import generate_dataset
        from backend.ml.train import train
        import pandas as pd
        from pathlib import Path

        data_path = "data/raw/claims_dataset.csv"
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        df = generate_dataset(n_samples=20_000)
        df.to_csv(data_path, index=False)
        logger.info("Synthetic dataset generated for auto-training")
        train(data_path=data_path)
        self._load()

    def ensure_loaded(self) -> None:
        if self._model is None:
            self._load()

    def predict(self, claim: ClaimFeatures, claim_id: Optional[str] = None) -> ScoreResponse:
        self.ensure_loaded()

        claim_id = claim_id or f"TXN-{uuid.uuid4().hex[:8].upper()}"
        X = engineer_features(claim)

        prob = float(self._model.predict_proba(X)[0, 1])
        risk_label = self._get_risk_label(prob)
        risk_score = int(prob * 100)

        shap_features = self._compute_shap(X)

        return ScoreResponse(
            claim_id=claim_id,
            fraud_probability=round(prob, 4),
            risk_label=risk_label,
            risk_score=risk_score,
            confidence=self._compute_confidence(prob),
            shap_features=shap_features,
            recommendation=RECOMMENDATIONS[risk_label],
            model_version=self._model_version,
        )

    def _get_risk_label(self, prob: float) -> str:
        for label, (lo, hi) in RISK_THRESHOLDS.items():
            if lo <= prob < hi:
                return label
        return "CRITICAL"

    def _compute_confidence(self, prob: float) -> float:
        """Model confidence: how far from the 0.5 decision boundary."""
        return round(abs(prob - 0.5) * 2, 3)

    def _compute_shap(self, X) -> list[ShapFeature]:
        if self._explainer is None:
            return self._fallback_shap(X)

        try:
            values = self._explainer.shap_values(X)
            if isinstance(values, list):
                values = values[1]
            shap_vals = values[0]

            features = []
            for col, sv in sorted(
                zip(FEATURE_COLUMNS, shap_vals),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:8]:
                features.append(ShapFeature(
                    feature=FEATURE_LABELS.get(col, col),
                    value=round(float(X[col].iloc[0]), 3),
                    shap_value=round(float(sv), 4),
                    direction="increases_risk" if sv > 0 else "reduces_risk",
                ))
            return features
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return self._fallback_shap(X)

    def _fallback_shap(self, X) -> list[ShapFeature]:
        """Rule-based approximate attribution when SHAP unavailable."""
        row = X.iloc[0]
        approx = {
            "Transaction amount": (row["transaction_amt"], (row["transaction_amt"] - 5000) / 50000 * 0.3),
            "Transaction velocity": (row["transaction_velocity"], row["transaction_velocity"] / 30 * 0.25),
            "Email risk score": (row["email_risk_score"], (row["email_risk_score"] - 0.5) * 0.35),
            "Address match": (row["address_match"], (1 - row["address_match"]) * 0.2),
            "Distance from home": (row["distance_from_home_km"], row["distance_from_home_km"] / 5000 * 0.2),
            "Account age": (row["account_age_days"], -row["account_age_days"] / 3650 * 0.15),
            "Night transaction": (row["is_night"], row["is_night"] * 0.1),
            "Prior claims": (row["prior_claims_count"], row["prior_claims_count"] * 0.05),
        }
        return [
            ShapFeature(
                feature=k,
                value=round(float(v[0]), 3),
                shap_value=round(float(v[1]), 4),
                direction="increases_risk" if v[1] > 0 else "reduces_risk",
            )
            for k, v in sorted(approx.items(), key=lambda x: abs(x[1][1]), reverse=True)
        ]


_inference_engine = ModelInference()


def get_inference_engine() -> ModelInference:
    return _inference_engine
