"""
Feature engineering pipeline for the Argus fraud detection model.

Transforms raw claim inputs into a fixed-length numeric feature vector
suitable for gradient boosting. Includes both direct features and
derived interaction/risk features informed by insurance fraud research.
"""

import numpy as np
import pandas as pd
from backend.ml.schemas import ClaimFeatures


# ── Feature column order (must match training) ──────────────────────
FEATURE_COLUMNS = [
    "transaction_amt",
    "card_type_encoded",
    "device_type_encoded",
    "hour_of_day",
    "transaction_velocity",
    "account_age_days",
    "address_match",
    "email_risk_score",
    "distance_from_home_km",
    "prior_claims_count",
    "amt_log",
    "is_night",
    "velocity_x_amt",
    "age_risk",
    "composite_risk",
]

# ── Ordinal encodings ────────────────────────────────────────────────
CARD_TYPE_MAP = {"credit": 0, "debit": 1, "prepaid": 2}
DEVICE_TYPE_MAP = {"desktop": 0, "mobile": 1, "tablet": 2}


def engineer_features(claim: ClaimFeatures) -> pd.DataFrame:
    """
    Transform a single ClaimFeatures instance into a model-ready DataFrame.

    Applies ordinal encoding to categorical fields and computes five
    derived features that capture interaction effects known to correlate
    with insurance fraud:
      - amt_log            : log(1 + amount) to reduce skew
      - is_night           : binary flag for off-hours transactions
      - velocity_x_amt     : interaction between volume and size
      - age_risk           : inverse function of account age (newer = riskier)
      - composite_risk     : weighted combination of top fraud signals

    Args:
        claim: Validated ClaimFeatures Pydantic model.

    Returns:
        Single-row DataFrame with columns matching FEATURE_COLUMNS.
    """
    raw = {
        "transaction_amt": claim.transaction_amt,
        "card_type_encoded": CARD_TYPE_MAP.get(claim.card_type.lower(), 1),
        "device_type_encoded": DEVICE_TYPE_MAP.get(claim.device_type.lower(), 0),
        "hour_of_day": claim.hour_of_day,
        "transaction_velocity": claim.transaction_velocity,
        "account_age_days": claim.account_age_days,
        "address_match": int(claim.address_match),
        "email_risk_score": claim.email_risk_score,
        "distance_from_home_km": claim.distance_from_home_km,
        "prior_claims_count": claim.prior_claims_count,
        # Derived features
        "amt_log": np.log1p(claim.transaction_amt),
        "is_night": int(claim.hour_of_day < 6 or claim.hour_of_day > 22),
        "velocity_x_amt": claim.transaction_velocity * claim.transaction_amt,
        "age_risk": 1.0 / (1.0 + claim.account_age_days / 365.0),
        "composite_risk": (
            claim.email_risk_score * 0.3
            + (1 - int(claim.address_match)) * 0.25
            + min(claim.transaction_velocity / 10.0, 1.0) * 0.25
            + min(claim.distance_from_home_km / 1000.0, 1.0) * 0.2
        ),
    }
    return pd.DataFrame([raw])[FEATURE_COLUMNS]


def engineer_features_bulk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to a full training DataFrame in-place.

    Mirrors the logic in engineer_features() but operates on pandas
    Series for vectorised performance on large datasets.

    Args:
        df: Raw training DataFrame. Must contain all columns listed
            in ClaimFeatures plus 'is_fraud'.

    Returns:
        DataFrame with exactly FEATURE_COLUMNS as column set.
    """
    df = df.copy()
    df["card_type_encoded"] = df["card_type"].map(CARD_TYPE_MAP).fillna(1)
    df["device_type_encoded"] = df["device_type"].map(DEVICE_TYPE_MAP).fillna(0)
    df["address_match"] = df["address_match"].astype(int)
    df["amt_log"] = np.log1p(df["transaction_amt"])
    df["is_night"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] > 22)).astype(int)
    df["velocity_x_amt"] = df["transaction_velocity"] * df["transaction_amt"]
    df["age_risk"] = 1.0 / (1.0 + df["account_age_days"] / 365.0)
    df["composite_risk"] = (
        df["email_risk_score"] * 0.3
        + (1 - df["address_match"]) * 0.25
        + (df["transaction_velocity"] / 10.0).clip(upper=1.0) * 0.25
        + (df["distance_from_home_km"] / 1000.0).clip(upper=1.0) * 0.2
    )
    return df[FEATURE_COLUMNS]
