from pydantic import BaseModel, Field
from typing import Optional


class ClaimFeatures(BaseModel):
    transaction_amt: float = Field(..., gt=0, description="Transaction amount in AUD")
    card_type: str = Field(..., description="Card type: credit | debit | prepaid")
    device_type: str = Field(..., description="Device: mobile | desktop | tablet")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of transaction")
    transaction_velocity: float = Field(..., ge=0, description="Transactions in last hour")
    account_age_days: int = Field(..., ge=0, description="Days since account opened")
    address_match: bool = Field(..., description="Billing address matches account")
    email_risk_score: float = Field(..., ge=0, le=1, description="Email domain risk score")
    distance_from_home_km: float = Field(..., ge=0, description="Distance from registered address")
    prior_claims_count: int = Field(..., ge=0, description="Number of prior claims")

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_amt": 4250.00,
                "card_type": "credit",
                "device_type": "mobile",
                "hour_of_day": 2,
                "transaction_velocity": 5.0,
                "account_age_days": 45,
                "address_match": False,
                "email_risk_score": 0.82,
                "distance_from_home_km": 847.0,
                "prior_claims_count": 0,
            }
        }


class ShapFeature(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str  # "increases_risk" | "reduces_risk"


class ScoreResponse(BaseModel):
    claim_id: str
    fraud_probability: float
    risk_label: str  # LOW | MEDIUM | HIGH | CRITICAL
    risk_score: int  # 0–100
    confidence: float
    shap_features: list[ShapFeature]
    recommendation: str
    model_version: str


class TrainResponse(BaseModel):
    status: str
    auc_roc: float
    precision: float
    recall: float
    f1_score: float
    n_samples: int
    model_path: str
