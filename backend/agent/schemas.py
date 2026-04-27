from pydantic import BaseModel, Field
from typing import Optional
from backend.ml.schemas import ScoreResponse
from backend.rag.schemas import QueryResponse


class AgentRequest(BaseModel):
    claim_description: str = Field(
        ..., min_length=10, max_length=1000,
        description="Natural language description of the claim"
    )
    claim_features: Optional[dict] = Field(
        default=None,
        description="Optional structured claim features for risk scoring"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "claim_description": "Policyholder reports vehicle sustained hail damage during last night's storm. Vehicle was parked in the driveway. Claiming $3,200 for panel repairs.",
                "claim_features": {
                    "transaction_amt": 3200.0,
                    "card_type": "credit",
                    "device_type": "mobile",
                    "hour_of_day": 9,
                    "transaction_velocity": 1.0,
                    "account_age_days": 720,
                    "address_match": True,
                    "email_risk_score": 0.12,
                    "distance_from_home_km": 0.0,
                    "prior_claims_count": 1,
                }
            }
        }


class ToolCall(BaseModel):
    tool_name: str
    input: str
    output: str


class AgentResponse(BaseModel):
    claim_id: str
    summary: str
    risk_assessment: Optional[ScoreResponse] = None
    policy_context: Optional[QueryResponse] = None
    tool_calls: list[ToolCall]
    final_recommendation: str
    processing_time_ms: int
