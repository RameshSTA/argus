"""
Claims intelligence agent that orchestrates the ML risk scorer and RAG policy
assistant to produce a unified claim triage decision.
"""
import time
import uuid
from typing import Optional

from backend.agent.schemas import AgentRequest, AgentResponse, ToolCall
from backend.ml.model import get_inference_engine
from backend.ml.schemas import ClaimFeatures
from backend.rag.chain import query as rag_query
from backend.rag.schemas import QueryRequest
from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

FINAL_RECOMMENDATION_TEMPLATE = """
Claim ID: {claim_id}
Risk Level: {risk_label} ({risk_score}/100)
Fraud Probability: {fraud_prob:.1%}
Policy Coverage: {coverage_summary}

Recommendation: {recommendation}
""".strip()


def _extract_policy_question(description: str) -> str:
    """Derive the most relevant policy question from a claim description."""
    description_lower = description.lower()
    if "hail" in description_lower:
        return "Is hail damage covered and what excess applies?"
    elif "storm" in description_lower or "weather" in description_lower:
        return "Does comprehensive cover include storm damage and what conditions apply?"
    elif "flood" in description_lower or "water" in description_lower:
        return "Is flood and water damage covered under the standard policy?"
    elif "theft" in description_lower or "stolen" in description_lower:
        return "What are the coverage conditions for theft and attempted theft?"
    elif "fire" in description_lower:
        return "Is fire damage covered and what exclusions apply?"
    elif "accident" in description_lower or "collision" in description_lower:
        return "What coverage applies for collision and how is at-fault liability determined?"
    elif "liability" in description_lower:
        return "What are the third-party liability coverage limits and notification requirements?"
    else:
        return f"What coverage applies to: {description[:120]}"


def _build_default_features(description: str, amount: float = 2500.0) -> ClaimFeatures:
    """Build default claim features from description when none provided."""
    return ClaimFeatures(
        transaction_amt=amount,
        card_type="credit",
        device_type="mobile",
        hour_of_day=10,
        transaction_velocity=1.0,
        account_age_days=365,
        address_match=True,
        email_risk_score=0.15,
        distance_from_home_km=5.0,
        prior_claims_count=0,
    )


def _compose_recommendation(
    description: str,
    risk_label: str,
    fraud_prob: float,
    policy_answer: str,
) -> str:
    """Compose the final agent recommendation using Claude or rule-based logic."""
    settings = get_settings()

    if settings.anthropic_api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            prompt = f"""You are a senior insurance claims analyst. Based on:

Claim Description: {description}
Fraud Risk Level: {risk_label} ({fraud_prob:.1%} probability)
Policy Coverage Analysis: {policy_answer}

Write a concise 2-3 sentence professional recommendation for how to handle this claim. 
Focus on: (1) whether to approve/escalate/investigate, (2) relevant policy conditions, 
(3) any required documentation. Be direct and actionable."""

            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            logger.error(f"LLM recommendation failed: {e}")

    # Rule-based fallback
    risk_actions = {
        "LOW": "Proceed with standard claim processing. No escalation required.",
        "MEDIUM": "Recommend secondary review by a senior assessor before approving.",
        "HIGH": "Flag for dedicated fraud investigator review. Do not approve without investigation.",
        "CRITICAL": "Suspend claim immediately and escalate to fraud team. Notify compliance.",
    }
    return f"{risk_actions[risk_label]} Policy analysis: {policy_answer[:200]}..."


def run_agent(request: AgentRequest) -> AgentResponse:
    start = time.time()
    claim_id = f"CLM-{uuid.uuid4().hex[:8].upper()}"
    tool_calls: list[ToolCall] = []

    logger.info(f"Agent processing claim {claim_id}")

    # Tool 1: Risk Scoring
    try:
        engine = get_inference_engine()
        if request.claim_features:
            features = ClaimFeatures(**request.claim_features)
        else:
            features = _build_default_features(request.claim_description)

        risk_result = engine.predict(features, claim_id=claim_id)

        tool_calls.append(ToolCall(
            tool_name="get_risk_score",
            input=f"claim_id={claim_id}, amt={features.transaction_amt}",
            output=f"fraud_prob={risk_result.fraud_probability:.3f}, label={risk_result.risk_label}",
        ))
        logger.info(f"Risk score: {risk_result.risk_label} ({risk_result.fraud_probability:.2%})")
    except Exception as e:
        logger.error(f"Risk scoring failed: {e}")
        risk_result = None

    # Tool 2: Policy Lookup
    try:
        policy_question = _extract_policy_question(request.claim_description)
        rag_result = rag_query(QueryRequest(question=policy_question, top_k=4))

        tool_calls.append(ToolCall(
            tool_name="lookup_policy",
            input=policy_question,
            output=f"answer_length={len(rag_result.answer)}, confidence={rag_result.confidence}",
        ))
        logger.info(f"Policy lookup: {rag_result.confidence} confidence, {rag_result.retrieved_chunks} chunks")
    except Exception as e:
        logger.error(f"Policy lookup failed: {e}")
        rag_result = None

    # Compose final recommendation
    risk_label = risk_result.risk_label if risk_result else "MEDIUM"
    fraud_prob = risk_result.fraud_probability if risk_result else 0.0
    policy_answer = rag_result.answer if rag_result else "Policy information unavailable."

    final_rec = _compose_recommendation(
        request.claim_description,
        risk_label,
        fraud_prob,
        policy_answer,
    )

    summary = (
        f"Claim {claim_id} analysed. "
        f"Risk: {risk_label} ({fraud_prob:.0%}). "
        f"Policy coverage assessed with {rag_result.confidence if rag_result else 'N/A'} confidence."
    )

    elapsed_ms = int((time.time() - start) * 1000)

    return AgentResponse(
        claim_id=claim_id,
        summary=summary,
        risk_assessment=risk_result,
        policy_context=rag_result,
        tool_calls=tool_calls,
        final_recommendation=final_rec,
        processing_time_ms=elapsed_ms,
    )
