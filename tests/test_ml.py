"""Tests for the ML risk scoring pipeline."""
import pytest
from backend.ml.schemas import ClaimFeatures
from backend.ml.features import engineer_features, FEATURE_COLUMNS


@pytest.fixture
def low_risk_claim():
    return ClaimFeatures(
        transaction_amt=500.0, card_type="credit", device_type="desktop",
        hour_of_day=14, transaction_velocity=0.5, account_age_days=1200,
        address_match=True, email_risk_score=0.05,
        distance_from_home_km=2.0, prior_claims_count=0,
    )


@pytest.fixture
def high_risk_claim():
    return ClaimFeatures(
        transaction_amt=15000.0, card_type="prepaid", device_type="mobile",
        hour_of_day=3, transaction_velocity=9.0, account_age_days=12,
        address_match=False, email_risk_score=0.95,
        distance_from_home_km=2000.0, prior_claims_count=3,
    )


def test_feature_engineering_shape(low_risk_claim):
    df = engineer_features(low_risk_claim)
    assert list(df.columns) == FEATURE_COLUMNS
    assert len(df) == 1


def test_feature_engineering_derived(low_risk_claim):
    df = engineer_features(low_risk_claim)
    import numpy as np
    assert abs(df["amt_log"].iloc[0] - np.log1p(500.0)) < 1e-6
    assert df["is_night"].iloc[0] == 0
    assert df["address_match"].iloc[0] == 1


def test_feature_engineering_night(high_risk_claim):
    df = engineer_features(high_risk_claim)
    assert df["is_night"].iloc[0] == 1
    assert df["address_match"].iloc[0] == 0


def test_composite_risk_range(low_risk_claim, high_risk_claim):
    low_df = engineer_features(low_risk_claim)
    high_df = engineer_features(high_risk_claim)
    assert 0 <= low_df["composite_risk"].iloc[0] <= 1
    assert 0 <= high_df["composite_risk"].iloc[0] <= 1
    assert high_df["composite_risk"].iloc[0] > low_df["composite_risk"].iloc[0]


def test_schema_validation():
    with pytest.raises(Exception):
        ClaimFeatures(
            transaction_amt=-100, card_type="credit", device_type="mobile",
            hour_of_day=25, transaction_velocity=1.0, account_age_days=100,
            address_match=True, email_risk_score=1.5,
            distance_from_home_km=10.0, prior_claims_count=0,
        )
