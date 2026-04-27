"""
Argus — Real Data Preparation Pipeline
=======================================
Downloads and preprocesses publicly available fraud detection datasets
and maps them to the Argus feature schema for model training.

Supported sources:
  1. IEEE-CIS Fraud Detection (Kaggle) — best choice for insurance fraud
  2. Credit Card Fraud Detection (ULB/Kaggle) — large volume, PCA-masked
  3. German Credit (UCI) — credit risk proxy, accessible without auth
  4. PaySim Mobile Money (Kaggle) — financial fraud simulation

Usage:
  # With Kaggle API configured (~/.kaggle/kaggle.json):
  python -m scripts.prepare_real_data --source ieee

  # Without Kaggle (uses UCI German Credit as credit risk proxy):
  python -m scripts.prepare_real_data --source uci

  # Download IEEE-CIS manually from:
  # https://www.kaggle.com/competitions/ieee-fraud-detection/data
  # Then run:
  python -m scripts.prepare_real_data --source ieee --path data/raw/ieee_fraud/

References:
  - Yao et al. (2019). "IEEE-CIS Fraud Detection Challenge." Kaggle.
  - Dal Pozzolo et al. (2015). "Calibrating Probability with Undersampling for
    Unbalanced Classification." CIDM, IEEE.
  - Lopez-Rojas & Axelsson (2016). "PaySim: A Financial Mobile Money Simulator
    for Fraud Detection." EMSS.
"""
import argparse
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path

# ── Output path ──────────────────────────────────────────────────────────────
OUTPUT_PATH = Path("data/raw/claims_dataset.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Target feature schema ────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "transaction_amt", "card_type", "device_type", "hour_of_day",
    "transaction_velocity", "account_age_days", "address_match",
    "email_risk_score", "distance_from_home_km", "prior_claims_count",
    "is_fraud",
]


# ── Source: IEEE-CIS Fraud Detection ─────────────────────────────────────────
def prepare_ieee(data_path: str) -> pd.DataFrame:
    """
    Map the IEEE-CIS Fraud Detection dataset to the Argus feature schema.

    Download from: https://www.kaggle.com/competitions/ieee-fraud-detection/data
    Required files: train_transaction.csv, train_identity.csv

    IEEE-CIS Feature Mapping:
      TransactionAmt   → transaction_amt (direct)
      TransactionDT    → hour_of_day     (seconds since epoch % 86400 // 3600)
      card4            → card_type       (visa/mc=credit, discover=debit, other=prepaid)
      D1               → account_age_days (days since last transaction)
      C1               → transaction_velocity (count-based feature)
      D10              → email_risk_score   (email domain age, clipped 0-1)
      addr1/addr2      → address_match      (billing zip == account zip)
      dist1            → distance_from_home_km
      C14              → prior_claims_count (chargeback count proxy)
      DeviceType       → device_type        (mobile/desktop)
      isFraud          → is_fraud
    """
    path = Path(data_path)
    print(f"Loading IEEE-CIS data from: {path}")

    df_tx = pd.read_csv(path / "train_transaction.csv")
    try:
        df_id = pd.read_csv(path / "train_identity.csv")
        df = df_tx.merge(df_id, on="TransactionID", how="left")
    except FileNotFoundError:
        df = df_tx

    mapped = pd.DataFrame()
    mapped["transaction_amt"]       = df["TransactionAmt"]
    mapped["hour_of_day"]           = (df["TransactionDT"] // 3600) % 24
    mapped["account_age_days"]      = df["D1"].fillna(0).clip(0, 3650)
    mapped["transaction_velocity"]  = df["C1"].fillna(1).clip(0, 30)
    mapped["email_risk_score"]      = (df["D10"].fillna(180) / 365).clip(0, 1)
    mapped["address_match"]         = (df["addr1"] == df["addr2"]).astype(int)
    mapped["distance_from_home_km"] = df["dist1"].fillna(0).clip(0, 5000)
    mapped["prior_claims_count"]    = df["C14"].fillna(0).clip(0, 20).astype(int)
    mapped["is_fraud"]              = df["isFraud"]

    # Card type mapping
    card_map = {
        "visa": "credit", "mastercard": "credit",
        "american express": "credit", "discover": "debit",
    }
    mapped["card_type"] = df["card4"].str.lower().map(card_map).fillna("prepaid")

    # Device type
    if "DeviceType" in df.columns:
        mapped["device_type"] = df["DeviceType"].map(
            {"mobile": "mobile", "desktop": "desktop"}
        ).fillna("desktop")
    else:
        mapped["device_type"] = "desktop"

    df_out = mapped.dropna(subset=["transaction_amt", "is_fraud"]).reset_index(drop=True)
    print(f"  Records: {len(df_out):,} | Fraud rate: {df_out['is_fraud'].mean():.2%}")
    return df_out


# ── Source: ULB Credit Card Fraud (Kaggle) ───────────────────────────────────
def prepare_creditcard(csv_path: str) -> pd.DataFrame:
    """
    Map the ULB Credit Card Fraud dataset to the Argus feature schema.
    Note: this dataset uses PCA-masked features (V1-V28) so direct mapping
    is an approximation. The IEEE-CIS source is preferred for insurance fraud.

    Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    """
    print(f"Loading ULB Credit Card data from: {csv_path}")
    df = pd.read_csv(csv_path)

    mapped = pd.DataFrame()
    mapped["transaction_amt"]       = df["Amount"]
    mapped["hour_of_day"]           = (df["Time"] // 3600) % 24
    mapped["account_age_days"]      = (df["V2"].clip(-3, 3) * 200 + 500).clip(0, 3650)
    mapped["transaction_velocity"]  = (df["V4"].clip(-3, 3) + 3).clip(0, 30) / 6 * 10
    mapped["email_risk_score"]      = ((df["V14"].clip(-3, 3) + 3) / 6).clip(0, 1)
    mapped["address_match"]         = ((df["V6"] > 0).astype(int))
    mapped["distance_from_home_km"] = ((df["V3"].clip(-3, 3) + 3) / 6 * 1000).clip(0, 5000)
    mapped["prior_claims_count"]    = ((df["V1"].clip(0, 5)).astype(int))
    mapped["card_type"]             = pd.cut(
        df["V8"], bins=3, labels=["credit", "debit", "prepaid"]
    ).astype(str)
    mapped["device_type"]           = np.where(df["V10"] > 0, "mobile", "desktop")
    mapped["is_fraud"]              = df["Class"]

    print(f"  Records: {len(mapped):,} | Fraud rate: {mapped['is_fraud'].mean():.2%}")
    return mapped


# ── Source: UCI German Credit (no auth required) ─────────────────────────────
def prepare_uci_german() -> pd.DataFrame:
    """
    Download and map the UCI Statlog German Credit dataset.
    This is a credit risk dataset — not insurance fraud — but useful as a
    no-auth-required real-world validation dataset.

    Source: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
    License: UCI ML Repository (free to use for research)
    Paper: Hofmann (1994). "Statlog (German Credit Data)." UCI ML Repository.
    """
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "statlog/german/german.data")
    print(f"Downloading UCI German Credit from:\n  {url}")
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            raw = r.read().decode()
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}\n"
                           f"Try manually downloading from: {url}")

    cols = [
        "checking_status", "duration", "credit_history", "purpose",
        "credit_amount", "savings_status", "employment", "installment_rate",
        "personal_status", "other_parties", "residence_since", "property_magnitude",
        "age", "other_payment_plans", "housing", "existing_credits", "job",
        "num_dependents", "own_telephone", "foreign_worker", "class",
    ]
    from io import StringIO
    df = pd.read_csv(StringIO(raw), sep=" ", header=None, names=cols)

    # Recode target: original 1=good, 2=bad → 0=good, 1=bad (fraud proxy)
    df["is_fraud"] = (df["class"] == 2).astype(int)

    # Map German Credit features → Argus insurance fraud schema (approximate)
    mapped = pd.DataFrame()
    mapped["transaction_amt"]       = df["credit_amount"].clip(100, 50_000)
    mapped["hour_of_day"]           = (df.index % 24)  # no time feature — use index cycle
    mapped["account_age_days"]      = (df["duration"] * 30).clip(0, 3650)
    mapped["transaction_velocity"]  = df["installment_rate"].clip(0, 15)
    mapped["email_risk_score"]      = (df["age"] / 60).clip(0, 1).rsub(1)  # younger = riskier
    mapped["address_match"]         = (df["residence_since"] >= 2).astype(int)
    mapped["distance_from_home_km"] = (df["existing_credits"] * 80).clip(0, 1000)
    mapped["prior_claims_count"]    = df["existing_credits"].clip(0, 10).astype(int)
    mapped["card_type"]             = df["checking_status"].map({
        "A11": "prepaid", "A12": "debit", "A13": "credit", "A14": "credit"
    }).fillna("debit")
    mapped["device_type"]           = np.where(df["own_telephone"] == "A192", "mobile", "desktop")
    mapped["is_fraud"]              = df["is_fraud"]

    print(f"  Records: {len(mapped):,} | Default rate: {mapped['is_fraud'].mean():.2%}")
    print("  Note: UCI German Credit is a credit DEFAULT dataset (not insurance fraud).")
    print("  Use as statistical validation only. IEEE-CIS is preferred for production.")
    return mapped


# ── Validation ───────────────────────────────────────────────────────────────
def validate_and_save(df: pd.DataFrame, output_path: Path) -> None:
    """Validate schema compliance and save."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after mapping: {missing}")

    # Type coercion
    df["transaction_amt"]       = df["transaction_amt"].astype(float)
    df["hour_of_day"]           = df["hour_of_day"].astype(int).clip(0, 23)
    df["account_age_days"]      = df["account_age_days"].astype(int).clip(0, 5000)
    df["transaction_velocity"]  = df["transaction_velocity"].astype(float).clip(0, 50)
    df["email_risk_score"]      = df["email_risk_score"].astype(float).clip(0, 1)
    df["address_match"]         = df["address_match"].astype(bool)
    df["distance_from_home_km"] = df["distance_from_home_km"].astype(float).clip(0, 10_000)
    df["prior_claims_count"]    = df["prior_claims_count"].astype(int).clip(0, 30)
    df["is_fraud"]              = df["is_fraud"].astype(int)

    final = df[REQUIRED_COLUMNS].dropna().reset_index(drop=True)
    final.to_csv(output_path, index=False)

    print(f"\n✓ Saved {len(final):,} records → {output_path}")
    print(f"  Fraud rate:       {final['is_fraud'].mean():.2%}")
    print(f"  Columns:          {list(final.columns)}")
    print(f"  Fraud count:      {final['is_fraud'].sum():,}")
    print(f"  Legitimate count: {(final['is_fraud']==0).sum():,}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a real fraud dataset for Argus training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--source", choices=["ieee", "creditcard", "uci"],
        default="uci",
        help="Data source to use (default: uci — no auth required)"
    )
    parser.add_argument(
        "--path", type=str, default="",
        help="Path to downloaded data files (for ieee/creditcard sources)"
    )
    args = parser.parse_args()

    if args.source == "ieee":
        if not args.path:
            print("ERROR: --path required for IEEE-CIS source.")
            print("Download from: https://www.kaggle.com/competitions/ieee-fraud-detection/data")
            print("Then run: python -m scripts.prepare_real_data --source ieee --path data/raw/ieee/")
            return
        df = prepare_ieee(args.path)

    elif args.source == "creditcard":
        if not args.path:
            print("ERROR: --path required for Credit Card source.")
            print("Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
            print("Then run: python -m scripts.prepare_real_data --source creditcard --path data/raw/creditcard.csv")
            return
        df = prepare_creditcard(args.path)

    else:  # uci — no auth required
        df = prepare_uci_german()

    validate_and_save(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
