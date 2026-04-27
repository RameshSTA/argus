"""
Standalone training script.
Usage:
    python -m scripts.train_model              # auto-generate data + train
    python -m scripts.train_model --data path  # train on existing CSV
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train the Argus fraud detection model")
    parser.add_argument("--data", default="data/raw/claims_dataset.csv", help="Path to training CSV")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic data first")
    parser.add_argument("--samples", type=int, default=50_000, help="Synthetic samples to generate")
    args = parser.parse_args()

    if args.generate or not Path(args.data).exists():
        print(f"Generating synthetic dataset ({args.samples:,} samples)...")
        from scripts.generate_data import generate_dataset
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        df = generate_dataset(n_samples=args.samples)
        df.to_csv(args.data, index=False)
        print(f"✓ Dataset saved → {args.data}")

    print("Training XGBoost model...")
    from backend.ml.train import train
    result = train(data_path=args.data)

    print("\n" + "─" * 40)
    print(f"  Status    : {result.status}")
    print(f"  AUC-ROC   : {result.auc_roc:.4f}")
    print(f"  Precision : {result.precision:.4f}")
    print(f"  Recall    : {result.recall:.4f}")
    print(f"  F1        : {result.f1_score:.4f}")
    print(f"  Samples   : {result.n_samples:,}")
    print(f"  Saved to  : {result.model_path}")
    print("─" * 40)


if __name__ == "__main__":
    main()
