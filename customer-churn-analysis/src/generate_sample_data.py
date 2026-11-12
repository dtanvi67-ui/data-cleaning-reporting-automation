"""Generate fictional subscription data for a reproducible learning demo."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def main(n: int = 1200, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    tenure = rng.integers(1, 73, n)
    internet = rng.choice(["Fiber optic", "DSL", "No"], n, p=[0.48, 0.38, 0.14])
    support = rng.choice(["Yes", "No"], n, p=[0.38, 0.62])
    payment = rng.choice(["Electronic check", "Credit card", "Bank transfer", "Mailed check"], n)
    monthly = np.clip(rng.normal(68, 22, n), 18, 125).round(2)
    support_calls = rng.poisson(1.2, n)
    # Synthetic relationship for demonstrating classification; not a real-world estimate.
    logit = (
        -1.9
        + 1.05 * (contract == "Month-to-month")
        - 0.8 * (contract == "Two year")
        - 0.025 * tenure
        + 0.65 * (support == "No")
        + 0.45 * (internet == "Fiber optic")
        + 0.22 * (payment == "Electronic check")
        + 0.28 * (support_calls >= 3)
        + 0.008 * (monthly - 60)
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(n) < probability, "Yes", "No")
    df = pd.DataFrame({
        "customer_id": [f"DEMO{i:05d}" for i in range(1, n + 1)],
        "tenure_months": tenure,
        "contract": contract,
        "monthly_charges": monthly,
        "total_charges": (monthly * tenure).round(2),
        "internet_service": internet,
        "tech_support": support,
        "payment_method": payment,
        "support_calls": support_calls,
        "churn": churn,
    })
    # Introduce a small number of realistic missing values for cleaning practice.
    missing_idx = rng.choice(n, size=max(1, n // 100), replace=False)
    df.loc[missing_idx, "monthly_charges"] = np.nan
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "customer_churn.csv", index=False)
    print(f"Created {len(df)} fictional demo records: {DATA_DIR / 'customer_churn.csv'}")


if __name__ == "__main__":
    main()
