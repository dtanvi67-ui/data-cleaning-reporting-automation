"""Analyze churn patterns and train baseline classification models."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "customer_churn.csv"
OUT = ROOT / "output"
TARGET = "churn"
NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "support_calls"]
CATEGORICAL = ["contract", "internet_service", "tech_support", "payment_method"]


def load_and_clean(path: Path = DATA_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("Dataset missing. Run: python src/generate_sample_data.py")
    df = pd.read_csv(path)
    required = ["customer_id", *NUMERIC, *CATEGORICAL, TARGET]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.drop_duplicates(subset="customer_id").copy()
    df["churn"] = df["churn"].astype(str).str.strip().str.title()
    df = df[df["churn"].isin(["Yes", "No"])].copy()
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["tenure_months"].between(0, 100, inclusive="both")]
    df = df[df["monthly_charges"].isna() | df["monthly_charges"].between(0, 1000)]
    for col in CATEGORICAL:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    return df.reset_index(drop=True)


def make_pipeline(model):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
    ])
    return Pipeline([("preprocess", prep), ("model", model)])


def save_charts(df: pd.DataFrame, importance: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    churn_counts = df[TARGET].value_counts().reindex(["No", "Yes"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    churn_counts.plot(kind="bar", ax=ax)
    ax.set(title="Customer Churn Distribution", xlabel="Churn", ylabel="Customers")
    fig.tight_layout()
    fig.savefig(OUT / "churn_distribution.png", dpi=150)
    plt.close(fig)

    for column, filename, title in [
        ("contract", "churn_by_contract.png", "Churn Rate by Contract"),
    ]:
        rates = df.groupby(column, observed=False)[TARGET].apply(lambda x: (x == "Yes").mean()).sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        rates.plot(kind="bar", ax=ax)
        ax.set(title=title, xlabel=column.replace("_", " ").title(), ylabel="Churn rate")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=150)
        plt.close(fig)

    tenure_group = pd.cut(df["tenure_months"], bins=[-1, 6, 12, 24, 48, 100],
                          labels=["0-6", "7-12", "13-24", "25-48", "49+"])
    rates = df.assign(tenure_group=tenure_group).groupby("tenure_group", observed=False)[TARGET].apply(
        lambda x: (x == "Yes").mean())
    fig, ax = plt.subplots(figsize=(7, 4))
    rates.plot(kind="bar", ax=ax)
    ax.set(title="Churn Rate by Customer Tenure", xlabel="Tenure (months)", ylabel="Churn rate")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "churn_by_tenure_group.png", dpi=150)
    plt.close(fig)

    top = importance.head(12).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["importance"])
    ax.set(title="Random Forest Feature Importance", xlabel="Importance", ylabel="Feature")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_and_clean()
    if len(df) < 30 or df[TARGET].nunique() < 2:
        raise ValueError("Need at least 30 valid rows and both churn classes to train models.")

    X = df[NUMERIC + CATEGORICAL]
    y = (df[TARGET] == "Yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced", min_samples_leaf=3
        ),
    }
    rows = []
    fitted = {}
    for name, estimator in models.items():
        pipe = make_pipeline(estimator)
        pipe.fit(X_train, y_train)
        prediction = pipe.predict(X_test)
        probability = pipe.predict_proba(X_test)[:, 1]
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, prediction),
            "precision": precision_score(y_test, prediction, zero_division=0),
            "recall": recall_score(y_test, prediction, zero_division=0),
            "f1": f1_score(y_test, prediction, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probability),
        })
        fitted[name] = pipe

    metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    metrics.to_csv(OUT / "model_comparison.csv", index=False)
    best_name = str(metrics.iloc[0]["model"])
    best_model = fitted[best_name]

    all_prob = best_model.predict_proba(X)[:, 1]
    risk = df[["customer_id", *NUMERIC, *CATEGORICAL, TARGET]].copy()
    risk["churn_probability"] = all_prob.round(4)
    risk["risk_band"] = pd.cut(
        risk["churn_probability"], bins=[-0.001, 0.33, 0.66, 1.001],
        labels=["Low", "Medium", "High"]
    ).astype(str)
    risk.to_csv(OUT / "customer_churn_risk.csv", index=False)

    forest = fitted["Random Forest"]
    feature_names = forest.named_steps["preprocess"].get_feature_names_out()
    importances = forest.named_steps["model"].feature_importances_
    importance = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance["feature"] = importance["feature"].str.replace(r"^(numeric|categorical)__", "", regex=True)
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(OUT / "feature_importance.csv", index=False)

    churn_rate = float((df[TARGET] == "Yes").mean())
    report = {
        "dataset_type": "fictional synthetic demonstration data",
        "customers_analyzed": int(len(df)),
        "churned_customers": int((df[TARGET] == "Yes").sum()),
        "churn_rate": churn_rate,
        "best_model_by_roc_auc": best_name,
        "best_model_test_metrics": metrics.iloc[0].to_dict(),
        "risk_band_counts": risk["risk_band"].value_counts().to_dict(),
        "top_associated_model_features": importance.head(5).to_dict(orient="records"),
        "interpretation_note": "Synthetic demo only; feature importance is association within this model, not proof of causation."
    }
    (OUT / "churn_analysis_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    save_charts(df, importance)
    print("Customer churn analysis completed.")
    print(f"Customers analyzed: {len(df)}")
    print(f"Observed churn rate in this demo dataset: {churn_rate:.1%}")
    print(f"Best baseline model by ROC-AUC: {best_name}")
    print(f"Outputs saved to: {OUT}")


if __name__ == "__main__":
    main()
