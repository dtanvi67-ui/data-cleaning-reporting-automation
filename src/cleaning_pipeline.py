"""Clean and structurally validate the sample customer dataset."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "raw_customers.csv"
OUTPUT_DIR = ROOT / "output"

REQUIRED_COLUMNS = [
    "customer_id", "customer_name", "email", "phone", "city",
    "signup_date", "age", "annual_income", "segment"
]
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def standardize_text(series: pd.Series) -> pd.Series:
    """Trim whitespace and collapse repeated internal spaces."""
    return series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)


def load_data(path: Path = INPUT_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    return pd.read_csv(path, dtype="string")


def check_structure(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    check_structure(raw)
    original_rows = len(raw)
    df = raw[REQUIRED_COLUMNS].copy()

    # Normalize whitespace and empty strings without changing the source file.
    for col in df.columns:
        if df[col].dtype == "string":
            df[col] = standardize_text(df[col])
            df[col] = df[col].replace("", pd.NA)

    # Standardize key text fields.
    df["customer_id"] = df["customer_id"].str.upper()
    df["customer_name"] = df["customer_name"].str.title()
    df["email"] = df["email"].str.lower()
    df["city"] = df["city"].str.title()
    df["segment"] = df["segment"].str.title()

    # Convert types and record essential structural failures.
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce", dayfirst=True)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["annual_income"] = pd.to_numeric(df["annual_income"], errors="coerce")

    invalid_id = df["customer_id"].isna()
    invalid_date = df["signup_date"].isna()
    invalid_age = df["age"].notna() & ~df["age"].between(13, 100)
    invalid_income = df["annual_income"].notna() & (df["annual_income"] < 0)

    rejected_mask = invalid_id | invalid_date | invalid_age | invalid_income
    rejected = df.loc[rejected_mask].copy()
    if not rejected.empty:
        reasons = []
        for idx in rejected.index:
            row_reasons = []
            if invalid_id.loc[idx]:
                row_reasons.append("missing_customer_id")
            if invalid_date.loc[idx]:
                row_reasons.append("invalid_or_missing_signup_date")
            if invalid_age.loc[idx]:
                row_reasons.append("age_out_of_allowed_range")
            if invalid_income.loc[idx]:
                row_reasons.append("negative_annual_income")
            reasons.append(";".join(row_reasons))
        rejected["rejection_reason"] = reasons

    valid = df.loc[~rejected_mask].copy()

    # Normalize phone values to digits only. Missing phone values are preserved.
    valid["phone"] = valid["phone"].str.replace(r"\D", "", regex=True)
    valid.loc[valid["phone"] == "", "phone"] = pd.NA

    # Flag malformed email values, but do not discard otherwise valid customers.
    valid["email_valid"] = valid["email"].fillna("").apply(
        lambda value: bool(EMAIL_PATTERN.match(value)) if value else False
    )

    # Separate duplicate customer IDs; keep the first valid record.
    duplicate_mask = valid.duplicated(subset=["customer_id"], keep="first")
    duplicates = valid.loc[duplicate_mask].copy()
    valid = valid.loc[~duplicate_mask].copy()

    # Fill missing descriptive values and numeric values using transparent rules.
    for col in ["customer_name", "city", "segment"]:
        valid[col] = valid[col].fillna("Unknown")
    valid["email"] = valid["email"].fillna("Unknown")
    valid["phone"] = valid["phone"].fillna("Unknown")

    for col in ["age", "annual_income"]:
        median_value = valid[col].median()
        valid[col] = valid[col].fillna(median_value if pd.notna(median_value) else 0)

    valid["signup_date"] = valid["signup_date"].dt.strftime("%Y-%m-%d")
    valid["age"] = valid["age"].round().astype("int64")
    valid["annual_income"] = valid["annual_income"].round(2)

    # Keep a stable, analysis-friendly column order.
    cleaned = valid[
        ["customer_id", "customer_name", "email", "email_valid", "phone",
         "city", "signup_date", "age", "annual_income", "segment"]
    ].reset_index(drop=True)

    report = {
        "input_file": str(INPUT_FILE.name),
        "rows_before": int(original_rows),
        "rows_after": int(len(cleaned)),
        "rejected_rows": int(len(rejected)),
        "duplicate_rows": int(len(duplicates)),
        "missing_values_before": {
            str(k): int(v) for k, v in raw[REQUIRED_COLUMNS].isna().sum().items()
        },
        "missing_values_after": {
            str(k): int(v) for k, v in cleaned.isna().sum().items()
        },
        "invalid_email_count": int((~cleaned["email_valid"]).sum()),
        "validation_checks": {
            "required_columns_present": True,
            "customer_id_not_missing": bool(cleaned["customer_id"].notna().all()),
            "signup_date_valid": bool(cleaned["signup_date"].notna().all()),
            "customer_id_unique": bool(cleaned["customer_id"].is_unique),
            "age_in_range_13_to_100": bool(cleaned["age"].between(13, 100).all()),
            "annual_income_non_negative": bool((cleaned["annual_income"] >= 0).all()),
        },
    }
    report["overall_validation_passed"] = all(report["validation_checks"].values())
    return cleaned, duplicates, rejected, report


def save_outputs(raw: pd.DataFrame, cleaned: pd.DataFrame,
                 duplicates: pd.DataFrame, rejected: pd.DataFrame,
                 report: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_DIR / "cleaned_customers.csv", index=False)
    duplicates.to_csv(OUTPUT_DIR / "duplicate_records.csv", index=False)
    rejected.to_csv(OUTPUT_DIR / "rejected_records.csv", index=False)

    with (OUTPUT_DIR / "data_quality_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    summary = pd.DataFrame([
        {"metric": "Rows before cleaning", "value": len(raw)},
        {"metric": "Rows after cleaning", "value": len(cleaned)},
        {"metric": "Duplicate rows removed", "value": len(duplicates)},
        {"metric": "Rejected rows", "value": len(rejected)},
        {"metric": "Invalid email values", "value": report["invalid_email_count"]},
    ])
    summary.to_csv(OUTPUT_DIR / "quality_summary.csv", index=False)


def main() -> None:
    raw = load_data()
    cleaned, duplicates, rejected, report = clean_data(raw)
    save_outputs(raw, cleaned, duplicates, rejected, report)
    print("Data cleaning and structural validation completed.")
    print(f"Input rows: {report['rows_before']}")
    print(f"Clean rows: {report['rows_after']}")
    print(f"Duplicates: {report['duplicate_rows']}")
    print(f"Rejected rows: {report['rejected_rows']}")
    print(f"All structural checks passed: {report['overall_validation_passed']}")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
