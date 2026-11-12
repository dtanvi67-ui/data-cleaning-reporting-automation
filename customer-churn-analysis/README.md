# Customer Churn Analysis & Prediction

A beginner-friendly end-to-end customer churn project using Python, pandas, scikit-learn and Matplotlib. It explores customer behavior, identifies churn patterns, trains classification models, and creates retention-focused outputs.

> The included data generator creates fictional demo records so the project runs immediately. Replace it with a properly licensed real dataset for a real-world case study. Synthetic demo results are not business findings.

## Features
- Generate a reproducible sample subscription dataset.
- Clean and validate data.
- Explore churn by contract, tenure, support, payment method and monthly charges.
- Train Logistic Regression and Random Forest models using a preprocessing pipeline.
- Evaluate with accuracy, precision, recall, F1 and ROC-AUC.
- Export churn probabilities and Low / Medium / High risk bands.
- Save charts, feature importance and a concise JSON report.

## Tools
Python 3.10+, pandas, NumPy, scikit-learn, Matplotlib.

## Run in VS Code
```bash
python -m venv .venv
```
Windows:
```bash
.venv\Scripts\activate
```
macOS/Linux:
```bash
source .venv/bin/activate
```
Install:
```bash
pip install -r requirements.txt
```
Generate data and run analysis:
```bash
python src/generate_sample_data.py
python src/churn_analysis.py
```

## Outputs
The script creates an `output/` folder containing:
- `model_comparison.csv`
- `customer_churn_risk.csv`
- `feature_importance.csv`
- `churn_analysis_report.json`
- `churn_distribution.png`
- `churn_by_contract.png`
- `churn_by_tenure_group.png`
- `feature_importance.png`

## Dataset fields
Customer ID, tenure months, contract type, monthly charges, total charges, internet service, tech support, payment method, support calls, and churn label.

## Suggested Power BI dashboard
Import `output/customer_churn_risk.csv`:
- KPI cards: customers, churned customers, churn rate, monthly revenue at risk.
- Bar chart: churn rate by contract.
- Column chart: churn rate by tenure group.
- Donut chart: churn by internet service.
- Table: high-risk customer list.
- Slicers: contract, internet service, payment method and risk band.

## Responsible interpretation
Churn associations do not prove causation. A risk score is a prioritization aid, not a certainty about an individual customer. Validate recommendations with real data, business context and fair treatment.
