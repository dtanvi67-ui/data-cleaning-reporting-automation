# Data Cleaning & Structural Validation

A beginner-friendly Python project that transforms a messy customer dataset into a consistent, analysis-ready CSV while keeping the original source data unchanged.

## Project objectives
- Detect and remove duplicate customer records.
- Standardize text, email, phone numbers, dates, and categories.
- Handle missing and invalid values carefully.
- Validate the dataset structure and required fields.
- Save a cleaned dataset, duplicate log, rejected-row log, and data-quality report.

## Tools
- Python 3.10+
- pandas

## Folder structure
```
data-cleaning-reporting-automation/
├── data/
│   └── raw_customers.csv
├── output/                 # Created automatically when the script runs
├── src/
│   └── cleaning_pipeline.py
├── tests/
│   └── test_pipeline.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Run in VS Code
1. Install Python 3.10 or newer.
2. Open this project folder in VS Code.
3. Open the terminal in VS Code.
4. (Optional) Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
   Activate it:
   - Windows: `.venv\\Scripts\\activate`
   - macOS/Linux: `source .venv/bin/activate`
5. Install the dependency:
   ```bash
   pip install -r requirements.txt
   ```
6. Run the pipeline from the project root:
   ```bash
   python src/cleaning_pipeline.py
   ```

## Output files
The script creates these files in `output/`:
- `cleaned_customers.csv` — cleaned, validated data.
- `duplicate_records.csv` — duplicate customer IDs removed from the main output.
- `rejected_records.csv` — rows that fail essential structural checks.
- `data_quality_report.json` — missing values, row counts, and validation summary.
- `quality_summary.csv` — compact before/after quality metrics.

## Cleaning and validation rules
1. Required columns are checked before processing.
2. Leading/trailing spaces are removed from text.
3. Emails are lowercased and checked for a basic email pattern.
4. Phone numbers are normalized to digits; blank values remain blank.
5. City and customer segment labels are standardized.
6. Dates are parsed into ISO format (YYYY-MM-DD).
7. Age and annual income are converted to numeric values.
8. Rows with a missing customer ID or invalid signup date are written to the rejected log.
9. Duplicate customer IDs are written to a separate log; the first valid occurrence is retained.
10. Missing age and income values are filled with the respective median. Missing descriptive values are labeled `Unknown`.

The raw CSV is never overwritten. Review the output logs before using cleaned data for decisions.

## Power BI dashboard idea
Import `output/cleaned_customers.csv` into Power BI and create:
- Cards: Total Customers, Average Age, Average Annual Income.
- Bar chart: Customers by City.
- Donut chart: Customers by Segment.
- Column chart: Customer signups by Month.
- Slicers: City and Segment.

## Run tests
```bash
python -m unittest discover -s tests -v
```

## Note
This is a learning project using a small sample dataset with fictional records. For real datasets, confirm business rules with the data owner before deleting or changing records.
