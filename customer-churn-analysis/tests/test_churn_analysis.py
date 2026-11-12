import sys
import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from churn_analysis import load_and_clean


class ChurnCleaningTests(unittest.TestCase):
    def test_cleaning_removes_duplicate_ids_and_invalid_target(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.csv"
            frame = pd.DataFrame([
                {"customer_id":"A1","tenure_months":12,"monthly_charges":50,
                 "total_charges":600,"support_calls":1,"contract":" Month-to-month ",
                 "internet_service":"DSL","tech_support":"No","payment_method":"Card","churn":"Yes"},
                {"customer_id":"A1","tenure_months":12,"monthly_charges":50,
                 "total_charges":600,"support_calls":1,"contract":"Month-to-month",
                 "internet_service":"DSL","tech_support":"No","payment_method":"Card","churn":"No"},
                {"customer_id":"A2","tenure_months":8,"monthly_charges":40,
                 "total_charges":320,"support_calls":0,"contract":"One year",
                 "internet_service":"No","tech_support":"Yes","payment_method":"Card","churn":"Unknown"},
            ])
            frame.to_csv(path, index=False)
            cleaned = load_and_clean(path)
            self.assertEqual(len(cleaned), 1)
            self.assertEqual(cleaned.iloc[0]["churn"], "Yes")


if __name__ == "__main__":
    unittest.main()
