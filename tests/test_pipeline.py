import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cleaning_pipeline import REQUIRED_COLUMNS, check_structure, clean_data


class CleaningPipelineTests(unittest.TestCase):
    def sample_data(self):
        return pd.DataFrame([
            ["C001", "  Asha Patil ", "ASHA@EXAMPLE.COM", "98765 43210",
             " mumbai ", "2025-01-12", "24", "450000", " regular "],
            ["C001", "Asha Duplicate", "asha2@example.com", "9876543211",
             "Pune", "2025-02-01", "25", "500000", "New"],
            ["", "No ID", "bad-email", "9876543212",
             "Pune", "2025-03-01", "22", "300000", "Regular"],
            ["C004", "Valid Person", "valid@example.com", "9876543213",
             "Nashik", "not-a-date", "30", "400000", "Premium"],
        ], columns=REQUIRED_COLUMNS, dtype="string")

    def test_missing_required_column_raises_error(self):
        with self.assertRaises(ValueError):
            check_structure(pd.DataFrame({"customer_id": ["C001"]}))

    def test_duplicate_is_separated_and_invalid_rows_rejected(self):
        cleaned, duplicates, rejected, report = clean_data(self.sample_data())
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(len(rejected), 2)
        self.assertEqual(len(cleaned), 1)
        self.assertTrue(cleaned["customer_id"].is_unique)
        self.assertTrue(report["overall_validation_passed"])

    def test_text_is_standardized(self):
        cleaned, _, _, _ = clean_data(self.sample_data())
        self.assertEqual(cleaned.iloc[0]["customer_name"], "Asha Patil")
        self.assertEqual(cleaned.iloc[0]["city"], "Mumbai")


if __name__ == "__main__":
    unittest.main()
