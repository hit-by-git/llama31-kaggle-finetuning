import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from llama31_finetune.data import load_splits, validate_splits
from llama31_finetune.metrics import score_predictions


def misinfo_frame(prefix, event):
    return pd.DataFrame({
        "article": [f"{prefix} article one", f"{prefix} article two"],
        "post": [f"{prefix} post one", f"{prefix} post two"],
        "label": ["not_misinfo", "misinfo"],
        "event_id": [event, event],
    })


class DataMetricsTests(unittest.TestCase):
    def test_invalid_outputs_count_as_errors_and_confusion_column(self):
        frame = pd.DataFrame({"label": ["negative", "positive"], "variant": ["neutral", "misinfo"]})
        report = score_predictions(frame, ["negative", "I cannot answer"], ["negative", "positive"])
        self.assertEqual(report["accuracy"], 0.5)
        self.assertEqual(report["macro_f1"], 0.5)
        self.assertEqual(report["invalid_rate"], 0.5)
        self.assertEqual(report["confusion_matrix"]["counts"], [[1, 0, 0], [0, 0, 1]])
        self.assertIn("note", report["subgroups"]["variant"]["neutral"])
        json.dumps(report, allow_nan=False)

    def test_bootstrap_resamples_event_groups_and_is_reproducible(self):
        frame = pd.DataFrame({"label": ["n", "p", "n", "p"], "event_id": [10, 10, 20, 20]})
        first = score_predictions(frame, ["n", "p", "p", "n"], ["n", "p"], bootstrap=40, seed=9)
        second = score_predictions(frame, ["n", "p", "p", "n"], ["n", "p"], bootstrap=40, seed=9)
        self.assertEqual(first, second)
        ci = first["confidence_intervals"]
        self.assertEqual(ci["resampling_unit"], "event_id")
        self.assertEqual(ci["n_units"], 2)
        self.assertEqual(ci["metrics"]["accuracy"], {"lower": 0.0, "upper": 1.0})

    def test_validator_detects_event_leakage_even_for_different_posts(self):
        parts = {"train": misinfo_frame("train", 1), "val": misinfo_frame("val", 2),
                 "test": misinfo_frame("test", 1)}
        with self.assertRaisesRegex(ValueError, "Event leakage"):
            validate_splits(parts, "misinfo")

    def test_validator_normalizes_duplicate_whitespace_and_case(self):
        parts = {"train": misinfo_frame("train", 1), "val": misinfo_frame("val", 2),
                 "test": misinfo_frame("test", 3)}
        parts["test"].loc[0, "article"] = "  TRAIN  ARTICLE ONE "
        parts["test"].loc[0, "post"] = "TRAIN POST ONE"
        with self.assertRaisesRegex(ValueError, "Duplicate inputs"):
            validate_splits(parts, "misinfo")

    def test_constraint_cleanup_preserves_test_and_exposes_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "constraint_aaai2021"
            directory.mkdir()
            pd.DataFrame({"tweet": ["train real", "train fake", "TEST FAKE"],
                          "label": ["real", "fake", "real"]}).to_csv(directory / "train.csv", index=False)
            pd.DataFrame({"tweet": ["val real", "val fake"], "label": ["real", "fake"]}).to_csv(
                directory / "validation.csv", index=False)
            pd.DataFrame({"tweet": ["test real", "test fake"], "label": ["real", "fake"]}).to_csv(
                directory / "test.csv", index=False)
            parts = load_splits(root, "constraint")
            self.assertEqual(len(parts["train"]), 2)
            self.assertEqual(len(parts["test"]), 2)
            self.assertEqual(parts["test"].label.tolist(), ["real", "fake"])
            self.assertEqual(parts["train"].attrs["cleaning_audit"]["exclusions"]["train_cross_split_duplicates"], 1)
            self.assertEqual(set(parts["train"].task), {"constraint"})

    def test_reject_prediction_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Prediction count"):
            score_predictions(pd.DataFrame({"label": ["n", "p"]}), ["n"], ["n", "p"])


if __name__ == "__main__":
    unittest.main()
