"""Focused checks for the combined Phase 1 results table."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.evaluation.accuracy import TABLE_FIELDS as ACCURACY_FIELDS
from src.evaluation.overlap_metrics import TABLE_FIELDS as OVERLAP_FIELDS
from src.evaluation.run_eval import ResultsTableError, combine_results, write_results_table
from src.evaluation.semantic_metrics import TABLE_FIELDS as SEMANTIC_FIELDS


class ResultsTableTests(unittest.TestCase):
    def test_joins_matching_tables_and_rejects_conflicting_counts(self) -> None:
        def write(path: Path, fields: tuple[str, ...], overrides: dict[str, str]) -> None:
            row = {field: "value" for field in fields}
            row.update(
                model_key="model",
                model_name="Model",
                setting_id="S1",
                sample_count="1000",
                parser_failure_count="0",
                scored_explanation_count="1000",
                missing_explanation_count="0",
            )
            row.update(overrides)
            with path.open("w", encoding="utf-8", newline="") as sink:
                writer = csv.DictWriter(sink, fieldnames=fields)
                writer.writeheader()
                writer.writerow({field: row[field] for field in fields})

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            paths = [base / name for name in ("accuracy.csv", "overlap.csv", "semantic.csv")]
            write(paths[0], ACCURACY_FIELDS, {"accuracy": "0.5"})
            write(paths[1], OVERLAP_FIELDS, {"mean_rouge_l": "0.2"})
            write(paths[2], SEMANTIC_FIELDS, {"mean_bertscore_f1": "0.6"})

            rows = combine_results(*paths)
            output = base / "combined.csv"
            write_results_table(rows, output)
            self.assertEqual(rows[0]["accuracy"], "0.5")
            self.assertEqual(rows[0]["mean_rouge_l"], "0.2")
            self.assertEqual(rows[0]["mean_bertscore_f1"], "0.6")
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)

            write(paths[2], SEMANTIC_FIELDS, {"sample_count": "999"})
            with self.assertRaisesRegex(ResultsTableError, "conflicting sample_count"):
                combine_results(*paths)


if __name__ == "__main__":
    unittest.main()
