"""Focused standard-library checks for C9 overlap and length metrics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.overlap_metrics import (
    evaluate_overlap,
    rouge_l,
    write_overlap_artifacts,
)


class OverlapMetricTests(unittest.TestCase):
    def test_rouge_l_and_missing_explanation_aggregation(self) -> None:
        self.assertAlmostEqual(rouge_l("Alpha beta gamma.", "alpha delta gamma"), 2 / 3)
        self.assertIsNone(rouge_l("", "reference text"))

        def record(sample_id: str, explanation: str, generated_tokens: int) -> dict[str, object]:
            return {
                "run_id": "run-model-S1",
                "sample_id": sample_id,
                "model_key": "model",
                "model_name": "Model",
                "setting_id": "S1",
                "decoding_method": "standard",
                "gold_final_decision": "yes",
                "parsed_final_answer": "yes",
                "parser_status": "error" if not explanation else "ok",
                "parsed_explanation": explanation,
                "gold_long_answer": "alpha delta gamma",
                "generated_tokens": generated_tokens,
            }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            path.write_text(
                "".join(
                    json.dumps(record(*values)) + "\n"
                    for values in (
                        ("sample-1", "alpha beta gamma", 10),
                        ("sample-2", "", 20),
                    )
                ),
                encoding="utf-8",
            )
            rows, table = evaluate_overlap([path])
            write_overlap_artifacts(
                rows,
                table,
                metrics_path=Path(directory) / "metrics.jsonl",
                table_path=Path(directory) / "table.csv",
            )

            self.assertEqual(
                len((Path(directory) / "metrics.jsonl").read_text().splitlines()), 2
            )
            self.assertIn(
                "mean_rouge_l", (Path(directory) / "table.csv").read_text()
            )

        self.assertAlmostEqual(rows[0]["rouge_l"], 2 / 3)
        self.assertIsNone(rows[1]["rouge_l"])
        self.assertEqual(table[0]["scored_explanation_count"], 1)
        self.assertEqual(table[0]["missing_explanation_count"], 1)
        self.assertAlmostEqual(table[0]["mean_rouge_l"], 2 / 3)
        self.assertEqual(table[0]["mean_generated_tokens"], 15)
        self.assertEqual(table[0]["parser_failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
