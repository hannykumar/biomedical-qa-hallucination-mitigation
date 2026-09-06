"""Dependency-light checks for C10 semantic metric orchestration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.semantic_metrics import (
    evaluate_semantic,
    load_semantic_config,
    write_semantic_artifacts,
)


class SemanticMetricTests(unittest.TestCase):
    def test_pinned_config_and_missing_explanation_aggregation(self) -> None:
        config = load_semantic_config()
        self.assertEqual(config["device"], "cpu")
        self.assertEqual(len(config["bertscore"]["model_revision"]), 40)

        records = []
        for sample_id, explanation in (("sample-1", "evidence"), ("sample-2", "")):
            records.append(
                {
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
                    "gold_long_answer": "gold evidence",
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            rows, table = evaluate_semantic(
                [path],
                config=config,
                score_function=lambda predictions, references, _: ([0.8], [0.6]),
            )
            write_semantic_artifacts(
                rows,
                table,
                metrics_path=Path(directory) / "metrics.jsonl",
                table_path=Path(directory) / "table.csv",
            )
            self.assertEqual(
                len((Path(directory) / "metrics.jsonl").read_text().splitlines()), 2
            )
            self.assertIn(
                "mean_bertscore_f1", (Path(directory) / "table.csv").read_text()
            )

        self.assertEqual(rows[0]["bertscore_f1"], 0.8)
        self.assertIsNone(rows[1]["cosine_similarity"])
        self.assertEqual(table[0]["scored_explanation_count"], 1)
        self.assertEqual(table[0]["missing_explanation_count"], 1)
        self.assertEqual(table[0]["mean_cosine_similarity"], 0.6)
        self.assertEqual(table[0]["parser_failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
