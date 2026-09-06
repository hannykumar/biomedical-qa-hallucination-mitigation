"""Focused standard-library checks for C8 final-label accuracy."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.accuracy import (
    AccuracyEvaluationError,
    evaluate_accuracy,
    write_accuracy_artifacts,
)


class AccuracyTests(unittest.TestCase):
    def test_scores_unknown_reports_failures_writes_artifacts_and_checks_alignment(self) -> None:
        def record(sample_id: str, model: str, setting: str, prediction: str) -> dict[str, object]:
            return {
                "run_id": f"run-{model}-{setting}",
                "sample_id": sample_id,
                "model_key": model,
                "model_name": model.title(),
                "setting_id": setting,
                "decoding_method": "standard",
                "gold_final_decision": "yes",
                "parsed_final_answer": prediction,
                "parser_status": "error" if prediction == "unknown" else "ok",
            }

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            paths = []
            for model, setting, predictions in (
                ("model-a", "S1", ("yes", "unknown")),
                ("model-a", "S2", ("no", "yes")),
            ):
                path = base / f"{model}-{setting}.jsonl"
                path.write_text(
                    "".join(
                        json.dumps(record(f"sample-{index}", model, setting, prediction)) + "\n"
                        for index, prediction in enumerate(predictions, start=1)
                    ),
                    encoding="utf-8",
                )
                paths.append(path)

            rows, table = evaluate_accuracy(paths)
            write_accuracy_artifacts(
                rows,
                table,
                metrics_path=base / "metrics" / "accuracy.jsonl",
                table_path=base / "tables" / "accuracy.csv",
            )
            self.assertEqual([row["label_accuracy"] for row in rows], [1, 0, 0, 1])
            self.assertEqual(table[0]["accuracy"], 0.5)
            self.assertEqual(table[0]["parser_failure_count"], 1)
            self.assertEqual(
                len((base / "metrics" / "accuracy.jsonl").read_text().splitlines()), 4
            )
            self.assertEqual(
                (base / "tables" / "accuracy.csv").read_text().splitlines()[0],
                "model_key,model_name,setting_id,sample_count,correct_count,accuracy,parser_failure_count",
            )

            paths[1].write_text(
                "".join(
                    json.dumps(record(sample_id, "model-a", "S2", "yes")) + "\n"
                    for sample_id in ("sample-2", "sample-1")
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AccuracyEvaluationError, "sample order differs"):
                evaluate_accuracy(paths)


if __name__ == "__main__":
    unittest.main()
