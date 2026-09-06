"""Focused dependency-light checks for C7 output parsing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.generation.parse_outputs import OutputParseError, parse_files, parse_raw_output


class OutputParserTests(unittest.TestCase):
    def test_exact_and_observed_normalized_formats(self) -> None:
        exact = parse_raw_output("Final answer: yes\nExplanation: Supported.")
        same_line = parse_raw_output(
            "Final answer: no. Explanation: Not supported."
        )
        joined = parse_raw_output("Final answer: maybeExplanation: Uncertain.")

        self.assertEqual(exact["parsed_final_answer"], "yes")
        self.assertEqual(exact["prompt_format_compliance"], "exact")
        self.assertEqual(same_line["parsed_explanation"], "Not supported.")
        self.assertEqual(same_line["prompt_format_compliance"], "normalized")
        self.assertEqual(joined["parsed_final_answer"], "maybe")
        self.assertEqual(joined["parsed_explanation"], "Uncertain.")

    def test_flags_ambiguous_missing_and_malformed_fields(self) -> None:
        ambiguous = parse_raw_output(
            "Final answer: yes or no\nExplanation: Conflicting."
        )
        missing_label = parse_raw_output("Explanation: Evidence only.")
        malformed = parse_raw_output("Final answer: certainly\nExplanation:")

        self.assertEqual(ambiguous["parsed_final_answer"], "unknown")
        self.assertIn("ambiguous_label", ambiguous["parser_errors"])
        self.assertIn("missing_label", missing_label["parser_errors"])
        self.assertEqual(malformed["parsed_final_answer"], "unknown")
        self.assertEqual(
            malformed["parser_errors"],
            ["malformed_label", "malformed_explanation"],
        )

    def test_preserves_raw_records_and_reports_group_failure_rate(self) -> None:
        records = [
            {
                "sample_id": "sample-1",
                "model_key": "model",
                "setting_id": "S1",
                "raw_output": "Final answer: yes\nExplanation: Evidence.",
                "gold_final_decision": "yes",
            },
            {
                "sample_id": "sample-2",
                "model_key": "model",
                "setting_id": "S1",
                "raw_output": "Final answer: no",
                "gold_final_decision": "no",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw_path = base / "run.jsonl"
            raw_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            summary = parse_files([raw_path], output_dir=base / "parsed")
            parsed = [
                json.loads(line)
                for line in (base / "parsed" / "run.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(parsed[0]["raw_output"], records[0]["raw_output"])
        self.assertEqual(parsed[0]["gold_final_decision"], "yes")
        self.assertEqual(parsed[1]["parsed_final_answer"], "no")
        self.assertEqual(summary["groups"][0]["parser_failures"], 1)
        self.assertEqual(summary["groups"][0]["parser_failure_rate"], 0.5)

    def test_refuses_to_overwrite_raw_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(OutputParseError, "must not overwrite"):
                parse_files([path], output_dir=path.parent)


if __name__ == "__main__":
    unittest.main()
