"""Focused dependency-light checks for C7 output parsing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.generation.parse_outputs import OutputParseError, parse_files, parse_raw_output, parse_record


class OutputParserTests(unittest.TestCase):
    def test_json_routes_by_record_metadata_and_preserves_raw(self) -> None:
        raw = '{"decision":"no","long_answer":"Evidence with \\"quotes\\" and β."}'
        record = {"sample_id": "one", "model_key": "model", "setting_id": "S2",
                  "prompt_format": "json", "raw_output": raw, "context": "Evidence."}
        parsed = parse_record(record)
        self.assertEqual(parsed["raw_output"], raw)
        self.assertEqual(parsed["context"], "Evidence.")
        self.assertEqual(parsed["parsed_final_answer"], "no")
        self.assertEqual(parsed["parsed_explanation"], 'Evidence with "quotes" and β.')
        self.assertEqual(parsed["parser_status"], "ok")
        self.assertEqual(parsed["prompt_format_compliance"], "exact")

    def test_json_rejects_syntax_wrappers_duplicates_and_wrong_schema(self) -> None:
        valid = '{"decision":"yes","long_answer":"Evidence."}'
        bad = [valid[:-1], '```json\n' + valid + '\n```', valid + ' trailing',
               '{"decision":"yes","decision":"no","long_answer":"Evidence."}',
               '[]', 'null', '{"decision":true,"long_answer":12}',
               '{"decision":"YES","long_answer":""}',
               '{"decision":"yes","long_answer":"Evidence.","extra":1}',
               '{"decision":"yes","long_answer":NaN}']
        for raw in bad:
            with self.subTest(raw=raw):
                parsed = parse_raw_output(raw, prompt_format="json")
                self.assertEqual(parsed["parser_status"], "error")
                self.assertEqual(parsed["prompt_format_compliance"], "noncompliant")
        self.assertEqual(parse_raw_output(valid[:-1], prompt_format="json")["parsed_final_answer"], "unknown")

    def test_json_keeps_explicit_label_when_answer_missing(self) -> None:
        parsed = parse_raw_output('{"decision":"maybe"}', prompt_format="json")
        self.assertEqual(parsed["parsed_final_answer"], "maybe")
        self.assertEqual(parsed["parsed_explanation"], "")
        self.assertIn("missing_explanation", parsed["parser_errors"])

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
