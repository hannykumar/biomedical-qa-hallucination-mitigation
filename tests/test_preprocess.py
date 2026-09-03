"""Unit tests for dependency-light PubMedQA normalization."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.data.preprocess import (
    PubMedQAValidationError,
    normalize_pubmedqa_record,
    normalize_pubmedqa_records,
    write_normalized_csv,
)


def source_record(
    *,
    pubid: int = 12345,
    label: str = "yes",
) -> dict[str, object]:
    return {
        "pubid": pubid,
        "question": " Does the intervention help? ",
        "context": {
            "contexts": [" Background evidence. ", " Results evidence. "],
            "labels": ["BACKGROUND", "RESULTS"],
            "meshes": ["Humans", "Treatment Outcome"],
            "reasoning_required_pred": ["yes", "yes", "yes"],
            "reasoning_free_pred": ["yes", "yes", "yes"],
        },
        "long_answer": " The intervention was beneficial. ",
        "final_decision": label,
    }


class NormalizePubMedQARecordTests(unittest.TestCase):
    def test_normalizes_expected_fields_and_preserves_section_order(self) -> None:
        example = normalize_pubmedqa_record(source_record(), source_row_index=0)

        self.assertEqual(example.sample_id, "pubmedqa-pqa_labeled-12345")
        self.assertEqual(example.pubid, 12345)
        self.assertEqual(example.question, "Does the intervention help?")
        self.assertEqual(
            example.context,
            "Background evidence.\n\nResults evidence.",
        )
        self.assertEqual(
            example.context_sections,
            ("Background evidence.", "Results evidence."),
        )
        self.assertEqual(example.context_labels, ("BACKGROUND", "RESULTS"))
        self.assertEqual(example.gold_long_answer, "The intervention was beneficial.")
        self.assertEqual(example.gold_final_decision, "yes")

    def test_normalizes_label_case(self) -> None:
        example = normalize_pubmedqa_record(
            source_record(label=" Maybe "), source_row_index=0
        )
        self.assertEqual(example.gold_final_decision, "maybe")

    def test_sample_id_does_not_depend_on_row_order(self) -> None:
        first = normalize_pubmedqa_record(source_record(), source_row_index=0)
        moved = normalize_pubmedqa_record(source_record(), source_row_index=99)
        self.assertEqual(first.sample_id, moved.sample_id)

    def test_rejects_unknown_label(self) -> None:
        with self.assertRaisesRegex(PubMedQAValidationError, "not one of"):
            normalize_pubmedqa_record(source_record(label="sometimes"), source_row_index=0)

    def test_rejects_missing_required_field(self) -> None:
        record = source_record()
        del record["long_answer"]
        with self.assertRaisesRegex(PubMedQAValidationError, "missing required fields"):
            normalize_pubmedqa_record(record, source_row_index=0)

    def test_rejects_empty_context_section(self) -> None:
        record = source_record()
        record["context"]["contexts"][1] = " "  # type: ignore[index]
        with self.assertRaisesRegex(PubMedQAValidationError, "must not be empty"):
            normalize_pubmedqa_record(record, source_row_index=0)

    def test_rejects_misaligned_context_labels(self) -> None:
        record = source_record()
        record["context"]["labels"] = ["BACKGROUND"]  # type: ignore[index]
        with self.assertRaisesRegex(PubMedQAValidationError, "has 1 items"):
            normalize_pubmedqa_record(record, source_row_index=0)


class NormalizePubMedQARecordsTests(unittest.TestCase):
    def test_rejects_duplicate_source_pubids(self) -> None:
        with self.assertRaisesRegex(PubMedQAValidationError, "duplicate sample_id"):
            normalize_pubmedqa_records([source_record(), source_record()])

    def test_rejects_empty_dataset(self) -> None:
        with self.assertRaisesRegex(PubMedQAValidationError, "no records"):
            normalize_pubmedqa_records([])

    def test_writes_structured_fields_as_json_in_csv(self) -> None:
        example = normalize_pubmedqa_record(source_record(), source_row_index=0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "clean.csv"
            write_normalized_csv([example], output)

            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            json.loads(rows[0]["context_sections_json"]),
            ["Background evidence.", "Results evidence."],
        )
        self.assertEqual(rows[0]["gold_final_decision"], "yes")


if __name__ == "__main__":
    unittest.main()
