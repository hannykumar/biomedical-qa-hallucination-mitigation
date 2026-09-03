"""Unit tests for C1 dataset statistics and reporting."""

from __future__ import annotations

import unittest

from src.data.load_pubmedqa import (
    build_pubmedqa_dataset,
    render_inspection_report,
    resolve_dataset_revision,
    summarize_examples,
)
from src.data.preprocess import normalize_pubmedqa_record

from tests.test_preprocess import source_record


class DatasetSummaryTests(unittest.TestCase):
    def test_limited_build_cannot_overwrite_canonical_artifacts(self) -> None:
        with self.assertRaisesRegex(ValueError, "limited smoke build"):
            build_pubmedqa_dataset(limit=3)

    def test_immutable_revision_does_not_require_hub_resolution(self) -> None:
        revision = "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
        self.assertEqual(resolve_dataset_revision("unused/dataset", revision), revision.lower())

    def test_summary_counts_labels_ids_and_context_ranges(self) -> None:
        yes = normalize_pubmedqa_record(
            source_record(pubid=1, label="yes"), source_row_index=0
        )
        no = normalize_pubmedqa_record(
            source_record(pubid=2, label="no"), source_row_index=1
        )

        summary = summarize_examples([yes, no])

        self.assertEqual(summary["label_counts"], {"no": 1, "yes": 1})
        self.assertEqual(summary["unique_sample_ids"], 2)
        self.assertEqual(summary["context_sections"]["minimum"], 2)
        self.assertEqual(summary["context_sections"]["maximum"], 2)

    def test_report_contains_revision_and_distribution(self) -> None:
        metadata = {
            "created_at_utc": "2026-08-11T00:00:00+00:00",
            "source": {
                "dataset_id": "qiaojin/PubMedQA",
                "subset": "pqa_labeled",
                "split": "train",
                "resolved_revision": "abc123",
                "dataset_fingerprint": "fingerprint",
            },
            "artifact": {"row_count": 2, "sha256": "checksum"},
            "statistics": {
                "label_counts": {"no": 1, "yes": 1},
                "unique_sample_ids": 2,
                "context_sections": {"minimum": 1, "mean": 2.0, "maximum": 3},
                "context_characters": {
                    "minimum": 100,
                    "mean": 200.0,
                    "maximum": 300,
                },
            },
        }

        report = render_inspection_report(metadata)

        self.assertIn("`abc123`", report)
        self.assertIn("| no | 1 |", report)
        self.assertIn("| yes | 1 |", report)


if __name__ == "__main__":
    unittest.main()
