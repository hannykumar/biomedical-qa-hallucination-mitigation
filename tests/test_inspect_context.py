"""Known reference/context relationships, without network or model loading."""

import tempfile
import unittest
from pathlib import Path

from src.data.inspect_context import inspect_context
from src.data.preprocess import normalize_pubmedqa_record, write_normalized_csv
from tests.test_preprocess import source_record


class ContextInspectionTests(unittest.TestCase):
    def test_exact_normalized_and_absent_reference(self) -> None:
        records = []
        for index, answer in enumerate(("Background evidence.", "BACKGROUND   evidence.", "Absent conclusion.")):
            record = source_record(pubid=index + 1)
            record["long_answer"] = answer
            records.append(normalize_pubmedqa_record(record, source_row_index=index))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.csv"
            write_normalized_csv(records, path)
            report = inspect_context(path)
        self.assertEqual(report["samples"], 3)
        self.assertEqual(report["exact_reference_in_context"], 1)
        self.assertEqual(report["normalized_reference_in_context"], 2)
        self.assertEqual(len(report["dataset_sha256"]), 64)
