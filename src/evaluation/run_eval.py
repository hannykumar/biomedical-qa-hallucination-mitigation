"""Join validated C8-C10 aggregate tables into one Phase 1 results table."""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.accuracy import TABLE_FIELDS as ACCURACY_FIELDS
from src.evaluation.overlap_metrics import TABLE_FIELDS as OVERLAP_FIELDS
from src.evaluation.semantic_metrics import TABLE_FIELDS as SEMANTIC_FIELDS


KEY_FIELDS = ("model_key", "setting_id")
RESULT_FIELDS = tuple(
    dict.fromkeys((*ACCURACY_FIELDS, *OVERLAP_FIELDS, *SEMANTIC_FIELDS))
)


class ResultsTableError(ValueError):
    """Raised when aggregate inputs cannot form one trustworthy table."""


def _read_table(
    path: str | Path, expected_fields: Sequence[str]
) -> dict[tuple[str, str], dict[str, str]]:
    source_path = Path(path)
    with source_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != tuple(expected_fields):
            raise ResultsTableError(f"unexpected columns in {source_path}")
        rows: dict[tuple[str, str], dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            key = tuple(row[field] for field in KEY_FIELDS)
            if not all(key):
                raise ResultsTableError(f"empty model/setting key at {source_path}:{line_number}")
            if key in rows:
                raise ResultsTableError(f"duplicate model/setting key in {source_path}: {key}")
            rows[key] = dict(row)
    if not rows:
        raise ResultsTableError(f"no aggregate rows in {source_path}")
    return rows


def combine_results(
    accuracy_path: str | Path,
    overlap_path: str | Path,
    semantic_path: str | Path,
) -> list[dict[str, str]]:
    """Validate and join C8-C10 aggregates by model and setting."""

    tables = (
        _read_table(accuracy_path, ACCURACY_FIELDS),
        _read_table(overlap_path, OVERLAP_FIELDS),
        _read_table(semantic_path, SEMANTIC_FIELDS),
    )
    keys = set(tables[0])
    if any(set(table) != keys for table in tables[1:]):
        raise ResultsTableError("model/setting keys differ across aggregate tables")

    results = []
    for key in sorted(keys):
        combined: dict[str, str] = {}
        for table in tables:
            for field, value in table[key].items():
                if field in combined and combined[field] != value:
                    raise ResultsTableError(f"conflicting {field} for {key}")
                combined[field] = value
        results.append({field: combined[field] for field in RESULT_FIELDS})
    return results


def write_results_table(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Atomically write the combined Phase 1 CSV."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as sink:
            writer = csv.DictWriter(sink, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine C8-C10 aggregate results.")
    parser.add_argument("--accuracy-table", required=True, type=Path)
    parser.add_argument("--overlap-table", required=True, type=Path)
    parser.add_argument("--semantic-table", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()
    rows = combine_results(
        args.accuracy_table, args.overlap_table, args.semantic_table
    )
    write_results_table(rows, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
