"""Compute deterministic C8 final-label accuracy from parsed JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


LABELS = {"yes", "no", "maybe"}
PREDICTIONS = LABELS | {"unknown"}
METRIC_FIELDS = (
    "run_id",
    "sample_id",
    "model_key",
    "model_name",
    "setting_id",
    "decoding_method",
    "gold_final_decision",
    "parsed_final_answer",
    "label_accuracy",
    "parser_failure",
)
TABLE_FIELDS = (
    "model_key",
    "model_name",
    "setting_id",
    "sample_count",
    "correct_count",
    "accuracy",
    "parser_failure_count",
)


class AccuracyEvaluationError(ValueError):
    """Raised when parsed generation inputs violate the C8 contract."""


def _require_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise AccuracyEvaluationError(f"{field} must be a nonempty string")
    return value


def score_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the identifiers and deterministic label score for one record."""

    values = {field: _require_string(record, field) for field in METRIC_FIELDS[:6]}
    gold = _require_string(record, "gold_final_decision")
    predicted = _require_string(record, "parsed_final_answer")
    if gold not in LABELS:
        raise AccuracyEvaluationError(f"invalid gold_final_decision: {gold!r}")
    if predicted not in PREDICTIONS:
        raise AccuracyEvaluationError(f"invalid parsed_final_answer: {predicted!r}")
    parser_status = _require_string(record, "parser_status")
    if parser_status not in {"ok", "error"}:
        raise AccuracyEvaluationError(f"invalid parser_status: {parser_status!r}")
    return {
        **values,
        "gold_final_decision": gold,
        "parsed_final_answer": predicted,
        "label_accuracy": int(predicted == gold),
        "parser_failure": int(parser_status == "error"),
    }


def evaluate_accuracy(input_paths: Sequence[str | Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score parsed runs and require identical ordered sample IDs across them."""

    if not input_paths:
        raise AccuracyEvaluationError("at least one parsed JSONL path is required")
    rows: list[dict[str, Any]] = []
    reference_ids: list[str] | None = None
    seen_groups: set[tuple[str, str]] = set()

    for input_path in map(Path, input_paths):
        run_rows: list[dict[str, Any]] = []
        with input_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, Mapping):
                        raise AccuracyEvaluationError("record must be an object")
                    run_rows.append(score_record(record))
                except (json.JSONDecodeError, AccuracyEvaluationError) as error:
                    raise AccuracyEvaluationError(
                        f"invalid record at {input_path}:{line_number}: {error}"
                    ) from error
        if not run_rows:
            raise AccuracyEvaluationError(f"no records in {input_path}")
        sample_ids = [row["sample_id"] for row in run_rows]
        if len(sample_ids) != len(set(sample_ids)):
            raise AccuracyEvaluationError(f"duplicate sample_id in {input_path}")
        if reference_ids is None:
            reference_ids = sample_ids
        elif sample_ids != reference_ids:
            raise AccuracyEvaluationError(f"sample order differs in {input_path}")
        group = (run_rows[0]["model_key"], run_rows[0]["setting_id"])
        if group in seen_groups:
            raise AccuracyEvaluationError(f"duplicate model/setting group: {group}")
        if any((row["model_key"], row["setting_id"]) != group for row in run_rows):
            raise AccuracyEvaluationError(f"mixed model/setting identities in {input_path}")
        seen_groups.add(group)
        rows.extend(run_rows)

    totals: defaultdict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"sample_count": 0, "correct_count": 0, "parser_failure_count": 0}
    )
    for row in rows:
        counts = totals[(row["model_key"], row["model_name"], row["setting_id"])]
        counts["sample_count"] += 1
        counts["correct_count"] += row["label_accuracy"]
        counts["parser_failure_count"] += row["parser_failure"]
    table = [
        {
            "model_key": model_key,
            "model_name": model_name,
            "setting_id": setting_id,
            **counts,
            "accuracy": counts["correct_count"] / counts["sample_count"],
        }
        for (model_key, model_name, setting_id), counts in sorted(totals.items())
    ]
    return rows, table


def write_accuracy_artifacts(
    rows: Sequence[Mapping[str, Any]],
    table: Sequence[Mapping[str, Any]],
    *,
    metrics_path: str | Path,
    table_path: str | Path,
) -> None:
    """Atomically write per-sample JSONL and aggregate CSV artifacts."""

    metrics_target, table_target = Path(metrics_path), Path(table_path)
    for target in (metrics_target, table_target):
        target.parent.mkdir(parents=True, exist_ok=True)
    metrics_temporary = metrics_target.with_suffix(f"{metrics_target.suffix}.tmp")
    table_temporary = table_target.with_suffix(f"{table_target.suffix}.tmp")
    try:
        with metrics_temporary.open("w", encoding="utf-8") as sink:
            for row in rows:
                sink.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
        with table_temporary.open("w", encoding="utf-8", newline="") as sink:
            writer = csv.DictWriter(sink, fieldnames=TABLE_FIELDS)
            writer.writeheader()
            writer.writerows(table)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(metrics_temporary, metrics_target)
        os.replace(table_temporary, table_target)
    finally:
        metrics_temporary.unlink(missing_ok=True)
        table_temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute C8 final-label accuracy.")
    parser.add_argument("input_paths", nargs="+", type=Path)
    parser.add_argument("--metrics-path", required=True, type=Path)
    parser.add_argument("--table-path", required=True, type=Path)
    args = parser.parse_args()
    rows, table = evaluate_accuracy(args.input_paths)
    write_accuracy_artifacts(
        rows, table, metrics_path=args.metrics_path, table_path=args.table_path
    )
    print(json.dumps(table, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
