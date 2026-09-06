"""Compute deterministic C9 ROUGE-L and generated-output length metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from src.evaluation.accuracy import AccuracyEvaluationError, evaluate_accuracy


TOKEN = re.compile(r"\w+", re.UNICODE)
TABLE_FIELDS = (
    "model_key",
    "model_name",
    "setting_id",
    "sample_count",
    "scored_explanation_count",
    "missing_explanation_count",
    "mean_rouge_l",
    "rouge_l_stddev",
    "mean_generated_tokens",
    "generated_tokens_stddev",
    "parser_failure_count",
)


class OverlapEvaluationError(ValueError):
    """Raised when a parsed record violates the C9 metric contract."""


def rouge_l(prediction: str, reference: str) -> float | None:
    """Return case-insensitive word-level ROUGE-L F1 without stemming."""

    if not isinstance(prediction, str) or not isinstance(reference, str):
        raise OverlapEvaluationError("ROUGE-L inputs must be strings")
    predicted = TOKEN.findall(prediction.casefold())
    expected = TOKEN.findall(reference.casefold())
    if not expected:
        raise OverlapEvaluationError("ROUGE-L reference must not be empty")
    if not predicted:
        return None

    previous = [0] * (len(expected) + 1)
    for predicted_token in predicted:
        current = [0]
        for index, expected_token in enumerate(expected, start=1):
            current.append(
                previous[index - 1] + 1
                if predicted_token == expected_token
                else max(previous[index], current[-1])
            )
        previous = current
    overlap = previous[-1]
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def _score_record(
    record: Mapping[str, Any], accuracy_row: Mapping[str, Any]
) -> dict[str, Any]:
    explanation = record.get("parsed_explanation")
    reference = record.get("gold_long_answer")
    generated_tokens = record.get("generated_tokens")
    if not isinstance(explanation, str):
        raise OverlapEvaluationError("parsed_explanation must be a string")
    if not isinstance(reference, str) or not reference.strip():
        raise OverlapEvaluationError("gold_long_answer must be a nonempty string")
    if (
        isinstance(generated_tokens, bool)
        or not isinstance(generated_tokens, int)
        or generated_tokens < 0
    ):
        raise OverlapEvaluationError("generated_tokens must be a nonnegative integer")
    return {
        **{field: accuracy_row[field] for field in accuracy_row if field not in {
            "gold_final_decision", "parsed_final_answer", "label_accuracy"
        }},
        "rouge_l": rouge_l(explanation, reference),
        "generated_tokens": generated_tokens,
        "missing_explanation": int(not explanation.strip()),
    }


def evaluate_overlap(
    input_paths: Sequence[str | Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score C9 metrics after reusing C8 record and alignment validation."""

    try:
        accuracy_rows, _ = evaluate_accuracy(input_paths)
    except AccuracyEvaluationError as error:
        raise OverlapEvaluationError(str(error)) from error
    # ponytail: reading 4,000 small records twice reuses C8's proven validation;
    # extract shared I/O only if later metric volume makes this measurable.
    records = []
    for path in map(Path, input_paths):
        with path.open(encoding="utf-8") as source:
            records.extend(json.loads(line) for line in source if line.strip())
    rows = [_score_record(record, accuracy_row) for record, accuracy_row in zip(records, accuracy_rows)]

    groups: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model_key"], row["model_name"], row["setting_id"])].append(row)
    table = []
    for (model_key, model_name, setting_id), group in sorted(groups.items()):
        rouge_scores = [row["rouge_l"] for row in group if row["rouge_l"] is not None]
        lengths = [row["generated_tokens"] for row in group]
        table.append(
            {
                "model_key": model_key,
                "model_name": model_name,
                "setting_id": setting_id,
                "sample_count": len(group),
                "scored_explanation_count": len(rouge_scores),
                "missing_explanation_count": len(group) - len(rouge_scores),
                "mean_rouge_l": fmean(rouge_scores) if rouge_scores else None,
                "rouge_l_stddev": pstdev(rouge_scores) if rouge_scores else None,
                "mean_generated_tokens": fmean(lengths),
                "generated_tokens_stddev": pstdev(lengths),
                "parser_failure_count": sum(row["parser_failure"] for row in group),
            }
        )
    return rows, table


def write_overlap_artifacts(
    rows: Sequence[Mapping[str, Any]],
    table: Sequence[Mapping[str, Any]],
    *,
    metrics_path: str | Path,
    table_path: str | Path,
) -> None:
    """Atomically write per-sample C9 JSONL and aggregate CSV."""

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
    parser = argparse.ArgumentParser(description="Compute C9 ROUGE-L and output length.")
    parser.add_argument("input_paths", nargs="+", type=Path)
    parser.add_argument("--metrics-path", required=True, type=Path)
    parser.add_argument("--table-path", required=True, type=Path)
    args = parser.parse_args()
    rows, table = evaluate_overlap(args.input_paths)
    write_overlap_artifacts(
        rows, table, metrics_path=args.metrics_path, table_path=args.table_path
    )
    print(json.dumps(table, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
