"""Pure, deterministic normalization for PubMedQA records.

This module deliberately has no Hugging Face, pandas, or GPU dependency. It
turns dictionary-like source rows into a strict internal data contract that can
be unit-tested without network access.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


ALLOWED_LABELS: Final[frozenset[str]] = frozenset({"yes", "no", "maybe"})
CONTEXT_SEPARATOR: Final[str] = "\n\n"
SAMPLE_ID_PREFIX: Final[str] = "pubmedqa-pqa_labeled"
REQUIRED_SOURCE_FIELDS: Final[tuple[str, ...]] = (
    "pubid",
    "question",
    "context",
    "long_answer",
    "final_decision",
)
CSV_FIELDNAMES: Final[tuple[str, ...]] = (
    "sample_id",
    "pubid",
    "question",
    "context",
    "gold_long_answer",
    "gold_final_decision",
    "context_sections_json",
    "context_labels_json",
    "meshes_json",
    "source_row_index",
)


class PubMedQAValidationError(ValueError):
    """Raised when a source record violates the normalized data contract."""


@dataclass(frozen=True, slots=True)
class NormalizedPubMedQAExample:
    """One validated, model-ready PubMedQA example."""

    sample_id: str
    pubid: int
    question: str
    context: str
    gold_long_answer: str
    gold_final_decision: str
    context_sections: tuple[str, ...]
    context_labels: tuple[str, ...]
    meshes: tuple[str, ...]
    source_row_index: int

    def to_csv_row(self) -> dict[str, str | int]:
        """Serialize the example without losing its structured context fields."""

        return {
            "sample_id": self.sample_id,
            "pubid": self.pubid,
            "question": self.question,
            "context": self.context,
            "gold_long_answer": self.gold_long_answer,
            "gold_final_decision": self.gold_final_decision,
            "context_sections_json": _json_array(self.context_sections),
            "context_labels_json": _json_array(self.context_labels),
            "meshes_json": _json_array(self.meshes),
            "source_row_index": self.source_row_index,
        }


def normalize_pubmedqa_record(
    record: Mapping[str, Any],
    *,
    source_row_index: int,
) -> NormalizedPubMedQAExample:
    """Validate and normalize one raw `pqa_labeled` record.

    Context section order is preserved and the model-ready context is formed by
    joining the source text sections with one blank line. Section labels remain
    in a separate field rather than being injected into the model input.
    """

    if not isinstance(record, Mapping):
        raise PubMedQAValidationError("record must be a mapping")
    if not isinstance(source_row_index, int) or source_row_index < 0:
        raise PubMedQAValidationError("source_row_index must be a non-negative integer")

    missing = [field for field in REQUIRED_SOURCE_FIELDS if field not in record]
    if missing:
        raise PubMedQAValidationError(
            f"row {source_row_index}: missing required fields: {', '.join(missing)}"
        )

    pubid = _normalize_pubid(record["pubid"], source_row_index)
    question = _required_text(record["question"], "question", source_row_index)
    long_answer = _required_text(
        record["long_answer"], "long_answer", source_row_index
    )
    final_decision = _normalize_label(record["final_decision"], source_row_index)

    raw_context = record["context"]
    if not isinstance(raw_context, Mapping):
        raise PubMedQAValidationError(
            f"row {source_row_index}: context must be a mapping"
        )

    context_sections = _required_text_sequence(
        raw_context.get("contexts"), "context.contexts", source_row_index
    )
    context_labels = _optional_text_sequence(
        raw_context.get("labels"), "context.labels", source_row_index
    )
    meshes = _optional_text_sequence(
        raw_context.get("meshes"), "context.meshes", source_row_index
    )

    if context_labels and len(context_labels) != len(context_sections):
        raise PubMedQAValidationError(
            f"row {source_row_index}: context.labels has {len(context_labels)} items "
            f"but context.contexts has {len(context_sections)}"
        )

    return NormalizedPubMedQAExample(
        sample_id=f"{SAMPLE_ID_PREFIX}-{pubid}",
        pubid=pubid,
        question=question,
        context=CONTEXT_SEPARATOR.join(context_sections),
        gold_long_answer=long_answer,
        gold_final_decision=final_decision,
        context_sections=context_sections,
        context_labels=context_labels,
        meshes=meshes,
        source_row_index=source_row_index,
    )


def normalize_pubmedqa_records(
    records: Iterable[Mapping[str, Any]],
) -> list[NormalizedPubMedQAExample]:
    """Normalize records in source order and reject duplicate stable IDs."""

    normalized: list[NormalizedPubMedQAExample] = []
    seen_sample_ids: set[str] = set()

    for source_row_index, record in enumerate(records):
        example = normalize_pubmedqa_record(
            record,
            source_row_index=source_row_index,
        )
        if example.sample_id in seen_sample_ids:
            raise PubMedQAValidationError(
                f"row {source_row_index}: duplicate sample_id {example.sample_id!r}"
            )
        seen_sample_ids.add(example.sample_id)
        normalized.append(example)

    if not normalized:
        raise PubMedQAValidationError("dataset contains no records")

    return normalized


def write_normalized_csv(
    examples: Sequence[NormalizedPubMedQAExample],
    output_path: str | Path,
) -> Path:
    """Atomically write normalized examples as UTF-8 CSV."""

    if not examples:
        raise PubMedQAValidationError("cannot write an empty normalized dataset")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")

    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(example.to_csv_row() for example in examples)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return path


def _normalize_pubid(value: Any, row_index: int) -> int:
    if isinstance(value, bool):
        raise PubMedQAValidationError(f"row {row_index}: pubid must be an integer")
    try:
        pubid = int(value)
    except (TypeError, ValueError) as error:
        raise PubMedQAValidationError(
            f"row {row_index}: pubid must be an integer"
        ) from error
    if pubid <= 0:
        raise PubMedQAValidationError(f"row {row_index}: pubid must be positive")
    return pubid


def _normalize_label(value: Any, row_index: int) -> str:
    label = _required_text(value, "final_decision", row_index).lower()
    if label not in ALLOWED_LABELS:
        allowed = ", ".join(sorted(ALLOWED_LABELS))
        raise PubMedQAValidationError(
            f"row {row_index}: final_decision {label!r} is not one of {allowed}"
        )
    return label


def _required_text(value: Any, field: str, row_index: int) -> str:
    if not isinstance(value, str):
        raise PubMedQAValidationError(
            f"row {row_index}: {field} must be a string"
        )
    text = value.strip()
    if not text:
        raise PubMedQAValidationError(f"row {row_index}: {field} must not be empty")
    return text


def _required_text_sequence(
    value: Any,
    field: str,
    row_index: int,
) -> tuple[str, ...]:
    items = _text_sequence(value, field, row_index, allow_empty_items=False)
    if not items:
        raise PubMedQAValidationError(f"row {row_index}: {field} must not be empty")
    return items


def _optional_text_sequence(
    value: Any,
    field: str,
    row_index: int,
) -> tuple[str, ...]:
    if value is None:
        return ()
    return _text_sequence(value, field, row_index, allow_empty_items=True)


def _text_sequence(
    value: Any,
    field: str,
    row_index: int,
    *,
    allow_empty_items: bool,
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PubMedQAValidationError(
            f"row {row_index}: {field} must be a sequence of strings"
        )

    normalized: list[str] = []
    for item_index, item in enumerate(value):
        if not isinstance(item, str):
            raise PubMedQAValidationError(
                f"row {row_index}: {field}[{item_index}] must be a string"
            )
        text = item.strip()
        if not text and not allow_empty_items:
            raise PubMedQAValidationError(
                f"row {row_index}: {field}[{item_index}] must not be empty"
            )
        normalized.append(text)
    return tuple(normalized)


def _json_array(values: Sequence[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
