"""Parse immutable generation JSONL into auditable C7 derivatives."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PARSER_VERSION = 1
DEFAULT_OUTPUT_DIR = Path("outputs/generations/parsed")
LABEL_PREFIX = re.compile(
    r"^\s*(?:final\s+answer\s*:\s*)?(yes|no|maybe)"
    r"(?=\s|[.,;:!?]|explanation\s*:|$)",
    re.IGNORECASE,
)
LABEL_WORD = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)
ANSWER_MARKER = re.compile(r"^\s*final\s+answer\s*:", re.IGNORECASE)
EXPLANATION_MARKER = re.compile(r"explanation\s*:\s*", re.IGNORECASE)
EXACT_FORMAT = re.compile(
    r"Final answer: (yes|no|maybe)\nExplanation: \S[^\n]*"
)


class OutputParseError(ValueError):
    """Raised when a raw JSONL artifact violates the C4 record contract."""


def parse_raw_output(raw_output: str) -> dict[str, Any]:
    """Extract the leading label and marked explanation without inferring either."""

    if not isinstance(raw_output, str):
        raise OutputParseError("raw_output must be a string")

    errors: list[str] = []
    explanation_matches = list(EXPLANATION_MARKER.finditer(raw_output))
    answer_segment = (
        raw_output[: explanation_matches[0].start()]
        if explanation_matches
        else raw_output.splitlines()[0] if raw_output.splitlines() else raw_output
    )
    labels = LABEL_WORD.findall(answer_segment)
    label_match = LABEL_PREFIX.match(raw_output)

    if len(labels) > 1:
        parsed_label = "unknown"
        errors.append("ambiguous_label")
    elif label_match:
        parsed_label = label_match.group(1).lower()
    else:
        parsed_label = "unknown"
        errors.append(
            "malformed_label" if ANSWER_MARKER.match(raw_output) else "missing_label"
        )

    if not explanation_matches:
        explanation = ""
        errors.append("missing_explanation")
    elif len(explanation_matches) > 1:
        explanation = ""
        errors.append("malformed_explanation")
    else:
        explanation = raw_output[explanation_matches[0].end() :].strip()
        if not explanation:
            errors.append("malformed_explanation")

    exact = bool(EXACT_FORMAT.fullmatch(raw_output.strip()))
    return {
        "parsed_final_answer": parsed_label,
        "parsed_explanation": explanation,
        "parser_version": PARSER_VERSION,
        "parser_status": "error" if errors else "ok",
        "parser_errors": errors,
        "prompt_format_compliance": (
            "exact" if exact else "normalized" if not errors else "noncompliant"
        ),
    }


def parse_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Retain the complete raw record and append parser-owned fields."""

    for field in ("sample_id", "model_key", "setting_id"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise OutputParseError(f"{field} must be a nonempty string")
    return {**record, **parse_raw_output(record.get("raw_output"))}


def parse_files(
    input_paths: Sequence[str | Path],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Parse raw files atomically and summarize failures by model and setting."""

    if not input_paths:
        raise OutputParseError("at least one raw JSONL path is required")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    groups: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    outputs: list[str] = []
    targets: set[Path] = set()

    for input_path in map(Path, input_paths):
        target = destination / input_path.name
        if input_path.resolve() == target.resolve():
            raise OutputParseError("parsed output must not overwrite raw input")
        if target in targets:
            raise OutputParseError(f"duplicate output filename: {target.name}")
        targets.add(target)
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        try:
            with input_path.open(encoding="utf-8") as source, temporary.open(
                "w", encoding="utf-8"
            ) as sink:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    try:
                        raw_record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise OutputParseError(
                            f"invalid JSONL at {input_path}:{line_number}: {error}"
                        ) from error
                    if not isinstance(raw_record, Mapping):
                        raise OutputParseError(
                            f"record at {input_path}:{line_number} must be an object"
                        )
                    try:
                        parsed = parse_record(raw_record)
                    except OutputParseError as error:
                        raise OutputParseError(
                            f"invalid record at {input_path}:{line_number}: {error}"
                        ) from error
                    sink.write(
                        json.dumps(
                            parsed,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    counts = groups[(parsed["model_key"], parsed["setting_id"])]
                    counts["records"] += 1
                    counts["parser_failures"] += parsed["parser_status"] == "error"
                    counts[f"format_{parsed['prompt_format_compliance']}"] += 1
                    for error in parsed["parser_errors"]:
                        counts[error] += 1
                sink.flush()
                os.fsync(sink.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        outputs.append(str(target))

    summaries = []
    for (model_key, setting_id), counts in sorted(groups.items()):
        total = counts["records"]
        summaries.append(
            {
                "model_key": model_key,
                "setting_id": setting_id,
                **{
                    key: counts[key]
                    for key in (
                        "records",
                        "parser_failures",
                        "format_exact",
                        "format_normalized",
                        "format_noncompliant",
                        "ambiguous_label",
                        "missing_label",
                        "malformed_label",
                        "missing_explanation",
                        "malformed_explanation",
                    )
                },
                "parser_failure_rate": counts["parser_failures"] / total,
            }
        )
    return {
        "parser_version": PARSER_VERSION,
        "input_files": [str(Path(path)) for path in input_paths],
        "output_files": outputs,
        "groups": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse C4 generation JSONL files.")
    parser.add_argument("input_paths", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "parser_summary.json",
    )
    args = parser.parse_args()
    summary = parse_files(args.input_paths, output_dir=args.output_dir)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
