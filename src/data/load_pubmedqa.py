"""Download, validate, normalize, and inspect PubMedQA `pqa_labeled`.

Run from the repository root:

    python3 -m src.data.load_pubmedqa

The processed CSV and its metadata manifest are intentionally ignored by Git;
the small inspection report is versioned as documentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from src.data.preprocess import (
    CSV_FIELDNAMES,
    NormalizedPubMedQAExample,
    normalize_pubmedqa_records,
    write_normalized_csv,
)


DEFAULT_DATASET_ID: Final[str] = "qiaojin/PubMedQA"
DEFAULT_SUBSET: Final[str] = "pqa_labeled"
DEFAULT_SPLIT: Final[str] = "train"
DEFAULT_REVISION: Final[str] = "9001f2853fb87cab8d220904e0de81ac6973b318"
DEFAULT_CACHE_DIR: Final[Path] = Path("data/raw/huggingface")
DEFAULT_OUTPUT: Final[Path] = Path("data/processed/pubmedqa_labeled_clean.csv")
DEFAULT_METADATA_OUTPUT: Final[Path] = Path(
    "data/processed/pubmedqa_labeled_clean.metadata.json"
)
DEFAULT_REPORT: Final[Path] = Path("reports/dataset_inspection.md")
COMMIT_SHA_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True, slots=True)
class DatasetBuildResult:
    """Paths and provenance produced by one successful C1 build."""

    csv_path: Path
    metadata_path: Path
    report_path: Path
    row_count: int
    resolved_revision: str
    dataset_fingerprint: str
    csv_sha256: str


def resolve_dataset_revision(dataset_id: str, revision: str) -> str:
    """Resolve a branch/tag/commit to the Hub's immutable commit SHA."""

    if COMMIT_SHA_PATTERN.fullmatch(revision):
        return revision.lower()

    try:
        from huggingface_hub import HfApi
    except ImportError as error:  # pragma: no cover - environment-specific path
        raise RuntimeError(
            "huggingface_hub is required to resolve the dataset revision"
        ) from error

    info = HfApi().dataset_info(repo_id=dataset_id, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Hugging Face returned no revision SHA for {dataset_id}")
    return info.sha


def load_pubmedqa_source(
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    subset: str = DEFAULT_SUBSET,
    split: str = DEFAULT_SPLIT,
    revision: str = DEFAULT_REVISION,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
) -> Any:
    """Load the pinned source split through Hugging Face Datasets."""

    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - environment-specific path
        raise RuntimeError(
            "The 'datasets' package is required. Install the project environment "
            "before building PubMedQA."
        ) from error

    return load_dataset(
        dataset_id,
        subset,
        split=split,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )


def build_pubmedqa_dataset(
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    subset: str = DEFAULT_SUBSET,
    split: str = DEFAULT_SPLIT,
    revision: str = DEFAULT_REVISION,
    output_path: str | Path = DEFAULT_OUTPUT,
    metadata_path: str | Path = DEFAULT_METADATA_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    limit: int | None = None,
) -> DatasetBuildResult:
    """Build all C1 artifacts from one resolved source revision."""

    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    if limit is not None and (
        Path(output_path) == DEFAULT_OUTPUT
        or Path(metadata_path) == DEFAULT_METADATA_OUTPUT
        or Path(report_path) == DEFAULT_REPORT
    ):
        raise ValueError(
            "a limited smoke build must use non-default --output, "
            "--metadata-output, and --report paths so it cannot overwrite the "
            "complete C1 artifacts"
        )

    resolved_revision = resolve_dataset_revision(dataset_id, revision)
    source_dataset = load_pubmedqa_source(
        dataset_id=dataset_id,
        subset=subset,
        split=split,
        revision=resolved_revision,
        cache_dir=cache_dir,
    )
    source_records = _select_source_records(source_dataset, limit)
    examples = normalize_pubmedqa_records(source_records)

    csv_path = write_normalized_csv(examples, output_path)
    csv_sha256 = _sha256_file(csv_path)
    dataset_fingerprint = str(getattr(source_dataset, "_fingerprint", "unknown"))
    created_at = datetime.now(timezone.utc).isoformat()
    statistics_summary = summarize_examples(examples)

    metadata = {
        "schema_version": 1,
        "created_at_utc": created_at,
        "source": {
            "dataset_id": dataset_id,
            "subset": subset,
            "split": split,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "dataset_fingerprint": dataset_fingerprint,
            "source_row_count": len(source_dataset),
            "limit": limit,
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
        },
        "artifact": {
            "path": str(csv_path),
            "sha256": csv_sha256,
            "columns": list(CSV_FIELDNAMES),
            "row_count": len(examples),
        },
        "statistics": statistics_summary,
        "environment": _environment_versions(),
    }

    metadata_output = _write_json_atomic(metadata, metadata_path)
    report_output = _write_text_atomic(
        render_inspection_report(metadata),
        report_path,
    )

    return DatasetBuildResult(
        csv_path=csv_path,
        metadata_path=metadata_output,
        report_path=report_output,
        row_count=len(examples),
        resolved_revision=resolved_revision,
        dataset_fingerprint=dataset_fingerprint,
        csv_sha256=csv_sha256,
    )


def summarize_examples(
    examples: Sequence[NormalizedPubMedQAExample],
) -> dict[str, Any]:
    """Calculate deterministic dataset checks used in reports and tests."""

    if not examples:
        raise ValueError("cannot summarize an empty dataset")

    label_counts = Counter(example.gold_final_decision for example in examples)
    section_counts = [len(example.context_sections) for example in examples]
    context_lengths = [len(example.context) for example in examples]

    return {
        "label_counts": dict(sorted(label_counts.items())),
        "context_sections": {
            "minimum": min(section_counts),
            "maximum": max(section_counts),
            "mean": round(statistics.fmean(section_counts), 3),
        },
        "context_characters": {
            "minimum": min(context_lengths),
            "maximum": max(context_lengths),
            "mean": round(statistics.fmean(context_lengths), 3),
        },
        "unique_sample_ids": len({example.sample_id for example in examples}),
    }


def render_inspection_report(metadata: Mapping[str, Any]) -> str:
    """Render the small, versioned human-readable C1 inspection report."""

    source = metadata["source"]
    artifact = metadata["artifact"]
    summary = metadata["statistics"]
    labels = summary["label_counts"]
    sections = summary["context_sections"]
    characters = summary["context_characters"]

    lines = [
        "# PubMedQA Labeled Dataset Inspection",
        "",
        f"- Generated: `{metadata['created_at_utc']}`",
        f"- Dataset: `{source['dataset_id']}`",
        f"- Subset/split: `{source['subset']}` / `{source['split']}`",
        f"- Resolved revision: `{source['resolved_revision']}`",
        f"- Hugging Face fingerprint: `{source['dataset_fingerprint']}`",
        f"- Processed rows: **{artifact['row_count']}**",
        f"- Unique sample IDs: **{summary['unique_sample_ids']}**",
        f"- CSV SHA-256: `{artifact['sha256']}`",
        "",
        "## Final-decision distribution",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {label} | {count} |" for label, count in labels.items())
    lines.extend(
        [
            "",
            "## Context checks",
            "",
            "| Measure | Minimum | Mean | Maximum |",
            "|---|---:|---:|---:|",
            (
                "| Abstract sections per example | "
                f"{sections['minimum']} | {sections['mean']} | {sections['maximum']} |"
            ),
            (
                "| Flattened context characters | "
                f"{characters['minimum']} | {characters['mean']} | "
                f"{characters['maximum']} |"
            ),
            "",
            "## Normalization performed",
            "",
            "- Stable sample IDs are derived from the source PubMed ID.",
            "- Leading and trailing whitespace is removed from text fields.",
            "- Final decisions are lowercased and validated against yes/no/maybe.",
            "- Context sections retain source order and are joined with one blank line.",
            "- Original context sections, section labels, and MeSH terms remain in JSON columns.",
            "- No train/validation/test split is created during C1.",
            "",
            "The processed CSV and metadata manifest are generated artifacts and are "
            "excluded from Git. Rebuild them with `python3 -m src.data.load_pubmedqa`.",
            "",
        ]
    )
    return "\n".join(lines)


def _select_source_records(dataset: Any, limit: int | None) -> Any:
    if limit is None:
        return dataset
    selected_count = min(limit, len(dataset))
    return dataset.select(range(selected_count))


def _environment_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    try:
        import datasets

        versions["datasets"] = datasets.__version__
    except ImportError:  # pragma: no cover - already guarded by loader
        versions["datasets"] = "unavailable"
    try:
        import huggingface_hub

        versions["huggingface_hub"] = huggingface_hub.__version__
    except ImportError:  # pragma: no cover - already guarded by resolver
        versions["huggingface_hub"] = "unavailable"
    return versions


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(payload: Mapping[str, Any], path: str | Path) -> Path:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return _write_text_atomic(text, path)


def _write_text_atomic(text: str, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the normalized PubMedQA pqa_labeled C1 artifact."
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "Build only the first N source rows for a smoke test; custom output, "
            "metadata, and report paths are required."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = build_pubmedqa_dataset(
        dataset_id=args.dataset_id,
        subset=args.subset,
        split=args.split,
        revision=args.revision,
        output_path=args.output,
        metadata_path=args.metadata_output,
        report_path=args.report,
        cache_dir=args.cache_dir,
        limit=args.limit,
    )
    result_payload = asdict(result)
    result_payload.update(
        csv_path=str(result.csv_path),
        metadata_path=str(result.metadata_path),
        report_path=str(result.report_path),
    )
    print(json.dumps(result_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
