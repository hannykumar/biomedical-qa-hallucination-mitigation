"""Run resumable S1/S2 generation inside an approved Slurm GPU job."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.preprocess import (
    CSV_FIELDNAMES,
    NormalizedPubMedQAExample,
    normalize_pubmedqa_record,
)
from src.decoding.standard import generate_standard
from src.models.config import load_model_registry
from src.models.load_model import load_model_bundle
from src.prompts.build_prompts import build_prompt, load_prompt_config


DEFAULT_DATASET = Path("data/processed/pubmedqa_labeled_clean.csv")
DEFAULT_METADATA = Path("data/processed/pubmedqa_labeled_clean.metadata.json")
DEFAULT_OUTPUT_DIR = Path("outputs/generations/standard")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class GenerationRunError(RuntimeError):
    """Raised when a C4 run would violate its reproducibility contract."""


@dataclass(frozen=True, slots=True)
class StandardRunConfig:
    setting_id: str
    prompt_type: str
    decoding_method: str
    seed: int
    max_new_tokens: int
    do_sample: bool
    temperature: None
    top_p: None


def load_standard_run_config(
    setting_id: str,
    *,
    experiments_path: str | Path = "configs/experiments.yaml",
    generation_path: str | Path = "configs/generation.yaml",
    max_new_tokens_override: int | None = None,
) -> StandardRunConfig:
    """Resolve and validate one deterministic S1/S2 configuration."""

    if setting_id not in {"S1", "S2"}:
        raise GenerationRunError("standard baseline setting must be S1 or S2")
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("PyYAML is required to load generation configuration") from error

    settings = _mapping(
        yaml.safe_load(Path(experiments_path).read_text(encoding="utf-8")),
        "experiment configuration",
    ).get("settings")
    setting = _mapping(_mapping(settings, "settings").get(setting_id), setting_id)
    generation = _mapping(
        yaml.safe_load(Path(generation_path).read_text(encoding="utf-8")),
        "generation configuration",
    )

    if setting.get("decoding_method") != "standard":
        raise GenerationRunError(f"{setting_id} must use standard decoding")
    prompt_type = setting.get("prompt_type")
    if prompt_type not in {"question_only", "question_context"}:
        raise GenerationRunError(f"{setting_id} has an invalid prompt_type")
    if generation.get("do_sample") is not False:
        raise GenerationRunError("C4 requires deterministic do_sample: false")
    if generation.get("temperature") is not None or generation.get("top_p") is not None:
        raise GenerationRunError("temperature and top_p must remain null for C4")

    seed = generation.get("seed")
    max_new_tokens = (
        generation.get("max_new_tokens")
        if max_new_tokens_override is None
        else max_new_tokens_override
    )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise GenerationRunError("seed must be an integer")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise GenerationRunError("max_new_tokens must be a positive integer")

    return StandardRunConfig(
        setting_id=setting_id,
        prompt_type=prompt_type,
        decoding_method="standard",
        seed=seed,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )


def load_normalized_examples(
    path: str | Path = DEFAULT_DATASET,
    *,
    limit: int | None = None,
) -> list[NormalizedPubMedQAExample]:
    """Read the validated C1 CSV and reconstruct its immutable examples."""

    if limit is not None and (isinstance(limit, bool) or limit <= 0):
        raise GenerationRunError("limit must be a positive integer")

    examples: list[NormalizedPubMedQAExample] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDNAMES:
            raise GenerationRunError("normalized CSV columns do not match C1 schema")
        for row_number, row in enumerate(reader, start=2):
            try:
                source_row_index = int(row["source_row_index"])
                source = {
                    "pubid": int(row["pubid"]),
                    "question": row["question"],
                    "context": {
                        "contexts": json.loads(row["context_sections_json"]),
                        "labels": json.loads(row["context_labels_json"]),
                        "meshes": json.loads(row["meshes_json"]),
                    },
                    "long_answer": row["gold_long_answer"],
                    "final_decision": row["gold_final_decision"],
                }
                example = normalize_pubmedqa_record(
                    source, source_row_index=source_row_index
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise GenerationRunError(
                    f"invalid normalized CSV row {row_number}: {error}"
                ) from error
            if example.sample_id != row["sample_id"] or example.context != row["context"]:
                raise GenerationRunError(
                    f"normalized CSV row {row_number} fails identity/context validation"
                )
            if example.sample_id in seen:
                raise GenerationRunError(
                    f"duplicate sample_id in normalized CSV: {example.sample_id}"
                )
            seen.add(example.sample_id)
            examples.append(example)
            if limit is not None and len(examples) == limit:
                break

    if not examples:
        raise GenerationRunError("normalized CSV contains no examples")
    return examples


def run_generation(
    *,
    model_key: str,
    setting_id: str,
    run_id: str,
    code_revision: str,
    cache_dir: str | Path,
    dataset_path: str | Path = DEFAULT_DATASET,
    metadata_path: str | Path = DEFAULT_METADATA,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    local_files_only: bool = False,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    """Generate missing samples and append each raw result exactly once."""

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise GenerationRunError("run_id must use only letters, digits, dot, dash, underscore")
    if not COMMIT_PATTERN.fullmatch(code_revision):
        raise GenerationRunError("code_revision must be a 40-character lowercase Git SHA")

    run_config = load_standard_run_config(
        setting_id, max_new_tokens_override=max_new_tokens
    )
    examples = load_normalized_examples(dataset_path, limit=limit)
    prompt_config = load_prompt_config()
    registry = load_model_registry()
    spec = registry.get(model_key)

    definition = _run_definition(
        run_id=run_id,
        code_revision=code_revision,
        run_config=run_config,
        examples=examples,
        dataset_path=Path(dataset_path),
        metadata_path=Path(metadata_path),
        prompt_config=prompt_config,
        registry=registry,
        spec=spec,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    raw_path = output_path / f"{run_id}.jsonl"
    manifest_path = output_path / f"{run_id}.manifest.json"
    _write_or_validate_manifest(manifest_path, definition)

    completed = _completed_sample_ids(
        raw_path,
        run_id=run_id,
        model_key=model_key,
        setting_id=setting_id,
    )
    selected_ids = {example.sample_id for example in examples}
    unknown = completed - selected_ids
    if unknown:
        raise GenerationRunError(
            f"raw output contains samples outside this run: {sorted(unknown)[:3]}"
        )

    remaining = [example for example in examples if example.sample_id not in completed]
    if remaining:
        bundle = load_model_bundle(
            registry,
            model_key,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        import torch

        torch.manual_seed(run_config.seed)
        torch.cuda.manual_seed_all(run_config.seed)
        # ponytail: one-at-a-time keeps append/resume exact; batch only if the pilot
        # proves throughput matters enough to justify more complex checkpointing.
        for example in remaining:
            prompt = build_prompt(
                example,
                prompt_type=run_config.prompt_type,
                config=prompt_config,
            )
            generated = generate_standard(
                bundle,
                prompt,
                max_new_tokens=run_config.max_new_tokens,
            )
            _append_record(
                raw_path,
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    **asdict(example),
                    "model_key": model_key,
                    "model_name": spec.model_id,
                    "model_revision": spec.revision,
                    "tokenizer_revision": spec.tokenizer_revision,
                    "setting_id": setting_id,
                    "decoding_method": run_config.decoding_method,
                    "prompt_text": prompt.text,
                    **prompt.metadata(),
                    "generation_parameters": {
                        "seed": run_config.seed,
                        "max_new_tokens": run_config.max_new_tokens,
                        "do_sample": False,
                        "temperature": None,
                        "top_p": None,
                    },
                    "raw_output": generated.raw_output,
                    "serialized_input_tokens": generated.input_tokens,
                    "generated_tokens": generated.generated_tokens,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                },
            )

    return {
        "status": "complete",
        "run_id": run_id,
        "model_key": model_key,
        "setting_id": setting_id,
        "selected_samples": len(examples),
        "previously_completed": len(completed),
        "generated_now": len(remaining),
        "raw_output": str(raw_path),
        "manifest": str(manifest_path),
    }


def _run_definition(
    *,
    run_id: str,
    code_revision: str,
    run_config: StandardRunConfig,
    examples: list[NormalizedPubMedQAExample],
    dataset_path: Path,
    metadata_path: Path,
    prompt_config: Any,
    registry: Any,
    spec: Any,
) -> dict[str, Any]:
    try:
        import torch
        import transformers
    except ImportError as error:  # pragma: no cover - cluster-only path
        raise GenerationRunError("C4 runtime dependencies are missing") from error
    if not torch.cuda.is_available():
        raise GenerationRunError("C4 model generation must run inside a GPU job")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset_sha256 = _sha256_file(dataset_path)
    expected_sha256 = metadata.get("artifact", {}).get("sha256")
    if dataset_sha256 != expected_sha256:
        raise GenerationRunError(
            f"dataset SHA-256 mismatch: expected {expected_sha256}, got {dataset_sha256}"
        )
    template = prompt_config.template_for(run_config.prompt_type)
    device = torch.cuda.get_device_properties(0)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "code_revision": code_revision,
        "dataset": {
            "path": str(dataset_path),
            "metadata_path": str(metadata_path),
            "sha256": dataset_sha256,
            "source": metadata.get("source"),
            "selected_samples": len(examples),
            "sample_ids_sha256": _sha256_text(
                "\n".join(example.sample_id for example in examples)
            ),
        },
        "model": asdict(spec),
        "model_runtime": asdict(registry.runtime),
        "prompt": {
            "type": run_config.prompt_type,
            "version": prompt_config.version,
            "format": prompt_config.prompt_format,
            "template_sha256": _sha256_text(template),
        },
        "generation": asdict(run_config),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": device.name,
            "gpu_total_memory_bytes": device.total_memory,
        },
    }


def _write_or_validate_manifest(path: Path, definition: dict[str, Any]) -> None:
    definition_sha256 = _sha256_text(_canonical_json(definition))
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("definition_sha256") != definition_sha256:
            raise GenerationRunError("existing manifest does not match this run definition")
        return

    payload = {
        **definition,
        "definition_sha256": definition_sha256,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "initial_slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _completed_sample_ids(
    path: Path,
    *,
    run_id: str,
    model_key: str,
    setting_id: str,
) -> set[str]:
    if not path.exists():
        return set()

    completed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise GenerationRunError(
                    f"invalid JSONL at {path}:{line_number}: {error}"
                ) from error
            expected = {
                "run_id": run_id,
                "model_key": model_key,
                "setting_id": setting_id,
            }
            if any(record.get(key) != value for key, value in expected.items()):
                raise GenerationRunError(
                    f"record identity mismatch at {path}:{line_number}"
                )
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise GenerationRunError(f"missing sample_id at {path}:{line_number}")
            if sample_id in completed:
                raise GenerationRunError(f"duplicate sample_id in raw output: {sample_id}")
            completed.add(sample_id)
    return completed


def _append_record(path: Path, record: dict[str, Any]) -> None:
    data = (_canonical_json(record) + "\n").encode("utf-8")
    # ponytail: a single append+fsync is enough for one Slurm writer; add locking
    # only if concurrent jobs are ever allowed to share a run_id.
    with path.open("ab") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GenerationRunError(f"{location} must be a mapping")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C4 S1/S2 on an allocated GPU.")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--setting-id", required=True, choices=("S1", "S2"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--code-revision", default=os.environ.get("CODE_REVISION"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.code_revision is None:
        raise GenerationRunError("pass --code-revision or set CODE_REVISION")
    result = run_generation(
        model_key=args.model_key,
        setting_id=args.setting_id,
        run_id=args.run_id,
        code_revision=args.code_revision,
        cache_dir=args.cache_dir,
        dataset_path=args.dataset_path,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        limit=args.limit,
        local_files_only=args.local_files_only,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
