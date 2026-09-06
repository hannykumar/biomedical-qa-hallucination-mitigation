"""Compute pinned C10 BERTScore F1 and embedding cosine similarity."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import yaml

from src.evaluation.accuracy import AccuracyEvaluationError, evaluate_accuracy


DEFAULT_CONFIG_PATH = Path("configs/evaluation.yaml")
PROVENANCE_FIELDS = (
    "bert_score_version",
    "torch_version",
    "transformers_version",
    "bertscore_model_id",
    "bertscore_model_revision",
    "bertscore_num_layers",
    "cosine_model_id",
    "cosine_model_revision",
    "cosine_max_length",
)
TABLE_FIELDS = (
    "model_key",
    "model_name",
    "setting_id",
    "sample_count",
    "scored_explanation_count",
    "missing_explanation_count",
    "mean_bertscore_f1",
    "bertscore_f1_stddev",
    "mean_cosine_similarity",
    "cosine_similarity_stddev",
    "parser_failure_count",
    *PROVENANCE_FIELDS,
)
ScoreFunction = Callable[[Sequence[str], Sequence[str], Mapping[str, Any]], tuple[Sequence[float], Sequence[float]]]


class SemanticEvaluationError(ValueError):
    """Raised when C10 configuration, input, or scores violate the contract."""


def load_semantic_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and strictly validate the pinned semantic-metric configuration."""

    root = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(root, Mapping) or not isinstance(root.get("semantic_metrics"), Mapping):
        raise SemanticEvaluationError("semantic_metrics must be a mapping")
    config = dict(root["semantic_metrics"])
    if set(config) != {
        "batch_size", "cache_dir", "device", "runtime_versions", "bertscore",
        "cosine_similarity"
    }:
        raise SemanticEvaluationError("semantic_metrics has unexpected keys")
    if isinstance(config["batch_size"], bool) or not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
        raise SemanticEvaluationError("batch_size must be a positive integer")
    if config["device"] != "cpu":
        raise SemanticEvaluationError("C10 device must be cpu without GPU authorization")
    if not isinstance(config["cache_dir"], str) or not config["cache_dir"]:
        raise SemanticEvaluationError("cache_dir must be a nonempty string")
    versions = config["runtime_versions"]
    if (
        not isinstance(versions, Mapping)
        or set(versions) != {"bert-score", "torch", "transformers"}
        or not all(isinstance(value, str) and value for value in versions.values())
    ):
        raise SemanticEvaluationError("invalid runtime_versions configuration")

    bertscore = config["bertscore"]
    cosine = config["cosine_similarity"]
    if not isinstance(bertscore, Mapping) or set(bertscore) != {
        "model_id", "model_revision", "num_layers", "idf", "rescale_with_baseline"
    }:
        raise SemanticEvaluationError("invalid bertscore configuration")
    if not isinstance(cosine, Mapping) or set(cosine) != {
        "model_id", "model_revision", "max_length", "pooling"
    }:
        raise SemanticEvaluationError("invalid cosine_similarity configuration")
    for section, field in ((bertscore, "model_id"), (cosine, "model_id")):
        if not isinstance(section[field], str) or not section[field]:
            raise SemanticEvaluationError(f"{field} must be a nonempty string")
    for section in (bertscore, cosine):
        revision = section["model_revision"]
        if not isinstance(revision, str) or len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise SemanticEvaluationError("model_revision must be a 40-character commit")
    if isinstance(bertscore["num_layers"], bool) or not isinstance(bertscore["num_layers"], int) or bertscore["num_layers"] <= 0:
        raise SemanticEvaluationError("num_layers must be a positive integer")
    if bertscore["idf"] is not False or bertscore["rescale_with_baseline"] is not False:
        raise SemanticEvaluationError("C10 uses raw, non-IDF BERTScore")
    if isinstance(cosine["max_length"], bool) or not isinstance(cosine["max_length"], int) or cosine["max_length"] <= 0:
        raise SemanticEvaluationError("max_length must be a positive integer")
    if cosine["pooling"] != "mean":
        raise SemanticEvaluationError("cosine pooling must be mean")
    return config


def _provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    bertscore, cosine = config["bertscore"], config["cosine_similarity"]
    versions = config["runtime_versions"]
    return {
        "bert_score_version": versions["bert-score"],
        "torch_version": versions["torch"],
        "transformers_version": versions["transformers"],
        "bertscore_model_id": bertscore["model_id"],
        "bertscore_model_revision": bertscore["model_revision"],
        "bertscore_num_layers": bertscore["num_layers"],
        "cosine_model_id": cosine["model_id"],
        "cosine_model_revision": cosine["model_revision"],
        "cosine_max_length": cosine["max_length"],
    }


def _compute_scores(
    predictions: Sequence[str], references: Sequence[str], config: Mapping[str, Any]
) -> tuple[list[float], list[float]]:
    """Run the heavyweight pinned models; imports stay optional for unit tests."""

    try:
        from importlib.metadata import version

        import torch
        from bert_score import score as bert_score
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise SemanticEvaluationError(
            "C10 requires the pinned packages in requirements-evaluation.txt"
        ) from error

    actual_versions = {
        package: version(package)
        for package in ("bert-score", "torch", "transformers")
    }
    if actual_versions != dict(config["runtime_versions"]):
        raise SemanticEvaluationError(
            f"runtime version mismatch: expected {dict(config['runtime_versions'])}, "
            f"found {actual_versions}"
        )

    bertscore_config = config["bertscore"]
    snapshot = snapshot_download(
        repo_id=bertscore_config["model_id"],
        revision=bertscore_config["model_revision"],
        cache_dir=config["cache_dir"],
    )
    _, _, bertscore_f1 = bert_score(
        list(predictions),
        list(references),
        model_type=snapshot,
        num_layers=bertscore_config["num_layers"],
        idf=False,
        rescale_with_baseline=False,
        device=config["device"],
        batch_size=config["batch_size"],
        verbose=True,
    )

    cosine_config = config["cosine_similarity"]
    tokenizer = AutoTokenizer.from_pretrained(
        cosine_config["model_id"],
        revision=cosine_config["model_revision"],
        cache_dir=config["cache_dir"],
    )
    model = AutoModel.from_pretrained(
        cosine_config["model_id"],
        revision=cosine_config["model_revision"],
        cache_dir=config["cache_dir"],
    ).to(config["device"])
    model.eval()

    def encode(texts: Sequence[str]) -> Any:
        batches = []
        for start in range(0, len(texts), config["batch_size"]):
            inputs = tokenizer(
                list(texts[start : start + config["batch_size"]]),
                padding=True,
                truncation=True,
                max_length=cosine_config["max_length"],
                return_tensors="pt",
            ).to(config["device"])
            with torch.no_grad():
                token_embeddings = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1)
            batches.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu())
        return torch.cat(batches)

    prediction_embeddings = encode(predictions)
    reference_embeddings = encode(references)
    cosine_scores = (prediction_embeddings * reference_embeddings).sum(1)
    return bertscore_f1.tolist(), cosine_scores.tolist()


def evaluate_semantic(
    input_paths: Sequence[str | Path],
    *,
    config: Mapping[str, Any] | None = None,
    score_function: ScoreFunction = _compute_scores,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute C10 metrics with C8's paired input validation."""

    resolved = dict(config or load_semantic_config())
    try:
        accuracy_rows, _ = evaluate_accuracy(input_paths)
    except AccuracyEvaluationError as error:
        raise SemanticEvaluationError(str(error)) from error
    records = []
    for path in map(Path, input_paths):
        with path.open(encoding="utf-8") as source:
            records.extend(json.loads(line) for line in source if line.strip())

    scored_indices, predictions, references = [], [], []
    for index, record in enumerate(records):
        prediction, reference = record.get("parsed_explanation"), record.get("gold_long_answer")
        if not isinstance(prediction, str):
            raise SemanticEvaluationError("parsed_explanation must be a string")
        if not isinstance(reference, str) or not reference.strip():
            raise SemanticEvaluationError("gold_long_answer must be a nonempty string")
        if prediction.strip():
            scored_indices.append(index)
            predictions.append(prediction)
            references.append(reference)

    bertscore_values, cosine_values = score_function(predictions, references, resolved)
    if len(bertscore_values) != len(scored_indices) or len(cosine_values) != len(scored_indices):
        raise SemanticEvaluationError("semantic scorer returned the wrong number of scores")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -1 <= value <= 1
        for value in (*bertscore_values, *cosine_values)
    ):
        raise SemanticEvaluationError("semantic scores must be finite values from -1 to 1")

    scores = {
        index: (float(bertscore), float(cosine))
        for index, bertscore, cosine in zip(scored_indices, bertscore_values, cosine_values)
    }
    provenance = _provenance(resolved)
    rows = []
    for index, accuracy_row in enumerate(accuracy_rows):
        bertscore, cosine = scores.get(index, (None, None))
        rows.append(
            {
                **{
                    field: accuracy_row[field]
                    for field in (
                        "run_id", "sample_id", "model_key", "model_name",
                        "setting_id", "decoding_method", "parser_failure"
                    )
                },
                "bertscore_f1": bertscore,
                "cosine_similarity": cosine,
                "missing_explanation": int(index not in scores),
                **provenance,
            }
        )

    groups: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model_key"], row["model_name"], row["setting_id"])].append(row)
    table = []
    for (model_key, model_name, setting_id), group in sorted(groups.items()):
        bertscores = [row["bertscore_f1"] for row in group if row["bertscore_f1"] is not None]
        cosines = [row["cosine_similarity"] for row in group if row["cosine_similarity"] is not None]
        table.append(
            {
                "model_key": model_key,
                "model_name": model_name,
                "setting_id": setting_id,
                "sample_count": len(group),
                "scored_explanation_count": len(bertscores),
                "missing_explanation_count": len(group) - len(bertscores),
                "mean_bertscore_f1": fmean(bertscores) if bertscores else None,
                "bertscore_f1_stddev": pstdev(bertscores) if bertscores else None,
                "mean_cosine_similarity": fmean(cosines) if cosines else None,
                "cosine_similarity_stddev": pstdev(cosines) if cosines else None,
                "parser_failure_count": sum(row["parser_failure"] for row in group),
                **provenance,
            }
        )
    return rows, table


def write_semantic_artifacts(
    rows: Sequence[Mapping[str, Any]], table: Sequence[Mapping[str, Any]], *,
    metrics_path: str | Path, table_path: str | Path
) -> None:
    """Atomically write per-sample C10 JSONL and aggregate CSV."""

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
    parser = argparse.ArgumentParser(description="Compute pinned C10 semantic metrics.")
    parser.add_argument("input_paths", nargs="+", type=Path)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--metrics-path", required=True, type=Path)
    parser.add_argument("--table-path", required=True, type=Path)
    args = parser.parse_args()
    rows, table = evaluate_semantic(
        args.input_paths, config=load_semantic_config(args.config_path)
    )
    write_semantic_artifacts(
        rows, table, metrics_path=args.metrics_path, table_path=args.table_path
    )
    print(json.dumps(table, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
