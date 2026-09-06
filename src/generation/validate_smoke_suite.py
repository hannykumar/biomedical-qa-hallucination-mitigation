"""Validate the four five-sample C4 baseline smoke runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.generation.parse_outputs import parse_raw_output


RUNS = (
    ("mistral_7b_instruct_v01", "S1"),
    ("mistral_7b_instruct_v01", "S2"),
    ("biomistral_7b", "S1"),
    ("biomistral_7b", "S2"),
)


def validate_smoke_suite(
    run_prefix: str,
    output_dir: str | Path,
    *,
    expected_samples: int = 5,
) -> dict[str, Any]:
    if isinstance(expected_samples, bool) or expected_samples <= 0:
        raise ValueError("expected_samples must be a positive integer")
    base = Path(output_dir)
    summaries: list[dict[str, Any]] = []
    reference_ids: list[str] | None = None

    for model_key, setting_id in RUNS:
        run_id = f"{run_prefix}-{model_key}-{setting_id}"
        with (base / f"{run_id}.jsonl").open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        manifest = json.loads(
            (base / f"{run_id}.manifest.json").read_text(encoding="utf-8")
        )
        sample_ids = [record["sample_id"] for record in records]
        parsed = [parse_raw_output(record["raw_output"]) for record in records]
        expected_max = manifest["generation"]["max_new_tokens"]
        expected_prompt = "question_only" if setting_id == "S1" else "question_context"
        configuration_matches = all(
            record["generation_parameters"]["max_new_tokens"] == expected_max
            and record["prompt_type"] == expected_prompt
            for record in records
        ) and manifest["prompt"]["type"] == expected_prompt
        identities_match = all(
            record["model_key"] == model_key
            and record["setting_id"] == setting_id
            and record["run_id"] == run_id
            for record in records
        )
        summary = {
            "run_id": run_id,
            "model_key": model_key,
            "setting_id": setting_id,
            "records": len(records),
            "unique_sample_ids": len(set(sample_ids)),
            "nonempty_outputs": sum(bool(record["raw_output"].strip()) for record in records),
            "parsed_labels": sum(
                item["parsed_final_answer"] != "unknown" for item in parsed
            ),
            "parser_failures": sum(item["parser_status"] == "error" for item in parsed),
            "exact_format_outputs": sum(
                item["prompt_format_compliance"] == "exact" for item in parsed
            ),
            "max_new_tokens": expected_max,
            "outputs_at_token_limit": sum(
                record["generated_tokens"] >= expected_max for record in records
            ),
            "configuration_matches": configuration_matches,
            "identities_match": identities_match,
        }
        summaries.append(summary)
        if reference_ids is None:
            reference_ids = sample_ids
        elif sample_ids != reference_ids:
            raise ValueError(f"sample order differs in {run_id}")

        if (
            len(records) != expected_samples
            or len(set(sample_ids)) != expected_samples
            or manifest["dataset"]["selected_samples"] != expected_samples
            or not configuration_matches
            or not identities_match
        ):
            raise ValueError(f"invalid {expected_samples}-sample run: {run_id}")

    ready = all(
        item["nonempty_outputs"] == expected_samples
        and item["parsed_labels"] == expected_samples
        and item["parser_failures"] == 0
        and item["outputs_at_token_limit"] == 0
        and item["configuration_matches"]
        for item in summaries
    )
    return {
        "run_prefix": run_prefix,
        "expected_samples_per_run": expected_samples,
        "runs": summaries,
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(reference_ids or []).encode("utf-8")
        ).hexdigest(),
        # ponytail: this prefix/length gate is only for smoke readiness; C7 owns
        # the full parser and evaluation contract.
        "ready_for_full_baseline": ready,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--expected-samples", type=int, default=5)
    parser.add_argument(
        "--output-dir", default="outputs/generations/standard", type=Path
    )
    parser.add_argument("--summary-path", required=True, type=Path)
    args = parser.parse_args()

    summary = validate_smoke_suite(
        args.run_prefix,
        args.output_dir,
        expected_samples=args.expected_samples,
    )
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
