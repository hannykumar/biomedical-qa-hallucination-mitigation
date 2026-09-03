"""One-sample GPU smoke generation for an approved C3 model."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any

from src.data.preprocess import normalize_pubmedqa_record
from src.models.config import load_model_registry
from src.models.formatting import serialize_mistral_v01_instruction
from src.models.load_model import load_model_bundle
from src.prompts.build_prompts import build_prompt, load_prompt_config


def run_smoke_test(
    model_key: str,
    *,
    cache_dir: str | Path,
    max_new_tokens: int = 16,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Load one model and generate a deterministic short completion."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    try:
        import torch
        import transformers
    except ImportError as error:  # pragma: no cover - cluster-only path
        raise RuntimeError("C3 runtime dependencies are missing") from error

    registry = load_model_registry()
    prompt_config = load_prompt_config()
    bundle = load_model_bundle(
        registry,
        model_key,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    example = normalize_pubmedqa_record(_smoke_source_record(), source_row_index=0)
    built_prompt = build_prompt(
        example,
        prompt_type="question_context",
        config=prompt_config,
    )
    serialized_prompt = serialize_mistral_v01_instruction(built_prompt)
    inputs = bundle.tokenizer(
        serialized_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(bundle.device)

    with torch.inference_mode():
        output_ids = bundle.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=bundle.tokenizer.pad_token_id,
            eos_token_id=bundle.tokenizer.eos_token_id,
        )

    input_length = int(inputs["input_ids"].shape[-1])
    new_ids = output_ids[0, input_length:]
    completion = bundle.tokenizer.decode(new_ids, skip_special_tokens=True)
    device_properties = torch.cuda.get_device_properties(0)

    return {
        "status": "ok",
        "model_key": model_key,
        "model_id": bundle.spec.model_id,
        "model_revision": bundle.spec.revision,
        "tokenizer_revision": bundle.spec.tokenizer_revision,
        "device_name": device_properties.name,
        "device_total_memory_bytes": device_properties.total_memory,
        "dtype": bundle.dtype,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "prompt": built_prompt.metadata(),
        "serialized_input_tokens": input_length,
        "generated_tokens": int(new_ids.shape[-1]),
        "completion": completion,
    }


def _smoke_source_record() -> dict[str, Any]:
    return {
        "pubid": 99999999,
        "question": "Does the reported intervention improve the measured outcome?",
        "context": {
            "contexts": [
                "Participants receiving the intervention improved more than controls."
            ],
            "labels": ["RESULTS"],
            "meshes": [],
        },
        "long_answer": "The intervention improved the measured outcome.",
        "final_decision": "yes",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the C3 model GPU smoke test.")
    parser.add_argument(
        "--model-key",
        required=True,
        choices=("biomistral_7b", "mistral_7b_instruct_v01"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "data/raw/huggingface-models")),
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_smoke_test(
        args.model_key,
        cache_dir=args.cache_dir,
        max_new_tokens=args.max_new_tokens,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
