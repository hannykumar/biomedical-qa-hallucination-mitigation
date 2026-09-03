"""Load and validate an approved causal model/tokenizer pair on CUDA."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models.config import ModelRegistry, ModelSpec


class ModelSetupError(RuntimeError):
    """Raised when the runtime or downloaded model violates the C3 contract."""


@dataclass(frozen=True, slots=True)
class LoadedModelBundle:
    spec: ModelSpec
    model: Any
    tokenizer: Any
    device: str
    dtype: str
    cache_dir: str


def load_model_bundle(
    registry: ModelRegistry,
    model_key: str,
    *,
    cache_dir: str | Path,
    local_files_only: bool = False,
) -> LoadedModelBundle:
    """Load one pinned model directly on one CUDA GPU and validate metadata."""

    try:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - exercised on cluster
        raise ModelSetupError(
            "C3 runtime dependencies are missing; install the cluster environment"
        ) from error

    runtime = registry.runtime
    spec = registry.get(model_key)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    if runtime.device != "cuda":
        raise ModelSetupError(f"unsupported configured device: {runtime.device}")
    if not torch.cuda.is_available():
        raise ModelSetupError("CUDA is not available; run this loader inside a GPU job")
    if runtime.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ModelSetupError(
            "the allocated GPU does not support bfloat16; request an A40/H200 node"
        )

    tokenizer_model_path = hf_hub_download(
        repo_id=spec.model_id,
        filename="tokenizer.model",
        revision=spec.tokenizer_revision,
        cache_dir=cache_path,
        local_files_only=local_files_only,
    )
    actual_checksum = _sha256_file(Path(tokenizer_model_path))
    if actual_checksum != spec.expected_tokenizer_model_sha256:
        raise ModelSetupError(
            f"tokenizer.model checksum mismatch for {spec.model_id}: "
            f"expected {spec.expected_tokenizer_model_sha256}, got {actual_checksum}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id,
        revision=spec.tokenizer_revision,
        cache_dir=cache_path,
        local_files_only=local_files_only,
        trust_remote_code=runtime.trust_remote_code,
        use_fast=runtime.use_fast_tokenizer,
    )
    tokenizer.padding_side = runtime.padding_side
    if runtime.pad_token_policy == "eos":
        if tokenizer.eos_token is None:
            raise ModelSetupError(f"{spec.model_id} tokenizer has no EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _resolve_dtype(torch, runtime.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        cache_dir=cache_path,
        local_files_only=local_files_only,
        trust_remote_code=runtime.trust_remote_code,
        torch_dtype=dtype,
        attn_implementation=runtime.attention_implementation,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
    )
    _validate_loaded_model(model, spec)
    model.eval()

    return LoadedModelBundle(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        device="cuda:0",
        dtype=runtime.dtype,
        cache_dir=str(cache_path),
    )


def _validate_loaded_model(model: Any, spec: ModelSpec) -> None:
    config = model.config
    checks = {
        "model_type": (getattr(config, "model_type", None), spec.expected_model_type),
        "num_hidden_layers": (
            getattr(config, "num_hidden_layers", None),
            spec.expected_layers,
        ),
        "hidden_size": (getattr(config, "hidden_size", None), spec.expected_hidden_size),
    }
    mismatches = [
        f"{field}: expected {expected!r}, got {actual!r}"
        for field, (actual, expected) in checks.items()
        if actual != expected
    ]
    if mismatches:
        raise ModelSetupError(
            f"loaded model metadata mismatch for {spec.model_id}: {'; '.join(mismatches)}"
        )


def _resolve_dtype(torch: Any, dtype_name: str) -> Any:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[dtype_name]
    except KeyError as error:
        raise ModelSetupError(f"unsupported dtype: {dtype_name}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
