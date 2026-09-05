"""Deterministic standard decoding for the S1/S2 baseline."""

from __future__ import annotations

from dataclasses import dataclass

from src.models.formatting import serialize_mistral_v01_instruction
from src.models.load_model import LoadedModelBundle
from src.prompts.build_prompts import BuiltPrompt


@dataclass(frozen=True, slots=True)
class StandardGeneration:
    raw_output: str
    input_tokens: int
    generated_tokens: int


def generate_standard(
    bundle: LoadedModelBundle,
    prompt: BuiltPrompt,
    *,
    max_new_tokens: int,
) -> StandardGeneration:
    """Generate one greedy completion on the bundle's CUDA device."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    try:
        import torch
    except ImportError as error:  # pragma: no cover - cluster-only dependency
        raise RuntimeError("PyTorch is required inside the GPU job") from error

    serialized = serialize_mistral_v01_instruction(prompt)
    inputs = bundle.tokenizer(
        serialized,
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

    input_tokens = len(inputs["input_ids"][0])
    new_ids = output_ids[0][input_tokens:]
    return StandardGeneration(
        raw_output=bundle.tokenizer.decode(new_ids, skip_special_tokens=True),
        input_tokens=input_tokens,
        generated_tokens=len(new_ids),
    )
