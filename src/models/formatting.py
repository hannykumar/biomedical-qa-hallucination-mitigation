"""Shared Mistral-v0.1 instruction serialization for the controlled comparison."""

from __future__ import annotations

from typing import Final

from src.prompts.build_prompts import BuiltPrompt


INSTRUCTION_PREFIX: Final[str] = "<s>[INST] "
INSTRUCTION_SUFFIX: Final[str] = " [/INST]"


def serialize_mistral_v01_instruction(prompt: BuiltPrompt) -> str:
    """Wrap C2 semantic text in the official single-turn Mistral format.

    Call the tokenizer with ``add_special_tokens=False`` because the serialized
    string already includes the begin-of-sequence token.
    """

    if not isinstance(prompt, BuiltPrompt):
        raise TypeError("prompt must be a BuiltPrompt")
    if not prompt.text:
        raise ValueError("prompt text must not be empty")
    return f"{INSTRUCTION_PREFIX}{prompt.text}{INSTRUCTION_SUFFIX}"
