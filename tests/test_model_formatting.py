"""Tests for shared model instruction serialization."""

from __future__ import annotations

import unittest

from src.data.preprocess import normalize_pubmedqa_record
from src.models.formatting import serialize_mistral_v01_instruction
from src.prompts.build_prompts import build_prompt, load_prompt_config
from tests.test_preprocess import source_record


class MistralFormattingTests(unittest.TestCase):
    def test_wraps_question_context_prompt_exactly_once(self) -> None:
        example = normalize_pubmedqa_record(source_record(), source_row_index=0)
        prompt = build_prompt(
            example,
            prompt_type="question_context",
            config=load_prompt_config(),
        )

        serialized = serialize_mistral_v01_instruction(prompt)

        self.assertEqual(
            serialized,
            "<s>[INST] Context: Background evidence.\n\nResults evidence.\n\n"
            "Question: Does the intervention help?\n\n"
            "Respond with exactly two lines and no other text.\n"
            "Line 1 must be exactly one of:\n"
            "Final answer: yes\n"
            "Final answer: no\n"
            "Final answer: maybe\n"
            'Line 2 must begin with "Explanation:" and contain exactly one concise '
            "sentence.\n"
            "Always include both lines. Do not stop after line 1. Do not begin with "
            '"Explanation:". [/INST]',
        )
        self.assertEqual(serialized.count("<s>"), 1)
        self.assertEqual(serialized.count("[INST]"), 1)
        self.assertEqual(serialized.count("[/INST]"), 1)

    def test_rejects_non_built_prompt(self) -> None:
        with self.assertRaisesRegex(TypeError, "BuiltPrompt"):
            serialize_mistral_v01_instruction("Question: test")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
