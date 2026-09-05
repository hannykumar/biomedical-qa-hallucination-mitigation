"""Tests for exact, versioned C2 prompt construction."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data.preprocess import normalize_pubmedqa_record
from src.prompts.build_prompts import (
    PromptBuildError,
    PromptConfigurationError,
    build_prompt,
    build_prompts,
    load_prompt_config,
)
from tests.test_preprocess import source_record


class PromptBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_prompt_config()
        cls.example = normalize_pubmedqa_record(source_record(), source_row_index=0)

    def test_loads_approved_version_two_configuration(self) -> None:
        self.assertEqual(self.config.version, 2)
        self.assertEqual(self.config.prompt_format, "structured")
        self.assertEqual(self.config.source_path, "configs/prompts.yaml")

    def test_builds_exact_question_only_prompt(self) -> None:
        built = build_prompt(
            self.example,
            prompt_type="question_only",
            config=self.config,
        )

        self.assertEqual(
            built.text,
            "Question: Does the intervention help?\n\n"
            "Your first line must be exactly one of:\n"
            "Final answer: yes\n"
            "Final answer: no\n"
            "Final answer: maybe\n"
            'Your second line must begin with "Explanation:" and contain no more '
            "than three concise sentences.",
        )
        self.assertNotIn("Background evidence", built.text)
        self.assertFalse(built.text.endswith("\n"))

    def test_builds_exact_question_context_prompt(self) -> None:
        built = build_prompt(
            self.example,
            prompt_type="question_context",
            config=self.config,
        )

        self.assertEqual(
            built.text,
            "Context: Background evidence.\n\nResults evidence.\n\n"
            "Question: Does the intervention help?\n\n"
            "Your first line must be exactly one of:\n"
            "Final answer: yes\n"
            "Final answer: no\n"
            "Final answer: maybe\n"
            'Your second line must begin with "Explanation:" and contain no more '
            "than three concise sentences.",
        )
        self.assertFalse(built.text.endswith("\n"))

    def test_attaches_stable_prompt_provenance(self) -> None:
        first = build_prompt(
            self.example,
            prompt_type="question_context",
            config=self.config,
        )
        second = build_prompt(
            self.example,
            prompt_type="question_context",
            config=self.config,
        )

        self.assertEqual(first.sample_id, self.example.sample_id)
        self.assertEqual(first.prompt_type, "question_context")
        self.assertEqual(first.prompt_version, 2)
        self.assertEqual(first.prompt_format, "structured")
        self.assertEqual(len(first.template_sha256), 64)
        self.assertEqual(len(first.prompt_sha256), 64)
        self.assertEqual(first.template_sha256, second.template_sha256)
        self.assertEqual(first.prompt_sha256, second.prompt_sha256)
        self.assertEqual(
            first.metadata(),
            {
                "prompt_type": "question_context",
                "prompt_version": 2,
                "prompt_format": "structured",
                "template_sha256": first.template_sha256,
                "prompt_sha256": first.prompt_sha256,
            },
        )

    def test_batch_builder_preserves_input_order(self) -> None:
        second_example = normalize_pubmedqa_record(
            source_record(pubid=67890), source_row_index=1
        )
        prompts = build_prompts(
            [second_example, self.example],
            prompt_type="question_only",
            config=self.config,
        )

        self.assertEqual(
            [prompt.sample_id for prompt in prompts],
            [second_example.sample_id, self.example.sample_id],
        )

    def test_rejects_unknown_prompt_type(self) -> None:
        with self.assertRaisesRegex(PromptBuildError, "unsupported prompt_type"):
            build_prompt(
                self.example,
                prompt_type="strict",
                config=self.config,
            )


class PromptConfigurationTests(unittest.TestCase):
    def test_rejects_missing_required_placeholder(self) -> None:
        payload = valid_config_payload()
        payload["prompts"]["question_context"] = "Question: {question}"

        with self.assertRaisesRegex(PromptConfigurationError, "placeholders"):
            load_temporary_config(payload)

    def test_rejects_unexpected_prompt_key(self) -> None:
        payload = valid_config_payload()
        payload["prompts"]["strict"] = "Question: {question}"

        with self.assertRaisesRegex(PromptConfigurationError, "unexpected"):
            load_temporary_config(payload)

    def test_rejects_format_specifications(self) -> None:
        payload = valid_config_payload()
        payload["prompts"]["question_only"] = "Question: {question!r}"

        with self.assertRaisesRegex(PromptConfigurationError, "conversions"):
            load_temporary_config(payload)

    def test_rejects_leading_or_trailing_template_whitespace(self) -> None:
        payload = valid_config_payload()
        payload["prompts"]["question_only"] = " Question: {question}"

        with self.assertRaisesRegex(PromptConfigurationError, "whitespace"):
            load_temporary_config(payload)


def valid_config_payload() -> dict[str, object]:
    return {
        "version": 1,
        "default_format": "minimal",
        "prompts": {
            "question_only": "Question: {question}",
            "question_context": "Context: {context}\n\nQuestion: {question}",
        },
    }


def load_temporary_config(payload: dict[str, object]) -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "prompts.yaml"
        path.write_text(json.dumps(payload), encoding="utf-8")
        load_prompt_config(path)


if __name__ == "__main__":
    unittest.main()
