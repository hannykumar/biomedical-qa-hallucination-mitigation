"""Dependency-light checks for C4 standard generation and resume safety."""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.data.preprocess import normalize_pubmedqa_record, write_normalized_csv
from src.decoding.standard import generate_standard
from src.generation.run_generation import (
    GenerationRunError,
    _append_record,
    _completed_sample_ids,
    _write_or_validate_manifest,
    load_normalized_examples,
    load_standard_run_config,
)
from src.prompts.build_prompts import build_prompt, load_prompt_config
from tests.test_preprocess import source_record


class _Batch(dict):
    def to(self, device: str) -> "_Batch":
        self.device = device
        return self


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text: str, **kwargs: object) -> _Batch:
        self.text = text
        self.kwargs = kwargs
        return _Batch(input_ids=[[10, 11]])

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        self.decode_kwargs = kwargs
        return "yes because evidence"


class _Model:
    def generate(self, **kwargs: object) -> list[list[int]]:
        self.kwargs = kwargs
        return [[10, 11, 12, 13]]


class StandardDecoderTests(unittest.TestCase):
    def test_generates_only_new_tokens_with_greedy_settings(self) -> None:
        example = normalize_pubmedqa_record(source_record(), source_row_index=0)
        prompt = build_prompt(
            example,
            prompt_type="question_only",
            config=load_prompt_config(),
        )
        tokenizer = _Tokenizer()
        model = _Model()
        bundle = SimpleNamespace(
            tokenizer=tokenizer,
            model=model,
            device="cuda:0",
        )
        fake_torch = SimpleNamespace(inference_mode=contextlib.nullcontext)

        with patch.dict(sys.modules, {"torch": fake_torch}):
            result = generate_standard(bundle, prompt, max_new_tokens=128)

        self.assertEqual(result.raw_output, "yes because evidence")
        self.assertEqual(result.input_tokens, 2)
        self.assertEqual(result.generated_tokens, 2)
        self.assertFalse(model.kwargs["do_sample"])
        self.assertEqual(model.kwargs["max_new_tokens"], 128)
        self.assertFalse(tokenizer.kwargs["add_special_tokens"])


class GenerationOrchestrationTests(unittest.TestCase):
    def test_resolves_only_deterministic_s1_s2(self) -> None:
        s1 = load_standard_run_config("S1")
        self.assertEqual(s1.prompt_type, "question_only")
        self.assertEqual(s1.max_new_tokens, 256)
        self.assertEqual(load_standard_run_config("S2").prompt_type, "question_context")
        with self.assertRaisesRegex(GenerationRunError, "S1 or S2"):
            load_standard_run_config("S3")

    def test_reloads_c1_csv_and_preserves_limit(self) -> None:
        examples = [
            normalize_pubmedqa_record(source_record(pubid=12345), source_row_index=0),
            normalize_pubmedqa_record(source_record(pubid=67890), source_row_index=1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clean.csv"
            write_normalized_csv(examples, path)
            loaded = load_normalized_examples(path, limit=1)

        self.assertEqual(loaded, examples[:1])

    def test_resume_reads_completed_ids_and_rejects_duplicates(self) -> None:
        record = {
            "run_id": "pilot",
            "model_key": "biomistral_7b",
            "setting_id": "S1",
            "sample_id": "sample-1",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.jsonl"
            _append_record(path, record)
            completed = _completed_sample_ids(
                path,
                run_id="pilot",
                model_key="biomistral_7b",
                setting_id="S1",
            )
            self.assertEqual(completed, {"sample-1"})

            _append_record(path, record)
            with self.assertRaisesRegex(GenerationRunError, "duplicate sample_id"):
                _completed_sample_ids(
                    path,
                    run_id="pilot",
                    model_key="biomistral_7b",
                    setting_id="S1",
                )

    def test_resume_rejects_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "run_id": "other",
                        "model_key": "biomistral_7b",
                        "setting_id": "S1",
                        "sample_id": "sample-1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(GenerationRunError, "identity mismatch"):
                _completed_sample_ids(
                    path,
                    run_id="pilot",
                    model_key="biomistral_7b",
                    setting_id="S1",
                )

    def test_manifest_is_immutable_for_one_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.manifest.json"
            _write_or_validate_manifest(path, {"run_id": "pilot", "limit": 5})
            _write_or_validate_manifest(path, {"run_id": "pilot", "limit": 5})

            with self.assertRaisesRegex(GenerationRunError, "does not match"):
                _write_or_validate_manifest(path, {"run_id": "pilot", "limit": 6})


if __name__ == "__main__":
    unittest.main()
