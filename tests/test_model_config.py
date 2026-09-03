"""Dependency-light tests for the strict C3 model registry."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.models.config import (
    ModelConfigurationError,
    load_model_registry,
)


class ModelRegistryTests(unittest.TestCase):
    def test_loads_pinned_controlled_model_registry(self) -> None:
        registry = load_model_registry()

        self.assertEqual(registry.schema_version, 1)
        self.assertEqual(registry.runtime.device, "cuda")
        self.assertEqual(registry.runtime.dtype, "bfloat16")
        self.assertFalse(registry.runtime.trust_remote_code)
        self.assertFalse(registry.runtime.use_fast_tokenizer)
        self.assertEqual(
            set(registry.models),
            {"biomistral_7b", "mistral_7b_instruct_v01"},
        )
        for spec in registry.models.values():
            self.assertEqual(len(spec.revision), 40)
            self.assertEqual(len(spec.tokenizer_revision), 40)
            self.assertEqual(spec.expected_model_type, "mistral")
            self.assertEqual(spec.expected_layers, 32)
            self.assertEqual(spec.expected_hidden_size, 4096)

    def test_two_models_share_tokenizer_model_checksum(self) -> None:
        registry = load_model_registry()
        checksums = {
            spec.expected_tokenizer_model_sha256
            for spec in registry.models.values()
        }
        self.assertEqual(
            checksums,
            {
                "dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055"
            },
        )

    def test_rejects_mutable_revision(self) -> None:
        payload = valid_model_payload()
        payload["models"]["biomistral_7b"]["revision"] = "main"
        with self.assertRaisesRegex(ModelConfigurationError, "immutable"):
            load_temporary_registry(payload)

    def test_rejects_remote_code(self) -> None:
        payload = valid_model_payload()
        payload["runtime"]["trust_remote_code"] = True
        with self.assertRaisesRegex(ModelConfigurationError, "must remain false"):
            load_temporary_registry(payload)

    def test_rejects_tokenizer_checksum_mismatch_between_models(self) -> None:
        payload = valid_model_payload()
        payload["models"]["biomistral_7b"][
            "expected_tokenizer_model_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(ModelConfigurationError, "identical"):
            load_temporary_registry(payload)

    def test_rejects_unknown_model_key_lookup(self) -> None:
        registry = load_model_registry()
        with self.assertRaisesRegex(ModelConfigurationError, "unknown model key"):
            registry.get("newer_model")


def valid_model_payload() -> dict[str, object]:
    # JSON round-tripping creates a mutable deep copy using only basic types.
    import yaml

    payload = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
    return copy.deepcopy(payload)


def load_temporary_registry(payload: dict[str, object]) -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "models.yaml"
        path.write_text(json.dumps(payload), encoding="utf-8")
        load_model_registry(path)


if __name__ == "__main__":
    unittest.main()
