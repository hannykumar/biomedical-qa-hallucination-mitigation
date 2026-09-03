"""Strict configuration contract for the two approved C3 models."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


DEFAULT_MODEL_CONFIG: Final[Path] = Path("configs/models.yaml")
COMMIT_SHA_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_MODEL_KEYS: Final[frozenset[str]] = frozenset(
    {"biomistral_7b", "mistral_7b_instruct_v01"}
)
EXPECTED_ROLES: Final[frozenset[str]] = frozenset(
    {"biomedical", "base_comparison"}
)
EXPECTED_RUNTIME_KEYS: Final[frozenset[str]] = frozenset(
    {
        "framework",
        "device",
        "dtype",
        "attention_implementation",
        "trust_remote_code",
        "use_fast_tokenizer",
        "padding_side",
        "pad_token_policy",
        "instruction_format",
    }
)
EXPECTED_SPEC_KEYS: Final[frozenset[str]] = frozenset(
    {
        "model_id",
        "revision",
        "tokenizer_revision",
        "role",
        "expected_model_type",
        "expected_layers",
        "expected_hidden_size",
        "expected_tokenizer_model_sha256",
    }
)


class ModelConfigurationError(ValueError):
    """Raised when `configs/models.yaml` violates the C3 contract."""


@dataclass(frozen=True, slots=True)
class ModelRuntimeConfig:
    framework: str
    device: str
    dtype: str
    attention_implementation: str
    trust_remote_code: bool
    use_fast_tokenizer: bool
    padding_side: str
    pad_token_policy: str
    instruction_format: str


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    model_id: str
    revision: str
    tokenizer_revision: str
    role: str
    expected_model_type: str
    expected_layers: int
    expected_hidden_size: int
    expected_tokenizer_model_sha256: str


@dataclass(frozen=True, slots=True)
class ModelRegistry:
    schema_version: int
    runtime: ModelRuntimeConfig
    models: Mapping[str, ModelSpec]
    source_path: str

    def get(self, model_key: str) -> ModelSpec:
        try:
            return self.models[model_key]
        except KeyError as error:
            supported = ", ".join(sorted(self.models))
            raise ModelConfigurationError(
                f"unknown model key {model_key!r}; expected one of: {supported}"
            ) from error


def load_model_registry(path: str | Path = DEFAULT_MODEL_CONFIG) -> ModelRegistry:
    """Read and strictly validate the versioned model registry."""

    config_path = Path(path)
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("PyYAML is required to load the model registry") from error

    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ModelConfigurationError(
            f"model configuration does not exist: {config_path}"
        ) from error
    except yaml.YAMLError as error:
        raise ModelConfigurationError(
            f"invalid YAML in model configuration {config_path}: {error}"
        ) from error

    root = _mapping(payload, "model configuration")
    _exact_keys(root, frozenset({"schema_version", "runtime", "models"}), "root")
    schema_version = _positive_int(root["schema_version"], "schema_version")
    runtime = _parse_runtime(_mapping(root["runtime"], "runtime"))
    raw_models = _mapping(root["models"], "models")
    _exact_keys(raw_models, EXPECTED_MODEL_KEYS, "models")

    parsed_models = {
        key: _parse_model_spec(key, _mapping(value, f"models.{key}"))
        for key, value in raw_models.items()
    }
    roles = {spec.role for spec in parsed_models.values()}
    if roles != EXPECTED_ROLES:
        raise ModelConfigurationError(
            f"model roles must be exactly {sorted(EXPECTED_ROLES)}, got {sorted(roles)}"
        )

    tokenizer_checksums = {
        spec.expected_tokenizer_model_sha256 for spec in parsed_models.values()
    }
    if len(tokenizer_checksums) != 1:
        raise ModelConfigurationError(
            "the controlled comparison requires identical tokenizer.model checksums"
        )

    return ModelRegistry(
        schema_version=schema_version,
        runtime=runtime,
        models=parsed_models,
        source_path=str(config_path),
    )


def _parse_runtime(raw: Mapping[str, Any]) -> ModelRuntimeConfig:
    _exact_keys(raw, EXPECTED_RUNTIME_KEYS, "runtime")
    runtime = ModelRuntimeConfig(
        framework=_text(raw["framework"], "runtime.framework"),
        device=_text(raw["device"], "runtime.device"),
        dtype=_text(raw["dtype"], "runtime.dtype"),
        attention_implementation=_text(
            raw["attention_implementation"], "runtime.attention_implementation"
        ),
        trust_remote_code=_boolean(raw["trust_remote_code"], "runtime.trust_remote_code"),
        use_fast_tokenizer=_boolean(
            raw["use_fast_tokenizer"], "runtime.use_fast_tokenizer"
        ),
        padding_side=_text(raw["padding_side"], "runtime.padding_side"),
        pad_token_policy=_text(
            raw["pad_token_policy"], "runtime.pad_token_policy"
        ),
        instruction_format=_text(
            raw["instruction_format"], "runtime.instruction_format"
        ),
    )
    expected_values = {
        "framework": "transformers",
        "device": "cuda",
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "padding_side": "left",
        "pad_token_policy": "eos",
        "instruction_format": "mistral_v01_shared",
    }
    for field, expected in expected_values.items():
        actual = getattr(runtime, field)
        if actual != expected:
            raise ModelConfigurationError(
                f"runtime.{field} must be {expected!r}, got {actual!r}"
            )
    if runtime.trust_remote_code:
        raise ModelConfigurationError("runtime.trust_remote_code must remain false")
    if runtime.use_fast_tokenizer:
        raise ModelConfigurationError(
            "runtime.use_fast_tokenizer must remain false for the shared tokenizer contract"
        )
    return runtime


def _parse_model_spec(key: str, raw: Mapping[str, Any]) -> ModelSpec:
    _exact_keys(raw, EXPECTED_SPEC_KEYS, f"models.{key}")
    revision = _commit_sha(raw["revision"], f"models.{key}.revision")
    tokenizer_revision = _commit_sha(
        raw["tokenizer_revision"], f"models.{key}.tokenizer_revision"
    )
    checksum = _text(
        raw["expected_tokenizer_model_sha256"],
        f"models.{key}.expected_tokenizer_model_sha256",
    ).lower()
    if not SHA256_PATTERN.fullmatch(checksum):
        raise ModelConfigurationError(
            f"models.{key}.expected_tokenizer_model_sha256 must be 64 lowercase hex characters"
        )
    return ModelSpec(
        key=key,
        model_id=_text(raw["model_id"], f"models.{key}.model_id"),
        revision=revision,
        tokenizer_revision=tokenizer_revision,
        role=_text(raw["role"], f"models.{key}.role"),
        expected_model_type=_text(
            raw["expected_model_type"], f"models.{key}.expected_model_type"
        ),
        expected_layers=_positive_int(
            raw["expected_layers"], f"models.{key}.expected_layers"
        ),
        expected_hidden_size=_positive_int(
            raw["expected_hidden_size"], f"models.{key}.expected_hidden_size"
        ),
        expected_tokenizer_model_sha256=checksum,
    )


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelConfigurationError(f"{location} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ModelConfigurationError(f"{location} keys must be strings")
    return value


def _exact_keys(
    mapping: Mapping[str, Any], expected: frozenset[str], location: str
) -> None:
    actual = frozenset(mapping)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise ModelConfigurationError(f"{location} keys invalid: {', '.join(details)}")


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelConfigurationError(f"{location} must be a nonempty string")
    if value != value.strip():
        raise ModelConfigurationError(f"{location} must not contain surrounding whitespace")
    return value


def _positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelConfigurationError(f"{location} must be a positive integer")
    return value


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise ModelConfigurationError(f"{location} must be a boolean")
    return value


def _commit_sha(value: Any, location: str) -> str:
    revision = _text(value, location).lower()
    if not COMMIT_SHA_PATTERN.fullmatch(revision):
        raise ModelConfigurationError(
            f"{location} must be an immutable 40-character commit SHA"
        )
    return revision
