"""Load, validate, and build versioned PubMedQA prompts.

The builder consumes the normalized C1 data contract. It does not know about a
tokenizer or model chat template; that model-specific boundary belongs to C3/C4.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any, Final, Literal, TypeAlias, cast

from src.data.preprocess import NormalizedPubMedQAExample


PromptType: TypeAlias = Literal["question_only", "question_context"]
SUPPORTED_PROMPT_TYPES: Final[tuple[PromptType, ...]] = (
    "question_only",
    "question_context",
)
DEFAULT_PROMPT_CONFIG: Final[Path] = Path("configs/prompts.yaml")
EXPECTED_PLACEHOLDERS: Final[dict[PromptType, frozenset[str]]] = {
    "question_only": frozenset({"question"}),
    "question_context": frozenset({"context", "question"}),
}
EXPECTED_TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset(
    {"version", "default_format", "prompts"}
)


class PromptConfigurationError(ValueError):
    """Raised when the prompt YAML does not satisfy the versioned contract."""


class PromptBuildError(ValueError):
    """Raised when a normalized example cannot produce the selected prompt."""


@dataclass(frozen=True, slots=True)
class PromptConfig:
    """Validated prompt configuration with immutable template fields."""

    version: int
    prompt_format: str
    question_only_template: str
    question_context_template: str
    source_path: str

    def template_for(self, prompt_type: PromptType) -> str:
        """Return the validated template for one supported prompt type."""

        if prompt_type == "question_only":
            return self.question_only_template
        if prompt_type == "question_context":
            return self.question_context_template
        raise PromptBuildError(f"unsupported prompt_type: {prompt_type!r}")


@dataclass(frozen=True, slots=True)
class BuiltPrompt:
    """Exact prompt text plus provenance required by later generation records."""

    sample_id: str
    text: str
    prompt_type: PromptType
    prompt_version: int
    prompt_format: str
    template_sha256: str
    prompt_sha256: str

    def metadata(self) -> dict[str, str | int]:
        """Return JSON-serializable prompt provenance for a run record."""

        return {
            "prompt_type": self.prompt_type,
            "prompt_version": self.prompt_version,
            "prompt_format": self.prompt_format,
            "template_sha256": self.template_sha256,
            "prompt_sha256": self.prompt_sha256,
        }


def load_prompt_config(
    path: str | Path = DEFAULT_PROMPT_CONFIG,
) -> PromptConfig:
    """Read and strictly validate the versioned YAML prompt configuration."""

    config_path = Path(path)
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - environment-specific path
        raise RuntimeError(
            "PyYAML is required to read configs/prompts.yaml; install the project "
            "environment before building prompts."
        ) from error

    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PromptConfigurationError(
            f"prompt configuration does not exist: {config_path}"
        ) from error
    except yaml.YAMLError as error:
        raise PromptConfigurationError(
            f"invalid YAML in prompt configuration {config_path}: {error}"
        ) from error

    root = _require_mapping(payload, "prompt configuration")
    _require_exact_keys(root, EXPECTED_TOP_LEVEL_KEYS, "prompt configuration")

    version = root["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise PromptConfigurationError("version must be a positive integer")

    prompt_format = _require_nonempty_string(
        root["default_format"], "default_format"
    )
    prompts = _require_mapping(root["prompts"], "prompts")
    _require_exact_keys(prompts, frozenset(SUPPORTED_PROMPT_TYPES), "prompts")

    question_only_template = _validate_template(
        prompts["question_only"], "question_only"
    )
    question_context_template = _validate_template(
        prompts["question_context"], "question_context"
    )

    return PromptConfig(
        version=version,
        prompt_format=prompt_format,
        question_only_template=question_only_template,
        question_context_template=question_context_template,
        source_path=str(config_path),
    )


def build_prompt(
    example: NormalizedPubMedQAExample,
    *,
    prompt_type: PromptType | str,
    config: PromptConfig,
) -> BuiltPrompt:
    """Build one exact prompt without mutating the normalized example."""

    if not isinstance(example, NormalizedPubMedQAExample):
        raise TypeError("example must be a NormalizedPubMedQAExample")
    normalized_prompt_type = _normalize_prompt_type(prompt_type)

    if not example.question:
        raise PromptBuildError(f"sample {example.sample_id}: question is empty")
    if normalized_prompt_type == "question_context" and not example.context:
        raise PromptBuildError(f"sample {example.sample_id}: context is empty")

    template = config.template_for(normalized_prompt_type)
    format_values = {
        "question": example.question,
        "context": example.context,
    }
    text = template.format_map(format_values)

    return BuiltPrompt(
        sample_id=example.sample_id,
        text=text,
        prompt_type=normalized_prompt_type,
        prompt_version=config.version,
        prompt_format=config.prompt_format,
        template_sha256=_sha256_text(template),
        prompt_sha256=_sha256_text(text),
    )


def build_prompts(
    examples: Iterable[NormalizedPubMedQAExample],
    *,
    prompt_type: PromptType | str,
    config: PromptConfig,
) -> list[BuiltPrompt]:
    """Build prompts in input order using one validated configuration."""

    normalized_prompt_type = _normalize_prompt_type(prompt_type)
    return [
        build_prompt(
            example,
            prompt_type=normalized_prompt_type,
            config=config,
        )
        for example in examples
    ]


def _normalize_prompt_type(value: PromptType | str) -> PromptType:
    if value not in SUPPORTED_PROMPT_TYPES:
        supported = ", ".join(SUPPORTED_PROMPT_TYPES)
        raise PromptBuildError(
            f"unsupported prompt_type {value!r}; expected one of: {supported}"
        )
    return cast(PromptType, value)


def _validate_template(value: Any, prompt_type: PromptType) -> str:
    template = _require_nonempty_string(value, f"prompts.{prompt_type}")
    if template != template.strip():
        raise PromptConfigurationError(
            f"prompts.{prompt_type} must not contain leading or trailing whitespace"
        )

    placeholders: set[str] = set()
    try:
        parsed_fields = Formatter().parse(template)
        for _, field_name, format_spec, conversion in parsed_fields:
            if field_name is None:
                continue
            if format_spec or conversion:
                raise PromptConfigurationError(
                    f"prompts.{prompt_type} placeholders must not use conversions "
                    "or format specifications"
                )
            placeholders.add(field_name)
    except ValueError as error:
        raise PromptConfigurationError(
            f"prompts.{prompt_type} contains invalid braces: {error}"
        ) from error

    expected = EXPECTED_PLACEHOLDERS[prompt_type]
    if placeholders != expected:
        raise PromptConfigurationError(
            f"prompts.{prompt_type} placeholders must be exactly "
            f"{sorted(expected)}, got {sorted(placeholders)}"
        )
    return template


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PromptConfigurationError(f"{location} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise PromptConfigurationError(f"{location} keys must be strings")
    return value


def _require_exact_keys(
    mapping: Mapping[str, Any],
    expected: frozenset[str],
    location: str,
) -> None:
    actual = frozenset(mapping)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise PromptConfigurationError(f"{location} keys invalid: {', '.join(details)}")


def _require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise PromptConfigurationError(f"{location} must be a string")
    if not value.strip():
        raise PromptConfigurationError(f"{location} must not be empty")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
