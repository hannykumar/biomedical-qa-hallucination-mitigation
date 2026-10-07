"""Check whether reference long answers can be extracted from model context."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.generation.run_generation import load_normalized_examples


def inspect_context(dataset_path: str | Path) -> dict:
    path = Path(dataset_path)
    examples = load_normalized_examples(path)
    def normalize(text: str) -> str:
        return " ".join(text.casefold().split())

    return {
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "samples": len(examples),
        "exact_reference_in_context": sum(e.gold_long_answer in e.context for e in examples),
        "normalized_reference_in_context": sum(
            normalize(e.gold_long_answer) in normalize(e.context) for e in examples
        ),
        "normalization": "casefold and collapse whitespace; no paraphrase matching",
        "interpretation": "Tests full-reference extractability, not evidence support or hallucination.",
        "examples": [
            {"sample_id": e.sample_id, "question": e.question,
             "context": e.context, "gold_long_answer": e.gold_long_answer}
            for e in examples[:2]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=Path("data/processed/pubmedqa_labeled_clean.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("outputs/metrics/context-extraction-audit.json"))
    args = parser.parse_args()
    report = inspect_context(args.dataset_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
