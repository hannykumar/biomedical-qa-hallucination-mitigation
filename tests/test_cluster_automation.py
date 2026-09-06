"""Checks for the local-to-Slurm smoke-suite automation."""

from __future__ import annotations

import subprocess
import json
import tempfile
import unittest
from pathlib import Path

from src.generation.validate_smoke_suite import RUNS, validate_smoke_suite


class ClusterAutomationTests(unittest.TestCase):
    def test_help_does_not_connect(self) -> None:
        result = subprocess.run(
            ["bash", "cluster/run_standard_smoke_suite_remote.sh", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn(
            "USER@HOST REMOTE_PROJECT_DIR RUN_PREFIX SAMPLE_LIMIT MAX_NEW_TOKENS",
            result.stderr,
        )

    def test_ready_summary_requires_four_complete_aligned_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for model_key, setting_id in RUNS:
                run_id = f"suite-{model_key}-{setting_id}"
                records = [
                    {
                        "run_id": run_id,
                        "model_key": model_key,
                        "setting_id": setting_id,
                        "sample_id": f"sample-{index}",
                        "prompt_type": (
                            "question_only" if setting_id == "S1" else "question_context"
                        ),
                        "generation_parameters": {"max_new_tokens": 128},
                        "raw_output": "Final answer: yes\nExplanation: Complete answer.",
                        "generated_tokens": 12,
                    }
                    for index in range(5)
                ]
                (base / f"{run_id}.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )
                (base / f"{run_id}.manifest.json").write_text(
                    json.dumps(
                        {
                            "dataset": {"selected_samples": 5},
                            "generation": {"max_new_tokens": 128},
                            "prompt": {
                                "type": (
                                    "question_only"
                                    if setting_id == "S1"
                                    else "question_context"
                                )
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            summary = validate_smoke_suite("suite", base)

        self.assertTrue(summary["ready_for_full_baseline"])
        self.assertEqual(len(summary["runs"]), 4)


if __name__ == "__main__":
    unittest.main()
