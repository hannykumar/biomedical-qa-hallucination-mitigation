"""Small interface checks for the local-to-Slurm automation script."""

from __future__ import annotations

import subprocess
import unittest


class ClusterAutomationTests(unittest.TestCase):
    def test_help_does_not_connect(self) -> None:
        result = subprocess.run(
            ["bash", "cluster/run_standard_pilot_remote.sh", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("USER@HOST REMOTE_PROJECT_DIR MODEL_KEY S1|S2 RUN_ID", result.stderr)


if __name__ == "__main__":
    unittest.main()
