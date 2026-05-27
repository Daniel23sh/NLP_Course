import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


def load_build_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_official_dataset.py"
    spec = importlib.util.spec_from_file_location("build_official_dataset", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuildOfficialDatasetScriptTests(unittest.TestCase):
    def test_phase_outputs_use_required_raw_generated_structure(self):
        script = load_build_script()
        output_root = Path("/tmp/interview_answer_scoring_data")

        self.assertEqual(
            script._phase_clean_path(output_root, "broad"),
            output_root / "raw" / "generated" / "broad" / "full_synthetic_clean_accepted.jsonl",
        )

    def test_build_script_validates_required_raw_processed_report_paths(self):
        script = load_build_script()
        output_root = Path("/tmp/interview_answer_scoring_data")

        required = script._required_validation_files(output_root)

        self.assertIn(output_root / "raw" / "full_synthetic_all.jsonl", required)
        self.assertIn(output_root / "processed" / "train.jsonl", required)
        self.assertIn(output_root / "processed" / "not_selected.jsonl", required)
        self.assertIn(output_root / "reports" / "data_quality.json", required)
        self.assertNotIn(output_root / "final" / "full_synthetic_all.jsonl", required)

    def test_run_injects_default_openai_timeout_and_retries(self):
        script = load_build_script()
        captured = {}

        def fake_run(cmd, cwd, check, env):
            captured["env"] = env

        with patch("subprocess.run", side_effect=fake_run):
            with patch.dict(os.environ, {}, clear=True):
                script._run([sys.executable, "--version"])

        self.assertEqual(captured["env"]["OPENAI_TIMEOUT_SECONDS"], "180")
        self.assertEqual(captured["env"]["OPENAI_MAX_RETRIES"], "2")

    def test_run_preserves_explicit_openai_timeout_and_retries(self):
        script = load_build_script()
        captured = {}

        def fake_run(cmd, cwd, check, env):
            captured["env"] = env

        with patch("subprocess.run", side_effect=fake_run):
            with patch.dict(os.environ, {"OPENAI_TIMEOUT_SECONDS": "240", "OPENAI_MAX_RETRIES": "4"}, clear=True):
                script._run([sys.executable, "--version"])

        self.assertEqual(captured["env"]["OPENAI_TIMEOUT_SECONDS"], "240")
        self.assertEqual(captured["env"]["OPENAI_MAX_RETRIES"], "4")


if __name__ == "__main__":
    unittest.main()
