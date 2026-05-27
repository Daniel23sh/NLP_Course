import tempfile
import unittest
from pathlib import Path

from src.io_utils import read_jsonl
from src.main_generate import run_pipeline


class OfficialPipelineTests(unittest.TestCase):
    def test_official_broad_generation_writes_direct_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "broad"

            result = run_pipeline(
                mode="mock",
                n=8,
                seed=42,
                root_dir=Path(tmp),
                dataset_version="official_v1",
                generation_kind="broad",
                output_dir=output_dir,
                target_size_min=1,
                target_size_max=20,
                manual_review_size=2,
            )

            self.assertEqual(result["dataset_version"], "official_v1")
            self.assertEqual(result["generation_kind"], "broad")
            self.assertEqual(result["output_dir"], str(output_dir))
            self.assertTrue((output_dir / "full_synthetic_all.jsonl").exists())
            self.assertTrue((output_dir / "full_synthetic_clean_accepted.jsonl").exists())
            self.assertTrue((output_dir / "raw" / "generated_answers.jsonl").exists())
            self.assertFalse((output_dir / "data" / "v4").exists())

            rows = read_jsonl(output_dir / "full_synthetic_all.jsonl")
            self.assertGreater(len(rows), 0)
            self.assertEqual(rows[0]["dataset_version"], "official_v1")
            self.assertEqual(rows[0]["metadata"]["generation_kind"], "broad")

    def test_official_weak_patch_uses_official_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "weak_patch"

            result = run_pipeline(
                mode="mock",
                n=0,
                seed=43,
                root_dir=Path(tmp),
                dataset_version="official_v1",
                generation_kind="weak_patch",
                targeted_patch=True,
                patch_target_n=1,
                patch_max_attempts_per_target=4,
                output_dir=output_dir,
                target_size_min=1,
                target_size_max=20,
                manual_review_size=2,
            )

            self.assertEqual(result["generation_kind"], "weak_patch")
            rows = read_jsonl(output_dir / "full_synthetic_all.jsonl")
            self.assertGreater(len(rows), 0)
            self.assertTrue(all(row["dataset_version"] == "official_v1" for row in rows))
            self.assertTrue(all(row["metadata"]["generation_kind"] == "weak_patch" for row in rows))

    def test_official_strong_patch_writes_clean_high_quality_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "strong_patch"

            result = run_pipeline(
                mode="mock",
                n=0,
                seed=44,
                root_dir=Path(tmp),
                dataset_version="official_v1",
                generation_kind="strong_patch",
                patch_target_n=1,
                patch_max_attempts_per_target=3,
                output_dir=output_dir,
                target_size_min=1,
                target_size_max=20,
                manual_review_size=2,
            )

            self.assertEqual(result["generation_kind"], "strong_patch")
            clean_rows = read_jsonl(output_dir / "full_synthetic_clean_accepted.jsonl")
            self.assertGreater(len(clean_rows), 0)
            self.assertTrue(all(row["metadata"]["generation_kind"] == "strong_patch" for row in clean_rows))
            self.assertTrue(any(max(row["final_scores"].values()) >= 4 for row in clean_rows))


if __name__ == "__main__":
    unittest.main()
