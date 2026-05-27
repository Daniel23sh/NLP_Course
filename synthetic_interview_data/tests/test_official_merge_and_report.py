import tempfile
import unittest
from pathlib import Path

from src.io_utils import read_json, read_jsonl, write_jsonl
from src.merge_datasets import merge_dataset_files
from src.quality_report import write_official_quality_report
from src.schemas import ASPECTS, DATASET_VERSION_OFFICIAL
from src.validator import apply_validation, validate_record
from tests.test_v4_profile_success import make_v4_record


def official_record(example_id, family, domain="backend API", profile_id="strong_project_story"):
    scores = {aspect: 4 for aspect in ASPECTS}
    record = make_v4_record(
        scores,
        "I implemented a backend API fix, debugged the issue with logs, tested the change, delivered the feature to users, and reduced errors.",
        profile_id=profile_id,
        constraints={},
    )
    record.example_id = example_id
    record.dataset_version = DATASET_VERSION_OFFICIAL
    record.project_domain = domain
    record.question_type = "debugging_story"
    record.scenario_family = family
    record.profile.expected_score_constraints = {}
    record.metadata["generation_kind"] = "broad"
    record.validation = validate_record(record)
    return apply_validation(record, record.validation)


class OfficialMergeAndReportTests(unittest.TestCase):
    def test_merge_writes_final_files_and_prevents_train_test_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broad = [official_record(f"b_{index}", f"family_b_{index}") for index in range(6)]
            weak = [official_record(f"w_{index}", f"family_w_{index}", domain="general coursework", profile_id="weak_patch_profile") for index in range(4)]
            strong = [official_record(f"s_{index}", f"family_s_{index}", profile_id="strong_patch_profile") for index in range(4)]
            broad_path = root / "broad.jsonl"
            weak_path = root / "weak.jsonl"
            strong_path = root / "strong.jsonl"
            write_jsonl(broad_path, broad)
            write_jsonl(weak_path, weak)
            write_jsonl(strong_path, strong)

            result = merge_dataset_files(
                broad_path=broad_path,
                weak_path=weak_path,
                strong_path=strong_path,
                output_dir=root / "final",
                max_domain_share=1.0,
                max_profile_share=1.0,
                max_question_type_share=1.0,
                split_group_key="scenario_family",
                train_ratio=0.70,
                dev_ratio=0.10,
                test_ratio=0.10,
                ood_ratio=0.10,
                ood_domains={"general coursework"},
                seed=42,
            )

            final_dir = root / "final"
            self.assertTrue((final_dir / "train.jsonl").exists())
            self.assertTrue((final_dir / "dev_review_candidates.jsonl").exists())
            self.assertTrue((final_dir / "test_review_candidates.jsonl").exists())
            self.assertTrue((final_dir / "ood_test_review_candidates.jsonl").exists())
            self.assertTrue((final_dir / "full_synthetic_all.jsonl").exists())
            self.assertFalse(result["leakage"]["train_test_leakage"])
            self.assertFalse(result["leakage"]["ood_train_leakage"])

            train = read_jsonl(final_dir / "train.jsonl")
            self.assertGreater(len(train), 0)
            self.assertTrue(all(row["validation"]["final_status"] == "accepted" for row in train))

    def test_merge_can_write_required_raw_and_processed_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broad = [official_record(f"b_{index}", f"family_b_{index}") for index in range(6)]
            weak = [official_record(f"w_{index}", f"family_w_{index}", domain="general coursework") for index in range(4)]
            strong = [official_record(f"s_{index}", f"family_s_{index}") for index in range(4)]
            broad_path = root / "broad.jsonl"
            weak_path = root / "weak.jsonl"
            strong_path = root / "strong.jsonl"
            write_jsonl(broad_path, broad)
            write_jsonl(weak_path, weak)
            write_jsonl(strong_path, strong)

            result = merge_dataset_files(
                broad_path=broad_path,
                weak_path=weak_path,
                strong_path=strong_path,
                output_dir=root / "processed",
                raw_output_dir=root / "raw",
                max_domain_share=1.0,
                max_profile_share=1.0,
                max_question_type_share=1.0,
                split_group_key="scenario_family",
                train_ratio=0.70,
                dev_ratio=0.10,
                test_ratio=0.10,
                ood_ratio=0.10,
                ood_domains={"general coursework"},
                seed=42,
            )

            self.assertTrue((root / "raw" / "full_synthetic_all.jsonl").exists())
            self.assertTrue((root / "processed" / "train.jsonl").exists())
            self.assertTrue((root / "processed" / "not_selected.jsonl").exists())
            self.assertFalse((root / "processed" / "full_synthetic_all.jsonl").exists())
            self.assertEqual(result["paths"]["all"], str(root / "raw" / "full_synthetic_all.jsonl"))
            self.assertEqual(result["paths"]["not_selected"], str(root / "processed" / "not_selected.jsonl"))

    def test_official_quality_report_writes_readiness_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [official_record(f"q_{index}", f"family_q_{index}") for index in range(5)]
            input_path = root / "full_synthetic_all.jsonl"
            report_path = root / "data_quality.md"
            json_path = root / "data_quality.json"
            write_jsonl(input_path, records)

            payload = write_official_quality_report(
                input_path=input_path,
                output_md=report_path,
                output_json=json_path,
            )

            self.assertTrue(report_path.exists())
            self.assertTrue(json_path.exists())
            self.assertIn("Ready to start baseline training", report_path.read_text(encoding="utf-8"))
            self.assertIn("readiness", payload)
            self.assertIn("readiness", read_json(json_path))


if __name__ == "__main__":
    unittest.main()
