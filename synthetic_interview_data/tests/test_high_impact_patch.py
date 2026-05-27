import tempfile
import unittest
from pathlib import Path

from src.generator import build_answer_prompt
from src.io_utils import read_jsonl, write_jsonl
from src.labeler import heuristic_label_answer
from src.main_generate import run_pipeline
from src.merge_datasets import merge_dataset_files
from src.quality_report import write_official_quality_report
from src.schemas import ASPECTS, DATASET_VERSION_OFFICIAL, GeneratedAnswer
from src.targeted_generation import official_high_impact_profile
from src.text_features import has_metric_outcome
from tests.test_official_merge_and_report import official_record


class HighImpactPatchTests(unittest.TestCase):
    def test_official_high_impact_profile_requires_only_impact(self):
        profile = official_high_impact_profile(seed=45)

        self.assertEqual(profile.desired_quality_hint, "high_impact")
        self.assertEqual(profile.expected_score_constraints, {"impact": {"min": 4}})
        self.assertNotIn("technical_depth", profile.expected_score_constraints)
        self.assertNotIn("clarity", profile.expected_score_constraints)

    def test_high_impact_prompt_asks_for_concrete_project_outcome_without_score_labels(self):
        profile = official_high_impact_profile(seed=46)
        prompt = build_answer_prompt(
            profile,
            {
                "question": "Tell me about a project where your work made a difference.",
                "project_domain": "automation script",
            },
        )
        lowered = prompt.lower()

        self.assertIn("concrete project-level result", lowered)
        self.assertIn("saved time", lowered)
        self.assertIn("reduced manual work", lowered)
        self.assertIn("before/after", lowered)
        self.assertIn("do not rely on vague personal learning", lowered)
        for aspect in ASPECTS:
            self.assertNotIn(aspect, lowered)
        self.assertNotIn("score", lowered)
        self.assertNotIn("rubric", lowered)

    def test_metric_outcome_detection_supports_before_after_evidence(self):
        self.assertTrue(has_metric_outcome("Validation accuracy increased from 62% to 76%."))
        self.assertTrue(has_metric_outcome("The script reduced the report from 30 minutes to 5 minutes."))
        self.assertTrue(has_metric_outcome("The QA checklist went from 18 errors to 3."))
        self.assertFalse(has_metric_outcome("I learned a lot and it was useful for me."))

    def test_heuristic_labeler_keeps_learning_only_below_high_impact(self):
        profile = official_high_impact_profile(seed=47)
        generated = GeneratedAnswer(
            example_id="learning_only",
            question_id="q_learning_only",
            question_type="project_overview",
            project_domain="automation script",
            question="Tell me about a project outcome.",
            profile=profile,
            answer="I learned a lot from the project and it helped me understand teamwork better.",
        )

        scoring = heuristic_label_answer(generated)

        self.assertLessEqual(scoring.scores["impact"], 3)

    def test_heuristic_labeler_allows_metric_answer_to_support_high_impact(self):
        profile = official_high_impact_profile(seed=48)
        generated = GeneratedAnswer(
            example_id="metric_outcome",
            question_id="q_metric_outcome",
            question_type="performance_story",
            project_domain="machine learning project",
            question="Tell me about a project outcome.",
            profile=profile,
            answer=(
                "I updated the preprocessing in a machine learning project and compared the validation runs. "
                "Validation accuracy increased from 62% to 76%, and the result was used in the final demo."
            ),
        )

        scoring = heuristic_label_answer(generated)

        self.assertGreaterEqual(scoring.scores["impact"], 4)

    def test_mock_high_impact_patch_writes_clean_generation_kind_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "high_impact_patch"

            result = run_pipeline(
                mode="mock",
                n=0,
                seed=45,
                root_dir=Path(tmp),
                dataset_version="official_v1",
                generation_kind="high_impact_patch",
                patch_target_n=1,
                patch_max_attempts_per_target=5,
                output_dir=output_dir,
                target_size_min=1,
                target_size_max=20,
                manual_review_size=2,
            )

            self.assertEqual(result["generation_kind"], "high_impact_patch")
            clean = read_jsonl(output_dir / "full_synthetic_clean_accepted.jsonl")
            self.assertGreater(len(clean), 0)
            self.assertTrue(all(row["metadata"]["generation_kind"] == "high_impact_patch" for row in clean))
            self.assertTrue(all(row["final_scores"]["impact"] >= 4 for row in clean))

    def test_merge_accepts_optional_high_impact_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broad = [official_record("b_1", "family_b_1")]
            weak = [official_record("w_1", "family_w_1", domain="general coursework")]
            strong = [official_record("s_1", "family_s_1")]
            high = [official_record("hi_1", "family_hi_1", profile_id="deployed_feature_with_usage")]
            high[0].metadata["generation_kind"] = "high_impact_patch"
            for name, rows in {"broad": broad, "weak": weak, "strong": strong, "high": high}.items():
                rows[0].answer = rows[0].answer + f" Unique merge marker {name}."
            paths = {}
            for name, rows in {"broad": broad, "weak": weak, "strong": strong, "high": high}.items():
                paths[name] = root / f"{name}.jsonl"
                write_jsonl(paths[name], rows)

            result = merge_dataset_files(
                broad_path=paths["broad"],
                weak_path=paths["weak"],
                strong_path=paths["strong"],
                high_impact_path=paths["high"],
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

            all_rows = read_jsonl(root / "final" / "full_synthetic_all.jsonl")
            self.assertEqual(result["input_records"], 4)
            self.assertTrue(any(row["metadata"].get("generation_kind") == "high_impact_patch" for row in all_rows))
            self.assertFalse(result["leakage"]["train_test_leakage"])
            self.assertFalse(result["leakage"]["ood_train_leakage"])

    def test_quality_report_shows_explicit_impact_high_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [official_record(f"q_{index}", f"family_q_{index}") for index in range(5)]
            input_path = root / "full_synthetic_all.jsonl"
            report_path = root / "data_quality.md"
            json_path = root / "data_quality.json"
            write_jsonl(input_path, records)

            write_official_quality_report(input_path, report_path, json_path)

            self.assertIn("Impact high coverage: 5 / 20", report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
