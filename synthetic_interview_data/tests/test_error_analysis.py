import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_error_analysis import DEFAULT_EVAL_PATHS, compute_example_error, run_error_analysis
from src.schemas import ASPECTS


def scores(value):
    return {aspect: value for aspect in ASPECTS}


def record(example_id, answer, value, split):
    return {
        "example_id": example_id,
        "answer": answer,
        "split": split,
        "project_domain": "backend API",
        "question_type": "debugging_story",
        "scenario_family": "toy_family",
        "profile": {"profile_id": "toy_profile"},
        "final_scores": scores(value),
    }


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def toy_files(root):
    paths = {}
    split_rows = {
        "train": [record("train1", "I debugged an API and wrote tests.", 5, "train")],
        "dev": [record("dev1", "I fixed a backend bug and added tests.", 4, "dev")],
        "test": [record("test1", "I helped with a project but impact was vague.", 3, "test")],
        "ood": [record("ood1", "I organized a non software school event.", 1, "ood")],
    }
    for split, rows in split_rows.items():
        path = root / f"{split}.jsonl"
        write_jsonl(path, rows)
        paths[split] = path
    return paths, split_rows


def prediction(example_id, split, true_value, pred_value):
    return {
        "example_id": example_id,
        "split": split,
        "true_final_scores": scores(true_value),
        "predicted_final_scores": scores(pred_value),
        "true_weak_aspects": [] if true_value >= 3 else list(ASPECTS),
        "predicted_weak_aspects": [] if pred_value >= 3 else list(ASPECTS),
        "true_strong_aspects": list(ASPECTS) if true_value >= 4 else [],
        "predicted_strong_aspects": list(ASPECTS) if pred_value >= 4 else [],
    }


def encoder_report():
    return {
        "encoder_baseline_results_version": "v1",
        "dataset_version": "official_v1",
        "model_name": "distilbert-base-uncased",
        "split_counts": {"train": 1, "dev": 1, "test": 1, "ood": 1},
        "summary_rows": [
            {
                "split": "dev",
                "mean_exact_accuracy": 1.0,
                "mean_macro_f1": 1.0,
                "mean_weighted_f1": 1.0,
                "mean_low_mid_high_macro_f1": 1.0,
                "mean_mae": 0.0,
                "weak_aspect_f1": 1.0,
            },
            {
                "split": "test",
                "mean_exact_accuracy": 0.0,
                "mean_macro_f1": 0.0,
                "mean_weighted_f1": 0.0,
                "mean_low_mid_high_macro_f1": 0.0,
                "mean_mae": 1.0,
                "weak_aspect_f1": 0.0,
            },
            {
                "split": "ood",
                "mean_exact_accuracy": 0.0,
                "mean_macro_f1": 0.0,
                "mean_weighted_f1": 0.0,
                "mean_low_mid_high_macro_f1": 0.0,
                "mean_mae": 2.0,
                "weak_aspect_f1": 0.0,
            },
        ],
        "splits": {
            "dev": {"predictions": [prediction("dev1", "dev", 4, 4)]},
            "test": {"predictions": [prediction("test1", "test", 3, 4)]},
            "ood": {"predictions": [prediction("ood1", "ood", 1, 3)]},
        },
    }


def llm_report():
    return {
        "llm_baseline_results_version": "v1",
        "dataset_version": "official_v1",
        "run_type": "real-api",
        "summary_rows": [
            {"mode": "zero-shot", "split": "dev", "mean_exact_accuracy": 1.0, "mean_low_mid_high_macro_f1": 1.0, "mean_mae": 0.0, "weak_aspect_f1": 1.0},
            {"mode": "zero-shot", "split": "test", "mean_exact_accuracy": 1.0, "mean_low_mid_high_macro_f1": 1.0, "mean_mae": 0.0, "weak_aspect_f1": 1.0},
            {"mode": "zero-shot", "split": "ood", "mean_exact_accuracy": 1.0, "mean_low_mid_high_macro_f1": 1.0, "mean_mae": 0.0, "weak_aspect_f1": 1.0},
        ],
        "results": {
            "zero-shot": {
                "splits": {
                    "dev": {"predictions": [{"example_id": "dev1", "parse_status": "ok", "final_scores": scores(4)}]},
                    "test": {"predictions": [{"example_id": "test1", "parse_status": "ok", "final_scores": scores(3)}]},
                    "ood": {"predictions": [{"example_id": "ood1", "parse_status": "ok", "final_scores": scores(1)}]},
                }
            }
        },
    }


def baseline_report():
    return {
        "baseline_results_version": "v1",
        "summary_rows": [
            {"model": "majority", "split": "dev", "mean_exact_accuracy": 0.5, "mean_low_mid_high_macro_f1": 0.4, "mean_mae": 0.8, "weak_aspect_f1": 0.3},
            {"model": "tfidf_logistic_regression", "split": "test", "mean_exact_accuracy": 0.6, "mean_low_mid_high_macro_f1": 0.5, "mean_mae": 0.7, "weak_aspect_f1": 0.4},
        ],
    }


class ErrorAnalysisTests(unittest.TestCase):
    def test_compute_example_error_tracks_signed_severe_and_weak_errors(self):
        row = record("ex1", "I fixed a bug.", 3, "dev")
        true_scores = {
            "technical_depth": 2,
            "personal_contribution": 3,
            "clarity": 4,
            "problem_solving": 5,
            "impact": 1,
            "role_relevance": 4,
        }
        pred_scores = {
            "technical_depth": 4,
            "personal_contribution": 2,
            "clarity": 4,
            "problem_solving": 3,
            "impact": 3,
            "role_relevance": 5,
        }
        row["final_scores"] = true_scores

        payload = compute_example_error(row, pred_scores, "encoder", "dev")

        self.assertEqual(payload["total_abs_error"], 8)
        self.assertEqual(payload["max_abs_error"], 2)
        self.assertEqual(payload["severe_error_count"], 3)
        self.assertEqual(payload["aspect_errors"]["technical_depth"]["signed_error"], 2)
        self.assertEqual(payload["false_weak_aspects"], ["personal_contribution"])
        self.assertEqual(payload["missed_weak_aspects"], ["technical_depth", "impact"])

    def test_runner_writes_reports_and_includes_encoder_llm_and_classical_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, _ = toy_files(root)
            encoder_path = root / "encoder.json"
            llm_path = root / "llm.json"
            baseline_path = root / "baseline.json"
            output_dir = root / "reports"
            write_json(encoder_path, encoder_report())
            write_json(llm_path, llm_report())
            write_json(baseline_path, baseline_report())

            result = run_error_analysis(
                encoder_report_path=encoder_path,
                llm_report_path=llm_path,
                baseline_report_path=baseline_path,
                train_path=paths["train"],
                dev_path=paths["dev"],
                test_path=paths["test"],
                ood_path=paths["ood"],
                output_dir=output_dir,
                top_n_examples=2,
                focus_model="encoder",
                print_summary=False,
            )

            self.assertTrue((output_dir / "error_analysis_results.json").exists())
            self.assertTrue((output_dir / "error_analysis_results.md").exists())
            self.assertIn("encoder", result["models"])
            self.assertIn("zero-shot", result["models"])
            self.assertIn("classical:majority", {row["model"] for row in result["aggregate_comparison"]})
            self.assertEqual(result["models"]["encoder"]["splits"]["ood"]["record_count"], 1)
            self.assertEqual(result["ood_drop"]["encoder"]["mean_exact_accuracy_delta"], -0.0)
            self.assertEqual(result["top_error_examples"][0]["example_id"], "ood1")

    def test_missing_llm_report_is_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, _ = toy_files(root)
            encoder_path = root / "encoder.json"
            baseline_path = root / "baseline.json"
            output_dir = root / "reports"
            write_json(encoder_path, encoder_report())
            write_json(baseline_path, baseline_report())

            result = run_error_analysis(
                encoder_report_path=encoder_path,
                llm_report_path=root / "missing_llm.json",
                baseline_report_path=baseline_path,
                train_path=paths["train"],
                dev_path=paths["dev"],
                test_path=paths["test"],
                ood_path=paths["ood"],
                output_dir=output_dir,
                print_summary=False,
            )

            self.assertNotIn("zero-shot", result["models"])
            self.assertEqual(result["model_availability"]["llm"]["status"], "missing")
            self.assertTrue(any("LLM" in note for note in result["notes"]))

    def test_runner_defaults_use_reviewed_evaluation_files(self):
        for path in DEFAULT_EVAL_PATHS.values():
            self.assertIn("data/reviewed", str(path))
            self.assertNotIn("review_candidates", str(path))


if __name__ == "__main__":
    unittest.main()
