import json
import tempfile
import unittest
from pathlib import Path

from scripts.make_final_visuals import (
    DEFAULT_ERROR_REPORT_PATH,
    DEFAULT_EVAL_PATHS,
    EXPECTED_FIGURES,
    EXPECTED_TABLES,
    normalize_aggregate_rows,
    resolve_error_report_path,
    run_final_visualizations,
)
from src.schemas import ASPECTS


def scores(value):
    return {aspect: value for aspect in ASPECTS}


def record(example_id, answer, value, split):
    return {
        "example_id": example_id,
        "answer": answer,
        "split": split,
        "final_scores": scores(value),
    }


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def toy_dataset(root):
    paths = {}
    rows = {
        "train": [record("train1", "I built an API.", 4, "train"), record("train2", "I watched.", 1, "train")],
        "dev": [record("dev1", "I fixed a bug.", 4, "dev")],
        "test": [record("test1", "I helped a feature.", 3, "test")],
        "ood": [record("ood1", "I planned an event.", 1, "ood")],
    }
    for split, split_rows in rows.items():
        path = root / f"{split}.jsonl"
        write_jsonl(path, split_rows)
        paths[split] = path
    return paths


def toy_reports(root):
    baseline = {
        "summary_rows": [
            {"model": "majority", "split": "dev", "mean_exact_accuracy": 0.3, "mean_low_mid_high_macro_f1": 0.2, "mean_mae": 1.0, "weak_aspect_f1": 0.4},
            {"model": "tfidf_logistic_regression", "split": "test", "mean_exact_accuracy": 0.6, "mean_low_mid_high_macro_f1": 0.5, "mean_mae": 0.4, "weak_aspect_f1": 0.7},
        ]
    }
    llm = {
        "summary_rows": [
            {"mode": "zero-shot", "split": "dev", "mean_exact_accuracy": 0.8, "mean_low_mid_high_macro_f1": 0.7, "mean_mae": 0.2, "weak_aspect_f1": 0.9},
            {"mode": "few-shot", "split": "ood", "mean_exact_accuracy": 0.5, "mean_low_mid_high_macro_f1": 0.3, "mean_mae": 0.6, "weak_aspect_f1": 0.8},
        ]
    }
    encoder = {
        "summary_rows": [
            {"split": "dev", "mean_exact_accuracy": 0.7, "mean_low_mid_high_macro_f1": 0.6, "mean_mae": 0.3, "weak_aspect_f1": 0.85},
            {"split": "test", "mean_exact_accuracy": 0.65, "mean_low_mid_high_macro_f1": 0.55, "mean_mae": 0.35, "weak_aspect_f1": 0.82},
            {"split": "ood", "mean_exact_accuracy": 0.25, "mean_low_mid_high_macro_f1": 0.25, "mean_mae": 0.75, "weak_aspect_f1": 0.80},
        ]
    }
    error = {
        "aggregate_comparison": [
            {"model": "encoder", "split": "dev", "mean_exact_accuracy": 0.7, "mean_low_mid_high_macro_f1": 0.6, "mean_mae": 0.3, "weak_aspect_f1": 0.85},
            {"model": "encoder", "split": "test", "mean_exact_accuracy": 0.65, "mean_low_mid_high_macro_f1": 0.55, "mean_mae": 0.35, "weak_aspect_f1": 0.82},
            {"model": "encoder", "split": "ood", "mean_exact_accuracy": 0.25, "mean_low_mid_high_macro_f1": 0.25, "mean_mae": 0.75, "weak_aspect_f1": 0.80},
        ],
        "ood_drop": {
            "encoder": {
                "mean_exact_accuracy_delta": -0.40,
                "mean_mae_delta": 0.40,
                "weak_aspect_f1_delta": -0.02,
                "per_aspect_mae_delta": {aspect: 0.1 for aspect in ASPECTS},
            }
        },
        "models": {
            "encoder": {
                "overall": {
                    "aspects": {
                        aspect: {
                            "mae": 0.2 + index / 10,
                            "mean_signed_error": -0.1 + index / 20,
                            "severe_error_rate": 0.01 * index,
                        }
                        for index, aspect in enumerate(ASPECTS)
                    }
                }
            }
        },
    }
    paths = {}
    for name, payload in {"baseline": baseline, "llm": llm, "encoder": encoder, "error_analysis_results": error}.items():
        path = root / f"{name}.json"
        write_json(path, payload)
        paths[name] = path
    return paths


class FinalVisualsTests(unittest.TestCase):
    def test_defaults_use_reviewed_eval_paths_and_requested_error_default(self):
        self.assertEqual(str(DEFAULT_ERROR_REPORT_PATH), "data/reports/error_analysis.json")
        for path in DEFAULT_EVAL_PATHS.values():
            self.assertIn("data/reviewed", str(path))
            self.assertNotIn("review_candidates", str(path))

    def test_error_report_fallback_resolves_results_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requested = root / "error_analysis.json"
            fallback = root / "error_analysis_results.json"
            write_json(fallback, {"ok": True})

            self.assertEqual(resolve_error_report_path(requested), fallback)

    def test_normalize_aggregate_rows_includes_all_model_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = toy_reports(Path(tmp))

            rows = normalize_aggregate_rows(
                baseline_report=json.loads(paths["baseline"].read_text()),
                llm_report=json.loads(paths["llm"].read_text()),
                encoder_report=json.loads(paths["encoder"].read_text()),
                error_report=json.loads(paths["error_analysis_results"].read_text()),
            )

            models = {row["model"] for row in rows}
            self.assertIn("majority", models)
            self.assertIn("tfidf_logistic_regression", models)
            self.assertIn("zero-shot", models)
            self.assertIn("few-shot", models)
            self.assertIn("encoder", models)

    def test_runner_generates_expected_figures_tables_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = toy_reports(root)
            dataset = toy_dataset(root)
            output_dir = root / "visuals"

            result = run_final_visualizations(
                baseline_report_path=reports["baseline"],
                llm_report_path=reports["llm"],
                encoder_report_path=reports["encoder"],
                error_report_path=root / "error_analysis.json",
                train_path=dataset["train"],
                dev_path=dataset["dev"],
                test_path=dataset["test"],
                ood_path=dataset["ood"],
                output_dir=output_dir,
                output_format="png",
                print_summary=False,
            )

            for filename in EXPECTED_FIGURES:
                self.assertTrue((output_dir / filename.format(format="png")).exists())
            for filename in EXPECTED_TABLES:
                self.assertTrue((output_dir / filename).exists())
            self.assertTrue((output_dir / "visuals_manifest.json").exists())
            self.assertTrue((output_dir / "visuals_summary.md").exists())
            self.assertEqual(len(result["figures"]), len(EXPECTED_FIGURES))


if __name__ == "__main__":
    unittest.main()
