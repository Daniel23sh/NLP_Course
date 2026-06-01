import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_llm_baselines import DEFAULT_EVAL_PATHS, _write_markdown, run_llm_baseline_experiments
from src.llm_baselines import DryRunLLMPredictor, LLMBaseline, parse_llm_prediction
from src.llm_prompting import build_few_shot_prompt, build_zero_shot_prompt, select_few_shot_examples
from src.metrics import compute_metrics
from src.schemas import ASPECTS


def scores(value):
    return {aspect: value for aspect in ASPECTS}


def record(example_id, answer, value, split="train"):
    return {
        "example_id": example_id,
        "target_role": "Junior Software Developer",
        "question": "Tell me about a project you worked on.",
        "answer": answer,
        "split": split,
        "final_scores": scores(value),
    }


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def toy_split_files(root):
    paths = {}
    split_rows = {
        "train": [
            record("train1", "I debugged an API and wrote tests.", 5),
            record("train2", "I watched planning.", 1),
        ],
        "dev": [record("dev1", "I fixed a backend bug.", 4, split="dev")],
        "test": [record("test1", "I helped with a project.", 3, split="test")],
        "ood": [record("ood1", "I watched a presentation.", 1, split="ood")],
    }
    for name, rows in split_rows.items():
        path = root / f"{name}.jsonl"
        write_jsonl(path, rows)
        paths[name] = path
    return paths


def report_payload(run_type="real-api"):
    rows = [
        {
            "mode": "zero-shot",
            "split": "dev",
            "coverage": 1.0,
            "mean_exact_accuracy": 0.7708,
            "mean_low_mid_high_macro_f1": 0.7150,
            "mean_mae": 0.2474,
            "weak_aspect_f1": 0.9024,
        },
        {
            "mode": "zero-shot",
            "split": "test",
            "coverage": 1.0,
            "mean_exact_accuracy": 0.6698,
            "mean_low_mid_high_macro_f1": 0.5789,
            "mean_mae": 0.3318,
            "weak_aspect_f1": 0.8163,
        },
        {
            "mode": "zero-shot",
            "split": "ood",
            "coverage": 1.0,
            "mean_exact_accuracy": 0.6121,
            "mean_low_mid_high_macro_f1": 0.2792,
            "mean_mae": 0.4182,
            "weak_aspect_f1": 0.8753,
        },
        {
            "mode": "few-shot",
            "split": "dev",
            "coverage": 1.0,
            "mean_exact_accuracy": 0.7448,
            "mean_low_mid_high_macro_f1": 0.7193,
            "mean_mae": 0.2812,
            "weak_aspect_f1": 0.9412,
        },
        {
            "mode": "few-shot",
            "split": "test",
            "coverage": 1.0,
            "mean_exact_accuracy": 0.6352,
            "mean_low_mid_high_macro_f1": 0.5808,
            "mean_mae": 0.3664,
            "weak_aspect_f1": 0.8092,
        },
        {
            "mode": "few-shot",
            "split": "ood",
            "coverage": 1.0,
            "mean_exact_accuracy": 0.5848,
            "mean_low_mid_high_macro_f1": 0.2780,
            "mean_mae": 0.5121,
            "weak_aspect_f1": 0.8518,
        },
    ]
    return {
        "run_type": run_type,
        "model": "gpt-5.4-mini" if run_type == "real-api" else "dry-run-mock",
        "prompt_version": "llm_baseline_v1",
        "modes": ["zero-shot", "few-shot"],
        "few_shot_k": 3,
        "split_paths": {
            "train": "data/processed/train.jsonl",
            "dev": "data/reviewed/dev_project_team_reviewed.jsonl",
            "test": "data/reviewed/test_project_team_reviewed.jsonl",
            "ood": "data/reviewed/ood_project_team_reviewed.jsonl",
        },
        "split_counts": {"train": 265, "dev": 64, "test": 106, "ood": 55},
        "summary_rows": rows,
        "classical_baseline_summary": [
            {
                "model": "majority",
                "split": "dev",
                "mean_exact_accuracy": 0.3411,
                "mean_low_mid_high_macro_f1": 0.2372,
                "mean_mae": 0.9453,
                "weak_aspect_f1": 0.4667,
            },
            {
                "model": "tfidf_logistic_regression",
                "split": "dev",
                "mean_exact_accuracy": 0.7318,
                "mean_low_mid_high_macro_f1": 0.6036,
                "mean_mae": 0.2917,
                "weak_aspect_f1": 0.8121,
            },
        ],
        "results": {
            mode: {
                "splits": {
                    split: {
                        "successful_predictions": 1,
                        "attempted_predictions": 1,
                        "failed_predictions": 0,
                    }
                    for split in ["dev", "test", "ood"]
                }
            }
            for mode in ["zero-shot", "few-shot"]
        },
    }


class LLMBaselineTests(unittest.TestCase):
    def test_zero_shot_prompt_contains_aspects_and_strict_json_schema(self):
        prompt = build_zero_shot_prompt(record("ex1", "I debugged an API and wrote tests.", 4))

        for aspect in ASPECTS:
            self.assertIn(aspect, prompt)
        self.assertIn("Junior Software Developer", prompt)
        self.assertIn("Score only based on evidence", prompt)
        self.assertIn('"final_scores"', prompt)
        self.assertIn('"weak_aspects"', prompt)
        self.assertIn('"strong_aspects"', prompt)

    def test_few_shot_prompt_uses_train_examples_only(self):
        train_records = [
            record("train_weak", "I watched planning.", 1, split="train"),
            record("train_strong", "I implemented and tested a backend API.", 5, split="train"),
            record("train_mid", "I helped with a small feature.", 3, split="train"),
            record("test_leak", "This test example must not appear.", 4, split="test"),
            record("ood_leak", "This OOD example must not appear.", 2, split="ood"),
        ]

        examples = select_few_shot_examples(train_records, k=3, seed=42)
        prompt = build_few_shot_prompt(record("eval", "I fixed a bug.", 3, split="dev"), examples)

        self.assertTrue(examples)
        self.assertTrue(all(item["split"] == "train" for item in examples))
        self.assertIn("train_weak", prompt)
        self.assertIn("train_strong", prompt)
        self.assertNotIn("test_leak", prompt)
        self.assertNotIn("ood_leak", prompt)

    def test_few_shot_selection_excludes_records_without_train_split(self):
        missing_split = record("missing_split", "This unlabeled split record must not appear.", 1)
        missing_split.pop("split")
        train_records = [
            missing_split,
            record("train_weak", "I watched planning.", 1, split="train"),
            record("train_strong", "I implemented and tested a backend API.", 5, split="train"),
        ]

        examples = select_few_shot_examples(train_records, k=3, seed=42)
        prompt = build_few_shot_prompt(record("eval", "I fixed a bug.", 3, split="dev"), examples)

        self.assertTrue(all(item.get("split") == "train" for item in examples))
        self.assertNotIn("missing_split", prompt)

    def test_parser_accepts_json_and_recomputes_derived_aspects(self):
        raw = """
        Here is the score:
        ```json
        {
          "final_scores": {
            "technical_depth": 1,
            "personal_contribution": 3,
            "clarity": 4,
            "problem_solving": 5,
            "impact": 2,
            "role_relevance": 4
          },
          "weak_aspects": [],
          "strong_aspects": [],
          "rationale": {
            "technical_depth": "little technical detail",
            "personal_contribution": "some role evidence",
            "clarity": "clear answer",
            "problem_solving": "strong process",
            "impact": "weak result",
            "role_relevance": "relevant"
          }
        }
        ```
        """

        parsed = parse_llm_prediction(raw)

        self.assertEqual(parsed["parse_status"], "ok")
        self.assertEqual(parsed["weak_aspects"], ["technical_depth", "impact"])
        self.assertEqual(parsed["strong_aspects"], ["clarity", "problem_solving", "role_relevance"])

    def test_parser_rejects_missing_non_integer_and_out_of_range_scores(self):
        no_scores = {"rationale": {}}
        missing = {"final_scores": {aspect: 3 for aspect in ASPECTS if aspect != "impact"}}
        extra = {"final_scores": scores(3)}
        extra["final_scores"]["extra_aspect"] = 4
        non_integer = {"final_scores": scores(3)}
        non_integer["final_scores"]["impact"] = "3"
        out_of_range = {"final_scores": scores(3)}
        out_of_range["final_scores"]["impact"] = 6

        self.assertEqual(parse_llm_prediction(json.dumps(no_scores))["parse_status"], "failed")
        self.assertEqual(parse_llm_prediction(json.dumps(missing))["parse_status"], "failed")
        self.assertEqual(parse_llm_prediction(json.dumps(extra))["parse_status"], "failed")
        self.assertEqual(parse_llm_prediction(json.dumps(non_integer))["parse_status"], "failed")
        self.assertEqual(parse_llm_prediction(json.dumps(out_of_range))["parse_status"], "failed")

    def test_dry_run_predictor_works_without_api_key(self):
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            baseline = LLMBaseline(mode="zero-shot", predictor=DryRunLLMPredictor())
            predictions = baseline.predict([record("ex1", "I debugged an API.", 4)])
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key

        self.assertEqual(predictions[0]["parse_status"], "ok")
        self.assertEqual(set(predictions[0]["final_scores"]), set(ASPECTS))

    def test_zero_and_few_shot_dry_run_predictions_work_with_metrics(self):
        rows = [
            record("ex1", "I debugged an API.", 4),
            record("ex2", "I only watched planning.", 2),
        ]
        for mode in ["zero-shot", "few-shot"]:
            baseline = LLMBaseline(
                mode=mode,
                predictor=DryRunLLMPredictor(),
                train_records=rows,
                few_shot_k=2,
            )
            predictions = baseline.predict(rows)
            y_true = {aspect: [row["final_scores"][aspect] for row in rows] for aspect in ASPECTS}
            y_pred = {aspect: [row["final_scores"][aspect] for row in predictions] for aspect in ASPECTS}

            payload = compute_metrics(y_true, y_pred)

            self.assertIn("summary", payload)
            self.assertTrue(all(item["parse_status"] == "ok" for item in predictions))

    def test_runner_dry_run_writes_temporary_reports_without_api_key(self):
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                paths = toy_split_files(root)

                output_dir = root / "reports"
                result = run_llm_baseline_experiments(
                    train_path=paths["train"],
                    dev_path=paths["dev"],
                    test_path=paths["test"],
                    ood_path=paths["ood"],
                    mode="all",
                    splits=["dev"],
                    limit=1,
                    dry_run=True,
                    output_dir=output_dir,
                    print_summary=False,
                )

                self.assertTrue((output_dir / "llm_baseline_results.json").exists())
                self.assertTrue((output_dir / "llm_baseline_results.md").exists())
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key

        self.assertEqual(result["run_type"], "dry-run")
        self.assertIn("zero-shot", result["modes"])
        self.assertIn("few-shot", result["modes"])

    def test_runner_defaults_use_reviewed_evaluation_files(self):
        for path in DEFAULT_EVAL_PATHS.values():
            self.assertIn("data/reviewed", str(path))
            self.assertNotIn("review_candidates", str(path))

    def test_runner_refuses_to_overwrite_existing_llm_reports_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = toy_split_files(root)
            output_dir = root / "reports"
            output_dir.mkdir()
            json_report = output_dir / "llm_baseline_results.json"
            md_report = output_dir / "llm_baseline_results.md"
            json_report.write_text("keep json\n", encoding="utf-8")
            md_report.write_text("keep md\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "already exists"):
                run_llm_baseline_experiments(
                    train_path=paths["train"],
                    dev_path=paths["dev"],
                    test_path=paths["test"],
                    ood_path=paths["ood"],
                    mode="all",
                    splits=["dev"],
                    limit=1,
                    dry_run=True,
                    output_dir=output_dir,
                    print_summary=False,
                )

            self.assertEqual(json_report.read_text(encoding="utf-8"), "keep json\n")
            self.assertEqual(md_report.read_text(encoding="utf-8"), "keep md\n")

    def test_large_real_api_runs_require_cost_confirmation_before_predictor_creation(self):
        old_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                paths = toy_split_files(root)
                write_jsonl(paths["dev"], [record(f"dev{i}", "I fixed a backend bug.", 4, split="dev") for i in range(11)])
                with patch(
                    "scripts.run_llm_baselines.OpenAILLMPredictor",
                    side_effect=AssertionError("predictor should not be constructed before cost confirmation"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "confirm-cost"):
                        run_llm_baseline_experiments(
                            train_path=paths["train"],
                            dev_path=paths["dev"],
                            test_path=paths["test"],
                            ood_path=paths["ood"],
                            mode="all",
                            splits=["dev"],
                            limit=11,
                            dry_run=False,
                            output_dir=root / "reports",
                            model="gpt-5.4-mini",
                            print_summary=False,
                        )
        finally:
            if old_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_real_api_markdown_report_uses_real_wording_and_method_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm_baseline_results.md"
            _write_markdown(path, report_payload(run_type="real-api"))
            text = path.read_text(encoding="utf-8")

        self.assertIn("initial real API LLM baseline results", text)
        self.assertIn("## Method Comparison", text)
        self.assertIn("zero-shot LLM", text)
        self.assertIn("tfidf_logistic_regression", text)
        self.assertIn("OOD is the hardest split", text)
        self.assertNotIn("Dry-run rows are pipeline checks", text)
        self.assertNotIn("Run real zero-shot and few-shot LLM experiments intentionally", text)

    def test_dry_run_markdown_report_keeps_dry_run_caveat(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm_baseline_results.md"
            _write_markdown(path, report_payload(run_type="dry-run"))
            text = path.read_text(encoding="utf-8")

        self.assertIn("dry-run/mock", text)
        self.assertIn("Dry-run results are not real model results.", text)


if __name__ == "__main__":
    unittest.main()
