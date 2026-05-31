import json
import tempfile
import unittest
from pathlib import Path

from src.baselines import MajorityScoreBaseline, TfidfLogisticRegressionBaseline
from src.experiment_data import extract_texts_and_labels, load_jsonl, validate_experiment_record
from src.metrics import compute_metrics, compute_weak_aspect_metrics, score_to_band
from src.schemas import ASPECTS
from scripts.run_baselines import run_baseline_experiments


def scores(value):
    return {aspect: value for aspect in ASPECTS}


def record(example_id, answer, value):
    return {
        "example_id": example_id,
        "answer": answer,
        "final_scores": scores(value),
    }


class BaselineInfrastructureTests(unittest.TestCase):
    def test_load_jsonl_reads_and_validates_tiny_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            rows = [
                record("ex1", "I debugged an API and wrote tests.", 4),
                record("ex2", "I only watched the team plan.", 2),
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            loaded = load_jsonl(path)
            for index, row in enumerate(loaded, 1):
                validate_experiment_record(row, path, index)
            texts, labels = extract_texts_and_labels(loaded)

            self.assertEqual(texts, ["I debugged an API and wrote tests.", "I only watched the team plan."])
            self.assertEqual(labels["technical_depth"], [4, 2])

    def test_load_jsonl_raises_for_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_jsonl(Path(tmp) / "missing.jsonl")

    def test_validate_record_rejects_missing_and_invalid_fields(self):
        source = Path("toy.jsonl")
        valid = record("ex1", "I built a backend API.", 3)

        missing_answer = dict(valid)
        missing_answer.pop("answer")
        with self.assertRaisesRegex(ValueError, "answer"):
            validate_experiment_record(missing_answer, source, 1)

        missing_scores = dict(valid)
        missing_scores.pop("final_scores")
        with self.assertRaisesRegex(ValueError, "final_scores"):
            validate_experiment_record(missing_scores, source, 2)

        missing_aspect = record("ex2", "I built a backend API.", 3)
        missing_aspect["final_scores"].pop("impact")
        with self.assertRaisesRegex(ValueError, "impact"):
            validate_experiment_record(missing_aspect, source, 3)

        invalid_score = record("ex3", "I built a backend API.", 3)
        invalid_score["final_scores"]["impact"] = 6
        with self.assertRaisesRegex(ValueError, "impact"):
            validate_experiment_record(invalid_score, source, 4)

    def test_score_to_band_maps_ordinal_scores(self):
        self.assertEqual(score_to_band(1), "low")
        self.assertEqual(score_to_band(2), "low")
        self.assertEqual(score_to_band(3), "mid")
        self.assertEqual(score_to_band(4), "high")
        self.assertEqual(score_to_band(5), "high")

    def test_metrics_are_json_serializable_and_include_required_keys(self):
        y_true = {aspect: [1, 3, 5] for aspect in ASPECTS}
        y_pred = {aspect: [1, 4, 4] for aspect in ASPECTS}

        payload = compute_metrics(y_true, y_pred)
        json.dumps(payload)

        self.assertIn("aspects", payload)
        for aspect in ASPECTS:
            self.assertIn("exact_accuracy", payload["aspects"][aspect])
            self.assertIn("macro_f1", payload["aspects"][aspect])
            self.assertIn("weighted_f1", payload["aspects"][aspect])
            self.assertIn("mae", payload["aspects"][aspect])
            self.assertIn("low_mid_high_macro_f1", payload["aspects"][aspect])
            self.assertIn("confusion_matrix", payload["aspects"][aspect])

    def test_weak_aspect_metrics_uses_scores_at_or_below_two(self):
        true_scores = [
            {"technical_depth": 1, "personal_contribution": 3, "clarity": 5, "problem_solving": 4, "impact": 2, "role_relevance": 4},
            {"technical_depth": 4, "personal_contribution": 4, "clarity": 4, "problem_solving": 4, "impact": 4, "role_relevance": 4},
        ]
        pred_scores = [
            {"technical_depth": 1, "personal_contribution": 2, "clarity": 5, "problem_solving": 4, "impact": 4, "role_relevance": 4},
            {"technical_depth": 4, "personal_contribution": 4, "clarity": 4, "problem_solving": 4, "impact": 4, "role_relevance": 4},
        ]

        payload = compute_weak_aspect_metrics(true_scores, pred_scores)

        self.assertAlmostEqual(payload["precision"], 0.5)
        self.assertAlmostEqual(payload["recall"], 0.5)
        self.assertAlmostEqual(payload["f1"], 0.5)

    def test_majority_baseline_fits_and_predicts_all_aspects_with_low_tie_break(self):
        train = [
            record("ex1", "weak answer", 2),
            record("ex2", "strong answer", 4),
        ]

        model = MajorityScoreBaseline().fit(train)
        predictions = model.predict([record("ex3", "new answer", 3)])

        self.assertEqual(predictions, [scores(2)])

    def test_tfidf_logistic_regression_baseline_fits_and_predicts(self):
        train = [
            record("ex1", "debugged api tests backend", 5),
            record("ex2", "fixed database query bug", 5),
            record("ex3", "watched planning meeting", 1),
            record("ex4", "only listened to discussion", 1),
        ]

        model = TfidfLogisticRegressionBaseline().fit(train)
        predictions = model.predict([record("ex5", "debugged backend bug", 3)])

        self.assertEqual(set(predictions[0]), set(ASPECTS))
        for value in predictions[0].values():
            self.assertIn(value, {1, 5})

    def test_runner_writes_reports_with_temporary_toy_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = [
                record("train1", "debugged api tests backend", 5),
                record("train2", "fixed database query bug", 5),
                record("train3", "watched planning meeting", 1),
                record("train4", "only listened to discussion", 1),
            ]
            dev = [record("dev1", "debugged backend bug", 5), record("dev2", "watched meeting", 1)]
            test = [record("test1", "fixed api bug", 5), record("test2", "listened to discussion", 1)]
            ood = [record("ood1", "database tests", 5), record("ood2", "planning only", 1)]
            paths = {}
            for name, rows in {"train": train, "dev": dev, "test": test, "ood": ood}.items():
                path = root / f"{name}.jsonl"
                path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
                paths[name] = path

            output_dir = root / "reports"
            result = run_baseline_experiments(
                train_path=paths["train"],
                dev_path=paths["dev"],
                test_path=paths["test"],
                ood_path=paths["ood"],
                output_dir=output_dir,
                print_summary=False,
            )

            self.assertTrue((output_dir / "baseline_results.json").exists())
            self.assertTrue((output_dir / "baseline_results.md").exists())
            self.assertIn("majority", result["baselines"])
            self.assertIn("tfidf_logistic_regression", result["baselines"])


if __name__ == "__main__":
    unittest.main()
