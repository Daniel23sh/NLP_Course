import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_encoder_baseline import (
    DEFAULT_EVAL_PATHS,
    _build_prediction_rows,
    class_to_score,
    run_encoder_baseline_experiment,
    score_to_class,
)
from src.schemas import ASPECTS


def scores(value):
    return {aspect: value for aspect in ASPECTS}


def record(example_id, answer, value, split="train"):
    return {
        "example_id": example_id,
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


class FakeEncoderBaseline:
    device = "cpu"

    def fit(self, train_records):
        self.train_count = len(train_records)
        return [{"epoch": 1, "train_loss": 0.0}]

    def predict(self, records):
        return [dict(row["final_scores"]) for row in records]


class EncoderBaselineTests(unittest.TestCase):
    def test_score_class_mapping_uses_zero_based_classes(self):
        self.assertEqual(score_to_class(1), 0)
        self.assertEqual(score_to_class(5), 4)
        self.assertEqual(class_to_score(0), 1)
        self.assertEqual(class_to_score(4), 5)

        with self.assertRaisesRegex(ValueError, "score"):
            score_to_class(0)
        with self.assertRaisesRegex(ValueError, "class"):
            class_to_score(5)

    def test_prediction_rows_recompute_derived_weak_and_strong_aspects(self):
        rows = [
            {
                "example_id": "ex1",
                "split": "dev",
                "answer": "I implemented an API but had no outcome.",
                "final_scores": {
                    "technical_depth": 4,
                    "personal_contribution": 3,
                    "clarity": 5,
                    "problem_solving": 2,
                    "impact": 1,
                    "role_relevance": 4,
                },
            }
        ]
        predictions = [
            {
                "technical_depth": 5,
                "personal_contribution": 2,
                "clarity": 3,
                "problem_solving": 2,
                "impact": 4,
                "role_relevance": 1,
            }
        ]

        payload = _build_prediction_rows(rows, predictions, "dev")

        self.assertEqual(payload[0]["example_id"], "ex1")
        self.assertEqual(payload[0]["split"], "dev")
        self.assertEqual(payload[0]["true_weak_aspects"], ["problem_solving", "impact"])
        self.assertEqual(payload[0]["predicted_weak_aspects"], ["personal_contribution", "problem_solving", "role_relevance"])
        self.assertEqual(payload[0]["true_strong_aspects"], ["technical_depth", "clarity", "role_relevance"])
        self.assertEqual(payload[0]["predicted_strong_aspects"], ["technical_depth", "impact"])

    def test_runner_writes_reports_with_injected_fake_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = toy_split_files(root)
            output_dir = root / "reports"

            result = run_encoder_baseline_experiment(
                train_path=paths["train"],
                dev_path=paths["dev"],
                test_path=paths["test"],
                ood_path=paths["ood"],
                output_dir=output_dir,
                epochs=1,
                batch_size=2,
                limit_train=1,
                limit_eval=1,
                encoder=FakeEncoderBaseline(),
                print_summary=False,
            )

            self.assertTrue((output_dir / "encoder_baseline_results.json").exists())
            self.assertTrue((output_dir / "encoder_baseline_results.md").exists())
            self.assertEqual(result["dataset_version"], "official_v1")
            self.assertEqual(result["split_counts"], {"train": 1, "dev": 1, "test": 1, "ood": 1})
            self.assertEqual(result["full_split_counts"], {"train": 2, "dev": 1, "test": 1, "ood": 1})
            self.assertEqual(result["training_history"], [{"epoch": 1, "train_loss": 0.0}])
            self.assertIn("predictions", result["splits"]["dev"])
            self.assertEqual(result["splits"]["dev"]["predictions"][0]["predicted_final_scores"], scores(4))

    def test_runner_defaults_use_reviewed_evaluation_files(self):
        for path in DEFAULT_EVAL_PATHS.values():
            self.assertIn("data/reviewed", str(path))
            self.assertNotIn("review_candidates", str(path))


if __name__ == "__main__":
    unittest.main()
