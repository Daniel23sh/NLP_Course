from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import MajorityScoreBaseline, TfidfLogisticRegressionBaseline  # noqa: E402
from src.experiment_data import (  # noqa: E402
    DEFAULT_DEV_PATH,
    DEFAULT_OOD_PATH,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    load_experiment_splits,
)
from src.metrics import compute_metrics, compute_weak_aspect_metrics  # noqa: E402
from src.schemas import ASPECTS  # noqa: E402


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _scores_by_aspect(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    return {aspect: [int(record["final_scores"][aspect]) for record in records] for aspect in ASPECTS}


def _predictions_by_aspect(predictions: list[dict[str, int]]) -> dict[str, list[int]]:
    return {aspect: [int(row[aspect]) for row in predictions] for aspect in ASPECTS}


def _evaluate(records: list[dict[str, Any]], predictions: list[dict[str, int]]) -> dict:
    true_scores = [record["final_scores"] for record in records]
    metric_payload = compute_metrics(_scores_by_aspect(records), _predictions_by_aspect(predictions))
    metric_payload["weak_aspects"] = compute_weak_aspect_metrics(true_scores, predictions)
    return metric_payload


def _rounded(value: float) -> float:
    return round(float(value), 4)


def _summary_row(model_name: str, split_name: str, metrics: dict) -> dict[str, float | str]:
    summary = metrics["summary"]
    weak = metrics["weak_aspects"]
    return {
        "model": model_name,
        "split": split_name,
        "mean_exact_accuracy": _rounded(summary["mean_exact_accuracy"]),
        "mean_macro_f1": _rounded(summary["mean_macro_f1"]),
        "mean_low_mid_high_macro_f1": _rounded(summary["mean_low_mid_high_macro_f1"]),
        "mean_mae": _rounded(summary["mean_mae"]),
        "weak_aspect_f1": _rounded(weak["f1"]),
    }


def _best_and_worst_aspects(metrics: dict) -> tuple[str, str]:
    values = {
        aspect: payload["exact_accuracy"]
        for aspect, payload in metrics["aspects"].items()
    }
    best = max(values.items(), key=lambda item: (item[1], item[0]))[0]
    worst = min(values.items(), key=lambda item: (item[1], item[0]))[0]
    return best, worst


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Baseline Experiment Results",
        "",
        "These are initial baseline results on project-team reviewed evaluation files.",
        "",
        "## Purpose",
        "",
        "Establish simple reference points before adding LLM or supervised encoder baselines.",
        "",
        "## Dataset Splits Used",
        "",
        "| Split | Path | Records |",
        "| --- | --- | ---: |",
    ]
    for split, path_value in payload["split_paths"].items():
        lines.append(f"| {split} | `{path_value}` | {payload['split_counts'][split]} |")
    lines.extend(
        [
            "",
            "Training uses synthetic accepted training data. Evaluation uses project-team reviewed evaluation files.",
            "",
            "## Baselines Implemented",
            "",
            "- `majority`: predicts the most common training score for each aspect.",
            "- `tfidf_logistic_regression`: answer-only TF-IDF features with one logistic regression classifier per aspect.",
            "",
            "## Metrics Used",
            "",
            "- Per-aspect exact accuracy, macro-F1, weighted-F1, MAE, and low/mid/high macro-F1.",
            "- Weak-aspect precision, recall, and F1 where weak means score `<= 2`.",
            "- Per-aspect confusion matrices are included in the JSON report.",
            "",
            "## Results Summary",
            "",
            "| Model | Split | Mean Exact Accuracy | Mean Macro-F1 | Mean Low/Mid/High Macro-F1 | Mean MAE | Weak-Aspect F1 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["summary_rows"]:
        lines.append(
            "| {model} | {split} | {mean_exact_accuracy:.4f} | {mean_macro_f1:.4f} | "
            "{mean_low_mid_high_macro_f1:.4f} | {mean_mae:.4f} | {weak_aspect_f1:.4f} |".format(**row)
        )
    lines.extend(["", "## Per-Aspect Observations", ""])
    for model_name, model_payload in payload["baselines"].items():
        for split_name, split_payload in model_payload["splits"].items():
            best, worst = _best_and_worst_aspects(split_payload["metrics"])
            lines.append(
                f"- `{model_name}` on `{split_name}`: highest exact accuracy aspect was `{best}`; "
                f"lowest exact accuracy aspect was `{worst}`."
            )
    lines.extend(["", "## OOD Observations", ""])
    for model_name, model_payload in payload["baselines"].items():
        test_exact = model_payload["splits"]["test"]["metrics"]["summary"]["mean_exact_accuracy"]
        ood_exact = model_payload["splits"]["ood"]["metrics"]["summary"]["mean_exact_accuracy"]
        delta = ood_exact - test_exact
        lines.append(
            f"- `{model_name}` OOD mean exact accuracy differs from reviewed test by {delta:.4f} "
            f"({ood_exact:.4f} OOD vs {test_exact:.4f} test)."
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- These are first-pass baselines, not tuned systems.",
            "- TF-IDF uses only the answer text and does not model ordinal distance directly.",
            "- The evaluation labels come from rubric-based project-team review of synthetic examples.",
            "",
            "## Next Planned Baselines",
            "",
            "- Add zero-shot LLM baseline using the same evaluation framework.",
            "- Add few-shot LLM baseline using the same evaluation framework.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(payload: dict) -> None:
    print("Initial baseline results on project-team reviewed evaluation files")
    print("model\tsplit\tmean_exact\tmean_lmh_f1\tmean_mae\tweak_f1")
    for row in payload["summary_rows"]:
        print(
            "{model}\t{split}\t{mean_exact_accuracy:.4f}\t{mean_low_mid_high_macro_f1:.4f}\t"
            "{mean_mae:.4f}\t{weak_aspect_f1:.4f}".format(**row)
        )


def run_baseline_experiments(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    dev_path: str | Path = DEFAULT_DEV_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
    ood_path: str | Path = DEFAULT_OOD_PATH,
    output_dir: str | Path = Path("data/reports"),
    print_summary: bool = True,
) -> dict:
    resolved_paths = {
        "train": _resolve_path(train_path),
        "dev": _resolve_path(dev_path),
        "test": _resolve_path(test_path),
        "ood": _resolve_path(ood_path),
    }
    splits = load_experiment_splits(
        train_path=resolved_paths["train"],
        dev_path=resolved_paths["dev"],
        test_path=resolved_paths["test"],
        ood_path=resolved_paths["ood"],
    )
    baselines = {
        "majority": {
            "description": "Predicts the most common training score for each aspect.",
            "model": MajorityScoreBaseline().fit(splits["train"]),
        },
        "tfidf_logistic_regression": {
            "description": "Answer-only TF-IDF with one logistic regression classifier per aspect.",
            "model": TfidfLogisticRegressionBaseline().fit(splits["train"]),
        },
    }
    result = {
        "baseline_results_version": "v1",
        "dataset_version": "official_v1",
        "training_data_description": "synthetic accepted training data",
        "evaluation_data_description": "project-team reviewed evaluation files",
        "split_paths": {name: _display_path(path) for name, path in resolved_paths.items()},
        "split_counts": {name: len(records) for name, records in splits.items()},
        "metrics_used": [
            "per_aspect_exact_accuracy",
            "per_aspect_macro_f1",
            "per_aspect_weighted_f1",
            "per_aspect_mae",
            "low_mid_high_macro_f1",
            "weak_aspect_precision_recall_f1",
            "confusion_matrix",
        ],
        "baselines": {},
        "summary_rows": [],
    }
    for model_name, model_payload in baselines.items():
        model = model_payload["model"]
        result["baselines"][model_name] = {
            "description": model_payload["description"],
            "splits": {},
        }
        for split_name in ["dev", "test", "ood"]:
            predictions = model.predict(splits[split_name])
            metrics = _evaluate(splits[split_name], predictions)
            result["baselines"][model_name]["splits"][split_name] = {
                "record_count": len(splits[split_name]),
                "metrics": metrics,
            }
            result["summary_rows"].append(_summary_row(model_name, split_name, metrics))
    output_path = _resolve_path(output_dir)
    _write_json(output_path / "baseline_results.json", result)
    _write_markdown(output_path / "baseline_results.md", result)
    if print_summary:
        _print_summary(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run initial baseline experiments for official_v1.")
    parser.add_argument("--train", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--dev", default=str(DEFAULT_DEV_PATH))
    parser.add_argument("--test", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--ood", default=str(DEFAULT_OOD_PATH))
    parser.add_argument("--output-dir", default="data/reports")
    args = parser.parse_args()
    run_baseline_experiments(
        train_path=args.train,
        dev_path=args.dev,
        test_path=args.test,
        ood_path=args.ood,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
