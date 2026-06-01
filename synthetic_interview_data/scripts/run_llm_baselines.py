from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_data import (  # noqa: E402
    DEFAULT_DEV_PATH,
    DEFAULT_OOD_PATH,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    load_experiment_splits,
)
from src.llm_baselines import DryRunLLMPredictor, LLMBaseline, OpenAILLMPredictor  # noqa: E402
from src.llm_client import load_default_env_files  # noqa: E402
from src.llm_prompting import PROMPT_VERSION  # noqa: E402
from src.metrics import compute_metrics, compute_weak_aspect_metrics  # noqa: E402
from src.schemas import ASPECTS  # noqa: E402


DEFAULT_EVAL_PATHS = {
    "dev": DEFAULT_DEV_PATH,
    "test": DEFAULT_TEST_PATH,
    "ood": DEFAULT_OOD_PATH,
}
VALID_MODES = ("zero-shot", "few-shot", "all")
VALID_SPLITS = ("dev", "test", "ood")
LLM_REPORT_FILENAMES = ("llm_baseline_results.json", "llm_baseline_results.md")
REAL_RUN_CONFIRMATION_THRESHOLD = 20


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


def _predictions_by_aspect(predictions: list[dict[str, Any]]) -> dict[str, list[int]]:
    return {aspect: [int(row["final_scores"][aspect]) for row in predictions] for aspect in ASPECTS}


def _evaluate_successful(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict | None:
    pairs = [(record, prediction) for record, prediction in zip(records, predictions) if prediction["parse_status"] == "ok"]
    if not pairs:
        return None
    success_records = [record for record, _ in pairs]
    success_predictions = [prediction for _, prediction in pairs]
    metrics = compute_metrics(_scores_by_aspect(success_records), _predictions_by_aspect(success_predictions))
    metrics["weak_aspects"] = compute_weak_aspect_metrics(
        [record["final_scores"] for record in success_records],
        [prediction["final_scores"] for prediction in success_predictions],
    )
    return metrics


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def _summary_row(mode: str, split: str, split_payload: dict[str, Any]) -> dict[str, Any]:
    metrics = split_payload["metrics"]
    if metrics is None:
        return {
            "mode": mode,
            "split": split,
            "coverage": _rounded(split_payload["coverage"]),
            "mean_exact_accuracy": None,
            "mean_low_mid_high_macro_f1": None,
            "mean_mae": None,
            "weak_aspect_f1": None,
        }
    return {
        "mode": mode,
        "split": split,
        "coverage": _rounded(split_payload["coverage"]),
        "mean_exact_accuracy": _rounded(metrics["summary"]["mean_exact_accuracy"]),
        "mean_low_mid_high_macro_f1": _rounded(metrics["summary"]["mean_low_mid_high_macro_f1"]),
        "mean_mae": _rounded(metrics["summary"]["mean_mae"]),
        "weak_aspect_f1": _rounded(metrics["weak_aspects"]["f1"]),
    }


def _load_classical_summary() -> list[dict[str, Any]]:
    path = PROJECT_ROOT / "data/reports/baseline_results.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("summary_rows", [])
    except Exception:
        return []


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _output_paths(output_dir: str | Path) -> tuple[Path, Path]:
    output_path = _resolve_path(output_dir)
    return output_path / LLM_REPORT_FILENAMES[0], output_path / LLM_REPORT_FILENAMES[1]


def _ensure_can_write_reports(output_dir: str | Path, force: bool) -> None:
    if force:
        return
    existing = [path for path in _output_paths(output_dir) if path.exists()]
    if existing:
        formatted = ", ".join(_display_path(path) for path in existing)
        raise RuntimeError(
            f"LLM baseline report already exists: {formatted}. "
            "Use --force to overwrite or choose a different --output-dir."
        )


def _selected_record_count(splits: dict[str, list[dict[str, Any]]], requested_splits: list[str], limit: int | None) -> int:
    total = 0
    for split in requested_splits:
        records = splits[split]
        total += len(records[:limit]) if limit is not None else len(records)
    return total


def _ensure_cost_confirmed(
    dry_run: bool,
    confirm_cost: bool,
    mode: str,
    splits: dict[str, list[dict[str, Any]]],
    requested_splits: list[str],
    limit: int | None,
) -> int:
    planned_calls = len(_mode_list(mode)) * _selected_record_count(splits, requested_splits, limit)
    if not dry_run and planned_calls > REAL_RUN_CONFIRMATION_THRESHOLD and not confirm_cost:
        raise RuntimeError(
            f"This real LLM baseline run would make {planned_calls} API calls. "
            "Use --confirm-cost to proceed, add --limit for a smaller smoke run, "
            "or use --dry-run."
        )
    return planned_calls


def _comparison_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in payload.get("classical_baseline_summary", []):
        rows.append(
            {
                "method": row["model"],
                "split": row["split"],
                "mean_exact_accuracy": row.get("mean_exact_accuracy"),
                "mean_low_mid_high_macro_f1": row.get("mean_low_mid_high_macro_f1"),
                "mean_mae": row.get("mean_mae"),
                "weak_aspect_f1": row.get("weak_aspect_f1"),
            }
        )
    for row in payload.get("summary_rows", []):
        rows.append(
            {
                "method": f"{row['mode']} LLM",
                "split": row["split"],
                "mean_exact_accuracy": row.get("mean_exact_accuracy"),
                "mean_low_mid_high_macro_f1": row.get("mean_low_mid_high_macro_f1"),
                "mean_mae": row.get("mean_mae"),
                "weak_aspect_f1": row.get("weak_aspect_f1"),
            }
        )
    return rows


def _rows_by_mode_and_split(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["mode"], row["split"]): row for row in payload.get("summary_rows", [])}


def _ood_observation_lines(payload: dict[str, Any]) -> list[str]:
    if payload["run_type"] == "dry-run":
        return ["Dry-run rows are pipeline checks, not real model behavior."]
    rows = _rows_by_mode_and_split(payload)
    lines = []
    zero = [rows.get(("zero-shot", split)) for split in VALID_SPLITS]
    few = [rows.get(("few-shot", split)) for split in VALID_SPLITS]
    if all(zero) and all(few):
        if all(z["mean_exact_accuracy"] >= f["mean_exact_accuracy"] for z, f in zip(zero, few)) and all(
            z["mean_mae"] <= f["mean_mae"] for z, f in zip(zero, few)
        ):
            lines.append("Zero-shot is stronger overall than few-shot on exact accuracy and MAE in this run.")
        if all(row["mean_low_mid_high_macro_f1"] <= rows[(row["mode"], "test")]["mean_low_mid_high_macro_f1"] for row in [zero[2], few[2]]):
            lines.append("OOD is the hardest split by low/mid/high macro-F1 for both LLM modes.")
        ood_weak = [zero[2]["weak_aspect_f1"], few[2]["weak_aspect_f1"]]
        lines.append(
            "Weak-aspect F1 remains comparatively strong on OOD "
            f"({min(ood_weak):.4f}-{max(ood_weak):.4f}) even while exact score metrics drop."
        )
    if not lines:
        lines.append("OOD rows should be compared against dev/test rows before drawing conclusions.")
    return lines


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    is_dry_run = payload["run_type"] == "dry-run"
    result_label = "dry-run/mock" if is_dry_run else "initial real API"
    lines = [
        "# LLM Baseline Results",
        "",
        f"These are {result_label} LLM baseline results on project-team reviewed evaluation files.",
        "",
        "## Purpose",
        "",
        "Evaluate zero-shot and few-shot LLM prompting infrastructure using the same metrics as the classical baselines.",
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
            "## Model And Prompt Configuration",
            "",
            f"- Run type: `{payload['run_type']}`",
            f"- Model: `{payload['model']}`",
            f"- Prompt version: `{payload['prompt_version']}`",
            f"- Modes: {', '.join(f'`{mode}`' for mode in payload['modes'])}",
            f"- Few-shot k: `{payload['few_shot_k']}`",
            "",
            "## Results Summary",
            "",
            "| Mode | Split | Coverage | Mean Exact Accuracy | Mean Low/Mid/High Macro-F1 | Mean MAE | Weak-Aspect F1 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["summary_rows"]:
        lines.append(
            f"| {row['mode']} | {row['split']} | {_format_metric(row['coverage'])} | "
            f"{_format_metric(row['mean_exact_accuracy'])} | {_format_metric(row['mean_low_mid_high_macro_f1'])} | "
            f"{_format_metric(row['mean_mae'])} | {_format_metric(row['weak_aspect_f1'])} |"
        )
    comparison_rows = _comparison_rows(payload)
    if comparison_rows:
        lines.extend(
            [
                "",
                "## Method Comparison",
                "",
                "| Method | Split | Mean Exact Accuracy | Mean Low/Mid/High Macro-F1 | Mean MAE | Weak-Aspect F1 |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in comparison_rows:
            lines.append(
                f"| {row['method']} | {row['split']} | {_format_metric(row['mean_exact_accuracy'])} | "
                f"{_format_metric(row['mean_low_mid_high_macro_f1'])} | {_format_metric(row['mean_mae'])} | "
                f"{_format_metric(row['weak_aspect_f1'])} |"
            )
    lines.extend(["", "## Prediction Coverage", ""])
    for mode, mode_payload in payload["results"].items():
        for split, split_payload in mode_payload["splits"].items():
            lines.append(
                f"- `{mode}` on `{split}`: {split_payload['successful_predictions']} / "
                f"{split_payload['attempted_predictions']} successful predictions; "
                f"{split_payload['failed_predictions']} parse or prediction failures."
            )
    if payload["classical_baseline_summary"]:
        lines.extend(["", "## Classical Baseline Reference", ""])
        lines.append("Existing majority and TF-IDF summary rows are included above and in the JSON report.")
    lines.extend(["", "## OOD Observations", ""])
    for observation in _ood_observation_lines(payload):
        lines.append(f"- {observation}")
    lines.extend(["", "## Limitations", ""])
    if is_dry_run:
        lines.append("- Dry-run results are not real model results.")
    else:
        lines.append("- These are initial baseline results, not final deployment performance.")
    lines.extend(
        [
            "- Real API runs may incur cost and require `OPENAI_API_KEY` plus a selected model.",
            "- Few-shot examples are selected deterministically from train only.",
            "- The evaluation labels are project-team reviewed, not external expert annotations.",
            "",
            "## Next Experiments",
            "",
            "- Perform error analysis by aspect, split, project domain, and answer length.",
            "- Compare LLM behavior against majority and TF-IDF baselines using representative mistakes.",
            "- Add and evaluate a supervised encoder baseline.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(payload: dict[str, Any]) -> None:
    print(f"LLM baseline run type: {payload['run_type']}")
    print("mode\tsplit\tcoverage\tmean_exact\tmean_lmh_f1\tmean_mae\tweak_f1")
    for row in payload["summary_rows"]:
        print(
            f"{row['mode']}\t{row['split']}\t{_format_metric(row['coverage'])}\t"
            f"{_format_metric(row['mean_exact_accuracy'])}\t{_format_metric(row['mean_low_mid_high_macro_f1'])}\t"
            f"{_format_metric(row['mean_mae'])}\t{_format_metric(row['weak_aspect_f1'])}"
        )


def _mode_list(mode: str) -> list[str]:
    if mode == "all":
        return ["zero-shot", "few-shot"]
    return [mode]


def _model_name(model: str | None, dry_run: bool) -> str:
    if model:
        return model
    if dry_run:
        return "dry-run-mock"
    env_model = os.environ.get("LLM_BASELINE_MODEL")
    if not env_model:
        raise RuntimeError("Set --model or LLM_BASELINE_MODEL for real LLM baseline runs.")
    return env_model


def _predictor(model_name: str, dry_run: bool):
    return DryRunLLMPredictor(model_name=model_name) if dry_run else OpenAILLMPredictor(model_name=model_name)


def run_llm_baseline_experiments(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    dev_path: str | Path = DEFAULT_DEV_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
    ood_path: str | Path = DEFAULT_OOD_PATH,
    mode: str = "zero-shot",
    splits: list[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    output_dir: str | Path = Path("data/reports"),
    model: str | None = None,
    few_shot_k: int = 3,
    seed: int = 42,
    force: bool = False,
    confirm_cost: bool = False,
    print_summary: bool = True,
) -> dict[str, Any]:
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}")
    requested_splits = splits or ["dev"]
    invalid_splits = [split for split in requested_splits if split not in VALID_SPLITS]
    if invalid_splits:
        raise ValueError(f"invalid splits: {', '.join(invalid_splits)}")
    _ensure_can_write_reports(output_dir, force=force)
    if not dry_run:
        load_default_env_files()
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for real LLM baseline runs. Use --dry-run for mock runs.")
    model_name = _model_name(model, dry_run=dry_run)
    resolved_paths = {
        "train": _resolve_path(train_path),
        "dev": _resolve_path(dev_path),
        "test": _resolve_path(test_path),
        "ood": _resolve_path(ood_path),
    }
    all_splits = load_experiment_splits(
        train_path=resolved_paths["train"],
        dev_path=resolved_paths["dev"],
        test_path=resolved_paths["test"],
        ood_path=resolved_paths["ood"],
    )
    planned_predictions = _ensure_cost_confirmed(
        dry_run=dry_run,
        confirm_cost=confirm_cost,
        mode=mode,
        splits=all_splits,
        requested_splits=requested_splits,
        limit=limit,
    )
    result: dict[str, Any] = {
        "llm_baseline_results_version": "v1",
        "dataset_version": "official_v1",
        "run_type": "dry-run" if dry_run else "real-api",
        "model": model_name,
        "prompt_version": PROMPT_VERSION,
        "modes": _mode_list(mode),
        "requested_splits": requested_splits,
        "few_shot_k": few_shot_k,
        "seed": seed,
        "planned_predictions": planned_predictions,
        "split_paths": {name: _display_path(path) for name, path in resolved_paths.items()},
        "split_counts": {name: len(records) for name, records in all_splits.items()},
        "classical_baseline_summary": _load_classical_summary(),
        "results": {},
        "summary_rows": [],
    }
    for current_mode in _mode_list(mode):
        baseline = LLMBaseline(
            mode=current_mode,
            predictor=_predictor(model_name, dry_run=dry_run),
            train_records=all_splits["train"],
            few_shot_k=few_shot_k,
            seed=seed,
        )
        result["results"][current_mode] = {"splits": {}}
        for split in requested_splits:
            records = all_splits[split][:limit] if limit is not None else list(all_splits[split])
            predictions = baseline.predict(records)
            successful = sum(1 for prediction in predictions if prediction["parse_status"] == "ok")
            failed = len(predictions) - successful
            split_payload = {
                "record_count": len(records),
                "attempted_predictions": len(predictions),
                "successful_predictions": successful,
                "failed_predictions": failed,
                "coverage": successful / len(predictions) if predictions else 0.0,
                "metrics": _evaluate_successful(records, predictions),
                "prediction_failures": [
                    {
                        "example_id": prediction.get("example_id"),
                        "error": prediction.get("error", "unknown error"),
                    }
                    for prediction in predictions
                    if prediction["parse_status"] != "ok"
                ],
                "predictions": predictions,
            }
            result["results"][current_mode]["splits"][split] = split_payload
            result["summary_rows"].append(_summary_row(current_mode, split, split_payload))
    json_report_path, markdown_report_path = _output_paths(output_dir)
    _write_json(json_report_path, result)
    _write_markdown(markdown_report_path, result)
    if print_summary:
        _print_summary(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-shot and few-shot LLM baselines for official_v1.")
    parser.add_argument("--mode", choices=VALID_MODES, default="zero-shot")
    parser.add_argument("--splits", nargs="+", choices=VALID_SPLITS, default=["dev"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="data/reports")
    parser.add_argument("--model")
    parser.add_argument("--few-shot-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite existing LLM baseline report files.")
    parser.add_argument(
        "--confirm-cost",
        action="store_true",
        help="Confirm intentional real API runs above the small smoke-test threshold.",
    )
    args = parser.parse_args()
    try:
        run_llm_baseline_experiments(
            mode=args.mode,
            splits=args.splits,
            limit=args.limit,
            dry_run=args.dry_run,
            output_dir=args.output_dir,
            model=args.model,
            few_shot_k=args.few_shot_k,
            seed=args.seed,
            force=args.force,
            confirm_cost=args.confirm_cost,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
