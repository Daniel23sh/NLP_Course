from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
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
from src.metrics import compute_weak_aspect_metrics  # noqa: E402
from src.schemas import ASPECTS, DATASET_VERSION_OFFICIAL, compute_weak_aspects  # noqa: E402


DEFAULT_ENCODER_REPORT_PATH = Path("data/reports/encoder_baseline_results.json")
DEFAULT_LLM_REPORT_PATH = Path("data/reports/llm_baseline_results.json")
DEFAULT_BASELINE_REPORT_PATH = Path("data/reports/baseline_results.json")
DEFAULT_OUTPUT_DIR = Path("data/reports")
DEFAULT_TOP_N_EXAMPLES = 20
DEFAULT_FOCUS_MODEL = "encoder"
DEFAULT_EVAL_PATHS = {
    "dev": DEFAULT_DEV_PATH,
    "test": DEFAULT_TEST_PATH,
    "ood": DEFAULT_OOD_PATH,
}
VALID_SPLITS = ("dev", "test", "ood")
ERROR_REPORT_FILENAMES = ("error_analysis_results.json", "error_analysis_results.md")


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON report not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report must contain an object: {path}")
    return payload


def _maybe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json(path)


def _round(value: float) -> float:
    return round(float(value), 4)


def _answer_excerpt(answer: str, max_chars: int = 260) -> str:
    text = " ".join(str(answer).split())
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


def _answer_length_bucket(answer: str) -> str:
    word_count = len(str(answer).split())
    if word_count < 60:
        return "short_<60"
    if word_count <= 120:
        return "medium_60_120"
    return "long_>120"


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    profile = record.get("profile") if isinstance(record.get("profile"), dict) else {}
    return {
        "project_domain": record.get("project_domain", ""),
        "question_type": record.get("question_type", ""),
        "scenario_family": record.get("scenario_family", ""),
        "profile_id": profile.get("profile_id", ""),
        "answer_length_bucket": _answer_length_bucket(record.get("answer", "")),
    }


def compute_example_error(
    record: dict[str, Any],
    predicted_scores: dict[str, int],
    model_name: str,
    split_name: str,
) -> dict[str, Any]:
    true_scores = {aspect: int(record["final_scores"][aspect]) for aspect in ASPECTS}
    predicted = {aspect: int(predicted_scores[aspect]) for aspect in ASPECTS}
    aspect_errors = {}
    total_abs_error = 0
    max_abs_error = 0
    severe_error_count = 0
    for aspect in ASPECTS:
        signed_error = predicted[aspect] - true_scores[aspect]
        abs_error = abs(signed_error)
        total_abs_error += abs_error
        max_abs_error = max(max_abs_error, abs_error)
        if abs_error >= 2:
            severe_error_count += 1
        aspect_errors[aspect] = {
            "true": true_scores[aspect],
            "predicted": predicted[aspect],
            "signed_error": signed_error,
            "abs_error": abs_error,
            "is_exact": signed_error == 0,
            "is_severe": abs_error >= 2,
        }
    true_weak = set(compute_weak_aspects(true_scores))
    predicted_weak = set(compute_weak_aspects(predicted))
    return {
        "model": model_name,
        "split": split_name,
        "example_id": record.get("example_id"),
        "metadata": _metadata(record),
        "answer_excerpt": _answer_excerpt(record.get("answer", "")),
        "true_final_scores": true_scores,
        "predicted_final_scores": predicted,
        "aspect_errors": aspect_errors,
        "total_abs_error": total_abs_error,
        "mean_abs_error": total_abs_error / len(ASPECTS),
        "max_abs_error": max_abs_error,
        "severe_error_count": severe_error_count,
        "has_severe_error": severe_error_count > 0,
        "exact_all_aspects": total_abs_error == 0,
        "false_weak_aspects": [aspect for aspect in ASPECTS if aspect in predicted_weak and aspect not in true_weak],
        "missed_weak_aspects": [aspect for aspect in ASPECTS if aspect in true_weak and aspect not in predicted_weak],
    }


def _records_by_split(splits: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        split: {str(record["example_id"]): record for record in records}
        for split, records in splits.items()
        if split in VALID_SPLITS
    }


def _prediction_scores_by_split_from_encoder(report: dict[str, Any]) -> dict[str, dict[str, dict[str, int]]]:
    predictions: dict[str, dict[str, dict[str, int]]] = {}
    for split in VALID_SPLITS:
        split_payload = report.get("splits", {}).get(split, {})
        rows = split_payload.get("predictions", [])
        predictions[split] = {
            str(row["example_id"]): {aspect: int(row["predicted_final_scores"][aspect]) for aspect in ASPECTS}
            for row in rows
            if row.get("example_id") is not None and isinstance(row.get("predicted_final_scores"), dict)
        }
    return predictions


def _prediction_scores_by_split_from_llm(
    report: dict[str, Any], mode: str
) -> dict[str, dict[str, dict[str, int]]] | None:
    mode_payload = report.get("results", {}).get(mode)
    if not isinstance(mode_payload, dict):
        return None
    predictions: dict[str, dict[str, dict[str, int]]] = {}
    for split in VALID_SPLITS:
        split_payload = mode_payload.get("splits", {}).get(split, {})
        rows = split_payload.get("predictions", [])
        split_predictions = {}
        for row in rows:
            if row.get("parse_status") != "ok":
                continue
            scores = row.get("final_scores")
            if row.get("example_id") is not None and isinstance(scores, dict) and all(aspect in scores for aspect in ASPECTS):
                split_predictions[str(row["example_id"])] = {aspect: int(scores[aspect]) for aspect in ASPECTS}
        predictions[split] = split_predictions
    return predictions


def _aligned_errors(
    model_name: str,
    predictions: dict[str, dict[str, dict[str, int]]],
    records_by_split: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[str]]:
    notes = []
    errors = []
    for split in VALID_SPLITS:
        expected_ids = set(records_by_split[split])
        actual_ids = set(predictions.get(split, {}))
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        if missing or extra:
            notes.append(
                f"{model_name} predictions for {split} are not fully aligned "
                f"({len(missing)} missing, {len(extra)} extra). Detailed analysis skipped for this model."
            )
            return [], notes
        for example_id in sorted(expected_ids):
            errors.append(compute_example_error(records_by_split[split][example_id], predictions[split][example_id], model_name, split))
    return errors, notes


def _split_summary(errors: list[dict[str, Any]]) -> dict[str, Any]:
    if not errors:
        return {
            "record_count": 0,
            "mean_exact_accuracy": 0.0,
            "mean_mae": 0.0,
            "exact_all_aspects_rate": 0.0,
            "severe_error_rate": 0.0,
            "severe_example_rate": 0.0,
            "weak_aspects": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        }
    label_count = len(errors) * len(ASPECTS)
    exact_count = sum(1 for error in errors for aspect in ASPECTS if error["aspect_errors"][aspect]["is_exact"])
    severe_count = sum(error["severe_error_count"] for error in errors)
    true_scores = [error["true_final_scores"] for error in errors]
    pred_scores = [error["predicted_final_scores"] for error in errors]
    return {
        "record_count": len(errors),
        "mean_exact_accuracy": exact_count / label_count,
        "mean_mae": mean(error["mean_abs_error"] for error in errors),
        "exact_all_aspects_rate": sum(1 for error in errors if error["exact_all_aspects"]) / len(errors),
        "severe_error_rate": severe_count / label_count,
        "severe_example_rate": sum(1 for error in errors if error["has_severe_error"]) / len(errors),
        "weak_aspects": compute_weak_aspect_metrics(true_scores, pred_scores),
    }


def _aspect_summary(errors: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for aspect in ASPECTS:
        signed = [int(error["aspect_errors"][aspect]["signed_error"]) for error in errors]
        absolute = [abs(value) for value in signed]
        confusions = Counter(
            (
                int(error["aspect_errors"][aspect]["true"]),
                int(error["aspect_errors"][aspect]["predicted"]),
            )
            for error in errors
            if error["aspect_errors"][aspect]["signed_error"] != 0
        )
        result[aspect] = {
            "record_count": len(errors),
            "exact_accuracy": sum(1 for value in signed if value == 0) / len(signed) if signed else 0.0,
            "mae": mean(absolute) if absolute else 0.0,
            "mean_signed_error": mean(signed) if signed else 0.0,
            "overprediction_rate": sum(1 for value in signed if value > 0) / len(signed) if signed else 0.0,
            "underprediction_rate": sum(1 for value in signed if value < 0) / len(signed) if signed else 0.0,
            "severe_error_rate": sum(1 for value in absolute if value >= 2) / len(absolute) if absolute else 0.0,
            "common_confusions": [
                {"true": true, "predicted": predicted, "count": count}
                for (true, predicted), count in confusions.most_common(5)
            ],
        }
    return result


def _slice_summary(errors: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for error in errors:
        value = str(error["metadata"].get(field) or "unknown")
        grouped[value].append(error)
    rows = []
    for value, group in grouped.items():
        summary = _split_summary(group)
        rows.append(
            {
                "value": value,
                "record_count": summary["record_count"],
                "mean_mae": summary["mean_mae"],
                "mean_exact_accuracy": summary["mean_exact_accuracy"],
                "severe_error_rate": summary["severe_error_rate"],
            }
        )
    return sorted(rows, key=lambda row: (-row["mean_mae"], -row["record_count"], row["value"]))


def _model_analysis(model_name: str, errors: list[dict[str, Any]]) -> dict[str, Any]:
    split_payload = {}
    for split in VALID_SPLITS:
        split_errors = [error for error in errors if error["split"] == split]
        split_payload[split] = {
            **_split_summary(split_errors),
            "aspects": _aspect_summary(split_errors),
        }
    return {
        "record_count": len(errors),
        "splits": split_payload,
        "overall": {
            **_split_summary(errors),
            "aspects": _aspect_summary(errors),
        },
        "slice_analysis": {
            "project_domain": _slice_summary(errors, "project_domain"),
            "question_type": _slice_summary(errors, "question_type"),
            "scenario_family": _slice_summary(errors, "scenario_family"),
            "answer_length_bucket": _slice_summary(errors, "answer_length_bucket"),
        },
    }


def _aggregate_row(model: str, split: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "split": split,
        "mean_exact_accuracy": _round(metrics.get("mean_exact_accuracy", 0.0)),
        "mean_low_mid_high_macro_f1": _round(metrics.get("mean_low_mid_high_macro_f1", 0.0)),
        "mean_mae": _round(metrics.get("mean_mae", 0.0)),
        "weak_aspect_f1": _round(metrics.get("weak_aspect_f1", 0.0)),
    }


def _aggregate_from_model_analysis(model_name: str, model_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split in VALID_SPLITS:
        split_payload = model_payload["splits"][split]
        rows.append(
            _aggregate_row(
                model_name,
                split,
                {
                    "mean_exact_accuracy": split_payload["mean_exact_accuracy"],
                    "mean_low_mid_high_macro_f1": 0.0,
                    "mean_mae": split_payload["mean_mae"],
                    "weak_aspect_f1": split_payload["weak_aspects"]["f1"],
                },
            )
        )
    return rows


def _aggregate_from_encoder_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [_aggregate_row("encoder", row["split"], row) for row in report.get("summary_rows", [])]


def _aggregate_from_llm_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [_aggregate_row(row["mode"], row["split"], row) for row in report.get("summary_rows", [])]


def _aggregate_from_baseline_report(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not report:
        return []
    return [_aggregate_row(f"classical:{row['model']}", row["split"], row) for row in report.get("summary_rows", [])]


def _ood_drop(models: dict[str, Any], aggregate_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    rows_by_model_split = {(row["model"], row["split"]): row for row in aggregate_rows}
    result = {}
    for model in sorted({row["model"] for row in aggregate_rows} | set(models)):
        test_row = rows_by_model_split.get((model, "test"))
        ood_row = rows_by_model_split.get((model, "ood"))
        if not test_row or not ood_row:
            continue
        payload = {
            "mean_exact_accuracy_delta": _round(ood_row["mean_exact_accuracy"] - test_row["mean_exact_accuracy"]),
            "mean_mae_delta": _round(ood_row["mean_mae"] - test_row["mean_mae"]),
            "weak_aspect_f1_delta": _round(ood_row["weak_aspect_f1"] - test_row["weak_aspect_f1"]),
        }
        if model in models:
            payload["per_aspect_mae_delta"] = {
                aspect: _round(
                    models[model]["splits"]["ood"]["aspects"][aspect]["mae"]
                    - models[model]["splits"]["test"]["aspects"][aspect]["mae"]
                )
                for aspect in ASPECTS
            }
        result[model] = payload
    return result


def _top_examples(errors: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    sorted_errors = sorted(
        errors,
        key=lambda error: (-error["total_abs_error"], -error["max_abs_error"], error["split"], str(error["example_id"])),
    )
    return sorted_errors[:limit]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _format_metric(value: float | int) -> str:
    return f"{float(value):.4f}"


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Error Analysis Report",
        "",
        "## Purpose",
        "",
        "Analyze existing prediction artifacts for the junior interview answer scoring task, with the supervised encoder baseline as the primary focus.",
        "",
        "## Inputs And Model Availability",
        "",
        "| Input | Path |",
        "| --- | --- |",
    ]
    for name, path_value in payload["input_paths"].items():
        lines.append(f"| {name} | `{path_value}` |")
    lines.extend(["", "| Model Source | Status | Detail |", "| --- | --- | --- |"])
    for name, status in payload["model_availability"].items():
        lines.append(f"| {name} | `{status['status']}` | {status.get('detail', '')} |")
    lines.extend(
        [
            "",
            "## Aggregate Results Comparison",
            "",
            "| Model | Split | Mean Exact Accuracy | Mean Low/Mid/High Macro-F1 | Mean MAE | Weak-Aspect F1 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["aggregate_comparison"]:
        lines.append(
            f"| {row['model']} | {row['split']} | {_format_metric(row['mean_exact_accuracy'])} | "
            f"{_format_metric(row['mean_low_mid_high_macro_f1'])} | {_format_metric(row['mean_mae'])} | "
            f"{_format_metric(row['weak_aspect_f1'])} |"
        )
    encoder = payload["models"].get("encoder")
    if encoder:
        lines.extend(
            [
                "",
                "## Encoder Error Summary",
                "",
                "| Split | Records | Mean Exact Accuracy | Mean MAE | Exact-All Rate | Severe Error Rate | Weak F1 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for split in VALID_SPLITS:
            split_payload = encoder["splits"][split]
            lines.append(
                f"| {split} | {split_payload['record_count']} | {_format_metric(split_payload['mean_exact_accuracy'])} | "
                f"{_format_metric(split_payload['mean_mae'])} | {_format_metric(split_payload['exact_all_aspects_rate'])} | "
                f"{_format_metric(split_payload['severe_error_rate'])} | {_format_metric(split_payload['weak_aspects']['f1'])} |"
            )
    lines.extend(["", "## OOD Drop", ""])
    for model, drop in payload["ood_drop"].items():
        lines.append(
            f"- `{model}`: exact accuracy delta {drop['mean_exact_accuracy_delta']:.4f}; "
            f"MAE delta {drop['mean_mae_delta']:.4f}; weak-aspect F1 delta {drop['weak_aspect_f1_delta']:.4f}."
        )
    if encoder:
        lines.extend(["", "## Per-Aspect Failure Patterns", ""])
        for aspect, aspect_payload in encoder["overall"]["aspects"].items():
            confusions = ", ".join(
                f"{item['true']}->{item['predicted']} ({item['count']})"
                for item in aspect_payload["common_confusions"][:3]
            ) or "none"
            lines.append(
                f"- `{aspect}`: MAE {aspect_payload['mae']:.4f}, signed error "
                f"{aspect_payload['mean_signed_error']:.4f}, severe rate "
                f"{aspect_payload['severe_error_rate']:.4f}; common confusions: {confusions}."
            )
    lines.extend(["", "## Top Error Examples", ""])
    for index, example in enumerate(payload["top_error_examples"], 1):
        lines.append(
            f"{index}. `{example['example_id']}` ({example['split']}): total absolute error "
            f"{example['total_abs_error']}, max error {example['max_abs_error']}; "
            f"missed weak: {', '.join(example['missed_weak_aspects']) or 'none'}; "
            f"false weak: {', '.join(example['false_weak_aspects']) or 'none'}."
        )
    llm_models = [name for name in payload["models"] if name in {"zero-shot", "few-shot"}]
    lines.extend(["", "## Optional LLM Comparison Notes", ""])
    if llm_models:
        lines.append(f"Detailed aligned LLM analysis is available for: {', '.join(f'`{name}`' for name in llm_models)}.")
    else:
        lines.append("Detailed per-example LLM error analysis was not available from the provided report.")
    for note in payload["notes"]:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This report analyzes existing predictions only; it does not run new models or create new experiments.",
            "- Classical baselines are aggregate-only here because their report does not include per-example predictions.",
            "- Slice summaries are descriptive and may be noisy for small groups.",
            "- No charts are generated in this stage.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(payload: dict[str, Any]) -> None:
    print("Error analysis report")
    print("model\tsplit\tmean_exact\tmean_mae\tweak_f1")
    for row in payload["aggregate_comparison"]:
        print(
            f"{row['model']}\t{row['split']}\t{row['mean_exact_accuracy']:.4f}\t"
            f"{row['mean_mae']:.4f}\t{row['weak_aspect_f1']:.4f}"
        )


def run_error_analysis(
    encoder_report_path: str | Path = DEFAULT_ENCODER_REPORT_PATH,
    llm_report_path: str | Path = DEFAULT_LLM_REPORT_PATH,
    baseline_report_path: str | Path = DEFAULT_BASELINE_REPORT_PATH,
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    dev_path: str | Path = DEFAULT_DEV_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
    ood_path: str | Path = DEFAULT_OOD_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_n_examples: int = DEFAULT_TOP_N_EXAMPLES,
    focus_model: str = DEFAULT_FOCUS_MODEL,
    print_summary: bool = True,
) -> dict[str, Any]:
    resolved_paths = {
        "encoder_report": _resolve_path(encoder_report_path),
        "llm_report": _resolve_path(llm_report_path),
        "baseline_report": _resolve_path(baseline_report_path),
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
    records_by_split = _records_by_split(splits)
    encoder_report = _load_json(resolved_paths["encoder_report"])
    llm_report = _maybe_load_json(resolved_paths["llm_report"])
    baseline_report = _maybe_load_json(resolved_paths["baseline_report"])
    notes: list[str] = []
    model_availability: dict[str, dict[str, str]] = {
        "encoder": {"status": "available", "detail": "per-example predictions loaded"},
        "baseline": {
            "status": "available" if baseline_report else "missing",
            "detail": "aggregate rows only" if baseline_report else "baseline report file not found",
        },
    }
    models: dict[str, Any] = {}
    model_errors: dict[str, list[dict[str, Any]]] = {}
    encoder_errors, encoder_notes = _aligned_errors(
        "encoder", _prediction_scores_by_split_from_encoder(encoder_report), records_by_split
    )
    notes.extend(encoder_notes)
    if not encoder_errors:
        raise ValueError("encoder predictions must align with reviewed evaluation records")
    models["encoder"] = _model_analysis("encoder", encoder_errors)
    model_errors["encoder"] = encoder_errors
    aggregate_rows = _aggregate_from_baseline_report(baseline_report)
    aggregate_rows.extend(_aggregate_from_encoder_report(encoder_report) or _aggregate_from_model_analysis("encoder", models["encoder"]))
    if llm_report is None:
        model_availability["llm"] = {"status": "missing", "detail": "LLM report file not found"}
        notes.append("LLM report was not found; detailed per-example LLM error analysis was not available.")
    else:
        model_availability["llm"] = {"status": "available", "detail": "attempting per-example alignment"}
        aggregate_rows.extend(_aggregate_from_llm_report(llm_report))
        included_modes = []
        for mode in ["zero-shot", "few-shot"]:
            predictions = _prediction_scores_by_split_from_llm(llm_report, mode)
            if predictions is None:
                notes.append(f"LLM mode {mode} was not present in the report.")
                continue
            errors, mode_notes = _aligned_errors(mode, predictions, records_by_split)
            notes.extend(mode_notes)
            if errors:
                models[mode] = _model_analysis(mode, errors)
                model_errors[mode] = errors
                included_modes.append(mode)
        if included_modes:
            model_availability["llm"]["status"] = "aligned"
            model_availability["llm"]["detail"] = "included detailed modes: " + ", ".join(included_modes)
        else:
            model_availability["llm"]["status"] = "not_aligned"
            model_availability["llm"]["detail"] = "no LLM modes could be aligned by example_id"
            notes.append("Detailed per-example LLM error analysis was not available after alignment checks.")
    selected_focus = focus_model if focus_model in model_errors else "encoder"
    if selected_focus != focus_model:
        notes.append(f"Focus model {focus_model} was unavailable; using encoder for top error examples.")
    result = {
        "error_analysis_version": "v1",
        "dataset_version": encoder_report.get("dataset_version", DATASET_VERSION_OFFICIAL),
        "focus_model": selected_focus,
        "input_paths": {name: _display_path(path) for name, path in resolved_paths.items()},
        "split_counts": {name: len(records) for name, records in splits.items()},
        "model_availability": model_availability,
        "aggregate_comparison": sorted(aggregate_rows, key=lambda row: (row["model"], row["split"])),
        "models": models,
        "ood_drop": _ood_drop(models, aggregate_rows),
        "top_error_examples": _top_examples(model_errors[selected_focus], top_n_examples),
        "notes": notes,
    }
    output_path = _resolve_path(output_dir)
    json_path = output_path / ERROR_REPORT_FILENAMES[0]
    md_path = output_path / ERROR_REPORT_FILENAMES[1]
    _write_json(json_path, result)
    _write_markdown(md_path, result)
    if print_summary:
        _print_summary(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run error analysis over existing official_v1 baseline reports.")
    parser.add_argument("--encoder-report", default=str(DEFAULT_ENCODER_REPORT_PATH))
    parser.add_argument("--llm-report", default=str(DEFAULT_LLM_REPORT_PATH))
    parser.add_argument("--baseline-report", default=str(DEFAULT_BASELINE_REPORT_PATH))
    parser.add_argument("--train", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--dev", default=str(DEFAULT_DEV_PATH))
    parser.add_argument("--test", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--ood", default=str(DEFAULT_OOD_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-n-examples", type=_positive_int, default=DEFAULT_TOP_N_EXAMPLES)
    parser.add_argument("--focus-model", default=DEFAULT_FOCUS_MODEL)
    args = parser.parse_args()
    run_error_analysis(
        encoder_report_path=args.encoder_report,
        llm_report_path=args.llm_report,
        baseline_report_path=args.baseline_report,
        train_path=args.train,
        dev_path=args.dev,
        test_path=args.test,
        ood_path=args.ood,
        output_dir=args.output_dir,
        top_n_examples=args.top_n_examples,
        focus_model=args.focus_model,
    )


if __name__ == "__main__":
    main()
