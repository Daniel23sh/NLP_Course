from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
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
from src.schemas import ASPECTS  # noqa: E402


DEFAULT_BASELINE_REPORT_PATH = Path("data/reports/baseline_results.json")
DEFAULT_LLM_REPORT_PATH = Path("data/reports/llm_baseline_results.json")
DEFAULT_ENCODER_REPORT_PATH = Path("data/reports/encoder_baseline_results.json")
DEFAULT_ERROR_REPORT_PATH = Path("data/reports/error_analysis.json")
ERROR_REPORT_FALLBACK_NAME = "error_analysis_results.json"
DEFAULT_OUTPUT_DIR = Path("data/visuals")
DEFAULT_FORMAT = "png"
DEFAULT_EVAL_PATHS = {
    "dev": DEFAULT_DEV_PATH,
    "test": DEFAULT_TEST_PATH,
    "ood": DEFAULT_OOD_PATH,
}
EXPECTED_FIGURES = [
    "model_comparison_mean_exact.{format}",
    "model_comparison_mae.{format}",
    "weak_aspect_f1_by_model.{format}",
    "ood_drop_by_model.{format}",
    "encoder_per_aspect_errors.{format}",
    "dataset_split_counts.{format}",
]
EXPECTED_TABLES = [
    "model_comparison_table.csv",
    "ood_drop_table.csv",
    "encoder_aspect_error_table.csv",
    "dataset_split_table.csv",
]
VALID_SPLITS = ("dev", "test", "ood")
MODEL_ORDER = [
    "majority",
    "tfidf_logistic_regression",
    "zero-shot",
    "few-shot",
    "encoder",
]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON report not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report must contain an object: {path}")
    return payload


def resolve_error_report_path(path: str | Path) -> Path:
    resolved = _resolve_path(path)
    if resolved.exists():
        return resolved
    if resolved.name == DEFAULT_ERROR_REPORT_PATH.name:
        fallback = resolved.with_name(ERROR_REPORT_FALLBACK_NAME)
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"JSON report not found: {resolved}")


def _require_plotting_dependencies():
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nlp_course_matplotlib_cache"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas, matplotlib, and seaborn are required for final visualizations. "
            "Install with: python3 -m pip install -r requirements.txt"
        ) from exc
    return pd, plt, sns


def _metric(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _model_sort_key(model: str) -> tuple[int, str]:
    try:
        return MODEL_ORDER.index(model), model
    except ValueError:
        return len(MODEL_ORDER), model


def _row(model: str, split: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "split": split,
        "mean_exact_accuracy": _metric(payload.get("mean_exact_accuracy")),
        "mean_low_mid_high_macro_f1": _metric(payload.get("mean_low_mid_high_macro_f1")),
        "mean_mae": _metric(payload.get("mean_mae")),
        "weak_aspect_f1": _metric(payload.get("weak_aspect_f1")),
    }


def normalize_aggregate_rows(
    baseline_report: dict[str, Any],
    llm_report: dict[str, Any],
    encoder_report: dict[str, Any],
    error_report: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for item in baseline_report.get("summary_rows", []):
        row = _row(str(item["model"]), str(item["split"]), item)
        rows_by_key[(row["model"], row["split"])] = row
    for item in llm_report.get("summary_rows", []):
        row = _row(str(item["mode"]), str(item["split"]), item)
        rows_by_key[(row["model"], row["split"])] = row
    for item in encoder_report.get("summary_rows", []):
        row = _row("encoder", str(item["split"]), item)
        rows_by_key[(row["model"], row["split"])] = row
    if error_report:
        for item in error_report.get("aggregate_comparison", []):
            model = str(item["model"]).replace("classical:", "")
            row = _row(model, str(item["split"]), item)
            rows_by_key[(row["model"], row["split"])] = row
    return sorted(rows_by_key.values(), key=lambda row: (_model_sort_key(row["model"]), row["split"]))


def _dataset_split_rows(splits: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {"split": split, "record_count": len(splits[split])}
        for split in ["train", "dev", "test", "ood"]
    ]


def _ood_drop_rows(error_report: dict[str, Any], aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    if isinstance(error_report.get("ood_drop"), dict) and error_report["ood_drop"]:
        for model, payload in error_report["ood_drop"].items():
            rows.append(
                {
                    "model": str(model).replace("classical:", ""),
                    "mean_exact_accuracy_delta": _metric(payload.get("mean_exact_accuracy_delta")),
                    "mean_mae_delta": _metric(payload.get("mean_mae_delta")),
                    "weak_aspect_f1_delta": _metric(payload.get("weak_aspect_f1_delta")),
                }
            )
        return sorted(rows, key=lambda row: _model_sort_key(row["model"]))
    by_key = {(row["model"], row["split"]): row for row in aggregate_rows}
    for model in sorted({row["model"] for row in aggregate_rows}, key=_model_sort_key):
        test = by_key.get((model, "test"))
        ood = by_key.get((model, "ood"))
        if not test or not ood:
            continue
        rows.append(
            {
                "model": model,
                "mean_exact_accuracy_delta": ood["mean_exact_accuracy"] - test["mean_exact_accuracy"],
                "mean_mae_delta": ood["mean_mae"] - test["mean_mae"],
                "weak_aspect_f1_delta": ood["weak_aspect_f1"] - test["weak_aspect_f1"],
            }
        )
    return rows


def _encoder_aspect_rows(error_report: dict[str, Any]) -> list[dict[str, Any]]:
    aspects = error_report.get("models", {}).get("encoder", {}).get("overall", {}).get("aspects", {})
    rows = []
    for aspect in ASPECTS:
        payload = aspects.get(aspect, {})
        rows.append(
            {
                "aspect": aspect,
                "mae": _metric(payload.get("mae")),
                "mean_signed_error": _metric(payload.get("mean_signed_error")),
                "severe_error_rate": _metric(payload.get("severe_error_rate")),
            }
        )
    return rows


def _save_figure(fig, path: Path, output_format: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, format=output_format, facecolor="white", bbox_inches="tight")


def _barplot_metric(pd, plt, sns, rows: list[dict[str, Any]], metric: str, title: str, ylabel: str, path: Path, output_format: str) -> None:
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    sns.barplot(data=frame, x="model", y=metric, hue="split", order=MODEL_ORDER, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Split")
    _save_figure(fig, path, output_format)
    plt.close(fig)


def _plot_ood_drop(pd, plt, sns, rows: list[dict[str, Any]], path: Path, output_format: str) -> None:
    frame = pd.DataFrame(rows)
    melted = frame.melt(
        id_vars=["model"],
        value_vars=["mean_exact_accuracy_delta", "mean_mae_delta", "weak_aspect_f1_delta"],
        var_name="metric",
        value_name="delta",
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    sns.barplot(data=melted, x="model", y="delta", hue="metric", ax=ax)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("OOD Delta Relative To Reviewed Test")
    ax.set_xlabel("Model")
    ax.set_ylabel("OOD minus test")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Metric")
    _save_figure(fig, path, output_format)
    plt.close(fig)


def _plot_encoder_aspects(pd, plt, sns, rows: list[dict[str, Any]], path: Path, output_format: str) -> None:
    frame = pd.DataFrame(rows)
    melted = frame.melt(
        id_vars=["aspect"],
        value_vars=["mae", "severe_error_rate"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    sns.barplot(data=melted, x="aspect", y="value", hue="metric", order=ASPECTS, ax=ax)
    ax.set_title("Encoder Per-Aspect Error Patterns")
    ax.set_xlabel("Rubric aspect")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Metric")
    _save_figure(fig, path, output_format)
    plt.close(fig)


def _plot_dataset_counts(pd, plt, sns, rows: list[dict[str, Any]], path: Path, output_format: str) -> None:
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    sns.barplot(data=frame, x="split", y="record_count", order=["train", "dev", "test", "ood"], ax=ax)
    ax.set_title("Official Dataset Split Counts")
    ax.set_xlabel("Split")
    ax.set_ylabel("Records")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, path, output_format)
    plt.close(fig)


def _write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Final Visuals Summary",
        "",
        "These files were generated from existing official_v1 reports and reviewed split files.",
        "",
        "## Figures",
        "",
    ]
    for figure in manifest["figures"]:
        lines.append(f"- `{figure['filename']}`: {figure['description']}")
    lines.extend(["", "## Tables", ""])
    for table in manifest["tables"]:
        lines.append(f"- `{table['filename']}`: {table['description']}")
    lines.extend(["", "## Inputs", ""])
    for name, value in manifest["input_paths"].items():
        lines.append(f"- `{name}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_final_visualizations(
    baseline_report_path: str | Path = DEFAULT_BASELINE_REPORT_PATH,
    llm_report_path: str | Path = DEFAULT_LLM_REPORT_PATH,
    encoder_report_path: str | Path = DEFAULT_ENCODER_REPORT_PATH,
    error_report_path: str | Path = DEFAULT_ERROR_REPORT_PATH,
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    dev_path: str | Path = DEFAULT_DEV_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
    ood_path: str | Path = DEFAULT_OOD_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_format: str = DEFAULT_FORMAT,
    print_summary: bool = True,
) -> dict[str, Any]:
    pd, plt, sns = _require_plotting_dependencies()
    resolved_paths = {
        "baseline_report": _resolve_path(baseline_report_path),
        "llm_report": _resolve_path(llm_report_path),
        "encoder_report": _resolve_path(encoder_report_path),
        "error_report": resolve_error_report_path(error_report_path),
        "train": _resolve_path(train_path),
        "dev": _resolve_path(dev_path),
        "test": _resolve_path(test_path),
        "ood": _resolve_path(ood_path),
    }
    baseline_report = _load_json(resolved_paths["baseline_report"])
    llm_report = _load_json(resolved_paths["llm_report"])
    encoder_report = _load_json(resolved_paths["encoder_report"])
    error_report = _load_json(resolved_paths["error_report"])
    splits = load_experiment_splits(
        train_path=resolved_paths["train"],
        dev_path=resolved_paths["dev"],
        test_path=resolved_paths["test"],
        ood_path=resolved_paths["ood"],
    )
    output_path = _resolve_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    aggregate_rows = normalize_aggregate_rows(baseline_report, llm_report, encoder_report, error_report)
    ood_rows = _ood_drop_rows(error_report, aggregate_rows)
    aspect_rows = _encoder_aspect_rows(error_report)
    split_rows = _dataset_split_rows(splits)
    tables = [
        ("model_comparison_table.csv", aggregate_rows, "Normalized model metric comparison."),
        ("ood_drop_table.csv", ood_rows, "OOD minus reviewed-test metric deltas."),
        ("encoder_aspect_error_table.csv", aspect_rows, "Encoder per-aspect MAE and severe-error rates."),
        ("dataset_split_table.csv", split_rows, "Official train/dev/test/OOD split counts."),
    ]
    for filename, rows, _description in tables:
        pd.DataFrame(rows).to_csv(output_path / filename, index=False)
    figure_paths = {
        "model_comparison_mean_exact.{format}": "Mean exact accuracy by model and split.",
        "model_comparison_mae.{format}": "Mean absolute error by model and split.",
        "weak_aspect_f1_by_model.{format}": "Weak-aspect F1 by model and split.",
        "ood_drop_by_model.{format}": "OOD deltas relative to reviewed test.",
        "encoder_per_aspect_errors.{format}": "Encoder per-aspect error patterns.",
        "dataset_split_counts.{format}": "Official dataset split counts.",
    }
    _barplot_metric(
        pd,
        plt,
        sns,
        aggregate_rows,
        "mean_exact_accuracy",
        "Mean Exact Accuracy By Model And Split",
        "Mean exact accuracy",
        output_path / f"model_comparison_mean_exact.{output_format}",
        output_format,
    )
    _barplot_metric(
        pd,
        plt,
        sns,
        aggregate_rows,
        "mean_mae",
        "Mean Absolute Error By Model And Split",
        "Mean absolute error",
        output_path / f"model_comparison_mae.{output_format}",
        output_format,
    )
    _barplot_metric(
        pd,
        plt,
        sns,
        aggregate_rows,
        "weak_aspect_f1",
        "Weak-Aspect F1 By Model And Split",
        "Weak-aspect F1",
        output_path / f"weak_aspect_f1_by_model.{output_format}",
        output_format,
    )
    _plot_ood_drop(pd, plt, sns, ood_rows, output_path / f"ood_drop_by_model.{output_format}", output_format)
    _plot_encoder_aspects(
        pd,
        plt,
        sns,
        aspect_rows,
        output_path / f"encoder_per_aspect_errors.{output_format}",
        output_format,
    )
    _plot_dataset_counts(
        pd,
        plt,
        sns,
        split_rows,
        output_path / f"dataset_split_counts.{output_format}",
        output_format,
    )
    manifest = {
        "visuals_version": "v1",
        "format": output_format,
        "input_paths": {name: _display_path(path) for name, path in resolved_paths.items()},
        "figures": [
            {"filename": template.format(format=output_format), "description": description}
            for template, description in figure_paths.items()
        ],
        "tables": [
            {"filename": filename, "description": description}
            for filename, _rows, description in tables
        ],
    }
    (output_path / "visuals_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_summary(output_path / "visuals_summary.md", manifest)
    if print_summary:
        print(f"Final visuals written to {_display_path(output_path)}")
        for figure in manifest["figures"]:
            print(f"figure\t{figure['filename']}")
        for table in manifest["tables"]:
            print(f"table\t{table['filename']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final course-ready figures from existing reports.")
    parser.add_argument("--baseline-report", default=str(DEFAULT_BASELINE_REPORT_PATH))
    parser.add_argument("--llm-report", default=str(DEFAULT_LLM_REPORT_PATH))
    parser.add_argument("--encoder-report", default=str(DEFAULT_ENCODER_REPORT_PATH))
    parser.add_argument("--error-report", default=str(DEFAULT_ERROR_REPORT_PATH))
    parser.add_argument("--train", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--dev", default=str(DEFAULT_DEV_PATH))
    parser.add_argument("--test", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--ood", default=str(DEFAULT_OOD_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--format", default=DEFAULT_FORMAT)
    args = parser.parse_args()
    try:
        run_final_visualizations(
            baseline_report_path=args.baseline_report,
            llm_report_path=args.llm_report,
            encoder_report_path=args.encoder_report,
            error_report_path=args.error_report,
            train_path=args.train,
            dev_path=args.dev,
            test_path=args.test,
            ood_path=args.ood,
            output_dir=args.output_dir,
            output_format=args.format,
        )
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
