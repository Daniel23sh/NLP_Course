from __future__ import annotations

import argparse
import json
import random
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
from src.metrics import compute_metrics, compute_weak_aspect_metrics  # noqa: E402
from src.schemas import (  # noqa: E402
    ASPECTS,
    DATASET_VERSION_OFFICIAL,
    SCORE_MAX,
    SCORE_MIN,
    compute_strong_aspects,
    compute_weak_aspects,
)


DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_MAX_LENGTH = 256
DEFAULT_SEED = 42
ENCODER_REPORT_FILENAMES = ("encoder_baseline_results.json", "encoder_baseline_results.md")
DEFAULT_EVAL_PATHS = {
    "dev": DEFAULT_DEV_PATH,
    "test": DEFAULT_TEST_PATH,
    "ood": DEFAULT_OOD_PATH,
}
SCORE_CLASS_COUNT = SCORE_MAX - SCORE_MIN + 1


def score_to_class(score: int) -> int:
    if type(score) is not int or not SCORE_MIN <= score <= SCORE_MAX:
        raise ValueError(f"score must be an integer from {SCORE_MIN} to {SCORE_MAX}")
    return score - SCORE_MIN


def class_to_score(label: int) -> int:
    if type(label) is not int or not 0 <= label < SCORE_CLASS_COUNT:
        raise ValueError(f"class label must be an integer from 0 to {SCORE_CLASS_COUNT - 1}")
    return label + SCORE_MIN


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


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


def _evaluate(records: list[dict[str, Any]], predictions: list[dict[str, int]]) -> dict[str, Any]:
    true_scores = [record["final_scores"] for record in records]
    metric_payload = compute_metrics(_scores_by_aspect(records), _predictions_by_aspect(predictions))
    metric_payload["weak_aspects"] = compute_weak_aspect_metrics(true_scores, predictions)
    return metric_payload


def _build_prediction_rows(
    records: list[dict[str, Any]], predictions: list[dict[str, int]], split_name: str
) -> list[dict[str, Any]]:
    if len(records) != len(predictions):
        raise ValueError("records and predictions lengths differ")
    rows = []
    for record, prediction in zip(records, predictions):
        true_scores = {aspect: int(record["final_scores"][aspect]) for aspect in ASPECTS}
        predicted_scores = {aspect: int(prediction[aspect]) for aspect in ASPECTS}
        rows.append(
            {
                "example_id": record.get("example_id"),
                "split": split_name,
                "true_final_scores": true_scores,
                "predicted_final_scores": predicted_scores,
                "true_weak_aspects": compute_weak_aspects(true_scores),
                "predicted_weak_aspects": compute_weak_aspects(predicted_scores),
                "true_strong_aspects": compute_strong_aspects(true_scores),
                "predicted_strong_aspects": compute_strong_aspects(predicted_scores),
            }
        )
    return rows


def _rounded(value: float) -> float:
    return round(float(value), 4)


def _summary_row(split_name: str, metrics: dict[str, Any]) -> dict[str, float | str]:
    summary = metrics["summary"]
    weak = metrics["weak_aspects"]
    return {
        "split": split_name,
        "mean_exact_accuracy": _rounded(summary["mean_exact_accuracy"]),
        "mean_macro_f1": _rounded(summary["mean_macro_f1"]),
        "mean_weighted_f1": _rounded(summary["mean_weighted_f1"]),
        "mean_low_mid_high_macro_f1": _rounded(summary["mean_low_mid_high_macro_f1"]),
        "mean_mae": _rounded(summary["mean_mae"]),
        "weak_aspect_precision": _rounded(weak["precision"]),
        "weak_aspect_recall": _rounded(weak["recall"]),
        "weak_aspect_f1": _rounded(weak["f1"]),
    }


def _best_and_worst_aspects(metrics: dict[str, Any]) -> tuple[str, str]:
    values = {aspect: payload["exact_accuracy"] for aspect, payload in metrics["aspects"].items()}
    best = max(values.items(), key=lambda item: (item[1], item[0]))[0]
    worst = min(values.items(), key=lambda item: (item[1], item[0]))[0]
    return best, worst


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _format_metric(value: float) -> str:
    return f"{float(value):.4f}"


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Supervised Encoder Baseline Results",
        "",
        "These results evaluate a lightweight supervised encoder baseline on the official project splits.",
        "",
        "## Purpose",
        "",
        "Train an answer-only neural baseline for six ordinal rubric scores and compare behavior across dev, test, and OOD splits.",
        "",
        "## Dataset Splits Used",
        "",
        "| Split | Path | Records Used | Full Records |",
        "| --- | --- | ---: | ---: |",
    ]
    for split_name, path_value in payload["split_paths"].items():
        lines.append(
            f"| {split_name} | `{path_value}` | {payload['split_counts'][split_name]} | "
            f"{payload['full_split_counts'][split_name]} |"
        )
    training = payload["training_parameters"]
    lines.extend(
        [
            "",
            "## Model Configuration",
            "",
            f"- Model: `{payload['model_name']}`",
            "- Formulation: one shared encoder with one five-class classification head per rubric aspect.",
            "- Input features: `answer` text only.",
            "- Labels: `final_scores` only, with scores `1`-`5` mapped to classes `0`-`4` during training.",
            f"- Epochs: `{training['epochs']}`",
            f"- Batch size: `{training['batch_size']}`",
            f"- Learning rate: `{training['learning_rate']}`",
            f"- Max length: `{training['max_length']}`",
            f"- Seed: `{training['seed']}`",
            f"- Device: `{training['device']}`",
            "",
            "## Result Summary",
            "",
            "| Split | Mean Exact Accuracy | Mean Macro-F1 | Mean Weighted-F1 | Mean Low/Mid/High Macro-F1 | Mean MAE | Weak Precision | Weak Recall | Weak F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["summary_rows"]:
        lines.append(
            f"| {row['split']} | {_format_metric(row['mean_exact_accuracy'])} | "
            f"{_format_metric(row['mean_macro_f1'])} | {_format_metric(row['mean_weighted_f1'])} | "
            f"{_format_metric(row['mean_low_mid_high_macro_f1'])} | {_format_metric(row['mean_mae'])} | "
            f"{_format_metric(row['weak_aspect_precision'])} | {_format_metric(row['weak_aspect_recall'])} | "
            f"{_format_metric(row['weak_aspect_f1'])} |"
        )
    lines.extend(["", "## Per-Split Observations", ""])
    for split_name in ["dev", "test", "ood"]:
        split_payload = payload["splits"][split_name]
        best, worst = _best_and_worst_aspects(split_payload["metrics"])
        summary = split_payload["metrics"]["summary"]
        lines.append(
            f"- `{split_name}`: mean exact accuracy is {summary['mean_exact_accuracy']:.4f}; "
            f"highest exact accuracy aspect is `{best}` and lowest is `{worst}`."
        )
    if "test" in payload["splits"] and "ood" in payload["splits"]:
        test_exact = payload["splits"]["test"]["metrics"]["summary"]["mean_exact_accuracy"]
        ood_exact = payload["splits"]["ood"]["metrics"]["summary"]["mean_exact_accuracy"]
        lines.append(
            f"- OOD mean exact accuracy differs from reviewed test by {ood_exact - test_exact:.4f} "
            f"({ood_exact:.4f} OOD vs {test_exact:.4f} test)."
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This is a baseline training run, not a tuned neural system.",
            "- The model treats each aspect score as a five-way class and does not explicitly optimize ordinal distance.",
            "- The dataset is synthetic and evaluation labels are project-team reviewed rather than external expert annotations.",
            "- Error analysis by answer type, domain, and aspect is intentionally left for a later project stage.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _output_paths(output_dir: str | Path) -> tuple[Path, Path]:
    output_path = _resolve_path(output_dir)
    return output_path / ENCODER_REPORT_FILENAMES[0], output_path / ENCODER_REPORT_FILENAMES[1]


def _print_summary(payload: dict[str, Any]) -> None:
    print("Supervised encoder baseline results")
    print("split\tmean_exact\tmean_macro_f1\tmean_lmh_f1\tmean_mae\tweak_f1")
    for row in payload["summary_rows"]:
        print(
            f"{row['split']}\t{row['mean_exact_accuracy']:.4f}\t{row['mean_macro_f1']:.4f}\t"
            f"{row['mean_low_mid_high_macro_f1']:.4f}\t{row['mean_mae']:.4f}\t{row['weak_aspect_f1']:.4f}"
        )


def _limit_records(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    return list(records[:limit]) if limit is not None else list(records)


def _load_encoder_dependencies():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch and Transformers are required for the encoder baseline. "
            "Install with: python3 -m pip install -r requirements.txt"
        ) from exc
    return torch, nn, DataLoader, Dataset, AutoModel, AutoTokenizer


def _select_device(torch_module) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _set_seed(seed: int, torch_module) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


class EncoderBaseline:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        max_length: int = DEFAULT_MAX_LENGTH,
        seed: int = DEFAULT_SEED,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.seed = seed
        self.device = device
        self.tokenizer = None
        self.model = None
        self._torch = None

    def _build_dataset(self, records: list[dict[str, Any]], include_labels: bool = True):
        torch_module = self._torch
        texts = [str(record["answer"]) for record in records]
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = None
        if include_labels:
            labels = torch_module.tensor(
                [[score_to_class(int(record["final_scores"][aspect])) for aspect in ASPECTS] for record in records],
                dtype=torch_module.long,
            )
        return encodings, labels

    def fit(self, train_records: list[dict[str, Any]]) -> list[dict[str, float | int]]:
        if not train_records:
            raise ValueError("train_records must not be empty")
        torch_module, nn, DataLoader, Dataset, AutoModel, AutoTokenizer = _load_encoder_dependencies()
        self._torch = torch_module
        _set_seed(self.seed, torch_module)
        self.device = self.device or _select_device(torch_module)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        class EncodedDataset(Dataset):
            def __init__(self, encodings, labels) -> None:
                self.encodings = encodings
                self.labels = labels

            def __len__(self) -> int:
                return len(self.labels)

            def __getitem__(self, index: int) -> dict[str, Any]:
                item = {key: value[index] for key, value in self.encodings.items()}
                item["labels"] = self.labels[index]
                return item

        class MultiTaskScoreModel(nn.Module):
            def __init__(self, model_name: str) -> None:
                super().__init__()
                self.encoder = AutoModel.from_pretrained(model_name)
                hidden_size = int(self.encoder.config.hidden_size)
                self.heads = nn.ModuleDict({aspect: nn.Linear(hidden_size, SCORE_CLASS_COUNT) for aspect in ASPECTS})

            def forward(self, **inputs):
                outputs = self.encoder(**inputs)
                pooled = outputs.last_hidden_state[:, 0]
                return {aspect: head(pooled) for aspect, head in self.heads.items()}

        encodings, labels = self._build_dataset(train_records, include_labels=True)
        generator = torch_module.Generator()
        generator.manual_seed(self.seed)
        train_loader = DataLoader(
            EncodedDataset(encodings, labels),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        self.model = MultiTaskScoreModel(self.model_name).to(self.device)
        optimizer = torch_module.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        history = []
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_rows = 0
            for batch in train_loader:
                labels = batch.pop("labels").to(self.device)
                inputs = {key: value.to(self.device) for key, value in batch.items()}
                optimizer.zero_grad()
                logits = self.model(**inputs)
                loss = sum(
                    loss_fn(logits[aspect], labels[:, aspect_index]) for aspect_index, aspect in enumerate(ASPECTS)
                ) / len(ASPECTS)
                loss.backward()
                optimizer.step()
                row_count = int(labels.shape[0])
                total_loss += float(loss.detach().cpu()) * row_count
                total_rows += row_count
            history.append({"epoch": epoch, "train_loss": total_loss / total_rows if total_rows else 0.0})
        return history

    def predict(self, records: list[dict[str, Any]]) -> list[dict[str, int]]:
        if not records:
            return []
        if self.model is None or self.tokenizer is None or self._torch is None:
            raise ValueError("EncoderBaseline must be fit before predict")
        torch_module = self._torch
        encodings, _ = self._build_dataset(records, include_labels=False)
        predictions: list[dict[str, int]] = []
        self.model.eval()
        with torch_module.no_grad():
            for start in range(0, len(records), self.batch_size):
                end = start + self.batch_size
                inputs = {key: value[start:end].to(self.device) for key, value in encodings.items()}
                logits = self.model(**inputs)
                batch_size = next(iter(logits.values())).shape[0]
                for row_index in range(batch_size):
                    predictions.append(
                        {
                            aspect: class_to_score(int(torch_module.argmax(logits[aspect][row_index]).cpu()))
                            for aspect in ASPECTS
                        }
                    )
        return predictions


def run_encoder_baseline_experiment(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    dev_path: str | Path = DEFAULT_DEV_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
    ood_path: str | Path = DEFAULT_OOD_PATH,
    output_dir: str | Path = Path("data/reports"),
    model_name: str = DEFAULT_MODEL_NAME,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_length: int = DEFAULT_MAX_LENGTH,
    seed: int = DEFAULT_SEED,
    limit_train: int | None = None,
    limit_eval: int | None = None,
    encoder: Any | None = None,
    print_summary: bool = True,
) -> dict[str, Any]:
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
    limited_splits = {
        "train": _limit_records(all_splits["train"], limit_train),
        "dev": _limit_records(all_splits["dev"], limit_eval),
        "test": _limit_records(all_splits["test"], limit_eval),
        "ood": _limit_records(all_splits["ood"], limit_eval),
    }
    model = encoder or EncoderBaseline(
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        seed=seed,
    )
    training_history = model.fit(limited_splits["train"])
    device = str(getattr(model, "device", "unknown"))
    result: dict[str, Any] = {
        "encoder_baseline_results_version": "v1",
        "dataset_version": DATASET_VERSION_OFFICIAL,
        "model_name": model_name,
        "training_parameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "seed": seed,
            "limit_train": limit_train,
            "limit_eval": limit_eval,
            "device": device,
        },
        "split_paths": {name: _display_path(path) for name, path in resolved_paths.items()},
        "split_counts": {name: len(records) for name, records in limited_splits.items()},
        "full_split_counts": {name: len(records) for name, records in all_splits.items()},
        "training_history": training_history,
        "summary_rows": [],
        "splits": {},
    }
    for split_name in ["dev", "test", "ood"]:
        records = limited_splits[split_name]
        predictions = model.predict(records)
        metrics = _evaluate(records, predictions)
        result["splits"][split_name] = {
            "record_count": len(records),
            "metrics": metrics,
            "predictions": _build_prediction_rows(records, predictions, split_name),
        }
        result["summary_rows"].append(_summary_row(split_name, metrics))
    json_report_path, markdown_report_path = _output_paths(output_dir)
    _write_json(json_report_path, result)
    _write_markdown(markdown_report_path, result)
    if print_summary:
        _print_summary(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the supervised encoder baseline for official_v1.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--train", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--dev", default=str(DEFAULT_DEV_PATH))
    parser.add_argument("--test", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--ood", default=str(DEFAULT_OOD_PATH))
    parser.add_argument("--output-dir", default="data/reports")
    parser.add_argument("--epochs", type=_positive_int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-length", type=_positive_int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit-train", type=_positive_int)
    parser.add_argument("--limit-eval", type=_positive_int)
    args = parser.parse_args()
    try:
        run_encoder_baseline_experiment(
            train_path=args.train,
            dev_path=args.dev,
            test_path=args.test,
            ood_path=args.ood,
            output_dir=args.output_dir,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            seed=args.seed,
            limit_train=args.limit_train,
            limit_eval=args.limit_eval,
        )
    except (ModuleNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
