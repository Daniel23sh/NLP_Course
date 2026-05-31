from __future__ import annotations

from statistics import mean

from src.schemas import ASPECTS


SCORE_LABELS = [1, 2, 3, 4, 5]
BAND_LABELS = ["low", "mid", "high"]


def score_to_band(score: int) -> str:
    if score in {1, 2}:
        return "low"
    if score == 3:
        return "mid"
    if score in {4, 5}:
        return "high"
    raise ValueError(f"score must be from 1 to 5, got {score}")


def _require_sklearn_metrics():
    try:
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for baseline metrics. Install with: python3 -m pip install -r requirements.txt"
        ) from exc
    return accuracy_score, confusion_matrix, f1_score, mean_absolute_error


def _float(value) -> float:
    return float(value)


def _validate_aspect_vectors(y_true_by_aspect: dict[str, list[int]], y_pred_by_aspect: dict[str, list[int]]) -> None:
    if set(y_true_by_aspect) != set(ASPECTS):
        raise ValueError(f"y_true_by_aspect must contain exactly these aspects: {ASPECTS}")
    if set(y_pred_by_aspect) != set(ASPECTS):
        raise ValueError(f"y_pred_by_aspect must contain exactly these aspects: {ASPECTS}")
    for aspect in ASPECTS:
        if len(y_true_by_aspect[aspect]) != len(y_pred_by_aspect[aspect]):
            raise ValueError(f"{aspect}: y_true and y_pred lengths differ")


def compute_metrics(y_true_by_aspect: dict[str, list[int]], y_pred_by_aspect: dict[str, list[int]]) -> dict:
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error = _require_sklearn_metrics()
    _validate_aspect_vectors(y_true_by_aspect, y_pred_by_aspect)
    aspect_metrics = {}
    exact_values = []
    macro_values = []
    weighted_values = []
    mae_values = []
    band_macro_values = []
    for aspect in ASPECTS:
        y_true = list(y_true_by_aspect[aspect])
        y_pred = list(y_pred_by_aspect[aspect])
        true_bands = [score_to_band(score) for score in y_true]
        pred_bands = [score_to_band(score) for score in y_pred]
        exact_accuracy = _float(accuracy_score(y_true, y_pred))
        macro_f1 = _float(f1_score(y_true, y_pred, labels=SCORE_LABELS, average="macro", zero_division=0))
        weighted_f1 = _float(f1_score(y_true, y_pred, labels=SCORE_LABELS, average="weighted", zero_division=0))
        mae = _float(mean_absolute_error(y_true, y_pred))
        band_macro_f1 = _float(
            f1_score(true_bands, pred_bands, labels=BAND_LABELS, average="macro", zero_division=0)
        )
        aspect_metrics[aspect] = {
            "exact_accuracy": exact_accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "mae": mae,
            "low_mid_high_macro_f1": band_macro_f1,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=SCORE_LABELS).astype(int).tolist(),
        }
        exact_values.append(exact_accuracy)
        macro_values.append(macro_f1)
        weighted_values.append(weighted_f1)
        mae_values.append(mae)
        band_macro_values.append(band_macro_f1)
    return {
        "aspects": aspect_metrics,
        "summary": {
            "mean_exact_accuracy": mean(exact_values) if exact_values else 0.0,
            "mean_macro_f1": mean(macro_values) if macro_values else 0.0,
            "mean_weighted_f1": mean(weighted_values) if weighted_values else 0.0,
            "mean_mae": mean(mae_values) if mae_values else 0.0,
            "mean_low_mid_high_macro_f1": mean(band_macro_values) if band_macro_values else 0.0,
        },
    }


def compute_weak_aspect_metrics(true_scores: list[dict[str, int]], pred_scores: list[dict[str, int]]) -> dict[str, float]:
    if len(true_scores) != len(pred_scores):
        raise ValueError("true_scores and pred_scores lengths differ")
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for true_row, pred_row in zip(true_scores, pred_scores):
        true_weak = {aspect for aspect in ASPECTS if true_row[aspect] <= 2}
        pred_weak = {aspect for aspect in ASPECTS if pred_row[aspect] <= 2}
        true_positive += len(true_weak & pred_weak)
        false_positive += len(pred_weak - true_weak)
        false_negative += len(true_weak - pred_weak)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
