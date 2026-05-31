from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.schemas import ASPECTS, SCORE_MAX, SCORE_MIN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = Path("data/processed/train.jsonl")
DEFAULT_DEV_PATH = Path("data/reviewed/dev_project_team_reviewed.jsonl")
DEFAULT_TEST_PATH = Path("data/reviewed/test_project_team_reviewed.jsonl")
DEFAULT_OOD_PATH = Path("data/reviewed/ood_project_team_reviewed.jsonl")


def _resolve_path(path: str | Path, base_dir: Path = PROJECT_ROOT) -> Path:
    value = Path(path)
    return value if value.is_absolute() else base_dir / value


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"JSONL file not found: {source}")
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object")
            rows.append(payload)
    return rows


def validate_experiment_record(record: dict[str, Any], source: Path, index: int) -> None:
    prefix = f"{source}:{index}"
    answer = record.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"{prefix}: missing or empty answer")
    final_scores = record.get("final_scores")
    if not isinstance(final_scores, dict):
        raise ValueError(f"{prefix}: missing final_scores")
    missing = [aspect for aspect in ASPECTS if aspect not in final_scores]
    if missing:
        raise ValueError(f"{prefix}: final_scores missing aspects: {', '.join(missing)}")
    extra = [aspect for aspect in final_scores if aspect not in ASPECTS]
    if extra:
        raise ValueError(f"{prefix}: final_scores has unknown aspects: {', '.join(extra)}")
    for aspect in ASPECTS:
        value = final_scores[aspect]
        if type(value) is not int or not SCORE_MIN <= value <= SCORE_MAX:
            raise ValueError(f"{prefix}: final_scores.{aspect} must be an integer from {SCORE_MIN} to {SCORE_MAX}")


def _load_validated_split(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    for index, row in enumerate(rows, 1):
        validate_experiment_record(row, path, index)
    return rows


def load_experiment_splits(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    dev_path: str | Path = DEFAULT_DEV_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
    ood_path: str | Path = DEFAULT_OOD_PATH,
    base_dir: Path = PROJECT_ROOT,
) -> dict[str, list[dict[str, Any]]]:
    paths = {
        "train": _resolve_path(train_path, base_dir),
        "dev": _resolve_path(dev_path, base_dir),
        "test": _resolve_path(test_path, base_dir),
        "ood": _resolve_path(ood_path, base_dir),
    }
    return {name: _load_validated_split(path) for name, path in paths.items()}


def extract_texts_and_labels(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, list[int]]]:
    texts = [str(record["answer"]) for record in records]
    labels = {aspect: [] for aspect in ASPECTS}
    for record in records:
        for aspect in ASPECTS:
            labels[aspect].append(int(record["final_scores"][aspect]))
    return texts, labels
