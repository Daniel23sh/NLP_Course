from __future__ import annotations

from pathlib import Path

from src.io_utils import read_jsonl
from src.schemas import DatasetRecord, compute_strong_aspects, compute_weak_aspects, validate_scores


def validate_manual_gold_entry(entry: dict) -> None:
    required = {"example_id", "manual_scores", "manual_notes", "reviewer_id", "review_status"}
    missing = required - set(entry)
    if missing:
        raise ValueError(f"manual gold entry missing fields: {sorted(missing)}")
    if entry["review_status"] != "verified":
        raise ValueError("manual gold entry must have review_status='verified'")
    validate_scores({aspect: int(value) for aspect, value in entry["manual_scores"].items()})


def load_manual_gold(path: Path) -> dict[str, dict]:
    entries = {}
    for entry in read_jsonl(path):
        validate_manual_gold_entry(entry)
        entries[entry["example_id"]] = entry
    return entries


def apply_manual_gold(records: list[DatasetRecord], manual_gold: dict[str, dict]) -> list[DatasetRecord]:
    updated: list[DatasetRecord] = []
    for record in records:
        entry = manual_gold.get(record.example_id)
        if entry:
            scores = {aspect: int(value) for aspect, value in entry["manual_scores"].items()}
            record.final_scores = scores
            record.weak_aspects = compute_weak_aspects(scores)
            record.strong_aspects = compute_strong_aspects(scores)
            record.metadata["manual_review_status"] = entry["review_status"]
            record.metadata["manual_notes"] = entry["manual_notes"]
            record.metadata["manual_reviewer_id"] = entry["reviewer_id"]
        updated.append(record)
    return updated

