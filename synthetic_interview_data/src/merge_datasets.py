from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from src.io_utils import read_jsonl, write_jsonl
from src.schemas import (
    STATUS_ACCEPTED,
    STATUS_ACCEPTED_BORDERLINE,
    STATUS_AUDIT_ONLY,
    STATUS_MANUAL_REVIEW,
    STATUS_PROFILE_MISMATCH,
    STATUS_REJECTED,
    DatasetRecord,
    record_from_dict,
)


FINAL_SPLITS = {"train", "dev_review_candidates", "test_review_candidates", "ood_test_review_candidates"}


def _normalize_answer(answer: str) -> str:
    return re.sub(r"\s+", " ", answer.strip().lower())


def _load_records(path: Path) -> list[DatasetRecord]:
    return [record_from_dict(row) for row in read_jsonl(path)]


def _review_payload(record: DatasetRecord) -> dict:
    payload = record.to_dict()
    payload["human_reviewed"] = False
    payload["human_final_scores"] = {}
    payload["human_notes"] = ""
    return payload


def _scenario_family(record: DatasetRecord, split_group_key: str) -> str:
    if split_group_key == "scenario_family" and record.scenario_family:
        return record.scenario_family
    return "|".join([record.question_type, record.project_domain, record.profile.profile_id])


def _dedupe(records: list[DatasetRecord]) -> list[DatasetRecord]:
    seen_ids: set[str] = set()
    seen_answers: set[str] = set()
    kept: list[DatasetRecord] = []
    for record in records:
        normalized = _normalize_answer(record.answer)
        if record.example_id in seen_ids or normalized in seen_answers:
            continue
        seen_ids.add(record.example_id)
        seen_answers.add(normalized)
        kept.append(record)
    return kept


def _mark_audit(record: DatasetRecord, reason: str) -> None:
    record.validation.final_status = STATUS_AUDIT_ONLY
    if reason not in record.validation.rejection_reasons:
        record.validation.rejection_reasons.append(reason)
    if reason not in record.validation.flags:
        record.validation.flags.append(reason)
    record.split = "audit_only"


def _apply_share_caps(
    records: list[DatasetRecord],
    max_domain_share: float,
    max_profile_share: float,
    max_question_type_share: float,
) -> tuple[list[DatasetRecord], list[DatasetRecord]]:
    total = max(len(records), 1)
    limits = {
        "domain": max(1, int(total * max_domain_share)),
        "profile": max(1, int(total * max_profile_share)),
        "question_type": max(1, int(total * max_question_type_share)),
    }
    counts = {"domain": Counter(), "profile": Counter(), "question_type": Counter()}
    kept: list[DatasetRecord] = []
    audit: list[DatasetRecord] = []
    for record in records:
        values = {
            "domain": record.project_domain,
            "profile": record.profile.profile_id,
            "question_type": record.question_type,
        }
        exceeded = [key for key, value in values.items() if counts[key][value] >= limits[key]]
        if exceeded:
            _mark_audit(record, "merge_diversity_cap_excess:" + ",".join(sorted(exceeded)))
            audit.append(record)
            continue
        kept.append(record)
        for key, value in values.items():
            counts[key][value] += 1
    return kept, audit


def _group_records(records: list[DatasetRecord], split_group_key: str) -> list[list[DatasetRecord]]:
    grouped: dict[str, list[DatasetRecord]] = {}
    for record in records:
        grouped.setdefault(_scenario_family(record, split_group_key), []).append(record)
    return list(grouped.values())


def _assign(records: list[DatasetRecord], split_name: str) -> list[DatasetRecord]:
    for record in records:
        record.split = split_name
    return records


def _split_records(
    records: list[DatasetRecord],
    split_group_key: str,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    ood_ratio: float,
    ood_domains: set[str],
    seed: int,
) -> tuple[dict[str, list[DatasetRecord]], dict[str, list[str]]]:
    rng = random.Random(seed)
    eligible = [record for record in records if record.validation.final_status == STATUS_ACCEPTED]
    ood_groups = _group_records([record for record in eligible if record.project_domain in ood_domains], split_group_key)
    in_groups = _group_records([record for record in eligible if record.project_domain not in ood_domains], split_group_key)
    rng.shuffle(ood_groups)
    rng.shuffle(in_groups)

    total = len(eligible)
    ood_target = int(total * ood_ratio)
    dev_target = max(1, int(total * dev_ratio)) if total >= 4 else 0
    test_target = max(1, int(total * test_ratio)) if total >= 4 else 0

    ood: list[DatasetRecord] = []
    for group in ood_groups:
        if len(ood) >= ood_target and ood:
            break
        ood.extend(group)

    dev: list[DatasetRecord] = []
    test: list[DatasetRecord] = []
    train: list[DatasetRecord] = []
    for group in in_groups:
        if len(dev) < dev_target:
            dev.extend(group)
        elif len(test) < test_target:
            test.extend(group)
        else:
            train.extend(group)

    if not train and test:
        train.append(test.pop())
    if not train and dev:
        train.append(dev.pop())

    splits = {
        "train": _assign(train, "train"),
        "dev_review_candidates": _assign(dev, "dev_review_candidates"),
        "test_review_candidates": _assign(test, "test_review_candidates"),
        "ood_test_review_candidates": _assign(ood, "ood_test_review_candidates"),
    }
    leakage = _leakage_report(splits, split_group_key)
    return splits, leakage


def _leakage_report(splits: dict[str, list[DatasetRecord]], split_group_key: str) -> dict[str, list[str]]:
    family_splits: dict[str, set[str]] = {}
    for split_name, records in splits.items():
        for record in records:
            family_splits.setdefault(_scenario_family(record, split_group_key), set()).add(split_name)
    return {
        "train_test_leakage": sorted(
            family
            for family, names in family_splits.items()
            if "train" in names and "test_review_candidates" in names
        ),
        "ood_train_leakage": sorted(
            family
            for family, names in family_splits.items()
            if "train" in names and "ood_test_review_candidates" in names
        ),
    }


def merge_dataset_files(
    broad_path: Path,
    weak_path: Path,
    strong_path: Path,
    output_dir: Path,
    max_domain_share: float,
    max_profile_share: float,
    max_question_type_share: float,
    split_group_key: str,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    ood_ratio: float,
    ood_domains: set[str] | None = None,
    high_impact_path: Path | None = None,
    raw_output_dir: Path | None = None,
    seed: int = 42,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_output_dir = raw_output_dir or output_dir
    all_output_dir.mkdir(parents=True, exist_ok=True)
    input_records = _load_records(broad_path) + _load_records(weak_path) + _load_records(strong_path)
    if high_impact_path is not None:
        input_records.extend(_load_records(high_impact_path))
    records = _dedupe(input_records)
    clean_candidates = [record for record in records if record.validation.final_status == STATUS_ACCEPTED]
    kept, cap_audit = _apply_share_caps(
        clean_candidates,
        max_domain_share=max_domain_share,
        max_profile_share=max_profile_share,
        max_question_type_share=max_question_type_share,
    )
    excluded_ids = {record.example_id for record in cap_audit}
    all_records = [record for record in records if record.example_id not in excluded_ids] + cap_audit
    splits, leakage = _split_records(
        kept,
        split_group_key=split_group_key,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        ood_ratio=ood_ratio,
        ood_domains=ood_domains or {"NLP course project", "general coursework"},
        seed=seed,
    )
    final_records = (
        splits["train"]
        + splits["dev_review_candidates"]
        + splits["test_review_candidates"]
        + splits["ood_test_review_candidates"]
    )
    final_ids = {record.example_id for record in final_records}
    for record in all_records:
        if record.example_id not in final_ids and record.validation.final_status == STATUS_ACCEPTED:
            record.split = "not_selected"

    paths = {
        "train": output_dir / "train.jsonl",
        "dev": output_dir / "dev_review_candidates.jsonl",
        "test": output_dir / "test_review_candidates.jsonl",
        "ood": output_dir / "ood_test_review_candidates.jsonl",
        "not_selected": output_dir / "not_selected.jsonl",
        "all": all_output_dir / "full_synthetic_all.jsonl",
        "clean_accepted": output_dir / "full_synthetic_clean_accepted.jsonl",
        "borderline_review": output_dir / "full_synthetic_borderline_review.jsonl",
        "manual_review": output_dir / "full_synthetic_manual_review.jsonl",
        "profile_mismatch": output_dir / "full_synthetic_profile_mismatch.jsonl",
        "rejected": output_dir / "full_synthetic_rejected.jsonl",
    }
    write_jsonl(paths["train"], splits["train"])
    write_jsonl(paths["dev"], [_review_payload(record) for record in splits["dev_review_candidates"]])
    write_jsonl(paths["test"], [_review_payload(record) for record in splits["test_review_candidates"]])
    write_jsonl(paths["ood"], [_review_payload(record) for record in splits["ood_test_review_candidates"]])
    write_jsonl(paths["not_selected"], [record for record in all_records if record.split == "not_selected"])
    write_jsonl(paths["all"], all_records)
    write_jsonl(paths["clean_accepted"], [record for record in all_records if record.validation.final_status == STATUS_ACCEPTED])
    write_jsonl(paths["borderline_review"], [record for record in all_records if record.validation.final_status == STATUS_ACCEPTED_BORDERLINE])
    write_jsonl(paths["manual_review"], [_review_payload(record) for record in all_records if record.validation.final_status == STATUS_MANUAL_REVIEW])
    write_jsonl(paths["profile_mismatch"], [record for record in all_records if record.validation.final_status == STATUS_PROFILE_MISMATCH])
    write_jsonl(paths["rejected"], [record for record in all_records if record.validation.final_status in {STATUS_REJECTED, STATUS_AUDIT_ONLY}])
    return {
        "input_records": len(records),
        "final_records": len(final_records),
        "paths": {key: str(path) for key, path in paths.items()},
        "split_counts": {key: len(value) for key, value in splits.items()},
        "status_counts": dict(Counter(record.validation.final_status for record in all_records)),
        "leakage": leakage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge official synthetic dataset phases.")
    parser.add_argument("--broad", required=True)
    parser.add_argument("--weak", required=True)
    parser.add_argument("--strong", required=True)
    parser.add_argument("--high-impact", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-output-dir", default=None)
    parser.add_argument("--max-domain-share", type=float, default=0.25)
    parser.add_argument("--max-profile-share", type=float, default=0.25)
    parser.add_argument("--max-question-type-share", type=float, default=0.30)
    parser.add_argument("--split-group-key", default="scenario_family")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--dev-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--ood-ratio", type=float, default=0.10)
    parser.add_argument("--ood-domains", default="NLP course project,general coursework")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ood_domains = {item.strip() for item in args.ood_domains.split(",") if item.strip()}
    result = merge_dataset_files(
        broad_path=Path(args.broad),
        weak_path=Path(args.weak),
        strong_path=Path(args.strong),
        output_dir=Path(args.output_dir),
        max_domain_share=args.max_domain_share,
        max_profile_share=args.max_profile_share,
        max_question_type_share=args.max_question_type_share,
        split_group_key=args.split_group_key,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        ood_ratio=args.ood_ratio,
        ood_domains=ood_domains,
        high_impact_path=Path(args.high_impact) if args.high_impact else None,
        raw_output_dir=Path(args.raw_output_dir) if args.raw_output_dir else None,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
