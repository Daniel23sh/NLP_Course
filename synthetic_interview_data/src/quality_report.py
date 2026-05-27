from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from src.distribution_analysis import low_mid_high_counts, low_mid_high_coverage_gaps, score_distribution
from src.io_utils import read_jsonl
from src.schemas import ASPECTS, STATUS_ACCEPTED, DatasetRecord, record_from_dict


def _records(path: Path) -> list[DatasetRecord]:
    return [record_from_dict(row) for row in read_jsonl(path)]


def _scenario_family(record: DatasetRecord) -> str:
    return record.scenario_family or "|".join([record.question_type, record.project_domain, record.profile.profile_id])


def _leakage_report(records: list[DatasetRecord]) -> dict[str, list[str]]:
    family_splits: dict[str, set[str]] = {}
    for record in records:
        if record.split:
            family_splits.setdefault(_scenario_family(record), set()).add(record.split)
    return {
        "train_test_leakage": sorted(
            family for family, splits in family_splits.items() if "train" in splits and "test_review_candidates" in splits
        ),
        "ood_train_leakage": sorted(
            family for family, splits in family_splits.items() if "train" in splits and "ood_test_review_candidates" in splits
        ),
    }


def _ready_for_training(payload: dict) -> bool:
    readiness = payload["readiness"]
    has_coverage_gaps = any(payload["remaining_low_mid_high_gaps"].values())
    return (
        readiness["train_non_empty"]
        and readiness["clean_accepted_non_empty"]
        and not has_coverage_gaps
        and payload["ood_count"] >= 20
        and not payload["split_leakage_report"]["train_test_leakage"]
        and not payload["split_leakage_report"]["ood_train_leakage"]
        and payload["readiness"]["accepted_delta_ge_2_count"] == 0
    )


def build_official_quality_payload(records: list[DatasetRecord], min_per_band: int = 20) -> dict:
    scored = [record for record in records if record.final_scores]
    accepted = [record for record in records if record.validation.final_status == STATUS_ACCEPTED]
    word_counts = [len(record.answer.split()) for record in records if record.answer]
    split_counts = dict(Counter(record.split for record in records))
    leakage = _leakage_report(records)
    accepted_delta_ge_2 = [
        record
        for record in accepted
        if any(delta >= 2 for delta in record.validation.score_deltas.values())
    ]
    payload = {
        "dataset_size": len(records),
        "clean_accepted_count": len(accepted),
        "split_counts": split_counts,
        "status_counts": dict(Counter(record.validation.final_status for record in records)),
        "score_distribution": score_distribution(scored),
        "low_mid_high_coverage": low_mid_high_counts(scored),
        "remaining_low_mid_high_gaps": low_mid_high_coverage_gaps(scored, min_per_band=min_per_band),
        "weak_aspect_frequency": dict(Counter(aspect for record in scored for aspect in record.weak_aspects)),
        "strong_aspect_frequency": dict(Counter(aspect for record in scored for aspect in record.strong_aspects)),
        "profile_distribution": dict(Counter(record.profile.profile_id for record in records)),
        "profile_success_rate": _profile_success_rate(records),
        "profile_mismatch_count": sum(1 for record in records if record.validation.final_status == "profile_mismatch"),
        "domain_distribution": dict(Counter(record.project_domain for record in records)),
        "question_type_distribution": dict(Counter(record.question_type for record in records)),
        "scenario_family_distribution": dict(Counter(_scenario_family(record) for record in records)),
        "answer_length": {
            "min": min(word_counts) if word_counts else 0,
            "mean": round(mean(word_counts), 2) if word_counts else 0,
            "max": max(word_counts) if word_counts else 0,
        },
        "labeler_validator_agreement": dict(Counter(str(delta) for record in records for delta in record.validation.score_deltas.values())),
        "deterministic_validation_flags": dict(Counter(flag for record in records for flag in record.validation.flags)),
        "split_leakage_report": leakage,
        "ood_count": split_counts.get("ood_test_review_candidates", 0),
        "manual_review_candidate_count": (
            split_counts.get("dev_review_candidates", 0)
            + split_counts.get("test_review_candidates", 0)
            + split_counts.get("ood_test_review_candidates", 0)
        ),
        "manual_review_examples": [
            record.example_id for record in records if record.split in {"dev_review_candidates", "test_review_candidates", "ood_test_review_candidates"}
        ][:20],
        "readiness": {
            "usable_for_synthetic_training": bool(accepted),
            "usable_for_final_evaluation_without_manual_labels": False,
            "manual_review_required_for_dev_test_ood": True,
            "train_non_empty": split_counts.get("train", 0) > 0,
            "clean_accepted_non_empty": bool(accepted),
            "accepted_delta_ge_2_count": len(accepted_delta_ge_2),
        },
        "recommended_next_actions": [
            "Manually verify dev/test/OOD review candidates before final evaluation.",
            "Inspect any remaining low/mid/high coverage gaps before model training.",
            "Keep profile_mismatch and rejected examples for error analysis only.",
        ],
    }
    payload["readiness"]["ready_to_start_baseline_training"] = _ready_for_training(payload)
    return payload


def _profile_success_rate(records: list[DatasetRecord]) -> float:
    constrained = [record for record in records if record.profile.expected_score_constraints and record.final_scores]
    if not constrained:
        return 0.0
    success = 0
    total = 0
    for record in constrained:
        for aspect, constraint in record.profile.expected_score_constraints.items():
            if aspect not in record.final_scores:
                continue
            total += 1
            score = record.final_scores[aspect]
            if "max" in constraint and score <= constraint["max"]:
                success += 1
            if "min" in constraint and score >= constraint["min"]:
                success += 1
    return success / max(total, 1)


def write_official_quality_report(input_path: Path, output_md: Path, output_json: Path, min_per_band: int = 20) -> dict:
    records = _records(input_path)
    payload = build_official_quality_payload(records, min_per_band=min_per_band)
    lines = [
        "# Official Synthetic Interview Data Quality Report",
        "",
        f"- Dataset size: {payload['dataset_size']}",
        f"- Clean accepted count: {payload['clean_accepted_count']}",
        f"- OOD count: {payload['ood_count']}",
        f"- Manual review candidate count: {payload['manual_review_candidate_count']}",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(payload["status_counts"].items()):
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## Split Counts", ""])
    for split, count in sorted(payload["split_counts"].items()):
        lines.append(f"- {split}: {count}")
    lines.extend(["", "## Score Distributions", ""])
    for aspect in ASPECTS:
        lines.append(f"- {aspect}: {payload['score_distribution'].get(aspect, {})}")
    lines.extend(["", "## Low/Mid/High Coverage", ""])
    for aspect in ASPECTS:
        lines.append(f"- {aspect}: {payload['low_mid_high_coverage'].get(aspect, {})}")
    impact_high = payload["low_mid_high_coverage"].get("impact", {}).get("high", 0)
    lines.append(f"- Impact high coverage: {impact_high} / {min_per_band}")
    lines.extend(["", "## Remaining Coverage Gaps", ""])
    for aspect, gaps in payload["remaining_low_mid_high_gaps"].items():
        lines.append(f"- {aspect}: {gaps}")
    lines.extend(["", "## Diversity", ""])
    lines.append(f"- Domains: {payload['domain_distribution']}")
    lines.append(f"- Profiles: {payload['profile_distribution']}")
    lines.append(f"- Question types: {payload['question_type_distribution']}")
    lines.extend(["", "## Split Leakage Report", ""])
    lines.append(f"- train_test_leakage: {payload['split_leakage_report']['train_test_leakage']}")
    lines.append(f"- ood_train_leakage: {payload['split_leakage_report']['ood_train_leakage']}")
    lines.extend(["", "## Labeler-Validator Agreement", ""])
    for delta, count in sorted(payload["labeler_validator_agreement"].items()):
        lines.append(f"- delta {delta}: {count}")
    lines.extend(["", "## Readiness Statement", ""])
    lines.append(f"- Usable for synthetic training: {'yes' if payload['readiness']['usable_for_synthetic_training'] else 'no'}")
    lines.append("- Usable for final evaluation without manual labels: no")
    lines.append("- Manual review required for dev/test/OOD: yes")
    lines.append(f"- Ready to start baseline training: {'yes' if payload['readiness']['ready_to_start_baseline_training'] else 'no'}")
    lines.extend(["", "## Recommended Next Actions", ""])
    for action in payload["recommended_next_actions"]:
        lines.append(f"- {action}")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Write official synthetic data quality report.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-per-band", type=int, default=20)
    args = parser.parse_args()
    payload = write_official_quality_report(
        input_path=Path(args.input),
        output_md=Path(args.output_md),
        output_json=Path(args.output_json),
        min_per_band=args.min_per_band,
    )
    print(json.dumps({"dataset_size": payload["dataset_size"], "readiness": payload["readiness"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
