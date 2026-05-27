from __future__ import annotations

import csv
import json
from pathlib import Path

from src.distribution_analysis import build_distribution_report
from src.io_utils import write_jsonl
from src.quality_checks import write_quality_report, write_svg_bar_chart
from src.quality_checks import write_v3_quality_report, write_v4_quality_reports
from src.schemas import (
    ASPECTS,
    STATUS_ACCEPTED,
    STATUS_ACCEPTED_BORDERLINE,
    STATUS_AUDIT_ONLY,
    STATUS_MANUAL_REVIEW,
    STATUS_PROFILE_MISMATCH,
    STATUS_REJECTED,
    DatasetRecord,
)
from src.split_dataset import apply_domain_cap, split_records, split_v3_records, split_v4_records


def write_csv(path: Path, records: list[DatasetRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "example_id",
        "target_role",
        "question_id",
        "question_type",
        "project_domain",
        "question",
        "profile_id",
        "answer",
        "weak_aspects",
        "strong_aspects",
        "final_scores",
        "split",
    ] + ASPECTS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {
                "example_id": record.example_id,
                "target_role": record.target_role,
                "question_id": record.question_id,
                "question_type": record.question_type,
                "project_domain": record.project_domain,
                "question": record.question,
                "profile_id": record.profile.profile_id,
                "answer": record.answer,
                "weak_aspects": "|".join(record.weak_aspects),
                "strong_aspects": "|".join(record.strong_aspects),
                "final_scores": json.dumps(record.final_scores, ensure_ascii=True),
                "split": record.split,
            }
            row.update(record.final_scores)
            writer.writerow(row)


def export_artifacts(
    root_dir: Path,
    generated: list[DatasetRecord],
    labeled: list[DatasetRecord],
    accepted: list[DatasetRecord],
    rejected: list[DatasetRecord],
    seed: int,
    manual_review_size: int,
    ood_domains: set[str],
    min_per_score: int,
) -> dict:
    data_dir = root_dir / "data"
    report_dir = root_dir / "reports"
    splits = split_records(accepted, seed=seed, manual_review_size=manual_review_size, ood_domains=ood_domains)
    manual_counts = {
        "dev": len(splits["dev_review_candidates"]),
        "test": len(splits["test_review_candidates"]),
        "ood_test": len(splits["ood_test_review_candidates"]),
    }
    report = build_distribution_report(accepted, rejected, min_per_score=min_per_score, manual_review_candidates=manual_counts)

    paths = {
        "raw": data_dir / "raw" / "generated_answers.jsonl",
        "labeled": data_dir / "labeled" / "labeled_examples.jsonl",
        "accepted": data_dir / "validated" / "accepted_synthetic.jsonl",
        "rejected": data_dir / "rejected" / "rejected_examples.jsonl",
        "train": data_dir / "final" / "train.jsonl",
        "dev": data_dir / "final" / "dev_review_candidates.jsonl",
        "test": data_dir / "final" / "test_review_candidates.jsonl",
        "ood": data_dir / "final" / "ood_test_review_candidates.jsonl",
        "full": data_dir / "final" / "full_synthetic_accepted.jsonl",
        "clean_accepted": data_dir / "final" / "full_synthetic_clean_accepted.jsonl",
        "borderline_review": data_dir / "final" / "full_synthetic_borderline_review.jsonl",
        "profile_mismatch_full": data_dir / "final" / "full_synthetic_profile_mismatch.jsonl",
        "all": data_dir / "final" / "full_synthetic_all.jsonl",
        "csv": data_dir / "final" / "full_synthetic_accepted.csv",
        "quality": report_dir / "quality_report.md",
        "score_plot": report_dir / "figures" / "score_distribution.svg",
        "rejection_plot": report_dir / "figures" / "rejection_reasons.svg",
    }
    write_jsonl(paths["raw"], generated)
    write_jsonl(paths["labeled"], labeled)
    write_jsonl(paths["accepted"], accepted)
    write_jsonl(paths["rejected"], rejected)
    write_jsonl(paths["train"], splits["train"])
    write_jsonl(paths["dev"], splits["dev_review_candidates"])
    write_jsonl(paths["test"], splits["test_review_candidates"])
    write_jsonl(paths["ood"], splits["ood_test_review_candidates"])
    write_jsonl(paths["full"], accepted)
    write_csv(paths["csv"], accepted)
    write_quality_report(report, paths["quality"])
    flat_scores = {
        f"{aspect}_{score}": count
        for aspect, counts in report.score_distribution.items()
        for score, count in counts.items()
    }
    write_svg_bar_chart(flat_scores, paths["score_plot"], "Score Distribution")
    write_svg_bar_chart(report.rejection_reasons, paths["rejection_plot"], "Rejection Reasons")
    return {"paths": {key: str(value) for key, value in paths.items()}, "splits": splits, "report": report}


def _reason_counts(records: list[DatasetRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for reason in record.validation.rejection_reasons or [record.validation.final_status]:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _unique_records(records: list[DatasetRecord]) -> list[DatasetRecord]:
    seen: set[str] = set()
    unique: list[DatasetRecord] = []
    for record in records:
        if record.example_id in seen:
            continue
        seen.add(record.example_id)
        unique.append(record)
    return unique


def export_v3_artifacts(
    root_dir: Path,
    generated: list[DatasetRecord],
    labeled: list[DatasetRecord],
    records: list[DatasetRecord],
    seed: int,
    manual_review_size: int,
    ood_domains: set[str],
    min_per_score: int,
    domain_cap: float,
    target_size_min: int | None = None,
) -> dict:
    data_dir = root_dir / "data" / "v3"
    report_dir = root_dir / "reports"
    split_candidates = [
        record
        for record in records
        if record.validation.final_status in {STATUS_ACCEPTED, STATUS_ACCEPTED_BORDERLINE, STATUS_MANUAL_REVIEW}
    ]
    kept_split_pool, cap_audit = apply_domain_cap(split_candidates, domain_cap=domain_cap, cap_base_size=target_size_min)
    for record in cap_audit:
        if record not in records:
            records.append(record)
    manual_review = [record for record in records if record.validation.final_status == STATUS_MANUAL_REVIEW]
    audit_only = [record for record in records if record.validation.final_status == STATUS_AUDIT_ONLY]
    rejected = [record for record in records if record.validation.final_status == STATUS_REJECTED]
    accepted = [record for record in kept_split_pool if record.validation.final_status == STATUS_ACCEPTED]
    borderline = [record for record in kept_split_pool if record.validation.final_status == STATUS_ACCEPTED_BORDERLINE]
    splits = split_v3_records(
        kept_split_pool,
        seed=seed,
        manual_review_size=manual_review_size,
        ood_domains=ood_domains,
        target_size_min=target_size_min,
    )
    paths = {
        "raw": data_dir / "raw" / "generated_answers.jsonl",
        "labeled": data_dir / "labeled" / "labeled_examples.jsonl",
        "accepted": data_dir / "validated" / "accepted_synthetic.jsonl",
        "borderline": data_dir / "validated" / "accepted_borderline.jsonl",
        "manual_review": data_dir / "review" / "manual_review_candidates.jsonl",
        "audit_only": data_dir / "audit" / "audit_only_examples.jsonl",
        "rejected": data_dir / "rejected" / "rejected_examples.jsonl",
        "train": data_dir / "final" / "train.jsonl",
        "dev": data_dir / "final" / "dev_review_candidates.jsonl",
        "test": data_dir / "final" / "test_review_candidates.jsonl",
        "ood": data_dir / "final" / "ood_test_review_candidates.jsonl",
        "full": data_dir / "final" / "full_synthetic_accepted.jsonl",
        "csv": data_dir / "final" / "full_synthetic_accepted.csv",
        "quality": report_dir / "v3_quality_report.md",
        "score_plot": report_dir / "v3_figures" / "score_distribution.svg",
        "rejection_plot": report_dir / "v3_figures" / "rejection_reasons.svg",
    }
    accepted_synthetic = accepted + borderline
    report_records = _unique_records(accepted_synthetic + manual_review + audit_only + rejected)
    write_jsonl(paths["raw"], generated)
    write_jsonl(paths["labeled"], labeled)
    write_jsonl(paths["accepted"], accepted)
    write_jsonl(paths["borderline"], borderline)
    write_jsonl(paths["manual_review"], manual_review)
    write_jsonl(paths["audit_only"], audit_only)
    write_jsonl(paths["rejected"], rejected)
    write_jsonl(paths["train"], splits["train"])
    write_jsonl(paths["dev"], splits["dev_review_candidates"])
    write_jsonl(paths["test"], splits["test_review_candidates"])
    write_jsonl(paths["ood"], splits["ood_test_review_candidates"])
    status_counts = {}
    write_jsonl(paths["full"], accepted_synthetic)
    write_csv(paths["csv"], accepted_synthetic)
    assigned_ids = {
        record.example_id
        for split_records in splits.values()
        for record in split_records
    }
    for record in manual_review:
        if record.example_id not in assigned_ids:
            record.split = "manual_review_pool"
    for record in audit_only:
        record.split = "audit_only"
    for record in rejected:
        record.split = "rejected"
    for record in report_records:
        status_counts[record.validation.final_status] = status_counts.get(record.validation.final_status, 0) + 1
    write_v3_quality_report(
        paths["quality"],
        report_records,
        min_per_score=min_per_score,
        status_counts=status_counts,
        manual_review_reasons=_reason_counts(manual_review),
        audit_reasons=_reason_counts(audit_only),
        rejection_reasons=_reason_counts(rejected),
        final_export_records=(
            splits["train"]
            + splits["dev_review_candidates"]
            + splits["test_review_candidates"]
            + splits["ood_test_review_candidates"]
        ),
    )
    manual_counts = {
        "dev": len(splits["dev_review_candidates"]),
        "test": len(splits["test_review_candidates"]),
        "ood_test": len(splits["ood_test_review_candidates"]),
    }
    report = build_distribution_report(
        accepted_synthetic,
        rejected + audit_only,
        min_per_score=min_per_score,
        manual_review_candidates=manual_counts,
    )
    flat_scores = {
        f"{aspect}_{score}": count
        for aspect, counts in report.score_distribution.items()
        for score, count in counts.items()
    }
    write_svg_bar_chart(flat_scores, paths["score_plot"], "V3 Score Distribution")
    write_svg_bar_chart(_reason_counts(rejected + audit_only + manual_review), paths["rejection_plot"], "V3 Review/Audit/Rejection Reasons")
    return {"paths": {key: str(value) for key, value in paths.items()}, "splits": splits, "report": report}


def _manual_review_payload(record: DatasetRecord) -> dict:
    payload = record.to_dict()
    payload["id"] = record.example_id
    payload["labeler_notes"] = record.labeler.rationale
    payload["validator_notes"] = record.validator.rationale
    payload["human_final_scores"] = {}
    payload["human_notes"] = ""
    payload["reviewed"] = False
    return payload


def export_official_generation_artifacts(
    output_dir: Path,
    generated: list[DatasetRecord],
    labeled: list[DatasetRecord],
    records: list[DatasetRecord],
) -> dict:
    accepted = [record for record in records if record.validation.final_status == STATUS_ACCEPTED]
    borderline = [record for record in records if record.validation.final_status == STATUS_ACCEPTED_BORDERLINE]
    manual_review = [record for record in records if record.validation.final_status == STATUS_MANUAL_REVIEW]
    profile_mismatch = [record for record in records if record.validation.final_status == STATUS_PROFILE_MISMATCH]
    rejected = [record for record in records if record.validation.final_status == STATUS_REJECTED]
    audit_only = [record for record in records if record.validation.final_status == STATUS_AUDIT_ONLY]
    paths = {
        "raw": output_dir / "raw" / "generated_answers.jsonl",
        "labeled": output_dir / "labeled" / "labeled_examples.jsonl",
        "all": output_dir / "full_synthetic_all.jsonl",
        "clean_accepted": output_dir / "full_synthetic_clean_accepted.jsonl",
        "borderline_review": output_dir / "full_synthetic_borderline_review.jsonl",
        "manual_review": output_dir / "full_synthetic_manual_review.jsonl",
        "profile_mismatch": output_dir / "full_synthetic_profile_mismatch.jsonl",
        "rejected": output_dir / "full_synthetic_rejected.jsonl",
        "audit_only": output_dir / "audit_only_examples.jsonl",
    }
    write_jsonl(paths["raw"], generated)
    write_jsonl(paths["labeled"], labeled)
    write_jsonl(paths["all"], records)
    write_jsonl(paths["clean_accepted"], accepted)
    write_jsonl(paths["borderline_review"], borderline)
    write_jsonl(paths["manual_review"], [_manual_review_payload(record) for record in manual_review])
    write_jsonl(paths["profile_mismatch"], profile_mismatch)
    write_jsonl(paths["rejected"], rejected)
    write_jsonl(paths["audit_only"], audit_only)
    return {"paths": {key: str(value) for key, value in paths.items()}}


def export_v4_artifacts(
    root_dir: Path,
    generated: list[DatasetRecord],
    labeled: list[DatasetRecord],
    records: list[DatasetRecord],
    seed: int,
    manual_review_size: int,
    ood_domains: set[str],
    min_per_band: int,
    domain_cap: float,
    question_type_cap: float,
    profile_cap: float,
    target_size_min: int,
    target_size_max: int,
) -> dict:
    data_dir = root_dir / "data" / "v4"
    report_dir = root_dir / "reports"
    results_dir = root_dir / "results"
    splits, leakage = split_v4_records(
        records,
        seed=seed,
        target_size_min=target_size_min,
        target_size_max=target_size_max,
        manual_review_size=manual_review_size,
        ood_domains=ood_domains,
        domain_cap=domain_cap,
        question_type_cap=question_type_cap,
        profile_cap=profile_cap,
    )
    final_review_records = (
        splits["dev_review_candidates"]
        + splits["test_review_candidates"]
        + splits["ood_test_review_candidates"]
    )
    final_export_records = splits["train"] + final_review_records
    accepted = [
        record
        for record in records
        if record.validation.final_status == STATUS_ACCEPTED
        and record.validation.final_status == STATUS_ACCEPTED
        and record.split in {"train", "dev_review_candidates", "test_review_candidates", "ood_test_review_candidates"}
    ]
    borderline = [
        record
        for record in records
        if record.validation.final_status == STATUS_ACCEPTED_BORDERLINE
        and record.split in {"train", "dev_review_candidates", "test_review_candidates", "ood_test_review_candidates"}
    ]
    manual_review = [record for record in records if record.validation.final_status == STATUS_MANUAL_REVIEW]
    profile_mismatch = [record for record in records if record.validation.final_status == STATUS_PROFILE_MISMATCH]
    audit_only = [record for record in records if record.validation.final_status == STATUS_AUDIT_ONLY]
    rejected = [record for record in records if record.validation.final_status == STATUS_REJECTED]
    paths = {
        "raw": data_dir / "raw" / "generated_answers.jsonl",
        "labeled": data_dir / "labeled" / "labeled_examples.jsonl",
        "accepted": data_dir / "validated" / "accepted_synthetic.jsonl",
        "borderline": data_dir / "validated" / "accepted_borderline.jsonl",
        "manual_review": data_dir / "review" / "manual_review_candidates.jsonl",
        "profile_mismatch": data_dir / "audit" / "profile_mismatch_examples.jsonl",
        "audit_only": data_dir / "audit" / "audit_only_examples.jsonl",
        "rejected": data_dir / "rejected" / "rejected_examples.jsonl",
        "train": data_dir / "final" / "train.jsonl",
        "dev": data_dir / "final" / "dev_review_candidates.jsonl",
        "test": data_dir / "final" / "test_review_candidates.jsonl",
        "ood": data_dir / "final" / "ood_test_review_candidates.jsonl",
        "full": data_dir / "final" / "full_synthetic_accepted.jsonl",
        "clean_accepted": data_dir / "final" / "full_synthetic_clean_accepted.jsonl",
        "borderline_review": data_dir / "final" / "full_synthetic_borderline_review.jsonl",
        "profile_mismatch_full": data_dir / "final" / "full_synthetic_profile_mismatch.jsonl",
        "all": data_dir / "final" / "full_synthetic_all.jsonl",
        "csv": data_dir / "final" / "full_synthetic_accepted.csv",
        "quality": report_dir / "data_quality_v4.md",
        "quality_json": results_dir / "data_quality_v4.json",
        "score_plot": report_dir / "v4_figures" / "score_distribution.svg",
        "status_plot": report_dir / "v4_figures" / "status_counts.svg",
    }
    write_jsonl(paths["raw"], generated)
    write_jsonl(paths["labeled"], labeled)
    write_jsonl(paths["accepted"], accepted)
    write_jsonl(paths["borderline"], borderline)
    write_jsonl(paths["manual_review"], [_manual_review_payload(record) for record in final_review_records])
    write_jsonl(paths["profile_mismatch"], profile_mismatch)
    write_jsonl(paths["audit_only"], audit_only)
    write_jsonl(paths["rejected"], rejected)
    write_jsonl(paths["train"], splits["train"])
    write_jsonl(paths["dev"], splits["dev_review_candidates"])
    write_jsonl(paths["test"], splits["test_review_candidates"])
    write_jsonl(paths["ood"], splits["ood_test_review_candidates"])
    accepted_synthetic = [record for record in final_export_records if record.validation.final_status in {STATUS_ACCEPTED, STATUS_ACCEPTED_BORDERLINE}]
    write_jsonl(paths["full"], accepted_synthetic)
    write_jsonl(paths["clean_accepted"], [record for record in records if record.validation.final_status == STATUS_ACCEPTED])
    write_jsonl(
        paths["borderline_review"],
        [record for record in records if record.validation.final_status in {STATUS_ACCEPTED_BORDERLINE, STATUS_MANUAL_REVIEW}],
    )
    write_jsonl(paths["profile_mismatch_full"], profile_mismatch)
    write_jsonl(paths["all"], records)
    write_csv(paths["csv"], accepted_synthetic)
    quality_payload = write_v4_quality_reports(
        report_path=paths["quality"],
        json_path=paths["quality_json"],
        records=records,
        final_export_records=final_export_records,
        splits=splits,
        leakage_report=leakage,
        min_per_band=min_per_band,
    )
    flat_scores = {
        f"{aspect}_{score}": count
        for aspect, counts in quality_payload["score_distribution"].items()
        for score, count in counts.items()
    }
    write_svg_bar_chart(flat_scores, paths["score_plot"], "V4 Score Distribution")
    write_svg_bar_chart(quality_payload["status_counts"], paths["status_plot"], "V4 Status Counts")
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "splits": splits,
        "leakage": leakage,
        "quality": quality_payload,
    }
