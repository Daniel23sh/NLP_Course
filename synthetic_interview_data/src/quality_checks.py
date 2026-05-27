from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
from statistics import mean

from src.distribution_analysis import find_underrepresented_bands, low_mid_high_counts, low_mid_high_coverage_gaps, score_distribution
from src.schemas import ASPECTS, DistributionReport, DatasetRecord


def write_svg_bar_chart(values: dict, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 760
    height = max(180, 44 + 28 * len(values))
    max_value = max(values.values()) if values else 1
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="20" y="24" font-size="18">{title}</text>',
    ]
    y = 58
    for label, value in sorted(values.items()):
        bar_width = int((value / max_value) * 420) if max_value else 0
        rows.append(f'<text x="20" y="{y}" font-size="12">{label}</text>')
        rows.append(f'<rect x="260" y="{y - 13}" width="{bar_width}" height="16" fill="#2563eb"/>')
        rows.append(f'<text x="{270 + bar_width}" y="{y}" font-size="12">{value}</text>')
        y += 28
    rows.append("</svg>")
    path.write_text("\n".join(rows), encoding="utf-8")


def write_quality_report(report: DistributionReport, path: Path) -> None:
    data = report.to_dict()
    lines = [
        "# Quality Report",
        "",
        f"- Generated examples: {data['generated_examples']}",
        f"- Accepted examples: {data['accepted_examples']}",
        f"- Rejected examples: {data['rejected_examples']}",
        f"- Acceptance rate: {data['acceptance_rate']:.2%}",
        "",
        "## Labeler-Validator Agreement Delta Distribution",
        "",
    ]
    for key, value in sorted(data["agreement_delta_distribution"].items()):
        lines.append(f"- delta {key}: {value}")
    lines.extend(["", "## Score Distribution", ""])
    for aspect, distribution in data["score_distribution"].items():
        lines.append(f"- {aspect}: {distribution}")
    lines.extend(["", "## Weak Aspect Frequency", ""])
    for aspect, count in data["weak_aspect_frequency"].items():
        lines.append(f"- {aspect}: {count}")
    lines.extend(["", "## Strong Aspect Frequency", ""])
    for aspect, count in data["strong_aspect_frequency"].items():
        lines.append(f"- {aspect}: {count}")
    lines.extend(["", "## Profiles", ""])
    for key, value in sorted(data["profile_distribution"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Question Types", ""])
    for key, value in sorted(data["question_type_distribution"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Project Domains", ""])
    for key, value in sorted(data["project_domain_distribution"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Rejection Reasons", ""])
    for key, value in sorted(data["rejection_reasons"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Underrepresented Bands", ""])
    for aspect, bands in data["underrepresented_bands"].items():
        lines.append(f"- {aspect}: {bands}")
    lines.extend(["", "## Manual Review Candidates", ""])
    for key, value in sorted(data["manual_review_candidates"].items()):
        lines.append(f"- {key}: {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pct(count: int, total: int) -> str:
    return f"{(count / max(total, 1)):.1%}"


def write_v3_quality_report(
    path: Path,
    records: list[DatasetRecord],
    min_per_score: int,
    status_counts: dict[str, int] | None = None,
    manual_review_reasons: dict[str, int] | None = None,
    audit_reasons: dict[str, int] | None = None,
    rejection_reasons: dict[str, int] | None = None,
    final_export_records: list[DatasetRecord] | None = None,
) -> None:
    status_counts = status_counts or dict(Counter(record.validation.final_status for record in records))
    manual_review_reasons = manual_review_reasons or {}
    audit_reasons = audit_reasons or {}
    rejection_reasons = rejection_reasons or {}
    scored = [record for record in records if record.final_scores]
    word_counts = [len(record.answer.split()) for record in records if record.answer]
    domains = Counter(record.project_domain for record in records)
    final_export_records = final_export_records or []
    final_domains = Counter(record.project_domain for record in final_export_records)
    profiles = Counter(record.profile.profile_id for record in records)
    splits = Counter(record.split for record in records)
    deltas = Counter(str(delta) for record in records for delta in record.validation.score_deltas.values())
    weak = Counter(aspect for record in scored for aspect in record.weak_aspects)
    strong = Counter(aspect for record in scored for aspect in record.strong_aspects)
    lines = [
        "# V3 Quality Report",
        "",
        f"- Generated records: {len(records)}",
        f"- Scored records: {len(scored)}",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## Score Distribution", ""])
    for aspect, distribution in score_distribution(scored).items():
        lines.append(f"- {aspect}: {distribution}")
    lines.extend(["", "## Weak Aspect Frequency", ""])
    for aspect in ASPECTS:
        lines.append(f"- {aspect}: {weak.get(aspect, 0)}")
    lines.extend(["", "## Strong Aspect Frequency", ""])
    for aspect in ASPECTS:
        lines.append(f"- {aspect}: {strong.get(aspect, 0)}")
    lines.extend(["", "## Domain Distribution", ""])
    for domain, count in domains.most_common():
        lines.append(f"- {domain}: {count} ({_pct(count, len(records))})")
    if final_export_records:
        lines.extend(["", "## Final Export Domain Distribution", ""])
        for domain, count in final_domains.most_common():
            lines.append(f"- {domain}: {count} ({_pct(count, len(final_export_records))})")
    lines.extend(["", "## Profile Distribution", ""])
    for profile, count in profiles.most_common():
        lines.append(f"- {profile}: {count}")
    lines.extend(["", "## Split Counts", ""])
    for split, count in sorted(splits.items()):
        lines.append(f"- {split}: {count}")
    lines.extend(["", "## OOD Count", "", f"- ood_test_review_candidates: {splits.get('ood_test_review_candidates', 0)}"])
    lines.extend(["", "## Word Count Stats", ""])
    lines.append(f"- min: {min(word_counts) if word_counts else 0}")
    lines.append(f"- mean: {round(mean(word_counts), 2) if word_counts else 0}")
    lines.append(f"- max: {max(word_counts) if word_counts else 0}")
    lines.extend(["", "## Labeler-Validator Delta Summary", ""])
    for delta, count in sorted(deltas.items()):
        lines.append(f"- delta {delta}: {count}")
    lines.extend(["", "## Deterministic Check Counts", ""])
    flag_counts = Counter(flag for record in records for flag in record.validation.flags)
    for flag, count in flag_counts.most_common():
        lines.append(f"- {flag}: {count}")
    lines.extend(["", "## Manual Review Candidates", ""])
    for reason, count in sorted(manual_review_reasons.items()):
        lines.append(f"- {reason}: {count}")
    lines.extend(["", "## Audit Only Reasons", ""])
    for reason, count in sorted(audit_reasons.items()):
        lines.append(f"- {reason}: {count}")
    lines.extend(["", "## Rejection Reasons", ""])
    for reason, count in sorted(rejection_reasons.items()):
        lines.append(f"- {reason}: {count}")
    lines.extend(["", "## Remaining Underrepresented Bands", ""])
    for aspect, bands in find_underrepresented_bands(scored, min_per_score).items():
        lines.append(f"- {aspect}: {bands}")
    lines.extend(
        [
            "",
            "## V3 Dataset Readiness",
            "",
            f"- Usable for synthetic training: {'yes' if status_counts.get('accepted', 0) >= 1 else 'no'}",
            "- Usable for final evaluation without manual labels: no",
            "- Manual review required for dev/test/OOD: yes",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _profile_constraint_outcomes(records: list[DatasetRecord]) -> dict[str, dict[str, int]]:
    outcomes: dict[str, dict[str, int]] = {}
    for record in records:
        constraints = record.profile.expected_score_constraints
        if not constraints or not record.final_scores:
            continue
        profile_id = record.profile.profile_id
        outcomes.setdefault(profile_id, {"success": 0, "borderline": 0, "mismatch": 0})
        for aspect, constraint in constraints.items():
            if aspect not in record.final_scores:
                continue
            score = record.final_scores[aspect]
            max_score = constraint.get("max")
            if max_score is not None:
                if score <= max_score:
                    outcomes[profile_id]["success"] += 1
                elif constraint.get("strict"):
                    outcomes[profile_id]["mismatch"] += 1
                elif score >= max_score + 2:
                    outcomes[profile_id]["mismatch"] += 1
                else:
                    outcomes[profile_id]["borderline"] += 1
    return outcomes


def write_v4_quality_reports(
    report_path: Path,
    json_path: Path,
    records: list[DatasetRecord],
    final_export_records: list[DatasetRecord],
    splits: dict[str, list[DatasetRecord]],
    leakage_report: dict[str, list[str]],
    min_per_band: int,
) -> dict:
    scored = [record for record in records if record.final_scores]
    word_counts = [len(record.answer.split()) for record in records if record.answer]
    profile_outcomes = _profile_constraint_outcomes(records)
    success = sum(item["success"] for item in profile_outcomes.values())
    borderline = sum(item["borderline"] for item in profile_outcomes.values())
    mismatch = sum(item["mismatch"] for item in profile_outcomes.values())
    profile_total = success + borderline + mismatch
    status_counts = dict(Counter(record.validation.final_status for record in records))
    clean_accepted_count = status_counts.get("accepted", 0)
    borderline_count = status_counts.get("accepted_borderline", 0)
    profile_mismatch_count = status_counts.get("profile_mismatch", 0)
    payload = {
        "dataset_size": len(records),
        "final_export_size": len(final_export_records),
        "clean_accepted_count": clean_accepted_count,
        "borderline_count": borderline_count,
        "constraint_violation_rate": profile_mismatch_count / max(profile_total, 1),
        "status_counts": status_counts,
        "split_counts": {name: len(items) for name, items in splits.items()},
        "score_distribution": score_distribution(scored),
        "low_mid_high_coverage": low_mid_high_counts(scored),
        "remaining_low_mid_high_gaps": low_mid_high_coverage_gaps(scored, min_per_band),
        "weak_aspect_frequency": dict(Counter(aspect for record in scored for aspect in record.weak_aspects)),
        "strong_aspect_frequency": dict(Counter(aspect for record in scored for aspect in record.strong_aspects)),
        "profile_distribution": dict(Counter(record.profile.profile_id for record in records)),
        "profile_success_by_profile": profile_outcomes,
        "profile_success_rate": success / max(profile_total, 1),
        "profile_mismatch_count": profile_mismatch_count,
        "domain_distribution": dict(Counter(record.project_domain for record in final_export_records)),
        "question_type_distribution": dict(Counter(record.question_type for record in final_export_records)),
        "scenario_family_distribution": dict(Counter(record.scenario_family for record in final_export_records)),
        "answer_length": {
            "min": min(word_counts) if word_counts else 0,
            "mean": round(mean(word_counts), 2) if word_counts else 0,
            "max": max(word_counts) if word_counts else 0,
        },
        "labeler_validator_agreement": dict(Counter(str(delta) for record in records for delta in record.validation.score_deltas.values())),
        "deterministic_validation_flags": dict(Counter(flag for record in records for flag in record.validation.flags)),
        "split_leakage_report": leakage_report,
        "ood_count": len(splits.get("ood_test_review_candidates", [])),
        "manual_review_examples": [
            record.example_id
            for record in records
            if record.validation.final_status == "manual_review"
        ][:20],
        "readiness": {
            "usable_for_synthetic_training": any(record.validation.final_status == "accepted" for record in records),
            "usable_for_final_evaluation_without_manual_labels": False,
            "manual_review_required_for_dev_test_ood": True,
        },
        "recommended_next_actions": [
            "Manually review dev/test/OOD candidates before final evaluation.",
            "Regenerate or inspect profile_mismatch examples before using them for training.",
            "Review remaining low/mid/high coverage gaps before scaling beyond the pilot.",
        ],
    }
    lines = [
        "# V4 Data Quality Report",
        "",
        f"- Dataset size: {payload['dataset_size']}",
        f"- Final export size: {payload['final_export_size']}",
        f"- Clean accepted count: {payload['clean_accepted_count']}",
        f"- Borderline count: {payload['borderline_count']}",
        f"- Profile mismatch count: {payload['profile_mismatch_count']}",
        f"- Constraint violation rate: {payload['constraint_violation_rate']:.2%}",
        f"- OOD count: {payload['ood_count']}",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(payload["status_counts"].items()):
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## Score Distribution", ""])
    for aspect, distribution in payload["score_distribution"].items():
        lines.append(f"- {aspect}: {distribution}")
    lines.extend(["", "## Low/Mid/High Coverage", ""])
    for aspect, coverage in payload["low_mid_high_coverage"].items():
        lines.append(f"- {aspect}: {coverage}")
    lines.extend(["", "## Key Target Distributions", ""])
    for aspect in ("role_relevance", "personal_contribution", "clarity"):
        lines.append(f"- {aspect}: {payload['score_distribution'].get(aspect, {})}")
    lines.extend(["", "## Profile Success Rate", "", f"- overall: {payload['profile_success_rate']:.2%}"])
    for profile_id, outcomes in sorted(profile_outcomes.items()):
        lines.append(f"- {profile_id}: {outcomes}")
    lines.extend(["", "## Profile Mismatch Count", "", f"- total: {payload['profile_mismatch_count']}"])
    lines.extend(["", "## Domain Distribution", ""])
    for domain, count in sorted(payload["domain_distribution"].items()):
        lines.append(f"- {domain}: {count}")
    lines.extend(["", "## Question Type Distribution", ""])
    for question_type, count in sorted(payload["question_type_distribution"].items()):
        lines.append(f"- {question_type}: {count}")
    lines.extend(["", "## Scenario Family Distribution", ""])
    for family, count in sorted(payload["scenario_family_distribution"].items()):
        lines.append(f"- {family}: {count}")
    lines.extend(["", "## Split Counts", ""])
    for split_name, count in sorted(payload["split_counts"].items()):
        lines.append(f"- {split_name}: {count}")
    lines.extend(["", "## Split Leakage Report", ""])
    for key, values in leakage_report.items():
        lines.append(f"- {key}: {values}")
    lines.extend(["", "## Answer Length", ""])
    for key, value in payload["answer_length"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Labeler-Validator Agreement", ""])
    for delta, count in sorted(payload["labeler_validator_agreement"].items()):
        lines.append(f"- delta {delta}: {count}")
    lines.extend(["", "## Deterministic Validation Flags", ""])
    for flag, count in sorted(payload["deterministic_validation_flags"].items()):
        lines.append(f"- {flag}: {count}")
    lines.extend(["", "## Remaining Coverage Gaps", ""])
    for aspect, gaps in payload["remaining_low_mid_high_gaps"].items():
        lines.append(f"- {aspect}: {gaps}")
    lines.extend(["", "## Readiness Statement", ""])
    lines.append(f"- Usable for synthetic training: {'yes' if payload['readiness']['usable_for_synthetic_training'] else 'no'}")
    lines.append("- Usable for final evaluation without manual labels: no")
    lines.append("- Manual review required for dev/test/OOD: yes")
    lines.extend(["", "## Recommended Next Actions", ""])
    for action in payload["recommended_next_actions"]:
        lines.append(f"- {action}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return payload
