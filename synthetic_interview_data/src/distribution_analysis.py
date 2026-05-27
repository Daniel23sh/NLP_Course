from __future__ import annotations

from src.schemas import ASPECTS, DatasetRecord, DistributionReport


def score_distribution(records: list[DatasetRecord]) -> dict[str, dict[int, int]]:
    distribution = {aspect: {score: 0 for score in range(1, 6)} for aspect in ASPECTS}
    for record in records:
        for aspect, score in record.final_scores.items():
            distribution[aspect][score] += 1
    return distribution


def find_underrepresented_bands(records: list[DatasetRecord], min_per_score: int) -> dict[str, list[int]]:
    distribution = score_distribution(records)
    return {
        aspect: [score for score, count in counts.items() if count < min_per_score]
        for aspect, counts in distribution.items()
    }


def low_mid_high_counts(records: list[DatasetRecord]) -> dict[str, dict[str, int]]:
    counts = {aspect: {"low": 0, "mid": 0, "high": 0} for aspect in ASPECTS}
    for record in records:
        for aspect, score in record.final_scores.items():
            if score <= 2:
                counts[aspect]["low"] += 1
            elif score == 3:
                counts[aspect]["mid"] += 1
            else:
                counts[aspect]["high"] += 1
    return counts


def low_mid_high_coverage_gaps(records: list[DatasetRecord], min_per_band: int) -> dict[str, list[str]]:
    counts = low_mid_high_counts(records)
    return {
        aspect: [band for band, count in band_counts.items() if count < min_per_band]
        for aspect, band_counts in counts.items()
    }


def _count_by(records: list[DatasetRecord], attr: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for record in records:
        value = getattr(record, attr)
        values[value] = values.get(value, 0) + 1
    return values


def _validation_reason_counts(records: list[DatasetRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for reason in record.validation.rejection_reasons or [record.validation.final_status]:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _delta_distribution(records: list[DatasetRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for delta in record.validation.score_deltas.values():
            key = str(delta)
            counts[key] = counts.get(key, 0) + 1
    return counts


def build_distribution_report(
    accepted: list[DatasetRecord],
    rejected: list[DatasetRecord],
    min_per_score: int = 1,
    manual_review_candidates: dict[str, int] | None = None,
) -> DistributionReport:
    generated = len(accepted) + len(rejected)
    weak = {aspect: 0 for aspect in ASPECTS}
    strong = {aspect: 0 for aspect in ASPECTS}
    profiles: dict[str, int] = {}
    for record in accepted:
        profiles[record.profile.profile_id] = profiles.get(record.profile.profile_id, 0) + 1
        for aspect in record.weak_aspects:
            weak[aspect] += 1
        for aspect in record.strong_aspects:
            strong[aspect] += 1
    return DistributionReport(
        generated_examples=generated,
        accepted_examples=len(accepted),
        rejected_examples=len(rejected),
        acceptance_rate=len(accepted) / max(generated, 1),
        score_distribution=score_distribution(accepted),
        weak_aspect_frequency=weak,
        strong_aspect_frequency=strong,
        profile_distribution=profiles,
        question_type_distribution=_count_by(accepted, "question_type"),
        project_domain_distribution=_count_by(accepted, "project_domain"),
        rejection_reasons=_validation_reason_counts(rejected),
        agreement_delta_distribution=_delta_distribution(accepted + rejected),
        underrepresented_bands=find_underrepresented_bands(accepted, min_per_score),
        manual_review_candidates=manual_review_candidates or {"dev": 0, "test": 0, "ood_test": 0},
    )
