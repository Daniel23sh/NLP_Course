from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.config_loader import ROOT, load_generation_config
from src.distribution_analysis import find_underrepresented_bands, low_mid_high_coverage_gaps
from src.export_dataset import export_artifacts, export_official_generation_artifacts, export_v3_artifacts, export_v4_artifacts
from src.generator import generate_answer
from src.labeler import label_answer
from src.profile_sampler import GenerationRequest, sample_generation_requests
from src.schemas import (
    ASPECTS,
    DATASET_VERSION_OFFICIAL,
    DATASET_VERSION_V3,
    DATASET_VERSION_V4,
    STATUS_ACCEPTED,
    STATUS_PROFILE_MISMATCH,
    STATUS_REJECTED,
    DatasetRecord,
)
from src.targeted_generation import (
    official_high_impact_profile,
    official_strong_profile,
    profile_for_underrepresented_band,
    profile_for_v4_weak_aspect,
    v3_weak_profile,
    v4_weak_profile,
)
from src.validator import apply_validation, independent_rescore, validate_record


V4_PATCH_TARGETS = [
    ("role_relevance", "low", 15),
    ("personal_contribution", "exact_1", 10),
    ("clarity", "low", 10),
    ("impact", "low", 10),
    ("technical_depth", "low", 10),
]

OFFICIAL_WEAK_PATCH_TARGETS = [
    ("role_relevance", "low", 20),
    ("personal_contribution", "exact_1", 20),
    ("clarity", "low", 20),
    ("impact", "low", 20),
    ("technical_depth", "low", 20),
    ("problem_solving", "low", 20),
]

OFFICIAL_STRONG_PATCH_TARGETS = [(aspect, "high", 15) for aspect in ASPECTS]
OFFICIAL_HIGH_IMPACT_PATCH_TARGETS = [("impact", "high", 25)]


def _v4_patch_targets(patch_target_n: int | None = None) -> list[tuple[str, str, int]]:
    if patch_target_n is None:
        return list(V4_PATCH_TARGETS)
    return [(aspect, target, patch_target_n) for aspect, target, _required in V4_PATCH_TARGETS]


def _record_from_request(
    request: GenerationRequest,
    mode: str,
    seed: int,
    model: str,
    dataset_version: str = "v2_label_after_generation",
    generation_kind: str | None = None,
) -> DatasetRecord:
    generated = generate_answer(request.profile, request.question_payload, mode=mode, seed=seed, model=model)
    record = DatasetRecord.from_generated(
        generated,
        metadata={
            "seed": seed,
            "generation_backend": mode,
            "model": model,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    record.dataset_version = dataset_version
    if generation_kind:
        record.metadata["generation_kind"] = generation_kind
    return record


def _score_and_validate(
    record: DatasetRecord,
    label_mode: str,
    validator_mode: str,
    model: str,
    tolerance: int,
) -> DatasetRecord:
    generated = record_to_generated(record)
    record.labeler = label_answer(generated, mode=label_mode, model=model)
    record.validator = independent_rescore(record, mode=validator_mode, model=model)
    record.metadata["labeler_backend"] = label_mode
    record.metadata["validator_backend"] = validator_mode
    validation = validate_record(record, tolerance=tolerance)
    return apply_validation(record, validation)


def record_to_generated(record: DatasetRecord):
    from src.schemas import GeneratedAnswer

    return GeneratedAnswer(
        example_id=record.example_id,
        question_id=record.question_id,
        question_type=record.question_type,
        project_domain=record.project_domain,
        question=record.question,
        profile=record.profile,
        answer=record.answer,
        metadata=record.metadata,
    )


def _targeted_requests(
    accepted: list[DatasetRecord],
    target_min_per_score: int,
    max_targeted_rounds: int,
    seed: int,
) -> list[GenerationRequest]:
    requests: list[GenerationRequest] = []
    config = load_generation_config()
    question_bank = config["question_bank"]
    next_index = 1
    for round_index in range(max_targeted_rounds):
        missing = find_underrepresented_bands(accepted, target_min_per_score)
        for aspect, scores in missing.items():
            for score in scores:
                profile = profile_for_underrepresented_band(aspect, score, seed + round_index + next_index)
                question = next((item for item in question_bank if item["project_domain"] in profile.likely_domains), question_bank[0])
                requests.append(
                    GenerationRequest(
                        example_id=f"targeted_{next_index:06d}",
                        question_id=f"q_{question['question_type']}_{question['project_domain'].lower().replace(' ', '_')}",
                        question_type=question["question_type"],
                        project_domain=question["project_domain"],
                        question=question["question"],
                        profile=profile,
                    )
                )
                next_index += 1
    return requests


def _v3_weak_requests(total: int, seed: int) -> list[GenerationRequest]:
    config = load_generation_config()
    question_bank = config["question_bank"]
    plan = [
        ("personal_contribution", 1),
        ("clarity", 1),
        ("clarity", 2),
        ("impact", 1),
        ("impact", 2),
        ("role_relevance", 1),
        ("role_relevance", 2),
        ("technical_depth", 1),
        ("technical_depth", 2),
    ]
    requests: list[GenerationRequest] = []
    for index in range(total):
        aspect, score = plan[index % len(plan)]
        profile = profile_for_underrepresented_band(aspect, score, seed + index)
        question = next((item for item in question_bank if item["project_domain"] in profile.likely_domains), question_bank[0])
        requests.append(
            GenerationRequest(
                example_id=f"v3_weak_{index + 1:06d}",
                question_id=f"q_{question['question_type']}_{question['project_domain'].lower().replace(' ', '_')}",
                question_type=question["question_type"],
                project_domain=question["project_domain"],
                question=question["question"],
                profile=profile,
            )
        )
    return requests


def _v3_ood_requests(total: int, seed: int) -> list[GenerationRequest]:
    config = load_generation_config()
    question_bank = config["question_bank"]
    ood_domains = set(config.get("ood_domains", ["NLP course project", "general coursework"]))
    ood_questions = [item for item in question_bank if item["project_domain"] in ood_domains]
    if not ood_questions:
        ood_questions = [
            {
                "question_type": "general_reflection",
                "project_domain": "general coursework",
                "question": "Tell me about something you learned during your studies.",
            }
        ]
    profiles = [
        v3_weak_profile("off_topic_learning_reflection"),
        v3_weak_profile("irrelevant_project_summary"),
        profile_for_underrepresented_band("technical_depth", 1, seed),
    ]
    requests: list[GenerationRequest] = []
    for index in range(total):
        profile = profiles[index % len(profiles)]
        question = ood_questions[index % len(ood_questions)]
        requests.append(
            GenerationRequest(
                example_id=f"v3_ood_{index + 1:06d}",
                question_id=f"q_{question['question_type']}_{question['project_domain'].lower().replace(' ', '_')}",
                question_type=question["question_type"],
                project_domain=question["project_domain"],
                question=question["question"],
                profile=profile,
            )
        )
    return requests


def _question_id(question: dict[str, str]) -> str:
    return f"q_{question['question_type']}_{question['project_domain'].lower().replace(' ', '_').replace('-', '_')}"


def _non_react_question_for_profile(profile, seed: int) -> dict[str, str]:
    question_bank = load_generation_config()["question_bank"]
    matching = [
        item
        for item in question_bank
        if item["project_domain"] in profile.likely_domains and item["project_domain"] != "React web app"
    ]
    fallback = [item for item in question_bank if item["project_domain"] != "React web app"]
    candidates = matching or fallback or question_bank
    return candidates[seed % len(candidates)]


def _request_for_profile(prefix: str, index: int, profile, question: dict) -> GenerationRequest:
    scenario_family = "|".join(
        [
            question["question_type"],
            question["project_domain"],
            profile.profile_id,
            profile.challenge_type,
            profile.desired_quality_hint,
        ]
    )
    return GenerationRequest(
        example_id=f"{prefix}_{index:06d}",
        question_id=_question_id(question),
        question_type=question["question_type"],
        project_domain=question["project_domain"],
        question=question["question"],
        profile=profile,
        scenario_family=scenario_family,
    )


def _v4_weak_requests(total: int, seed: int) -> list[GenerationRequest]:
    config = load_generation_config()
    question_bank = config["question_bank"]
    plan = [
        "personal_contribution",
        "clarity",
        "clarity",
        "impact",
        "impact",
        "role_relevance",
        "technical_depth",
        "technical_depth",
    ]
    requests: list[GenerationRequest] = []
    for index in range(total):
        aspect = plan[index % len(plan)]
        profile = profile_for_v4_weak_aspect(aspect, seed + index)
        question = next((item for item in question_bank if item["project_domain"] in profile.likely_domains), question_bank[0])
        requests.append(_request_for_profile("v4_weak", index + 1, profile, question))
    return requests


def _v4_ood_requests(total: int, seed: int) -> list[GenerationRequest]:
    config = load_generation_config()
    question_bank = config["question_bank"]
    ood_domains = set(config.get("ood_domains", ["NLP course project", "general coursework"]))
    ood_questions = [item for item in question_bank if item["project_domain"] in ood_domains]
    if not ood_questions:
        ood_questions = [
            {
                "question_type": "general_reflection",
                "project_domain": "general coursework",
                "question": "Tell me about something you learned during your studies.",
            }
        ]
    profiles = [
        v4_weak_profile("irrelevant_non_software_project"),
        v4_weak_profile("vague_tools_no_explanation"),
        profile_for_underrepresented_band("problem_solving", 1, seed),
    ]
    requests: list[GenerationRequest] = []
    for index in range(total):
        profile = profiles[index % len(profiles)]
        question = ood_questions[index % len(ood_questions)]
        requests.append(_request_for_profile("v4_ood", index + 1, profile, question))
    return requests


def _v4_patch_profile(aspect: str, seed: int):
    if aspect == "role_relevance":
        return profile_for_v4_weak_aspect("role_relevance", seed)
    if aspect == "personal_contribution":
        return v4_weak_profile("observer_no_personal_task")
    if aspect == "clarity":
        return v4_weak_profile("severely_unclear_but_realistic_answer")
    if aspect == "impact":
        return v4_weak_profile("unfinished_project_no_result" if seed % 2 else "no_outcome_answer")
    if aspect == "technical_depth":
        return v4_weak_profile("vague_tools_no_explanation")
    if aspect == "problem_solving":
        profile = profile_for_underrepresented_band("problem_solving", 1, seed)
        profile.profile_id = f"official_weak_problem_solving_{seed}"
        profile.profile_group = "official_weak"
        profile.challenge_type = "overview"
        profile.desired_quality_hint = "weak_problem_solving"
        profile.expected_score_constraints = {"problem_solving": {"max": 2}}
        return profile
    return profile_for_v4_weak_aspect(aspect, seed)


def _matches_v4_patch_target(record: DatasetRecord, aspect: str, target: str) -> bool:
    if not record.final_scores or aspect not in record.final_scores:
        return False
    score = record.final_scores[aspect]
    if target == "exact_1":
        return score == 1
    if target == "low":
        return score <= 2
    if target == "high":
        return score >= 4
    return False


def _mark_v4_patch_profile_mismatch(record: DatasetRecord, aspect: str, target: str) -> None:
    record.validation.final_status = STATUS_PROFILE_MISMATCH
    reason = f"targeted_patch_missed_required_band:{aspect}:{target}"
    if reason not in record.validation.rejection_reasons:
        record.validation.rejection_reasons.append(reason)
    if reason not in record.validation.flags:
        record.validation.flags.append(reason)
    record.split = "profile_mismatch"


def _resolve_output_dir(root: Path, output_dir: Path | str | None, generation_kind: str) -> Path:
    if output_dir is None:
        return root / "data" / "raw" / "generated" / generation_kind
    path = Path(output_dir)
    return path if path.is_absolute() else root / path


def _official_targets(generation_kind: str, patch_target_n: int | None) -> list[tuple[str, str, int]]:
    if generation_kind == "high_impact_patch":
        targets = OFFICIAL_HIGH_IMPACT_PATCH_TARGETS
    elif generation_kind == "strong_patch":
        targets = OFFICIAL_STRONG_PATCH_TARGETS
    else:
        targets = OFFICIAL_WEAK_PATCH_TARGETS
    if patch_target_n is None:
        return list(targets)
    return [(aspect, target, patch_target_n) for aspect, target, _required in targets]


def _official_profile_for_target(generation_kind: str, aspect: str, seed: int):
    if generation_kind == "high_impact_patch":
        return official_high_impact_profile(seed)
    if generation_kind == "strong_patch":
        return official_strong_profile(aspect, seed)
    return _v4_patch_profile(aspect, seed)


def _mark_official_metadata(records: list[DatasetRecord], generation_kind: str) -> None:
    for record in records:
        record.dataset_version = DATASET_VERSION_OFFICIAL
        record.metadata["generation_kind"] = generation_kind


def _trim_v4_requests(
    base: list[GenerationRequest],
    weak: list[GenerationRequest],
    ood: list[GenerationRequest],
    target_size_max: int,
) -> list[GenerationRequest]:
    if len(base) + len(weak) + len(ood) <= target_size_max:
        return base + weak + ood
    kept: list[GenerationRequest] = []
    kept.extend(ood[:target_size_max])
    remaining = target_size_max - len(kept)
    kept.extend(weak[:max(remaining, 0)])
    remaining = target_size_max - len(kept)
    kept.extend(base[:max(remaining, 0)])
    return kept


def run_pipeline(
    mode: str,
    n: int,
    seed: int,
    root_dir: Path | str | None = None,
    model: str = "gpt-4.1-mini",
    rescore_mode: str | None = None,
    target_min_per_score: int = 25,
    max_targeted_rounds: int = 2,
    agreement_tolerance: int = 1,
    manual_review_size: int = 120,
    ood_domains: str | set[str] | None = None,
    dataset_version: str = "v2_label_after_generation",
    weak_target_n: int = 90,
    ood_n: int = 35,
    target_size_min: int = 200,
    target_size_max: int = 300,
    domain_cap: float = 0.20,
    question_type_cap: float = 0.25,
    profile_cap: float = 0.15,
    targeted_patch: bool = False,
    patch_target_n: int | None = None,
    patch_max_attempts_per_target: int | None = None,
    generation_kind: str = "broad",
    output_dir: Path | str | None = None,
) -> dict:
    root = Path(root_dir) if root_dir is not None else ROOT
    validator_mode = rescore_mode or ("openai" if mode == "openai" else "heuristic")
    label_mode = "openai" if mode == "openai" else "heuristic"
    configured_ood = set(load_generation_config().get("ood_domains", ["NLP course project"]))
    if isinstance(ood_domains, str):
        configured_ood = {item.strip() for item in ood_domains.split(",") if item.strip()}
    elif ood_domains:
        configured_ood = set(ood_domains)

    if dataset_version == DATASET_VERSION_OFFICIAL:
        return run_official_generation_pipeline(
            mode=mode,
            n=n,
            seed=seed,
            root_dir=root,
            model=model,
            rescore_mode=rescore_mode,
            agreement_tolerance=agreement_tolerance,
            generation_kind=generation_kind,
            output_dir=output_dir,
            targeted_patch=targeted_patch,
            patch_target_n=patch_target_n,
            patch_max_attempts_per_target=patch_max_attempts_per_target,
        )

    if dataset_version == DATASET_VERSION_V3:
        return run_v3_pipeline(
            mode=mode,
            n=n,
            seed=seed,
            root_dir=root,
            model=model,
            rescore_mode=rescore_mode,
            target_min_per_score=target_min_per_score,
            agreement_tolerance=agreement_tolerance,
            manual_review_size=manual_review_size,
            ood_domains=configured_ood,
            weak_target_n=weak_target_n,
            ood_n=ood_n,
            target_size_min=target_size_min,
            target_size_max=target_size_max,
            domain_cap=domain_cap,
        )
    if dataset_version == DATASET_VERSION_V4:
        return run_v4_pipeline(
            mode=mode,
            n=n,
            seed=seed,
            root_dir=root,
            model=model,
            rescore_mode=rescore_mode,
            target_min_per_score=target_min_per_score,
            agreement_tolerance=agreement_tolerance,
            manual_review_size=manual_review_size,
            ood_domains=configured_ood,
            weak_target_n=weak_target_n,
            ood_n=ood_n,
            target_size_min=target_size_min,
            target_size_max=target_size_max,
            domain_cap=domain_cap,
            question_type_cap=question_type_cap,
            profile_cap=profile_cap,
            targeted_patch=targeted_patch,
            patch_target_n=patch_target_n,
            patch_max_attempts_per_target=patch_max_attempts_per_target,
        )

    generated_records: list[DatasetRecord] = []
    labeled_records: list[DatasetRecord] = []
    accepted: list[DatasetRecord] = []
    rejected: list[DatasetRecord] = []

    for index, request in enumerate(sample_generation_requests(n, seed), start=1):
        record = _record_from_request(request, mode=mode, seed=seed + index, model=model)
        generated_records.append(record)
        scored = _score_and_validate(record, label_mode, validator_mode, model, agreement_tolerance)
        labeled_records.append(scored)
        if scored.validation.final_status == "accepted":
            accepted.append(scored)
        else:
            rejected.append(scored)

    targeted = _targeted_requests(accepted, target_min_per_score, max_targeted_rounds, seed)
    for index, request in enumerate(targeted, start=1):
        record = _record_from_request(request, mode=mode, seed=seed + 100000 + index, model=model)
        generated_records.append(record)
        scored = _score_and_validate(record, label_mode, validator_mode, model, agreement_tolerance)
        labeled_records.append(scored)
        if scored.validation.final_status == "accepted":
            accepted.append(scored)
        else:
            rejected.append(scored)

    export = export_artifacts(
        root_dir=root,
        generated=generated_records,
        labeled=labeled_records,
        accepted=accepted,
        rejected=rejected,
        seed=seed,
        manual_review_size=manual_review_size,
        ood_domains=configured_ood,
        min_per_score=target_min_per_score,
    )
    underrepresented = find_underrepresented_bands(accepted, target_min_per_score)
    return {
        "requested_initial": n,
        "generated_initial": n,
        "generated_targeted": len(targeted),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": len(accepted) / max(len(accepted) + len(rejected), 1),
        "manual_review_candidates": export["report"].manual_review_candidates,
        "underrepresented_after_targeting": underrepresented,
        "output_dir": str(root / "data" / "final"),
    }


def run_official_generation_pipeline(
    mode: str,
    n: int,
    seed: int,
    root_dir: Path,
    model: str,
    rescore_mode: str | None,
    agreement_tolerance: int,
    generation_kind: str,
    output_dir: Path | str | None,
    targeted_patch: bool,
    patch_target_n: int | None,
    patch_max_attempts_per_target: int | None,
) -> dict:
    if generation_kind not in {"broad", "weak_patch", "strong_patch", "high_impact_patch"}:
        raise ValueError("generation_kind must be broad, weak_patch, strong_patch, or high_impact_patch")
    validator_mode = rescore_mode or ("openai" if mode == "openai" else "heuristic")
    label_mode = "openai" if mode == "openai" else "heuristic"
    destination = _resolve_output_dir(root_dir, output_dir, generation_kind)
    if generation_kind in {"weak_patch", "strong_patch", "high_impact_patch"} or targeted_patch:
        return run_official_targeted_generation_pipeline(
            mode=mode,
            seed=seed,
            output_dir=destination,
            model=model,
            label_mode=label_mode,
            validator_mode=validator_mode,
            agreement_tolerance=agreement_tolerance,
            generation_kind=generation_kind,
            patch_target_n=patch_target_n,
            patch_max_attempts_per_target=patch_max_attempts_per_target,
        )

    generated_records: list[DatasetRecord] = []
    labeled_records: list[DatasetRecord] = []
    all_records: list[DatasetRecord] = []
    for index, request in enumerate(sample_generation_requests(n, seed), start=1):
        record = _record_from_request(
            request,
            mode=mode,
            seed=seed + index,
            model=model,
            dataset_version=DATASET_VERSION_OFFICIAL,
            generation_kind=generation_kind,
        )
        generated_records.append(record)
        scored = _score_and_validate(record, label_mode, validator_mode, model, agreement_tolerance)
        labeled_records.append(scored)
        all_records.append(scored)
    _mark_official_metadata(generated_records + labeled_records + all_records, generation_kind)
    export_official_generation_artifacts(
        output_dir=destination,
        generated=generated_records,
        labeled=labeled_records,
        records=all_records,
    )
    status_counts: dict[str, int] = {}
    for record in all_records:
        status_counts[record.validation.final_status] = status_counts.get(record.validation.final_status, 0) + 1
    return {
        "dataset_version": DATASET_VERSION_OFFICIAL,
        "generation_kind": generation_kind,
        "requested_initial": n,
        "total_generated": len(generated_records),
        "status_counts": status_counts,
        "output_dir": str(destination),
    }


def run_official_targeted_generation_pipeline(
    mode: str,
    seed: int,
    output_dir: Path,
    model: str,
    label_mode: str,
    validator_mode: str,
    agreement_tolerance: int,
    generation_kind: str,
    patch_target_n: int | None,
    patch_max_attempts_per_target: int | None,
) -> dict:
    generated_records: list[DatasetRecord] = []
    labeled_records: list[DatasetRecord] = []
    all_records: list[DatasetRecord] = []
    patch_targets = _official_targets(generation_kind, patch_target_n)
    clean_target_hits = {f"{aspect}:{target}": 0 for aspect, target, _required in patch_targets}
    attempted_by_target = {f"{aspect}:{target}": 0 for aspect, target, _required in patch_targets}
    next_index = 1

    for aspect, target, required_clean in patch_targets:
        target_key = f"{aspect}:{target}"
        max_attempts = patch_max_attempts_per_target or required_clean * 3
        while clean_target_hits[target_key] < required_clean and attempted_by_target[target_key] < max_attempts:
            attempted_by_target[target_key] += 1
            profile = _official_profile_for_target(generation_kind, aspect, seed + next_index)
            question = _non_react_question_for_profile(profile, seed + next_index)
            prefix = f"official_{generation_kind}_{aspect}"
            request = _request_for_profile(prefix, next_index, profile, question)
            record = _record_from_request(
                request,
                mode=mode,
                seed=seed + next_index,
                model=model,
                dataset_version=DATASET_VERSION_OFFICIAL,
                generation_kind=generation_kind,
            )
            generated_records.append(record)
            scored = _score_and_validate(record, label_mode, validator_mode, model, agreement_tolerance)
            if (
                scored.validation.final_status != STATUS_REJECTED
                and not _matches_v4_patch_target(scored, aspect, target)
            ):
                _mark_v4_patch_profile_mismatch(scored, aspect, target)
            if scored.validation.final_status == STATUS_ACCEPTED and _matches_v4_patch_target(scored, aspect, target):
                clean_target_hits[target_key] += 1
            labeled_records.append(scored)
            all_records.append(scored)
            next_index += 1

    _mark_official_metadata(generated_records + labeled_records + all_records, generation_kind)
    export_official_generation_artifacts(
        output_dir=output_dir,
        generated=generated_records,
        labeled=labeled_records,
        records=all_records,
    )
    status_counts: dict[str, int] = {}
    for record in all_records:
        status_counts[record.validation.final_status] = status_counts.get(record.validation.final_status, 0) + 1
    return {
        "dataset_version": DATASET_VERSION_OFFICIAL,
        "generation_kind": generation_kind,
        "targeted_patch": generation_kind in {"weak_patch", "strong_patch", "high_impact_patch"},
        "total_generated": len(generated_records),
        "target_attempts": attempted_by_target,
        "clean_target_hits": clean_target_hits,
        "status_counts": status_counts,
        "output_dir": str(output_dir),
    }


def run_v3_pipeline(
    mode: str,
    n: int,
    seed: int,
    root_dir: Path,
    model: str,
    rescore_mode: str | None,
    target_min_per_score: int,
    agreement_tolerance: int,
    manual_review_size: int,
    ood_domains: set[str],
    weak_target_n: int,
    ood_n: int,
    target_size_min: int,
    target_size_max: int,
    domain_cap: float,
) -> dict:
    validator_mode = rescore_mode or ("openai" if mode == "openai" else "heuristic")
    label_mode = "openai" if mode == "openai" else "heuristic"
    requests = sample_generation_requests(n, seed)
    requests.extend(_v3_weak_requests(weak_target_n, seed + 1000))
    requests.extend(_v3_ood_requests(ood_n, seed + 2000))
    requests = requests[: max(target_size_max, n)]
    generated_records: list[DatasetRecord] = []
    labeled_records: list[DatasetRecord] = []
    all_records: list[DatasetRecord] = []
    for index, request in enumerate(requests, start=1):
        record = _record_from_request(request, mode=mode, seed=seed + index, model=model, dataset_version=DATASET_VERSION_V3)
        generated_records.append(record)
        scored = _score_and_validate(record, label_mode, validator_mode, model, agreement_tolerance)
        labeled_records.append(scored)
        all_records.append(scored)
    export = export_v3_artifacts(
        root_dir=root_dir,
        generated=generated_records,
        labeled=labeled_records,
        records=all_records,
        seed=seed,
        manual_review_size=manual_review_size,
        ood_domains=ood_domains,
        min_per_score=target_min_per_score,
        domain_cap=domain_cap,
        target_size_min=target_size_min,
    )
    status_counts: dict[str, int] = {}
    for record in all_records:
        status_counts[record.validation.final_status] = status_counts.get(record.validation.final_status, 0) + 1
    underrepresented = find_underrepresented_bands([record for record in all_records if record.final_scores], target_min_per_score)
    return {
        "dataset_version": DATASET_VERSION_V3,
        "requested_initial": n,
        "generated_initial": n,
        "generated_weak_targeted": weak_target_n,
        "generated_ood": ood_n,
        "total_generated": len(generated_records),
        "status_counts": status_counts,
        "manual_review_candidates": export["report"].manual_review_candidates,
        "underrepresented_after_targeting": underrepresented,
        "target_size_min": target_size_min,
        "target_size_max": target_size_max,
        "output_dir": str(root_dir / "data" / "v3" / "final"),
    }


def run_v4_targeted_patch_pipeline(
    mode: str,
    seed: int,
    root_dir: Path,
    model: str,
    label_mode: str,
    validator_mode: str,
    agreement_tolerance: int,
    manual_review_size: int,
    ood_domains: set[str],
    target_min_per_score: int,
    target_size_min: int,
    target_size_max: int,
    domain_cap: float,
    question_type_cap: float,
    profile_cap: float,
    patch_target_n: int | None,
    patch_max_attempts_per_target: int | None,
) -> dict:
    generated_records: list[DatasetRecord] = []
    labeled_records: list[DatasetRecord] = []
    all_records: list[DatasetRecord] = []
    patch_targets = _v4_patch_targets(patch_target_n)
    clean_target_hits = {f"{aspect}:{target}": 0 for aspect, target, _required in patch_targets}
    attempted_by_target = {f"{aspect}:{target}": 0 for aspect, target, _required in patch_targets}
    next_index = 1

    for aspect, target, required_clean in patch_targets:
        target_key = f"{aspect}:{target}"
        max_attempts = patch_max_attempts_per_target or required_clean * 3
        while clean_target_hits[target_key] < required_clean and attempted_by_target[target_key] < max_attempts:
            attempted_by_target[target_key] += 1
            profile = _v4_patch_profile(aspect, seed + next_index)
            question = _non_react_question_for_profile(profile, seed + next_index)
            request = _request_for_profile(f"v4_patch_{aspect}", next_index, profile, question)
            record = _record_from_request(
                request,
                mode=mode,
                seed=seed + next_index,
                model=model,
                dataset_version=DATASET_VERSION_V4,
            )
            generated_records.append(record)
            scored = _score_and_validate(record, label_mode, validator_mode, model, agreement_tolerance)
            if (
                scored.validation.final_status != STATUS_REJECTED
                and not _matches_v4_patch_target(scored, aspect, target)
            ):
                _mark_v4_patch_profile_mismatch(scored, aspect, target)
            if scored.validation.final_status == STATUS_ACCEPTED and _matches_v4_patch_target(scored, aspect, target):
                clean_target_hits[target_key] += 1
            labeled_records.append(scored)
            all_records.append(scored)
            next_index += 1

    coverage_target = min(target_min_per_score, 20)
    export = export_v4_artifacts(
        root_dir=root_dir,
        generated=generated_records,
        labeled=labeled_records,
        records=all_records,
        seed=seed,
        manual_review_size=manual_review_size,
        ood_domains=ood_domains,
        min_per_band=coverage_target,
        domain_cap=domain_cap,
        question_type_cap=question_type_cap,
        profile_cap=profile_cap,
        target_size_min=target_size_min,
        target_size_max=target_size_max,
    )
    status_counts: dict[str, int] = {}
    for record in all_records:
        status_counts[record.validation.final_status] = status_counts.get(record.validation.final_status, 0) + 1
    return {
        "dataset_version": DATASET_VERSION_V4,
        "targeted_patch": True,
        "requested_initial": 0,
        "generated_initial": 0,
        "total_generated": len(generated_records),
        "target_attempts": attempted_by_target,
        "clean_target_hits": clean_target_hits,
        "status_counts": status_counts,
        "manual_review_candidates": {
            "dev": len(export["splits"]["dev_review_candidates"]),
            "test": len(export["splits"]["test_review_candidates"]),
            "ood_test": len(export["splits"]["ood_test_review_candidates"]),
        },
        "remaining_low_mid_high_gaps": export["quality"]["remaining_low_mid_high_gaps"],
        "split_leakage_report": export["leakage"],
        "target_size_min": target_size_min,
        "target_size_max": target_size_max,
        "output_dir": str(root_dir / "data" / "v4" / "final"),
        "quality_report": str(root_dir / "reports" / "data_quality_v4.md"),
    }


def run_v4_pipeline(
    mode: str,
    n: int,
    seed: int,
    root_dir: Path,
    model: str,
    rescore_mode: str | None,
    target_min_per_score: int,
    agreement_tolerance: int,
    manual_review_size: int,
    ood_domains: set[str],
    weak_target_n: int,
    ood_n: int,
    target_size_min: int,
    target_size_max: int,
    domain_cap: float,
    question_type_cap: float,
    profile_cap: float,
    targeted_patch: bool = False,
    patch_target_n: int | None = None,
    patch_max_attempts_per_target: int | None = None,
) -> dict:
    validator_mode = rescore_mode or ("openai" if mode == "openai" else "heuristic")
    label_mode = "openai" if mode == "openai" else "heuristic"
    if targeted_patch:
        return run_v4_targeted_patch_pipeline(
            mode=mode,
            seed=seed,
            root_dir=root_dir,
            model=model,
            label_mode=label_mode,
            validator_mode=validator_mode,
            agreement_tolerance=agreement_tolerance,
            manual_review_size=manual_review_size,
            ood_domains=ood_domains,
            target_min_per_score=target_min_per_score,
            target_size_min=target_size_min,
            target_size_max=target_size_max,
            domain_cap=domain_cap,
            question_type_cap=question_type_cap,
            profile_cap=profile_cap,
            patch_target_n=patch_target_n,
            patch_max_attempts_per_target=patch_max_attempts_per_target,
        )
    base_requests = sample_generation_requests(n, seed)
    weak_requests = _v4_weak_requests(weak_target_n, seed + 1000)
    ood_requests = _v4_ood_requests(ood_n, seed + 2000)
    requests = _trim_v4_requests(base_requests, weak_requests, ood_requests, target_size_max)
    generated_records: list[DatasetRecord] = []
    labeled_records: list[DatasetRecord] = []
    all_records: list[DatasetRecord] = []

    for index, request in enumerate(requests, start=1):
        record = _record_from_request(request, mode=mode, seed=seed + index, model=model, dataset_version=DATASET_VERSION_V4)
        generated_records.append(record)
        scored = _score_and_validate(record, label_mode, validator_mode, model, agreement_tolerance)
        labeled_records.append(scored)
        all_records.append(scored)

    coverage_target = min(target_min_per_score, 20)
    next_index = len(requests) + 1
    while len(all_records) < target_size_max:
        training_like = [
            record
            for record in all_records
            if record.final_scores and record.validation.final_status in {"accepted", "accepted_borderline", "manual_review"}
        ]
        gaps = low_mid_high_coverage_gaps(training_like, min_per_band=coverage_target)
        missing_aspect = next((aspect for aspect in ASPECTS if "low" in gaps.get(aspect, [])), None)
        if not missing_aspect:
            break
        profile = profile_for_v4_weak_aspect(missing_aspect, seed + next_index)
        question_bank = load_generation_config()["question_bank"]
        question = next((item for item in question_bank if item["project_domain"] in profile.likely_domains), question_bank[0])
        request = _request_for_profile("v4_extra", next_index, profile, question)
        record = _record_from_request(request, mode=mode, seed=seed + next_index, model=model, dataset_version=DATASET_VERSION_V4)
        generated_records.append(record)
        scored = _score_and_validate(record, label_mode, validator_mode, model, agreement_tolerance)
        labeled_records.append(scored)
        all_records.append(scored)
        next_index += 1

    export = export_v4_artifacts(
        root_dir=root_dir,
        generated=generated_records,
        labeled=labeled_records,
        records=all_records,
        seed=seed,
        manual_review_size=manual_review_size,
        ood_domains=ood_domains,
        min_per_band=coverage_target,
        domain_cap=domain_cap,
        question_type_cap=question_type_cap,
        profile_cap=profile_cap,
        target_size_min=target_size_min,
        target_size_max=target_size_max,
    )
    status_counts: dict[str, int] = {}
    for record in all_records:
        status_counts[record.validation.final_status] = status_counts.get(record.validation.final_status, 0) + 1
    return {
        "dataset_version": DATASET_VERSION_V4,
        "requested_initial": n,
        "generated_initial": len(base_requests[:n]),
        "generated_weak_targeted": weak_target_n,
        "generated_ood": ood_n,
        "total_generated": len(generated_records),
        "status_counts": status_counts,
        "manual_review_candidates": {
            "dev": len(export["splits"]["dev_review_candidates"]),
            "test": len(export["splits"]["test_review_candidates"]),
            "ood_test": len(export["splits"]["ood_test_review_candidates"]),
        },
        "remaining_low_mid_high_gaps": export["quality"]["remaining_low_mid_high_gaps"],
        "split_leakage_report": export["leakage"],
        "target_size_min": target_size_min,
        "target_size_max": target_size_max,
        "output_dir": str(root_dir / "data" / "v4" / "final"),
        "quality_report": str(root_dir / "reports" / "data_quality_v4.md"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate label-after-generation synthetic interview data.")
    parser.add_argument("--mode", choices=["mock", "openai"], required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--rescore-mode", choices=["heuristic", "openai"], default=None)
    parser.add_argument("--target-min-per-score", type=int, default=20)
    parser.add_argument("--max-targeted-rounds", type=int, default=2)
    parser.add_argument("--agreement-tolerance", type=int, default=1)
    parser.add_argument("--manual-review-size", type=int, default=120)
    parser.add_argument("--ood-domains", default=None)
    parser.add_argument("--dataset-version", choices=["v2_label_after_generation", "v3", "v4", "official_v1"], default="v2_label_after_generation")
    parser.add_argument("--generation-kind", choices=["broad", "weak_patch", "strong_patch", "high_impact_patch"], default="broad")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--weak-target-n", type=int, default=90)
    parser.add_argument("--ood-n", type=int, default=35)
    parser.add_argument("--target-size-min", type=int, default=200)
    parser.add_argument("--target-size-max", type=int, default=300)
    parser.add_argument("--domain-cap", type=float, default=0.20)
    parser.add_argument("--question-type-cap", type=float, default=0.25)
    parser.add_argument("--profile-cap", type=float, default=0.15)
    parser.add_argument("--targeted-patch", action="store_true")
    parser.add_argument("--patch-target-n", type=int, default=None)
    parser.add_argument("--patch-max-attempts-per-target", type=int, default=None)
    args = parser.parse_args()
    result = run_pipeline(
        mode=args.mode,
        n=args.n,
        seed=args.seed,
        model=args.model,
        rescore_mode=args.rescore_mode,
        target_min_per_score=args.target_min_per_score,
        max_targeted_rounds=args.max_targeted_rounds,
        agreement_tolerance=args.agreement_tolerance,
        manual_review_size=args.manual_review_size,
        ood_domains=args.ood_domains,
        dataset_version=args.dataset_version,
        weak_target_n=args.weak_target_n,
        ood_n=args.ood_n,
        target_size_min=args.target_size_min,
        target_size_max=args.target_size_max,
        domain_cap=args.domain_cap,
        question_type_cap=args.question_type_cap,
        profile_cap=args.profile_cap,
        targeted_patch=args.targeted_patch,
        patch_target_n=args.patch_target_n,
        patch_max_attempts_per_target=args.patch_max_attempts_per_target,
        generation_kind=args.generation_kind,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
