from __future__ import annotations

import json
import re

from src.config_loader import load_rubrics
from src.labeler import calibrate_scoring_pass, heuristic_label_answer
from src.llm_client import LLMClient
from src.schemas import (
    ASPECTS,
    DATASET_VERSION_V3,
    DATASET_VERSION_V4,
    DATASET_VERSION_OFFICIAL,
    STATUS_ACCEPTED,
    STATUS_ACCEPTED_BORDERLINE,
    STATUS_AUDIT_ONLY,
    STATUS_MANUAL_REVIEW,
    STATUS_PROFILE_MISMATCH,
    STATUS_REJECTED,
    DatasetRecord,
    ScoringPass,
    ValidationMetadata,
    compute_strong_aspects,
    compute_weak_aspects,
    validate_scores,
)
from src.text_features import (
    CONCRETE_PROJECT_OUTCOME_TERMS,
    FIRST_PERSON_ACTIONS,
    IMPLEMENTATION_TERMS,
    IMPACT_TERMS,
    PROBLEM_TERMS,
    PROJECT_TERMS,
    SOFTWARE_DEVELOPMENT_EVIDENCE_TERMS,
    SOLUTION_TERMS,
    TECH_TERMS,
    TECHNICAL_PROBLEM_SOLVING_EVIDENCE_TERMS,
    contains_any,
    contains_positive_any,
    count_any,
    has_metric_outcome,
)


LEAKAGE_TERMS = {
    "technical_depth",
    "personal_contribution",
    "role_relevance",
    "problem_solving",
    "rubric",
    "generation profile",
    "hidden attribute",
    "weak aspect",
    "strong aspect",
    "numeric rating",
}

SCORE_LEAKAGE_PATTERNS = [
    r"\bscore(?:s)?\s+(?:should|would|will|is|are|=|:)\s*(?:be\s*)?[1-5]\b",
    r"\b(?:my|the|this)\s+(?:answer\s+)?score(?:s)?\b",
    r"\b(?:rubric|rating|label)\s+score(?:s)?\b",
    r"\bscore(?:s)?\s+(?:for|on)\s+(?:technical|clarity|impact|role|problem|personal)",
]

IMPLEMENTATION_EVIDENCE_TERMS = [
    "implemented",
    "debugged",
    "tested",
    "technical decision",
    "designed",
    "refactored",
    "integrated",
    "optimized",
    "api",
    "database",
    "backend",
    "frontend",
    "code",
    "logs",
]

CONCRETE_OUTCOME_TERMS = [
    "delivered",
    "finished",
    "completed",
    "improved",
    "reduced",
    "faster",
    "fewer",
    "stable",
    "users",
    "user",
    "milestone",
    "performance",
    "accuracy",
    "load time",
    "response time",
]

OWNED_ACTION_TERMS = [
    "i implemented",
    "i built",
    "i created",
    "i wrote",
    "i tested",
    "i debugged",
    "i fixed",
    "i optimized",
    "i designed",
    "i refactored",
    "i integrated",
    "i owned",
    "i was responsible",
    "my responsibility",
    "my main task",
    "my specific task",
    "i handled",
]


def _has_software_evidence(answer: str) -> bool:
    return contains_positive_any(answer, SOFTWARE_DEVELOPMENT_EVIDENCE_TERMS)


def _has_project_outcome(answer: str) -> bool:
    return contains_positive_any(answer, CONCRETE_PROJECT_OUTCOME_TERMS) or has_metric_outcome(answer)


def _has_technical_problem_solving(answer: str) -> bool:
    return contains_positive_any(answer, TECHNICAL_PROBLEM_SOLVING_EVIDENCE_TERMS)


def _low_score_contradiction_supported(record: DatasetRecord, aspect: str) -> bool:
    answer = record.answer.lower()
    if aspect == "personal_contribution":
        return count_any(answer, FIRST_PERSON_ACTIONS) >= 2
    if aspect == "technical_depth":
        return count_any(answer, TECH_TERMS) >= 2 and count_any(answer, IMPLEMENTATION_TERMS) >= 1
    if aspect == "impact":
        return _has_project_outcome(answer)
    if aspect == "problem_solving":
        return _has_technical_problem_solving(answer)
    if aspect == "clarity":
        return contains_any(answer, PROJECT_TERMS) and count_any(answer, FIRST_PERSON_ACTIONS) > 0 and (
            _has_technical_problem_solving(answer) or _has_project_outcome(answer)
        )
    return True


def _unsupported_high_score_supported(record: DatasetRecord, aspect: str) -> bool:
    answer = record.answer.lower()
    if aspect == "role_relevance":
        return not _has_software_evidence(answer)
    if aspect == "impact":
        return not _has_project_outcome(answer)
    if aspect == "problem_solving":
        return not _has_technical_problem_solving(answer)
    if aspect == "technical_depth":
        return count_any(answer, TECH_TERMS) < 2 and count_any(answer, IMPLEMENTATION_TERMS) == 0
    if aspect == "personal_contribution":
        return count_any(answer, FIRST_PERSON_ACTIONS) == 0
    return True

V3_WEAK_PROFILE_TARGETS = {
    "vague_tool_mentions_without_explanation": "technical_depth",
    "generic_team_summary_no_personal_role": "personal_contribution",
    "unclear_rambling_answer": "clarity",
    "no_outcome_answer": "impact",
    "off_topic_learning_reflection": "role_relevance",
    "irrelevant_project_summary": "role_relevance",
}


def compute_score_deltas(labeler_scores: dict[str, int], validator_scores: dict[str, int]) -> dict[str, int]:
    return {aspect: abs(labeler_scores[aspect] - validator_scores[aspect]) for aspect in ASPECTS}


def detect_label_leakage(answer: str) -> list[str]:
    lowered = answer.lower()
    terms = {term for term in LEAKAGE_TERMS if term in lowered}
    if any(re.search(pattern, lowered) for pattern in SCORE_LEAKAGE_PATTERNS):
        terms.add("score")
    return sorted(terms)


def build_validation_prompt(question: str, answer: str, rubric: dict) -> str:
    return json.dumps(
        {
            "task": "Perform a strict audit of proposed interview-answer quality. Rescore independently using only the question and answer.",
            "question": question,
            "answer": answer,
            "rubric": rubric,
            "audit_instructions": [
                "Challenge unsupported high scores. Prefer lower scores when evidence is vague or missing.",
                "Do not reward fluent wording unless the answer contains concrete evidence.",
                "For scores 4 or 5, quote the observable evidence that justifies the high score.",
                "Generic teamwork, communication, planning, presentation, organization, task tracking, and personal learning do not justify role_relevance 4 or 5 without concrete software-development evidence.",
                "Personal learning alone is not high project impact; impact 4 requires project-level delivery, user/team value, deployment, measurable improvement, performance, accuracy, saved time, reduced errors, or adoption evidence.",
                "Impact 5 requires stronger evidence such as before/after metrics, real user/team/client usage, reduced manual work, QA passing, improved accuracy/performance, or a deployed/used deliverable with clear value.",
                "Vague words like useful, successful, or improved do not justify high impact without observable project-level evidence.",
                "Generic words like problem, solution, issue, or plan are not technical problem-solving evidence without debugging, testing, implementation, root-cause, log, reproduction, or verification details.",
                "For scores 1 or 2, check whether the answer contains contradictory evidence that should raise the score.",
                "Flag label leakage if the candidate answer mentions scoring, rubric language, hidden controls, or aspect names.",
                "Return strict JSON only.",
            ],
            "schema": {
                "scores": {aspect: "integer 1-5" for aspect in ASPECTS},
                "evidence": {aspect: ["quote or absence explanation"] for aspect in ASPECTS},
                "rationale": {aspect: "audit reason for this score" for aspect in ASPECTS},
                "confidence": {aspect: "float 0-1" for aspect in ASPECTS},
                "evidence_checks": {
                    "unsupported_high_scores": ["aspect names where high score lacks evidence"],
                    "low_score_contradictions": ["aspect names where low score conflicts with answer evidence"],
                    "label_leakage_terms": ["leaked terms from answer"],
                },
                "mismatch_flags": ["short audit flags"],
            },
        },
        ensure_ascii=True,
        indent=2,
    )


def _extract_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("validator response did not contain JSON")
    return json.loads(match.group(0))


def parse_validation_scoring_pass(raw_text: str) -> ScoringPass:
    payload = _extract_json_object(raw_text)
    scores = {aspect: int(payload["scores"][aspect]) for aspect in ASPECTS}
    validate_scores(scores)
    rationale = {aspect: str(payload.get("rationale", {}).get(aspect, "")) for aspect in ASPECTS}
    evidence_checks = payload.get("evidence_checks", {})
    mismatch_flags = payload.get("mismatch_flags", [])
    rationale["validator_flags"] = json.dumps(
        {
            "evidence_checks": evidence_checks,
            "mismatch_flags": mismatch_flags,
        },
        ensure_ascii=True,
    )
    return ScoringPass(
        scores=scores,
        evidence={aspect: list(payload.get("evidence", {}).get(aspect, [])) for aspect in ASPECTS},
        rationale=rationale,
        confidence={aspect: float(payload.get("confidence", {}).get(aspect, 0.65)) for aspect in ASPECTS},
    )


def independent_rescore(record: DatasetRecord, mode: str = "heuristic", model: str = "gpt-4.1-mini") -> ScoringPass:
    generated = record_to_generated(record)
    if mode in {"mock", "heuristic"}:
        return heuristic_label_answer(generated)
    prompt = build_validation_prompt(record.question, record.answer, load_rubrics()["aspects"])
    try:
        return calibrate_scoring_pass(generated, parse_validation_scoring_pass(LLMClient(mode="openai", model=model).generate(prompt)))
    except Exception:
        return heuristic_label_answer(generated)


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


def detect_evidence_contradictions(record: DatasetRecord) -> list[str]:
    if not record.labeler.scores:
        return []
    answer = record.answer.lower()
    scores = record.labeler.scores
    contradictions: list[str] = []
    first_person_actions = count_any(answer, FIRST_PERSON_ACTIONS)
    tech_count = count_any(answer, TECH_TERMS)
    implementation_count = count_any(answer, IMPLEMENTATION_TERMS)
    has_project_context = contains_any(answer, PROJECT_TERMS)
    has_problem = contains_any(answer, PROBLEM_TERMS)
    has_solution = contains_any(answer, SOLUTION_TERMS)
    has_impact = _has_project_outcome(answer)

    if scores["personal_contribution"] <= 2 and first_person_actions >= 2:
        contradictions.append("personal_contribution_low_but_first_person_actions_present")
    if scores["technical_depth"] <= 2 and tech_count >= 2 and implementation_count >= 1:
        contradictions.append("technical_depth_low_but_technical_implementation_present")
    if scores["impact"] <= 2 and has_impact:
        contradictions.append("impact_low_but_outcome_present")
    if scores["problem_solving"] <= 2 and has_problem and has_solution and _has_technical_problem_solving(answer):
        contradictions.append("problem_solving_low_but_debugging_process_present")
    if scores["clarity"] <= 2 and has_project_context and first_person_actions and (has_solution or has_impact):
        if not record.validator.scores or record.validator.scores.get("clarity", 1) >= 3:
            contradictions.append("clarity_low_but_answer_has_context_action_result")
    for aspect in ASPECTS:
        if scores[aspect] >= 4 and not record.labeler.evidence.get(aspect):
            contradictions.append(f"{aspect}_high_score_missing_evidence")
    return contradictions


def targeted_weak_failure_flags(record: DatasetRecord) -> list[str]:
    profile_id = record.profile.profile_id
    if profile_id in V3_WEAK_PROFILE_TARGETS and record.profile.desired_quality_hint.startswith("weak_"):
        intended_aspect = V3_WEAK_PROFILE_TARGETS[profile_id]
        if record.labeler.scores.get(intended_aspect, 5) > 2:
            return ["targeted_weak_profile_missed_band"]
        return []
    if not profile_id.startswith("targeted_"):
        return []
    parts = profile_id.split("_")
    if len(parts) < 3:
        return []
    aspect = "_".join(parts[1:-2]) if parts[-2].isdigit() else "_".join(parts[1:-1])
    score_part = parts[-2] if parts[-2].isdigit() else parts[-1]
    if aspect not in ASPECTS or not score_part.isdigit():
        return []
    intended_score = int(score_part)
    if intended_score <= 2 and record.labeler.scores.get(aspect, 5) > 2:
        return ["targeted_weak_profile_missed_band"]
    return []


def deterministic_review_flags(record: DatasetRecord) -> list[str]:
    if not record.labeler.scores:
        return []
    answer = record.answer.lower()
    flags: list[str] = []
    scores = record.labeler.scores
    has_impl = _has_software_evidence(answer)
    has_outcome = _has_project_outcome(answer)
    has_owned_action = contains_any(answer, OWNED_ACTION_TERMS)
    is_general_reflection = (
        record.question_type == "general_reflection"
        or record.project_domain == "general coursework"
        or record.profile.profile_id in {"off_topic_answer", "off_topic_learning_reflection", "irrelevant_project_summary"}
    )
    if scores.get("role_relevance", 0) >= 4 and not has_impl:
        flags.append("role_relevance_high_without_technical_evidence")
    if scores.get("impact", 0) >= 4 and not has_outcome:
        flags.append("impact_high_without_concrete_outcome")
    if scores.get("personal_contribution", 0) >= 3 and not has_owned_action:
        flags.append("personal_contribution_high_without_owned_action")
    if is_general_reflection and (
        scores.get("impact", 0) >= 4
        or scores.get("role_relevance", 0) >= 4
        or (scores.get("personal_contribution", 0) >= 3 and not has_owned_action)
    ):
        flags.append("general_reflection_generous_scores")
    flags.extend(targeted_weak_failure_flags(record))
    return sorted(set(flags))


def profile_success_flags(record: DatasetRecord) -> list[str]:
    if record.dataset_version not in {DATASET_VERSION_V4, DATASET_VERSION_OFFICIAL} or not record.labeler.scores:
        return []
    flags: list[str] = []
    for aspect, constraint in record.profile.expected_score_constraints.items():
        if aspect not in record.labeler.scores:
            continue
        score = record.labeler.scores[aspect]
        max_score = constraint.get("max")
        if max_score is not None:
            if constraint.get("strict") and score > max_score:
                flags.append(f"profile_mismatch:{aspect}")
            elif score >= max_score + 2:
                flags.append(f"profile_mismatch:{aspect}")
            elif score > max_score:
                flags.append(f"profile_borderline:{aspect}")
        min_score = constraint.get("min")
        if min_score is not None:
            if score <= min_score - 2:
                flags.append(f"profile_mismatch:{aspect}")
            elif score < min_score:
                flags.append(f"profile_borderline:{aspect}")
    return sorted(set(flags))


def validator_audit_flags(record: DatasetRecord) -> list[str]:
    raw_flags = record.validator.rationale.get("validator_flags", "")
    if not raw_flags:
        return []
    try:
        payload = json.loads(raw_flags)
    except json.JSONDecodeError:
        return ["validator_flags_unparseable"]
    checks = payload.get("evidence_checks", {})
    flags: list[str] = []
    unsupported_high = checks.get("unsupported_high_scores", [])
    low_contradictions = checks.get("low_score_contradictions", [])
    leakage_terms = checks.get("label_leakage_terms", [])
    for aspect in unsupported_high:
        if (
            aspect in record.labeler.scores
            and record.labeler.scores[aspect] >= 4
            and _unsupported_high_score_supported(record, aspect)
        ):
            flags.append(f"unsupported_high_score:{aspect}")
    for aspect in low_contradictions:
        if (
            aspect in record.labeler.scores
            and record.labeler.scores[aspect] <= 2
            and _low_score_contradiction_supported(record, aspect)
        ):
            flags.append(f"low_score_contradiction:{aspect}")
    for term in leakage_terms:
        flags.append(f"validator_label_leakage:{term}")
    # Free-form validator mismatch notes are useful in the audit trail, but in
    # targeted weak generation they often describe the intended weakness. Let
    # structured high/low evidence checks and deterministic rules control status.
    return flags


def validate_record(record: DatasetRecord, tolerance: int = 1) -> ValidationMetadata:
    reasons: list[str] = []
    flags: list[str] = []
    try:
        validate_scores(record.labeler.scores)
        validate_scores(record.validator.scores)
    except ValueError:
        reasons.append("schema_invalid")
    word_count = len(record.answer.split())
    if not record.answer.strip():
        reasons.append("empty_answer")
    leakage = detect_label_leakage(record.answer)
    deltas = compute_score_deltas(record.labeler.scores, record.validator.scores) if record.labeler.scores and record.validator.scores else {}
    contradictions = detect_evidence_contradictions(record)
    flags.extend(validator_audit_flags(record))
    flags.extend(deterministic_review_flags(record))
    flags.extend(profile_success_flags(record))
    flags = sorted(set(flags))
    is_v3_or_newer = record.dataset_version in {DATASET_VERSION_V3, DATASET_VERSION_V4, DATASET_VERSION_OFFICIAL}

    if leakage:
        status = STATUS_REJECTED if is_v3_or_newer else "rejected_label_leakage"
        reasons.append("label_leakage")
    elif "schema_invalid" in reasons or "empty_answer" in reasons:
        status = STATUS_REJECTED if is_v3_or_newer else "rejected_schema_invalid"
    elif any(delta >= 2 for delta in deltas.values()) or any(delta > tolerance for delta in deltas.values()):
        status = STATUS_REJECTED if is_v3_or_newer else "rejected_score_disagreement"
        reasons.append("score_delta_ge_2")
    elif contradictions:
        status = STATUS_REJECTED if is_v3_or_newer else "rejected_evidence_contradiction"
        reasons.append("evidence_contradiction")
    elif record.dataset_version in {DATASET_VERSION_V4, DATASET_VERSION_OFFICIAL} and any(flag.startswith("profile_mismatch:") for flag in flags):
        status = STATUS_PROFILE_MISMATCH
        reasons.append("profile_mismatch")
    elif record.dataset_version in {DATASET_VERSION_V4, DATASET_VERSION_OFFICIAL} and any(flag.startswith("profile_borderline:") for flag in flags):
        status = STATUS_ACCEPTED_BORDERLINE
        reasons.append("profile_borderline")
    elif record.dataset_version == DATASET_VERSION_V3 and "targeted_weak_profile_missed_band" in flags:
        status = STATUS_AUDIT_ONLY
        reasons.append("targeted_weak_profile_missed_band")
    elif is_v3_or_newer and (
        "role_relevance_high_without_technical_evidence" in flags
        or "personal_contribution_high_without_owned_action" in flags
        or "general_reflection_generous_scores" in flags
    ):
        status = STATUS_MANUAL_REVIEW
        reasons.append("deterministic_manual_review")
    elif is_v3_or_newer and "impact_high_without_concrete_outcome" in flags:
        status = STATUS_ACCEPTED_BORDERLINE
        reasons.append("deterministic_borderline")
    elif flags:
        status = STATUS_MANUAL_REVIEW if is_v3_or_newer else "manual_review_required"
        reasons.append("validator_audit_flag")
    else:
        status = STATUS_ACCEPTED

    return ValidationMetadata(
        final_status=status,
        agreement_tolerance=tolerance,
        rejection_reasons=sorted(set(reasons)),
        label_leakage_terms=leakage,
        evidence_contradictions=contradictions,
        actual_word_count=word_count,
        score_deltas=deltas,
        flags=flags,
    )


def apply_validation(record: DatasetRecord, validation: ValidationMetadata) -> DatasetRecord:
    record.validation = validation
    if validation.final_status in {STATUS_ACCEPTED, STATUS_ACCEPTED_BORDERLINE, STATUS_MANUAL_REVIEW, STATUS_PROFILE_MISMATCH, STATUS_AUDIT_ONLY}:
        record.final_scores = dict(record.labeler.scores)
        record.weak_aspects = compute_weak_aspects(record.final_scores)
        record.strong_aspects = compute_strong_aspects(record.final_scores)
    else:
        record.final_scores = {}
        record.weak_aspects = []
        record.strong_aspects = []
    return record
