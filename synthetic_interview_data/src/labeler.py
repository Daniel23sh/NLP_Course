from __future__ import annotations

import json
import re

from src.config_loader import load_rubrics
from src.llm_client import LLMClient
from src.schemas import ASPECTS, GeneratedAnswer, ScoringPass, validate_scores
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
    contains_any,
    contains_positive_any,
    count_any,
    has_metric_outcome,
)


def build_labeling_prompt(question: str, answer: str, rubric: dict) -> str:
    return json.dumps(
        {
            "task": "Score this junior developer interview answer using only the question and answer.",
            "question": question,
            "answer": answer,
            "rubric": rubric,
            "instructions": [
                "Return strict JSON with scores, evidence, rationale, and confidence.",
                "Do not infer unstated work.",
                "A score of 1 means absent or almost absent.",
                "Scores 4 or 5 require observable evidence.",
                "Generic teamwork, communication, planning, presentation, organization, task tracking, or personal learning does not justify role_relevance 4 or 5 unless connected to concrete software-development evidence such as coding, implementation, debugging, testing, APIs, databases, frontend/backend work, model training, deployment, technical design decisions, architecture, or performance optimization.",
                "Personal learning alone is not high project impact; impact 4 requires concrete project, user, team, deployment, delivery, metric, or performance value.",
                "Impact 5 requires stronger evidence such as a before/after metric, real user or team usage, saved time, reduced errors, improved accuracy or performance, or a deployed/used deliverable with clear value.",
                "Vague words like useful, successful, or improved do not justify high impact unless the answer explains the observable project-level result.",
                "Evidence should quote or summarize answer text. For absent evidence, state the absence.",
            ],
            "schema": {
                "scores": {aspect: "integer 1-5" for aspect in ASPECTS},
                "evidence": {aspect: ["supporting evidence strings"] for aspect in ASPECTS},
                "rationale": {aspect: "short reason" for aspect in ASPECTS},
                "confidence": {aspect: "float 0-1" for aspect in ASPECTS},
            },
        },
        ensure_ascii=True,
        indent=2,
    )


def _extract_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("scoring response did not contain JSON")
    return json.loads(match.group(0))


def parse_scoring_pass(raw_text: str) -> ScoringPass:
    payload = _extract_json_object(raw_text)
    scores = {aspect: int(payload["scores"][aspect]) for aspect in ASPECTS}
    validate_scores(scores)
    return ScoringPass(
        scores=scores,
        evidence={aspect: list(payload.get("evidence", {}).get(aspect, [])) for aspect in ASPECTS},
        rationale={aspect: str(payload.get("rationale", {}).get(aspect, "")) for aspect in ASPECTS},
        confidence={aspect: float(payload.get("confidence", {}).get(aspect, 0.75)) for aspect in ASPECTS},
    )


def _evidence(text: str, terms: list[str], fallback: str) -> list[str]:
    lowered = text.lower()
    found = [term for term in terms if term in lowered]
    return found[:4] or [fallback]


def _low_relevance_profile(record: GeneratedAnswer) -> bool:
    constraint = record.profile.expected_score_constraints.get("role_relevance", {})
    return constraint.get("max", 5) <= 2 or record.profile.desired_quality_hint == "weak_relevance"


def _low_impact_profile(record: GeneratedAnswer) -> bool:
    constraint = record.profile.expected_score_constraints.get("impact", {})
    return constraint.get("max", 5) <= 2 or record.profile.desired_quality_hint == "weak_impact"


def _low_technical_profile(record: GeneratedAnswer) -> bool:
    constraint = record.profile.expected_score_constraints.get("technical_depth", {})
    return constraint.get("max", 5) <= 2 or record.profile.desired_quality_hint == "weak_technical_depth"


def _low_clarity_profile(record: GeneratedAnswer) -> bool:
    constraint = record.profile.expected_score_constraints.get("clarity", {})
    return constraint.get("max", 5) <= 2 or record.profile.desired_quality_hint == "weak_clarity"


def _only_personal_learning_outcome(answer: str) -> bool:
    has_learning = any(term in answer for term in ["i learned", "helped me understand", "personal growth", "valuable experience"])
    return has_learning and not contains_positive_any(answer, CONCRETE_PROJECT_OUTCOME_TERMS) and not has_metric_outcome(answer)


def calibrate_scoring_pass(record: GeneratedAnswer, scoring: ScoringPass) -> ScoringPass:
    if not scoring.scores:
        return scoring
    answer = record.answer.lower()
    scores = dict(scoring.scores)
    rationale = dict(scoring.rationale)
    software_evidence = contains_positive_any(answer, SOFTWARE_DEVELOPMENT_EVIDENCE_TERMS)
    generic_non_software = contains_any(
        answer,
        [
            "communication",
            "planning",
            "presentation",
            "organizing",
            "organization",
            "team activity",
            "class activity",
            "school assignment",
            "volunteer",
            "not a software project",
            "no coding",
            "not software",
            "without implementation",
        ],
    )
    no_personal_markers = [
        "no specific personal task",
        "no separate personal task",
        "did not own a specific task",
        "the team handled the work",
        "mostly watched",
        "mostly listened",
        "listened during meetings",
        "attended meetings",
        "tried to understand",
        "mostly observed",
        "only listened",
        "followed along",
    ]
    if contains_any(answer, no_personal_markers) and count_any(answer, FIRST_PERSON_ACTIONS) == 0:
        scores["personal_contribution"] = 1
        rationale["personal_contribution"] = (
            rationale.get("personal_contribution", "")
            + " Calibrated to 1 because the answer explicitly describes observation without an owned task."
        )
    elif count_any(answer, FIRST_PERSON_ACTIONS) == 0 and contains_any(
        answer,
        ["i helped", "my part was mostly", "my role was mostly", "mostly helping", "assist", "support"],
    ):
        scores["personal_contribution"] = min(scores["personal_contribution"], 2)
        rationale["personal_contribution"] = (
            rationale.get("personal_contribution", "")
            + " Calibrated because helper/support language without owned actions is weak personal contribution."
        )
    if not software_evidence:
        if _low_relevance_profile(record) or generic_non_software:
            scores["role_relevance"] = min(scores["role_relevance"], 2)
        else:
            scores["role_relevance"] = min(scores["role_relevance"], 3)
        rationale["role_relevance"] = (
            rationale.get("role_relevance", "")
            + " Calibrated because generic teamwork/planning without software-development evidence cannot support high role relevance."
        )
    if _only_personal_learning_outcome(answer):
        scores["impact"] = min(scores["impact"], 3)
        rationale["impact"] = (
            rationale.get("impact", "")
            + " Calibrated because personal learning alone is not high project impact."
        )
    if _low_impact_profile(record) and not contains_positive_any(answer, CONCRETE_PROJECT_OUTCOME_TERMS) and not has_metric_outcome(answer):
        scores["impact"] = min(scores["impact"], 2)
    if _low_technical_profile(record) and not contains_positive_any(answer, SOFTWARE_DEVELOPMENT_EVIDENCE_TERMS):
        scores["technical_depth"] = min(scores["technical_depth"], 2)
    if _low_clarity_profile(record) and contains_any(answer, ["hard to explain", "not enough context", "the thing", "some part"]):
        scores["clarity"] = min(scores["clarity"], 2)
    return ScoringPass(
        scores=scores,
        evidence=scoring.evidence,
        rationale=rationale,
        confidence=scoring.confidence,
    )


def heuristic_label_answer(record: GeneratedAnswer) -> ScoringPass:
    answer = record.answer.lower()
    words = len(record.answer.split())
    first_person = count_any(answer, FIRST_PERSON_ACTIONS)
    tech_count = count_any(answer, TECH_TERMS)
    implementation_count = count_any(answer, IMPLEMENTATION_TERMS)
    has_problem = contains_any(answer, PROBLEM_TERMS)
    has_solution = contains_any(answer, SOLUTION_TERMS)
    has_metric = has_metric_outcome(answer)
    has_impact = contains_positive_any(answer, CONCRETE_PROJECT_OUTCOME_TERMS) or has_metric
    has_project = contains_any(answer, PROJECT_TERMS)
    weak_technical_markers = [
        "do not remember the implementation details",
        "do not remember many technical details",
        "do not remember technical details",
        "without technical details",
        "cannot explain the architecture",
        "naming the stack",
        "cannot explain the implementation",
    ]
    weak_relevance_markers = [
        "not describing one specific software project",
        "not describing software work",
        "more about planning and communication than building software",
        "not building software",
        "not a software project",
        "no coding",
        "not about software",
        "not related to software development",
        "presentation for class",
        "presentation order",
        "planning task",
        "planning conversation",
        "design mockup",
        "without implementation",
        "volunteer event",
        "general course",
        "class activity",
    ]
    weak_clarity_markers = [
        "hard to explain the order",
        "different things happened at the same time",
        "kind of about",
        "not enough context",
        "disconnected",
        "hard to explain",
        "do not really remember what the main project was",
        "some part changed",
    ]
    severe_clarity_markers = [
        "do not really remember what the main project was",
        "not enough context to know the task",
        "the order is hard to explain",
    ]
    no_personal_markers = [
        "no specific personal task",
        "no separate personal task",
        "did not own a specific task",
        "the team handled the work",
        "mostly watched",
        "mostly listened",
        "mostly observed",
        "only listened",
        "listened during meetings",
        "attended meetings",
        "tried to understand",
        "followed along",
    ]
    question_terms = [term for term in record.question.lower().split() if len(term) > 5]
    overlap = sum(1 for term in question_terms if term in answer)

    technical_depth = 1
    if tech_count >= 5 and implementation_count >= 2:
        technical_depth = 5
    elif tech_count >= 3 and implementation_count >= 1:
        technical_depth = 4
    elif tech_count >= 2 or implementation_count >= 2:
        technical_depth = 3
    elif tech_count or implementation_count:
        technical_depth = 2
    if contains_any(answer, weak_technical_markers):
        technical_depth = min(technical_depth, 2)

    personal_contribution = 1
    if contains_any(answer, no_personal_markers):
        personal_contribution = 1
    elif first_person >= 4:
        personal_contribution = 5
    elif first_person >= 2:
        personal_contribution = 4
    elif first_person == 1:
        personal_contribution = 3
    elif "team" in answer or "we " in answer:
        personal_contribution = 2

    clarity = 1
    if has_project and words >= 45:
        clarity = 3
    if has_project and first_person and (has_solution or has_impact) and words >= 55:
        clarity = 4
    if "hard to follow" in answer or "not very organized" in answer or contains_any(answer, weak_clarity_markers):
        clarity = min(clarity, 2)
    if contains_any(answer, severe_clarity_markers):
        clarity = 1
    if words >= 90 and clarity >= 4 and "first" in answer and ("after" in answer or "before" in answer):
        clarity = 5

    problem_solving = 1
    if has_problem and has_solution and ("assumption" in answer or "compared" in answer or "one at a time" in answer):
        problem_solving = 5
    elif has_problem and has_solution:
        problem_solving = 4
    elif has_solution:
        problem_solving = 3
    elif has_problem:
        problem_solving = 2

    impact = 1
    if "30 percent" in answer or "measured" in answer or has_metric:
        impact = 5
    elif has_impact:
        impact = 4
    elif "learned" in answer or "move forward" in answer:
        impact = 3
    elif "result" in answer:
        impact = 2

    role_relevance = 1
    has_software_evidence = contains_positive_any(answer, SOFTWARE_DEVELOPMENT_EVIDENCE_TERMS)
    generic_non_software = contains_any(
        answer,
        weak_relevance_markers + ["communication", "planning", "presentation", "school assignment", "volunteer"],
    )
    if has_project and overlap >= 2 and has_software_evidence:
        role_relevance = 5
    elif has_project and overlap >= 1 and has_software_evidence:
        role_relevance = 4
    elif has_project:
        role_relevance = 3
    elif "coursework" in answer:
        role_relevance = 2
    if not has_software_evidence and (generic_non_software or _low_relevance_profile(record)):
        role_relevance = min(role_relevance, 2)

    scores = {
        "technical_depth": technical_depth,
        "personal_contribution": personal_contribution,
        "clarity": clarity,
        "problem_solving": problem_solving,
        "impact": impact,
        "role_relevance": role_relevance,
    }
    evidence = {
        "technical_depth": _evidence(answer, TECH_TERMS + IMPLEMENTATION_TERMS, "No concrete technical evidence."),
        "personal_contribution": _evidence(answer, FIRST_PERSON_ACTIONS, "Personal action is absent or vague."),
        "clarity": ["project context and sequence present" if clarity >= 3 else "Answer organization is weak."],
        "problem_solving": _evidence(answer, PROBLEM_TERMS + SOLUTION_TERMS, "No concrete problem-solving process."),
        "impact": _evidence(answer, IMPACT_TERMS + ["learned", "move forward"], "No clear result or impact."),
        "role_relevance": _evidence(answer, PROJECT_TERMS, "Weak connection to the interview question."),
    }
    rationale = {aspect: f"Heuristic rubric score {scores[aspect]} based on observed answer evidence." for aspect in ASPECTS}
    confidence = {aspect: 0.7 for aspect in ASPECTS}
    return calibrate_scoring_pass(record, ScoringPass(scores=scores, evidence=evidence, rationale=rationale, confidence=confidence))


def label_answer(record: GeneratedAnswer, mode: str, model: str) -> ScoringPass:
    if mode in {"mock", "heuristic"}:
        return heuristic_label_answer(record)
    prompt = build_labeling_prompt(record.question, record.answer, load_rubrics()["aspects"])
    try:
        return calibrate_scoring_pass(record, parse_scoring_pass(LLMClient(mode="openai", model=model).generate(prompt)))
    except Exception:
        return heuristic_label_answer(record)
