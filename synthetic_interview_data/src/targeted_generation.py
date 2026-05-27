from __future__ import annotations

import random

from src.schemas import ProfileSpec


BASE = {
    "profile_id": "targeted_profile",
    "profile_group": "targeted",
    "candidate_level": "junior",
    "project_type": "course_or_portfolio_project",
    "technologies": ["Python", "REST API", "logs"],
    "challenge_type": "debugging",
    "ownership_level": "clear_personal_task",
    "communication_quality": "clear",
    "technical_detail": "medium",
    "specificity": "moderate",
    "outcome_strength": "vague",
    "likely_domains": ["backend API", "NLP course project", "React web app"],
    "desired_quality_hint": "targeted",
}


V3_WEAK_PROFILE_MAP = {
    "technical_depth": "vague_tool_mentions_without_explanation",
    "personal_contribution": "generic_team_summary_no_personal_role",
    "clarity": "unclear_rambling_answer",
    "impact": "no_outcome_answer",
    "role_relevance": "off_topic_learning_reflection",
}


V3_PROFILE_CONTROLS = {
    "vague_tool_mentions_without_explanation": {
        "profile_group": "v3_weak",
        "candidate_level": "junior",
        "project_type": "course_or_portfolio_project",
        "technologies": ["React", "Python", "API"],
        "challenge_type": "overview",
        "ownership_level": "assisted",
        "communication_quality": "understandable",
        "technical_detail": "low",
        "specificity": "generic",
        "outcome_strength": "vague",
        "likely_domains": ["React web app", "backend API"],
        "desired_quality_hint": "weak_technical_depth",
    },
    "generic_team_summary_no_personal_role": {
        "profile_group": "v3_weak",
        "candidate_level": "junior",
        "project_type": "team_course_project",
        "technologies": ["Python", "pandas"],
        "challenge_type": "collaboration",
        "ownership_level": "hidden_team_only",
        "communication_quality": "understandable",
        "technical_detail": "medium",
        "specificity": "limited",
        "outcome_strength": "vague",
        "likely_domains": ["data analysis dashboard"],
        "desired_quality_hint": "weak_personal_contribution",
    },
    "unclear_rambling_answer": {
        "profile_group": "v3_weak",
        "candidate_level": "junior",
        "project_type": "course_or_portfolio_project",
        "technologies": ["JavaScript"],
        "challenge_type": "overview",
        "ownership_level": "assisted",
        "communication_quality": "poor",
        "technical_detail": "low",
        "specificity": "generic",
        "outcome_strength": "vague",
        "likely_domains": ["automation script", "React web app"],
        "desired_quality_hint": "weak_clarity",
    },
    "no_outcome_answer": {
        "profile_group": "v3_weak",
        "candidate_level": "junior",
        "project_type": "course_or_portfolio_project",
        "technologies": ["React", "JavaScript"],
        "challenge_type": "overview",
        "ownership_level": "clear_personal_task",
        "communication_quality": "clear",
        "technical_detail": "medium",
        "specificity": "moderate",
        "outcome_strength": "none",
        "likely_domains": ["React web app", "automation script"],
        "desired_quality_hint": "weak_impact",
    },
    "off_topic_learning_reflection": {
        "profile_group": "v3_weak",
        "candidate_level": "junior",
        "project_type": "general_coursework",
        "technologies": [],
        "challenge_type": "overview",
        "ownership_level": "hidden_team_only",
        "communication_quality": "clear",
        "technical_detail": "absent",
        "specificity": "generic",
        "outcome_strength": "clear_learning",
        "likely_domains": ["general coursework"],
        "desired_quality_hint": "weak_relevance",
    },
    "irrelevant_project_summary": {
        "profile_group": "v3_weak",
        "candidate_level": "junior",
        "project_type": "non_software_class_project",
        "technologies": [],
        "challenge_type": "overview",
        "ownership_level": "hidden_team_only",
        "communication_quality": "understandable",
        "technical_detail": "absent",
        "specificity": "generic",
        "outcome_strength": "vague",
        "likely_domains": ["general coursework"],
        "desired_quality_hint": "weak_relevance",
    },
}


V4_WEAK_PROFILE_MAP = {
    "technical_depth": ["vague_tools_no_explanation", "polished_but_shallow_answer"],
    "personal_contribution": ["observer_no_personal_task", "team_only_no_personal_ownership"],
    "clarity": ["severely_unclear_but_realistic_answer", "rambling_unclear_answer", "fragmented_missing_context_answer"],
    "impact": ["no_outcome_answer", "unfinished_project_no_result"],
    "role_relevance": [
        "non_software_team_activity",
        "generic_school_assignment_no_coding",
        "project_manager_observer_role",
        "presentation_only_project",
        "learning_reflection_without_project",
        "design_only_no_implementation",
        "irrelevant_volunteer_project",
    ],
}


V4_PROFILE_CONTROLS = {
    "team_only_no_personal_ownership": {
        "profile_group": "v4_weak",
        "candidate_level": "junior",
        "project_type": "team_course_project",
        "technologies": ["Python", "pandas", "dashboard"],
        "challenge_type": "collaboration",
        "ownership_level": "hidden_team_only",
        "communication_quality": "understandable",
        "technical_detail": "low",
        "specificity": "limited",
        "outcome_strength": "vague",
        "likely_domains": ["data analysis dashboard", "analytics reporting tool"],
        "desired_quality_hint": "weak_personal_contribution",
        "expected_score_constraints": {"personal_contribution": {"max": 2}},
    },
    "observer_no_personal_task": {
        "profile_group": "v4_weak",
        "candidate_level": "junior",
        "project_type": "team_course_project",
        "technologies": [],
        "challenge_type": "collaboration",
        "ownership_level": "hidden_team_only",
        "communication_quality": "understandable",
        "technical_detail": "absent",
        "specificity": "generic",
        "outcome_strength": "vague",
        "likely_domains": ["planning-only project", "non-software team activity"],
        "desired_quality_hint": "weak_personal_contribution",
        "expected_score_constraints": {"personal_contribution": {"max": 1, "strict": 1}},
    },
    "rambling_unclear_answer": {
        "profile_group": "v4_weak",
        "candidate_level": "junior",
        "project_type": "course_or_portfolio_project",
        "technologies": ["JavaScript"],
        "challenge_type": "overview",
        "ownership_level": "assisted",
        "communication_quality": "poor",
        "technical_detail": "low",
        "specificity": "generic",
        "outcome_strength": "vague",
        "likely_domains": ["automation script", "simple game project", "React web app"],
        "desired_quality_hint": "weak_clarity",
        "expected_score_constraints": {"clarity": {"max": 2}},
    },
    "fragmented_missing_context_answer": {
        "profile_group": "v4_weak",
        "candidate_level": "junior",
        "project_type": "course_or_portfolio_project",
        "technologies": ["Python"],
        "challenge_type": "overview",
        "ownership_level": "assisted",
        "communication_quality": "poor",
        "technical_detail": "absent",
        "specificity": "generic",
        "outcome_strength": "none",
        "likely_domains": ["file processing utility", "CLI tool"],
        "desired_quality_hint": "weak_clarity",
        "expected_score_constraints": {"clarity": {"max": 2}},
    },
    "severely_unclear_but_realistic_answer": {
        "profile_group": "v4_weak",
        "candidate_level": "junior",
        "project_type": "course_or_portfolio_project",
        "technologies": [],
        "challenge_type": "overview",
        "ownership_level": "assisted",
        "communication_quality": "poor",
        "technical_detail": "absent",
        "specificity": "generic",
        "outcome_strength": "none",
        "likely_domains": ["simple school assignment", "planning-only project"],
        "desired_quality_hint": "weak_clarity",
        "expected_score_constraints": {"clarity": {"max": 2, "strict": 1}},
    },
    "no_outcome_answer": {
        "profile_group": "v4_weak",
        "candidate_level": "junior",
        "project_type": "course_or_portfolio_project",
        "technologies": ["React", "JavaScript"],
        "challenge_type": "overview",
        "ownership_level": "clear_personal_task",
        "communication_quality": "clear",
        "technical_detail": "medium",
        "specificity": "moderate",
        "outcome_strength": "none",
        "likely_domains": ["React web app", "automation script"],
        "desired_quality_hint": "weak_impact",
        "expected_score_constraints": {"impact": {"max": 2}},
    },
    "unfinished_project_no_result": {
        "profile_group": "v4_weak",
        "candidate_level": "junior",
        "project_type": "portfolio_project",
        "technologies": ["Node.js", "SQLite"],
        "challenge_type": "failure_learning",
        "ownership_level": "clear_personal_task",
        "communication_quality": "understandable",
        "technical_detail": "medium",
        "specificity": "moderate",
        "outcome_strength": "none",
        "likely_domains": ["API integration", "browser extension"],
        "desired_quality_hint": "weak_impact",
        "expected_score_constraints": {"impact": {"max": 2}},
    },
    "irrelevant_non_software_project": {
        "profile_group": "v4_weak",
        "candidate_level": "junior",
        "project_type": "non_software_class_project",
        "technologies": [],
        "challenge_type": "overview",
        "ownership_level": "hidden_team_only",
        "communication_quality": "clear",
        "technical_detail": "absent",
        "specificity": "generic",
        "outcome_strength": "clear_learning",
        "likely_domains": ["general coursework"],
        "desired_quality_hint": "weak_relevance",
        "expected_score_constraints": {"role_relevance": {"max": 2}},
    },
    "non_software_team_activity": {
        "profile_group": "v4_weak",
        "candidate_level": "junior",
        "project_type": "non_software_class_project",
        "technologies": [],
        "challenge_type": "overview",
        "ownership_level": "hidden_team_only",
        "communication_quality": "clear",
        "technical_detail": "absent",
        "specificity": "generic",
        "outcome_strength": "vague",
        "likely_domains": ["non-software team activity"],
        "desired_quality_hint": "weak_relevance",
        "expected_score_constraints": {"role_relevance": {"max": 2, "strict": 1}},
    },
    "generic_school_assignment_no_coding": {
        "profile_group": "v4_weak",
        "candidate_level": "junior",
        "project_type": "general_coursework",
        "technologies": [],
        "challenge_type": "overview",
        "ownership_level": "hidden_team_only",
        "communication_quality": "understandable",
        "technical_detail": "absent",
        "specificity": "generic",
        "outcome_strength": "clear_learning",
        "likely_domains": ["simple school assignment"],
        "desired_quality_hint": "weak_relevance",
        "expected_score_constraints": {"role_relevance": {"max": 2, "strict": 1}},
    },
    "project_manager_observer_role": {
        "profile_group": "v4_weak",
        "candidate_level": "junior",
        "project_type": "planning_only_project",
        "technologies": [],
        "challenge_type": "collaboration",
        "ownership_level": "hidden_team_only",
        "communication_quality": "clear",
        "technical_detail": "absent",
        "specificity": "limited",
        "outcome_strength": "vague",
        "likely_domains": ["planning-only project"],
        "desired_quality_hint": "weak_relevance",
        "expected_score_constraints": {"role_relevance": {"max": 2, "strict": 1}},
    },
    "presentation_only_project": {
        "profile_group": "v4_weak",
        "candidate_level": "junior",
        "project_type": "presentation_project",
        "technologies": [],
        "challenge_type": "overview",
        "ownership_level": "assisted",
        "communication_quality": "clear",
        "technical_detail": "absent",
        "specificity": "generic",
        "outcome_strength": "vague",
        "likely_domains": ["presentation project"],
        "desired_quality_hint": "weak_relevance",
        "expected_score_constraints": {"role_relevance": {"max": 2, "strict": 1}},
    },
    "learning_reflection_without_project": {
        "profile_group": "v4_weak",
        "candidate_level": "junior",
        "project_type": "general_coursework",
        "technologies": [],
        "challenge_type": "overview",
        "ownership_level": "hidden_team_only",
        "communication_quality": "clear",
        "technical_detail": "absent",
        "specificity": "generic",
        "outcome_strength": "clear_learning",
        "likely_domains": ["general coursework"],
        "desired_quality_hint": "weak_relevance",
        "expected_score_constraints": {"role_relevance": {"max": 2, "strict": 1}},
    },
    "design_only_no_implementation": {
        "profile_group": "v4_weak",
        "candidate_level": "junior",
        "project_type": "design_mockup",
        "technologies": [],
        "challenge_type": "design_decision",
        "ownership_level": "assisted",
        "communication_quality": "understandable",
        "technical_detail": "absent",
        "specificity": "limited",
        "outcome_strength": "vague",
        "likely_domains": ["design mockup without implementation"],
        "desired_quality_hint": "weak_relevance",
        "expected_score_constraints": {"role_relevance": {"max": 2, "strict": 1}},
    },
    "irrelevant_volunteer_project": {
        "profile_group": "v4_weak",
        "candidate_level": "junior",
        "project_type": "volunteer_event",
        "technologies": [],
        "challenge_type": "collaboration",
        "ownership_level": "assisted",
        "communication_quality": "clear",
        "technical_detail": "absent",
        "specificity": "generic",
        "outcome_strength": "vague",
        "likely_domains": ["volunteer event organization"],
        "desired_quality_hint": "weak_relevance",
        "expected_score_constraints": {"role_relevance": {"max": 2, "strict": 1}},
    },
    "vague_tools_no_explanation": {
        "profile_group": "v4_weak",
        "candidate_level": "junior",
        "project_type": "course_or_portfolio_project",
        "technologies": ["React", "Firebase", "Python"],
        "challenge_type": "overview",
        "ownership_level": "assisted",
        "communication_quality": "clear",
        "technical_detail": "low",
        "specificity": "generic",
        "outcome_strength": "vague",
        "likely_domains": ["API integration", "browser extension", "mobile app"],
        "desired_quality_hint": "weak_technical_depth",
        "expected_score_constraints": {"technical_depth": {"max": 2}},
    },
    "polished_but_shallow_answer": {
        "profile_group": "v4_weak",
        "candidate_level": "junior",
        "project_type": "portfolio_project",
        "technologies": ["React"],
        "challenge_type": "overview",
        "ownership_level": "clear_personal_task",
        "communication_quality": "excellent",
        "technical_detail": "low",
        "specificity": "generic",
        "outcome_strength": "vague",
        "likely_domains": ["React web app", "browser extension", "simple game project"],
        "desired_quality_hint": "weak_technical_depth",
        "expected_score_constraints": {"technical_depth": {"max": 2}, "impact": {"max": 3}},
    },
}


OFFICIAL_STRONG_DOMAINS = [
    "backend API",
    "database system",
    "mobile app",
    "data pipeline",
    "API integration",
    "testing framework",
    "DevOps/deployment task",
    "CLI tool",
    "machine learning project",
    "NLP course project",
]


OFFICIAL_HIGH_IMPACT_PROFILE_IDS = [
    "deployed_feature_with_usage",
    "measurable_performance_improvement",
    "reduced_manual_work",
    "accuracy_or_quality_improvement",
    "successful_demo_or_client_value",
    "qa_or_error_reduction_fix",
    "before_after_metric_improvement",
]


OFFICIAL_HIGH_IMPACT_PROFILE_CONTROLS = {
    "deployed_feature_with_usage": {
        "project_type": "course_or_portfolio_project",
        "technologies": ["Python", "FastAPI", "SQLite"],
        "challenge_type": "design_decision",
        "ownership_level": "clear_personal_task",
        "communication_quality": "clear",
        "technical_detail": "medium",
        "specificity": "concrete",
        "outcome_strength": "clear_project_value",
        "likely_domains": ["backend API", "API integration", "mobile app"],
    },
    "measurable_performance_improvement": {
        "project_type": "course_or_portfolio_project",
        "technologies": ["Python", "SQL", "profiling"],
        "challenge_type": "performance",
        "ownership_level": "personally_owned",
        "communication_quality": "clear",
        "technical_detail": "high",
        "specificity": "highly_specific",
        "outcome_strength": "measurable",
        "likely_domains": ["backend API", "database system", "data pipeline"],
    },
    "reduced_manual_work": {
        "project_type": "team_course_project",
        "technologies": ["Python", "CSV", "cron"],
        "challenge_type": "overview",
        "ownership_level": "clear_personal_task",
        "communication_quality": "understandable",
        "technical_detail": "medium",
        "specificity": "concrete",
        "outcome_strength": "measurable",
        "likely_domains": ["automation script", "data analysis dashboard", "analytics reporting tool"],
    },
    "accuracy_or_quality_improvement": {
        "project_type": "course_or_portfolio_project",
        "technologies": ["Python", "scikit-learn", "validation set"],
        "challenge_type": "design_decision",
        "ownership_level": "clear_personal_task",
        "communication_quality": "clear",
        "technical_detail": "high",
        "specificity": "highly_specific",
        "outcome_strength": "measurable",
        "likely_domains": ["machine learning project", "NLP course project", "data pipeline"],
    },
    "successful_demo_or_client_value": {
        "project_type": "team_course_project",
        "technologies": ["Python", "pandas", "dashboard"],
        "challenge_type": "collaboration",
        "ownership_level": "clear_personal_task",
        "communication_quality": "clear",
        "technical_detail": "medium",
        "specificity": "concrete",
        "outcome_strength": "clear_project_value",
        "likely_domains": ["data analysis dashboard", "API integration", "mobile app"],
    },
    "qa_or_error_reduction_fix": {
        "project_type": "course_or_portfolio_project",
        "technologies": ["Python", "pytest", "logs"],
        "challenge_type": "debugging",
        "ownership_level": "personally_owned",
        "communication_quality": "understandable",
        "technical_detail": "high",
        "specificity": "concrete",
        "outcome_strength": "measurable",
        "likely_domains": ["testing framework", "backend API", "CLI tool"],
    },
    "before_after_metric_improvement": {
        "project_type": "course_or_portfolio_project",
        "technologies": ["Python", "REST API", "benchmarks"],
        "challenge_type": "performance",
        "ownership_level": "clear_personal_task",
        "communication_quality": "clear",
        "technical_detail": "medium",
        "specificity": "highly_specific",
        "outcome_strength": "measurable",
        "likely_domains": ["CLI tool", "data pipeline", "API integration"],
    },
}


def v3_weak_profile(profile_id: str) -> ProfileSpec:
    payload = dict(V3_PROFILE_CONTROLS[profile_id])
    payload["profile_id"] = profile_id
    return ProfileSpec.from_dict(payload)


def v4_weak_profile(profile_id: str) -> ProfileSpec:
    payload = dict(V4_PROFILE_CONTROLS[profile_id])
    payload["profile_id"] = profile_id
    return ProfileSpec.from_dict(payload)


def profile_for_v4_weak_aspect(aspect: str, seed: int) -> ProfileSpec:
    profile_ids = V4_WEAK_PROFILE_MAP[aspect]
    return v4_weak_profile(profile_ids[seed % len(profile_ids)])


def official_strong_profile(aspect: str, seed: int) -> ProfileSpec:
    domains = list(OFFICIAL_STRONG_DOMAINS)
    random.Random(seed).shuffle(domains)
    payload = {
        "profile_id": f"official_strong_{aspect}_{seed}",
        "profile_group": "official_strong",
        "candidate_level": "junior",
        "project_type": "course_or_portfolio_project",
        "technologies": ["Python", "REST API", "SQL", "tests"],
        "challenge_type": "debugging" if seed % 2 else "design_decision",
        "ownership_level": "personally_owned",
        "communication_quality": "excellent",
        "technical_detail": "very_high",
        "specificity": "highly_specific",
        "outcome_strength": "measurable",
        "likely_domains": domains[:4],
        "desired_quality_hint": "strong_official",
        "expected_score_constraints": {target_aspect: {"min": 4} for target_aspect in [
            "technical_depth",
            "personal_contribution",
            "clarity",
            "problem_solving",
            "impact",
            "role_relevance",
        ]},
    }
    return ProfileSpec.from_dict(payload)


def official_high_impact_profile(seed: int) -> ProfileSpec:
    profile_id = OFFICIAL_HIGH_IMPACT_PROFILE_IDS[seed % len(OFFICIAL_HIGH_IMPACT_PROFILE_IDS)]
    controls = dict(OFFICIAL_HIGH_IMPACT_PROFILE_CONTROLS[profile_id])
    domains = list(controls["likely_domains"])
    random.Random(seed).shuffle(domains)
    controls["likely_domains"] = domains
    payload = {
        "profile_id": profile_id,
        "profile_group": "official_high_impact",
        "candidate_level": "junior",
        "desired_quality_hint": "high_impact",
        "expected_score_constraints": {"impact": {"min": 4}},
        **controls,
    }
    return ProfileSpec.from_dict(payload)


def profile_for_underrepresented_band(aspect: str, score: int, seed: int) -> ProfileSpec:
    if score <= 2 and aspect in V3_WEAK_PROFILE_MAP:
        if aspect == "role_relevance" and seed % 2 == 0:
            return v3_weak_profile("irrelevant_project_summary")
        return v3_weak_profile(V3_WEAK_PROFILE_MAP[aspect])
    payload = dict(BASE)
    payload["profile_id"] = f"targeted_{aspect}_{score}_{seed}"
    if aspect == "technical_depth":
        payload["technical_detail"] = "absent" if score <= 1 else "very_high" if score >= 5 else "medium"
    elif aspect == "personal_contribution":
        payload["ownership_level"] = "hidden_team_only" if score <= 1 else "personally_owned" if score >= 5 else "clear_personal_task"
    elif aspect == "clarity":
        payload["communication_quality"] = "poor" if score <= 1 else "excellent" if score >= 5 else "understandable"
    elif aspect == "problem_solving":
        payload["challenge_type"] = "overview" if score <= 1 else "debugging" if score >= 5 else "failure_learning"
    elif aspect == "impact":
        payload["outcome_strength"] = "none" if score <= 1 else "measurable" if score >= 5 else "clear_learning"
    elif aspect == "role_relevance":
        payload["likely_domains"] = ["general coursework"] if score <= 1 else ["backend API", "React web app"]
    if score <= 2:
        payload["specificity"] = "generic"
    elif score >= 5:
        payload["specificity"] = "highly_specific"
    rng = random.Random(seed)
    rng.shuffle(payload["likely_domains"])
    return ProfileSpec.from_dict(payload)
