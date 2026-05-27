from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from src.config_loader import load_generation_config, load_profiles
from src.schemas import ProfileSpec


@dataclass
class GenerationRequest:
    example_id: str
    question_id: str
    question_type: str
    project_domain: str
    question: str
    profile: ProfileSpec
    scenario_family: str = ""

    @property
    def question_payload(self) -> dict[str, Any]:
        scenario_family = self.scenario_family or "|".join(
            [
                self.question_type,
                self.project_domain,
                self.profile.profile_id,
                self.profile.challenge_type,
                self.profile.desired_quality_hint,
            ]
        )
        return {
            "example_id": self.example_id,
            "question_id": self.question_id,
            "question_type": self.question_type,
            "project_domain": self.project_domain,
            "question": self.question,
            "scenario_family": scenario_family,
        }


def _weighted_groups(total: int, profile_mix: dict[str, float], rng: random.Random) -> list[str]:
    ordered = list(profile_mix)
    groups: list[str] = []
    allocated = 0
    for group in ordered[:-1]:
        count = int(total * float(profile_mix[group]))
        groups.extend([group] * count)
        allocated += count
    if ordered:
        groups.extend([ordered[-1]] * (total - allocated))
    rng.shuffle(groups)
    return groups


def _question_id(question: dict[str, str]) -> str:
    domain = question["project_domain"].lower().replace(" ", "_").replace("-", "_")
    return f"q_{question['question_type']}_{domain}"


def _profile_from_config(profile_id: str, payload: dict) -> ProfileSpec:
    return ProfileSpec(
        profile_id=profile_id,
        profile_group=payload["profile_group"],
        candidate_level=payload.get("candidate_level", "junior"),
        project_type=payload.get("project_type", "course_or_portfolio_project"),
        technologies=list(payload.get("technologies", [])),
        challenge_type=payload["challenge_type"],
        ownership_level=payload["ownership_level"],
        communication_quality=payload["communication_quality"],
        technical_detail=payload["technical_detail"],
        specificity=payload["specificity"],
        outcome_strength=payload["outcome_strength"],
        likely_domains=list(payload.get("likely_domains", [])),
        desired_quality_hint=payload.get("desired_quality_hint", "mixed"),
        expected_score_constraints=dict(payload.get("expected_score_constraints", {})),
    )


def sample_generation_requests(total: int, seed: int) -> list[GenerationRequest]:
    rng = random.Random(seed)
    generation_config = load_generation_config()
    profiles_config = load_profiles()
    profile_mix = generation_config["profile_mix"]
    questions = generation_config["question_bank"]
    groups = _weighted_groups(total, profile_mix, rng)
    group_offsets = {group: 0 for group in profile_mix}
    requests: list[GenerationRequest] = []

    for index, group in enumerate(groups, start=1):
        profile_ids = profiles_config["profile_groups"][group]
        offset = group_offsets[group]
        profile_id = profile_ids[offset % len(profile_ids)]
        group_offsets[group] = offset + 1
        profile = _profile_from_config(profile_id, profiles_config["profiles"][profile_id])
        matching_questions = [item for item in questions if item["project_domain"] in profile.likely_domains]
        question = rng.choice(matching_questions or questions)
        requests.append(
            GenerationRequest(
                example_id=f"ex_{index:06d}",
                question_id=_question_id(question),
                question_type=question["question_type"],
                project_domain=question["project_domain"],
                question=question["question"],
                profile=profile,
            )
        )
    return requests
