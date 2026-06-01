from __future__ import annotations

import json
import random
from statistics import mean
from typing import Any

from src.schemas import ASPECTS, compute_strong_aspects, compute_weak_aspects


PROMPT_VERSION = "llm_baseline_v1"

ASPECT_DEFINITIONS = {
    "technical_depth": "Junior-appropriate technical reasoning, implementation detail, debugging, tradeoffs, constraints, or meaningful tool use.",
    "personal_contribution": "How clearly the candidate explains what they personally did.",
    "clarity": "Organization, coherence, concreteness, and interview-readiness.",
    "problem_solving": "Challenge, reasoning, troubleshooting, decisions, iteration, or solution quality.",
    "impact": "Project outcome, user/team value, delivery, metrics, improvement, or meaningful learning.",
    "role_relevance": "Fit to the interview question and to junior software-development work.",
}

SCORE_SCALE = [
    "1 = absent or almost absent",
    "2 = weak",
    "3 = partial / acceptable",
    "4 = good",
    "5 = strong",
]


def _json_schema_example() -> str:
    return json.dumps(
        {
            "final_scores": {aspect: 3 for aspect in ASPECTS},
            "weak_aspects": [],
            "strong_aspects": [],
            "rationale": {aspect: "short explanation based only on answer evidence" for aspect in ASPECTS},
        },
        ensure_ascii=True,
        indent=2,
    )


def _rubric_section() -> str:
    lines = ["Rubric aspects:"]
    for aspect in ASPECTS:
        lines.append(f"- {aspect}: {ASPECT_DEFINITIONS[aspect]}")
    return "\n".join(lines)


def _base_instructions() -> str:
    return "\n".join(
        [
            "You are scoring a junior software developer interview answer.",
            "Score the answer on six aspects from 1 to 5.",
            "",
            _rubric_section(),
            "",
            "Score scale:",
            *[f"- {line}" for line in SCORE_SCALE],
            "",
            "Important scoring rules:",
            "- Score only based on evidence in the answer.",
            "- Do not infer generously.",
            "- A fluent answer should not receive high scores if it lacks substance.",
            "- Technology names alone do not justify high technical depth.",
            "- Using 'we' should lower personal contribution only when the candidate's own role is unclear.",
            "",
            "Return strict JSON only. Do not include markdown or prose outside the JSON.",
            "The JSON must match this shape:",
            _json_schema_example(),
        ]
    )


def _record_input(record: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Input:",
            f"target_role: {record.get('target_role', 'Junior Software Developer')}",
            f"question: {record.get('question', '')}",
            "answer:",
            str(record.get("answer", "")),
        ]
    )


def _label_json(record: dict[str, Any]) -> str:
    scores = {aspect: int(record["final_scores"][aspect]) for aspect in ASPECTS}
    return json.dumps(
        {
            "final_scores": scores,
            "weak_aspects": compute_weak_aspects(scores),
            "strong_aspects": compute_strong_aspects(scores),
            "rationale": {aspect: "training example label" for aspect in ASPECTS},
        },
        ensure_ascii=True,
        indent=2,
    )


def build_zero_shot_prompt(record: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            f"Prompt version: {PROMPT_VERSION}",
            _base_instructions(),
            _record_input(record),
            "Score this answer now.",
        ]
    )


def _is_train_record(record: dict[str, Any]) -> bool:
    return record.get("split") == "train"


def _is_strong_record(record: dict[str, Any]) -> bool:
    return mean(int(record["final_scores"][aspect]) for aspect in ASPECTS) >= 4


def select_few_shot_examples(train_records: list[dict[str, Any]], k: int = 3, seed: int = 42) -> list[dict[str, Any]]:
    candidates = sorted((record for record in train_records if _is_train_record(record)), key=lambda row: row.get("example_id", ""))
    selected: list[dict[str, Any]] = []

    def add_first(predicate) -> None:
        for candidate in candidates:
            if candidate in selected:
                continue
            if predicate(candidate):
                selected.append(candidate)
                return

    add_first(lambda row: int(row["final_scores"]["impact"]) <= 2)
    add_first(lambda row: int(row["final_scores"]["personal_contribution"]) <= 2 or int(row["final_scores"]["technical_depth"]) <= 2)
    add_first(_is_strong_record)

    remaining = [candidate for candidate in candidates if candidate not in selected]
    random.Random(seed).shuffle(remaining)
    for candidate in remaining:
        if len(selected) >= k:
            break
        selected.append(candidate)
    return selected[:k]


def build_few_shot_prompt(record: dict[str, Any], examples: list[dict[str, Any]]) -> str:
    example_blocks = []
    for index, example in enumerate(examples, 1):
        example_blocks.append(
            "\n".join(
                [
                    f"Example {index} ({example.get('example_id', 'unknown')}):",
                    _record_input(example),
                    "Expected JSON:",
                    _label_json(example),
                ]
            )
        )
    return "\n\n".join(
        [
            f"Prompt version: {PROMPT_VERSION}",
            _base_instructions(),
            "Few-shot training examples:",
            "\n\n".join(example_blocks),
            "Now score the evaluation answer.",
            _record_input(record),
            "Return strict JSON only.",
        ]
    )
