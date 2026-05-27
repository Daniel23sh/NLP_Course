from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


ASPECTS = [
    "technical_depth",
    "personal_contribution",
    "clarity",
    "problem_solving",
    "impact",
    "role_relevance",
]

TARGET_ROLE = "Junior Software Developer"
DATASET_VERSION = "v2_label_after_generation"
DATASET_VERSION_V3 = "v3"
DATASET_VERSION_V4 = "v4"
DATASET_VERSION_OFFICIAL = "official_v1"
SCORE_MIN = 1
SCORE_MAX = 5
STATUS_ACCEPTED = "accepted"
STATUS_ACCEPTED_BORDERLINE = "accepted_borderline"
STATUS_MANUAL_REVIEW = "manual_review"
STATUS_PROFILE_MISMATCH = "profile_mismatch"
STATUS_AUDIT_ONLY = "audit_only"
STATUS_REJECTED = "rejected"
TRAIN_ELIGIBLE_STATUSES = {STATUS_ACCEPTED, STATUS_ACCEPTED_BORDERLINE}
REVIEW_STATUSES = {STATUS_MANUAL_REVIEW}
AUDIT_STATUSES = {STATUS_AUDIT_ONLY, STATUS_PROFILE_MISMATCH}
REJECTED_STATUSES = {STATUS_REJECTED}


def validate_scores(scores: dict[str, int]) -> None:
    if set(scores) != set(ASPECTS):
        raise ValueError(f"scores must contain exactly these aspects: {ASPECTS}")
    for aspect, value in scores.items():
        if not isinstance(value, int) or not SCORE_MIN <= value <= SCORE_MAX:
            raise ValueError(f"score for {aspect} must be an integer from {SCORE_MIN} to {SCORE_MAX}")


def compute_weak_aspects(scores: dict[str, int]) -> list[str]:
    return [aspect for aspect in ASPECTS if scores.get(aspect, 0) <= 2]


def compute_strong_aspects(scores: dict[str, int]) -> list[str]:
    return [aspect for aspect in ASPECTS if scores.get(aspect, 0) >= 4]


@dataclass
class ProfileSpec:
    profile_id: str
    profile_group: str
    candidate_level: str
    project_type: str
    technologies: list[str]
    challenge_type: str
    ownership_level: str
    communication_quality: str
    technical_detail: str
    specificity: str
    outcome_strength: str
    likely_domains: list[str]
    desired_quality_hint: str = "mixed"
    expected_score_constraints: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProfileSpec":
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in allowed})


@dataclass
class GeneratedAnswer:
    example_id: str
    question_id: str
    question_type: str
    project_domain: str
    question: str
    profile: ProfileSpec
    answer: str
    scenario_family: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AspectScore:
    score: int
    evidence: list[str] = field(default_factory=list)
    rationale: str = ""
    confidence: float = 0.0


@dataclass
class ScoringPass:
    scores: dict[str, int] = field(default_factory=dict)
    evidence: dict[str, list[str]] = field(default_factory=dict)
    rationale: dict[str, str] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scores:
            validate_scores(self.scores)
        for aspect in ASPECTS:
            self.evidence.setdefault(aspect, [])
            self.rationale.setdefault(aspect, "")
            self.confidence.setdefault(aspect, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ScoringPass":
        if not payload:
            return cls()
        return cls(
            scores={aspect: int(value) for aspect, value in payload.get("scores", {}).items()},
            evidence={aspect: list(value) for aspect, value in payload.get("evidence", {}).items()},
            rationale=dict(payload.get("rationale", {})),
            confidence={aspect: float(value) for aspect, value in payload.get("confidence", {}).items()},
        )


@dataclass
class ValidationMetadata:
    final_status: str = "pending"
    agreement_tolerance: int = 1
    rejection_reasons: list[str] = field(default_factory=list)
    label_leakage_terms: list[str] = field(default_factory=list)
    evidence_contradictions: list[str] = field(default_factory=list)
    actual_word_count: int = 0
    score_deltas: dict[str, int] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ValidationMetadata":
        if not payload:
            return cls()
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in allowed})


@dataclass
class DatasetRecord:
    example_id: str
    question_id: str
    question_type: str
    project_domain: str
    question: str
    profile: ProfileSpec
    answer: str
    scenario_family: str = ""
    dataset_version: str = DATASET_VERSION
    target_role: str = TARGET_ROLE
    labeler: ScoringPass = field(default_factory=ScoringPass)
    validator: ScoringPass = field(default_factory=ScoringPass)
    final_scores: dict[str, int] = field(default_factory=dict)
    weak_aspects: list[str] = field(default_factory=list)
    strong_aspects: list[str] = field(default_factory=list)
    validation: ValidationMetadata = field(default_factory=ValidationMetadata)
    split: str = "unassigned"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target_role != TARGET_ROLE:
            raise ValueError(f"target_role must be {TARGET_ROLE}")
        if self.final_scores:
            validate_scores(self.final_scores)
            self.weak_aspects = compute_weak_aspects(self.final_scores)
            self.strong_aspects = compute_strong_aspects(self.final_scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "dataset_version": self.dataset_version,
            "target_role": self.target_role,
            "question_id": self.question_id,
            "question_type": self.question_type,
            "project_domain": self.project_domain,
            "question": self.question,
            "scenario_family": self.scenario_family,
            "profile": self.profile.to_dict(),
            "answer": self.answer,
            "labeler": self.labeler.to_dict(),
            "validator": self.validator.to_dict(),
            "final_scores": dict(self.final_scores),
            "weak_aspects": list(self.weak_aspects),
            "strong_aspects": list(self.strong_aspects),
            "validation": self.validation.to_dict(),
            "split": self.split,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_generated(cls, generated: GeneratedAnswer, metadata: dict[str, Any] | None = None) -> "DatasetRecord":
        return cls(
            example_id=generated.example_id,
            question_id=generated.question_id,
            question_type=generated.question_type,
            project_domain=generated.project_domain,
            question=generated.question,
            profile=generated.profile,
            answer=generated.answer,
            scenario_family=generated.scenario_family,
            metadata=metadata or dict(generated.metadata),
        )


@dataclass
class DistributionReport:
    generated_examples: int
    accepted_examples: int
    rejected_examples: int
    acceptance_rate: float
    score_distribution: dict[str, dict[int, int]]
    weak_aspect_frequency: dict[str, int]
    strong_aspect_frequency: dict[str, int]
    profile_distribution: dict[str, int]
    question_type_distribution: dict[str, int]
    project_domain_distribution: dict[str, int]
    rejection_reasons: dict[str, int]
    agreement_delta_distribution: dict[str, int]
    underrepresented_bands: dict[str, list[int]]
    manual_review_candidates: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_to_dict(record: DatasetRecord) -> dict[str, Any]:
    return record.to_dict()


def record_from_dict(payload: dict[str, Any]) -> DatasetRecord:
    return DatasetRecord(
        example_id=payload["example_id"],
        dataset_version=payload.get("dataset_version", DATASET_VERSION),
        target_role=payload.get("target_role", TARGET_ROLE),
        question_id=payload["question_id"],
        question_type=payload["question_type"],
        project_domain=payload["project_domain"],
        question=payload["question"],
        scenario_family=payload.get("scenario_family", ""),
        profile=ProfileSpec.from_dict(payload["profile"]),
        answer=payload.get("answer", ""),
        labeler=ScoringPass.from_dict(payload.get("labeler")),
        validator=ScoringPass.from_dict(payload.get("validator")),
        final_scores={aspect: int(value) for aspect, value in payload.get("final_scores", {}).items()},
        weak_aspects=list(payload.get("weak_aspects", [])),
        strong_aspects=list(payload.get("strong_aspects", [])),
        validation=ValidationMetadata.from_dict(payload.get("validation")),
        split=payload.get("split", "unassigned"),
        metadata=dict(payload.get("metadata", {})),
    )
