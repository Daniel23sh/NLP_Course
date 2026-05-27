import unittest

from src.schemas import ASPECTS, DATASET_VERSION_V4, DatasetRecord, ProfileSpec, ScoringPass
from src.validator import apply_validation, validate_record


def make_v4_record(scores, answer, profile_id="no_outcome_answer", constraints=None, validator_scores=None):
    profile = ProfileSpec(
        profile_id=profile_id,
        profile_group="v4_weak",
        candidate_level="junior",
        project_type="course_or_portfolio_project",
        technologies=["React"],
        challenge_type="overview",
        ownership_level="assisted",
        communication_quality="clear",
        technical_detail="low",
        specificity="generic",
        outcome_strength="none",
        likely_domains=["React web app"],
        desired_quality_hint="weak_impact",
        expected_score_constraints=constraints or {"impact": {"max": 2}},
    )
    return DatasetRecord(
        example_id="v4_test_001",
        dataset_version=DATASET_VERSION_V4,
        question_id="q_project_overview_react_web_app",
        question_type="project_overview",
        project_domain="React web app",
        question="Tell me about a software project you worked on.",
        profile=profile,
        answer=answer,
        labeler=ScoringPass(scores=scores, evidence={aspect: ["evidence"] for aspect in ASPECTS}),
        validator=ScoringPass(scores=validator_scores or scores, evidence={aspect: ["evidence"] for aspect in ASPECTS}),
        scenario_family="project_overview|React web app|no_outcome_answer|overview|weak_impact",
    )


class V4ProfileSuccessTests(unittest.TestCase):
    def test_low_impact_profile_with_high_impact_routes_to_profile_mismatch(self):
        scores = {aspect: 3 for aspect in ASPECTS}
        scores["impact"] = 4
        record = make_v4_record(scores, "I helped on a React project and users saw a faster delivered feature.")

        validation = validate_record(record, tolerance=1)
        updated = apply_validation(record, validation)

        self.assertEqual(validation.final_status, "profile_mismatch")
        self.assertIn("profile_mismatch:impact", validation.flags)
        self.assertEqual(updated.final_scores["impact"], 4)

    def test_low_clarity_profile_with_high_clarity_routes_to_profile_mismatch(self):
        scores = {aspect: 3 for aspect in ASPECTS}
        scores["clarity"] = 5
        record = make_v4_record(
            scores,
            "I described the project clearly from context to action to result.",
            profile_id="rambling_unclear_answer",
            constraints={"clarity": {"max": 2}},
        )

        validation = validate_record(record, tolerance=1)

        self.assertEqual(validation.final_status, "profile_mismatch")
        self.assertIn("profile_mismatch:clarity", validation.flags)

    def test_low_personal_contribution_profile_with_score_one_can_be_accepted(self):
        scores = {aspect: 3 for aspect in ASPECTS}
        scores["personal_contribution"] = 1
        record = make_v4_record(
            scores,
            "The group had a dashboard task. The work was shared generally and no separate personal part was described.",
            profile_id="team_only_no_personal_ownership",
            constraints={"personal_contribution": {"max": 2}},
        )

        validation = validate_record(record, tolerance=1)

        self.assertEqual(validation.final_status, "accepted")

    def test_strict_low_role_profile_with_score_three_routes_to_profile_mismatch(self):
        scores = {aspect: 3 for aspect in ASPECTS}
        record = make_v4_record(
            scores,
            "The answer is about a presentation activity with no coding.",
            profile_id="presentation_only_project",
            constraints={"role_relevance": {"max": 2, "strict": 1}},
        )

        validation = validate_record(record, tolerance=1)

        self.assertEqual(validation.final_status, "profile_mismatch")
        self.assertIn("profile_mismatch:role_relevance", validation.flags)

    def test_strict_personal_contribution_one_profile_rejects_score_two(self):
        scores = {aspect: 3 for aspect in ASPECTS}
        scores["personal_contribution"] = 2
        record = make_v4_record(
            scores,
            "The team discussed the work while the candidate mostly observed.",
            profile_id="observer_no_personal_task",
            constraints={"personal_contribution": {"max": 1, "strict": 1}},
        )

        validation = validate_record(record, tolerance=1)

        self.assertEqual(validation.final_status, "profile_mismatch")
        self.assertIn("profile_mismatch:personal_contribution", validation.flags)

    def test_low_clarity_patch_target_accepts_clarity_two(self):
        scores = {aspect: 2 for aspect in ASPECTS}
        scores["clarity"] = 2
        record = make_v4_record(
            scores,
            "The thing was from class and the order is hard to explain. Some part changed, but not enough context is here.",
            profile_id="severely_unclear_but_realistic_answer",
            constraints={"clarity": {"max": 2, "strict": 1}},
        )

        validation = validate_record(record, tolerance=1)

        self.assertEqual(validation.final_status, "accepted")


if __name__ == "__main__":
    unittest.main()
