import unittest

from src.schemas import (
    ASPECTS,
    DatasetRecord,
    ProfileSpec,
    ScoringPass,
    ValidationMetadata,
    compute_strong_aspects,
    compute_weak_aspects,
    record_from_dict,
    validate_scores,
)


class SchemaTests(unittest.TestCase):
    def test_score_validation_rejects_missing_or_out_of_range_aspects(self):
        with self.assertRaises(ValueError):
            validate_scores({"technical_depth": 3})
        scores = {aspect: 3 for aspect in ASPECTS}
        scores["impact"] = 6
        with self.assertRaises(ValueError):
            validate_scores(scores)

    def test_weak_and_strong_aspects_derive_from_final_scores(self):
        scores = {
            "technical_depth": 4,
            "personal_contribution": 2,
            "clarity": 1,
            "problem_solving": 3,
            "impact": 5,
            "role_relevance": 4,
        }
        self.assertEqual(compute_weak_aspects(scores), ["personal_contribution", "clarity"])
        self.assertEqual(compute_strong_aspects(scores), ["technical_depth", "impact", "role_relevance"])

    def test_record_round_trips_nested_schema(self):
        scores = {aspect: 3 for aspect in ASPECTS}
        scores["impact"] = 2
        profile = ProfileSpec(
            profile_id="p1",
            profile_group="mixed_plausible",
            candidate_level="junior",
            project_type="course_or_portfolio_project",
            technologies=["Python"],
            challenge_type="debugging",
            ownership_level="clear_personal_task",
            communication_quality="clear",
            technical_detail="medium",
            specificity="moderate",
            outcome_strength="vague",
            likely_domains=["backend API"],
        )
        record = DatasetRecord(
            example_id="ex_000001",
            question_id="q1",
            question_type="debugging_story",
            project_domain="backend API",
            question="Describe a bug.",
            profile=profile,
            answer="I debugged a backend API issue.",
            labeler=ScoringPass(scores=scores, evidence={aspect: ["e"] for aspect in ASPECTS}),
            validator=ScoringPass(scores=scores, evidence={aspect: ["e"] for aspect in ASPECTS}),
            final_scores=scores,
            validation=ValidationMetadata(final_status="accepted"),
        )

        payload = record.to_dict()
        loaded = record_from_dict(payload)

        self.assertEqual(loaded.example_id, "ex_000001")
        self.assertEqual(loaded.final_scores["impact"], 2)
        self.assertEqual(loaded.weak_aspects, ["impact"])


if __name__ == "__main__":
    unittest.main()
