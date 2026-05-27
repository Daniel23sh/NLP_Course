import unittest

from src.schemas import ASPECTS, DatasetRecord, ProfileSpec, ScoringPass
from src.labeler import build_labeling_prompt
from src.validator import (
    apply_validation,
    build_validation_prompt,
    compute_score_deltas,
    detect_label_leakage,
    parse_validation_scoring_pass,
    validator_audit_flags,
    validate_record,
)


def make_record(scores, validator_scores=None, answer=None):
    profile = ProfileSpec(
        profile_id="test_profile",
        profile_group="test",
        candidate_level="junior",
        project_type="course_or_portfolio_project",
        technologies=["Python", "REST API", "logs"],
        challenge_type="debugging",
        ownership_level="clear_personal_task",
        communication_quality="clear",
        technical_detail="high",
        specificity="concrete",
        outcome_strength="clear_project_value",
        likely_domains=["backend API"],
    )
    answer = answer or (
        "In a backend API project, I debugged a slow endpoint. I checked logs, reproduced the issue locally, "
        "tested a smaller query change, fixed the code, and confirmed the response was faster for users."
    )
    return DatasetRecord(
        example_id="ex_test",
        question_id="q_debugging_backend_api",
        question_type="debugging_story",
        project_domain="backend API",
        question="Describe a project where you had to debug a difficult issue.",
        profile=profile,
        answer=answer,
        labeler=ScoringPass(scores=scores, evidence={aspect: ["evidence"] for aspect in ASPECTS}),
        validator=ScoringPass(scores=validator_scores or scores, evidence={aspect: ["evidence"] for aspect in ASPECTS}),
    )


class ValidatorTests(unittest.TestCase):
    def test_validation_prompt_is_distinct_from_labeling_prompt_and_asks_for_audit(self):
        answer = "I built a small React feature and tested it."
        rubric = {"technical_depth": {"definition": "depth"}}

        label_prompt = build_labeling_prompt("Tell me about a project.", answer, rubric)
        validator_prompt = build_validation_prompt("Tell me about a project.", answer, rubric)

        self.assertNotEqual(label_prompt, validator_prompt)
        self.assertIn("strict audit", validator_prompt.lower())
        self.assertIn("challenge", validator_prompt.lower())
        self.assertIn("evidence_checks", validator_prompt)
        self.assertIn("mismatch_flags", validator_prompt)

    def test_label_leakage_allows_ml_metric_score_but_catches_rubric_score_language(self):
        self.assertEqual(detect_label_leakage("The F1 score improved from 0.68 to 0.76."), [])
        self.assertIn("score", detect_label_leakage("I think the score should be 5 because I did well."))

    def test_parse_validation_scoring_pass_preserves_flags_in_rationale(self):
        raw = """
        {
          "scores": {
            "technical_depth": 3,
            "personal_contribution": 2,
            "clarity": 4,
            "problem_solving": 2,
            "impact": 2,
            "role_relevance": 4
          },
          "evidence": {
            "technical_depth": ["React feature"],
            "personal_contribution": ["helped"],
            "clarity": ["clear sequence"],
            "problem_solving": ["no process"],
            "impact": ["limited outcome"],
            "role_relevance": ["project answer"]
          },
          "rationale": {
            "technical_depth": "basic",
            "personal_contribution": "unclear",
            "clarity": "clear",
            "problem_solving": "weak",
            "impact": "weak",
            "role_relevance": "relevant"
          },
          "confidence": {
            "technical_depth": 0.6,
            "personal_contribution": 0.6,
            "clarity": 0.7,
            "problem_solving": 0.6,
            "impact": 0.6,
            "role_relevance": 0.8
          },
          "evidence_checks": {
            "unsupported_high_scores": ["impact"],
            "low_score_contradictions": ["personal_contribution"],
            "label_leakage_terms": []
          },
          "mismatch_flags": ["validator_lowered_personal_contribution"]
        }
        """

        scoring = parse_validation_scoring_pass(raw)

        self.assertEqual(scoring.scores["personal_contribution"], 2)
        self.assertIn("validator_flags", scoring.rationale)
        self.assertIn("validator_lowered_personal_contribution", scoring.rationale["validator_flags"])

    def test_accepts_exact_and_adjacent_agreement(self):
        labeler = {aspect: 3 for aspect in ASPECTS}
        validator = dict(labeler)
        validator["clarity"] = 4
        record = make_record(labeler, validator)

        validation = validate_record(record, tolerance=1)
        accepted = apply_validation(record, validation)

        self.assertEqual(validation.final_status, "accepted")
        self.assertEqual(accepted.final_scores, labeler)
        self.assertEqual(compute_score_deltas(labeler, validator)["clarity"], 1)

    def test_rejects_score_delta_ge_two(self):
        labeler = {aspect: 3 for aspect in ASPECTS}
        validator = dict(labeler)
        validator["impact"] = 5
        record = make_record(labeler, validator)

        validation = validate_record(record, tolerance=1)

        self.assertEqual(validation.final_status, "rejected_score_disagreement")
        self.assertIn("score_delta_ge_2", validation.rejection_reasons)

    def test_validator_audit_flags_force_manual_review_when_they_apply_to_labeler_scores(self):
        scores = {aspect: 3 for aspect in ASPECTS}
        scores["impact"] = 5
        record = make_record(scores, answer="I worked with a group and learned a lot, but no project result was described.")
        record.validator.rationale["validator_flags"] = (
            '{"evidence_checks":{"unsupported_high_scores":["impact"],'
            '"low_score_contradictions":[],"label_leakage_terms":[]},"mismatch_flags":[]}'
        )

        validation = validate_record(record)

        self.assertEqual(validation.final_status, "manual_review_required")
        self.assertIn("validator_audit_flag", validation.rejection_reasons)
        self.assertIn("unsupported_high_score:impact", validation.flags)

    def test_validator_audit_flags_ignore_non_applicable_aspects(self):
        scores = {aspect: 3 for aspect in ASPECTS}
        scores["problem_solving"] = 1
        record = make_record(scores)
        record.validator.rationale["validator_flags"] = (
            '{"evidence_checks":{"unsupported_high_scores":["problem_solving"],'
            '"low_score_contradictions":[],"label_leakage_terms":[]},"mismatch_flags":[]}'
        )

        self.assertEqual(validator_audit_flags(record), [])

    def test_freeform_validator_mismatch_notes_do_not_force_manual_review(self):
        scores = {aspect: 2 for aspect in ASPECTS}
        record = make_record(scores, answer="This was a generic planning activity with no coding or software work.")
        record.validator.rationale["validator_flags"] = (
            '{"evidence_checks":{"unsupported_high_scores":[],"low_score_contradictions":[],'
            '"label_leakage_terms":[]},"mismatch_flags":["Answer is generic and non-technical."]}'
        )

        self.assertEqual(validator_audit_flags(record), [])

    def test_rejects_low_personal_contribution_with_first_person_actions(self):
        scores = {aspect: 3 for aspect in ASPECTS}
        scores["personal_contribution"] = 1
        record = make_record(scores)

        validation = validate_record(record)

        self.assertEqual(validation.final_status, "rejected_evidence_contradiction")
        self.assertIn("personal_contribution_low_but_first_person_actions_present", validation.evidence_contradictions)

    def test_rejects_low_technical_depth_with_technical_evidence(self):
        scores = {aspect: 3 for aspect in ASPECTS}
        scores["technical_depth"] = 1
        record = make_record(scores)

        validation = validate_record(record)

        self.assertEqual(validation.final_status, "rejected_evidence_contradiction")
        self.assertIn("technical_depth_low_but_technical_implementation_present", validation.evidence_contradictions)

    def test_rejects_low_impact_with_clear_outcome(self):
        scores = {aspect: 3 for aspect in ASPECTS}
        scores["impact"] = 1
        record = make_record(scores)

        validation = validate_record(record)

        self.assertEqual(validation.final_status, "rejected_evidence_contradiction")
        self.assertIn("impact_low_but_outcome_present", validation.evidence_contradictions)

    def test_rejects_low_problem_solving_with_challenge_and_solution(self):
        scores = {aspect: 3 for aspect in ASPECTS}
        scores["problem_solving"] = 1
        record = make_record(scores)

        validation = validate_record(record)

        self.assertEqual(validation.final_status, "rejected_evidence_contradiction")
        self.assertIn("problem_solving_low_but_debugging_process_present", validation.evidence_contradictions)

    def test_personal_learning_outcome_does_not_trigger_low_impact_contradiction(self):
        scores = {aspect: 3 for aspect in ASPECTS}
        scores["impact"] = 1
        record = make_record(
            scores,
            answer=(
                "The group discussed a class topic. I learned that communication matters, "
                "but there was no delivered feature, user result, metric, or project outcome."
            ),
        )

        validation = validate_record(record)

        self.assertNotIn("impact_low_but_outcome_present", validation.evidence_contradictions)

    def test_generic_problem_solution_wording_does_not_trigger_debugging_contradiction(self):
        scores = {aspect: 3 for aspect in ASPECTS}
        scores["problem_solving"] = 1
        record = make_record(
            scores,
            answer=(
                "For a class presentation, the group talked about a community problem and a possible solution. "
                "It was a discussion activity, not coding, debugging, testing, or implementation work."
            ),
        )

        validation = validate_record(record)

        self.assertNotIn("problem_solving_low_but_debugging_process_present", validation.evidence_contradictions)


if __name__ == "__main__":
    unittest.main()
