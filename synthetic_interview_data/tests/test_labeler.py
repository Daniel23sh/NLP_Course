import json
import unittest

from src.generator import generate_answer
from src.labeler import heuristic_label_answer, parse_scoring_pass
from src.profile_sampler import sample_generation_requests
from src.schemas import ASPECTS, GeneratedAnswer
from src.targeted_generation import v4_weak_profile


class LabelerTests(unittest.TestCase):
    def test_parse_scoring_pass_reads_scores_evidence_and_rationale(self):
        payload = {
            "scores": {aspect: 3 for aspect in ASPECTS},
            "evidence": {aspect: ["some evidence"] for aspect in ASPECTS},
            "rationale": {aspect: "partial evidence" for aspect in ASPECTS},
            "confidence": {aspect: 0.7 for aspect in ASPECTS},
        }

        scoring = parse_scoring_pass(json.dumps(payload))

        self.assertEqual(scoring.scores["clarity"], 3)
        self.assertEqual(scoring.evidence["impact"], ["some evidence"])
        self.assertEqual(scoring.confidence["technical_depth"], 0.7)

    def test_heuristic_labeler_scores_generated_answer_with_evidence(self):
        request = sample_generation_requests(total=1, seed=11)[0]
        generated = generate_answer(request.profile, request.question_payload, mode="mock", seed=11, model="mock")

        scoring = heuristic_label_answer(generated)

        self.assertEqual(set(scoring.scores), set(ASPECTS))
        for aspect in ASPECTS:
            self.assertGreaterEqual(scoring.scores[aspect], 1)
            self.assertLessEqual(scoring.scores[aspect], 5)
            self.assertTrue(scoring.evidence[aspect])

    def test_non_software_planning_answer_has_low_role_relevance(self):
        profile = v4_weak_profile("project_manager_observer_role")
        generated = GeneratedAnswer(
            example_id="role_low_test",
            question_id="q_planning_activity",
            question_type="planning_activity",
            project_domain="planning-only project",
            question="Tell me about a project where planning was the main activity.",
            profile=profile,
            answer=(
                "The group talked about a class activity and how people would present different parts. "
                "It was mostly communication, planning, and deciding the order of the discussion. "
                "There was no coding, testing, technical decision, or software work."
            ),
        )

        scoring = heuristic_label_answer(generated)

        self.assertLessEqual(scoring.scores["role_relevance"], 2)

    def test_generic_school_reflection_has_low_role_relevance(self):
        profile = v4_weak_profile("generic_school_assignment_no_coding")
        generated = GeneratedAnswer(
            example_id="school_reflection_test",
            question_id="q_school_assignment",
            question_type="school_assignment",
            project_domain="simple school assignment",
            question="Tell me about a school assignment you worked on.",
            profile=profile,
            answer=(
                "One assignment was a short written reflection for a general course. "
                "The main part was reading, discussing ideas, and submitting a simple response. "
                "It was not a software project and did not involve coding or implementation."
            ),
        )

        scoring = heuristic_label_answer(generated)

        self.assertLessEqual(scoring.scores["role_relevance"], 2)

    def test_observer_only_answer_has_personal_contribution_one(self):
        profile = v4_weak_profile("observer_no_personal_task")
        generated = GeneratedAnswer(
            example_id="observer_test",
            question_id="q_team_activity",
            question_type="team_activity",
            project_domain="non-software team activity",
            question="Tell me about a team activity or group project from school.",
            profile=profile,
            answer=(
                "The team handled the work. I mostly watched, listened during meetings, "
                "and tried to understand what others were doing. I did not own a specific task."
            ),
        )

        scoring = heuristic_label_answer(generated)

        self.assertEqual(scoring.scores["personal_contribution"], 1)


if __name__ == "__main__":
    unittest.main()
