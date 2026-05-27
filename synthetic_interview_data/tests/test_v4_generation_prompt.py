import unittest

from src.generator import build_answer_prompt
from src.schemas import ASPECTS
from src.targeted_generation import v4_weak_profile


class V4GenerationPromptTests(unittest.TestCase):
    def test_v4_weak_prompt_uses_behavioral_constraints_without_score_labels(self):
        profile = v4_weak_profile("team_only_no_personal_ownership")
        prompt = build_answer_prompt(
            profile,
            {
                "question": "Tell me about a software project you worked on.",
                "project_domain": "data analysis dashboard",
            },
        )
        lowered = prompt.lower()

        self.assertIn("team-level", lowered)
        self.assertIn("avoid", lowered)
        for aspect in ASPECTS:
            self.assertNotIn(aspect, lowered)
        self.assertNotIn("score", lowered)
        self.assertNotIn("rubric", lowered)
        self.assertNotIn("numeric", lowered)

    def test_low_role_relevance_prompt_avoids_developer_framing_and_technical_work(self):
        profile = v4_weak_profile("presentation_only_project")
        prompt = build_answer_prompt(
            profile,
            {
                "question": "Tell me about a presentation or class project you worked on.",
                "project_domain": "presentation project",
            },
        )
        lowered = prompt.lower()

        self.assertNotIn("write a natural first-person junior developer interview answer", lowered)
        self.assertIn("junior developer framing", lowered)
        self.assertNotIn("software interview", lowered)
        self.assertIn("forbid", lowered)
        self.assertIn("coding", lowered)
        self.assertIn("task boards", lowered)
        self.assertIn("strong project narrative", lowered)

    def test_observer_prompt_forbids_candidate_owned_action_phrases(self):
        profile = v4_weak_profile("observer_no_personal_task")
        prompt = build_answer_prompt(
            profile,
            {
                "question": "Tell me about a team activity or group project from school.",
                "project_domain": "non-software team activity",
            },
        )
        lowered = prompt.lower()

        self.assertIn("do not use", lowered)
        for phrase in ["i helped", "i contributed", "my role", "i organized", "i kept track"]:
            self.assertIn(phrase, lowered)


if __name__ == "__main__":
    unittest.main()
