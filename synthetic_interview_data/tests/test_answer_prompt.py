import unittest

from src.generator import build_answer_prompt
from src.profile_sampler import sample_generation_requests
from src.schemas import ASPECTS


class AnswerPromptTests(unittest.TestCase):
    def test_answer_prompt_contains_profile_controls_but_no_score_labels(self):
        request = sample_generation_requests(total=1, seed=3)[0]

        prompt = build_answer_prompt(request.profile, request.question_payload)

        self.assertIn("junior developer interview answer", prompt.lower())
        self.assertIn(request.project_domain, prompt)
        for aspect in ASPECTS:
            self.assertNotIn(aspect, prompt)
        self.assertNotIn("rubric", prompt.lower())
        self.assertNotIn("score", prompt.lower())
        self.assertNotIn("target", prompt.lower())


if __name__ == "__main__":
    unittest.main()
