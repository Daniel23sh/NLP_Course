import unittest

from src.schemas import ASPECTS
from src.targeted_generation import profile_for_underrepresented_band


class TargetedGenerationTests(unittest.TestCase):
    def test_targeted_profile_uses_controls_not_trusted_final_labels(self):
        profile = profile_for_underrepresented_band("impact", 1, seed=5)

        payload = profile.to_dict()
        self.assertEqual(profile.outcome_strength, "none")
        self.assertNotIn("target_scores", payload)
        for aspect in ASPECTS:
            self.assertNotIn(aspect + "_score", payload)


if __name__ == "__main__":
    unittest.main()
