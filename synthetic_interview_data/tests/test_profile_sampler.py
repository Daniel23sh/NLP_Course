import unittest

from src.profile_sampler import sample_generation_requests


class ProfileSamplerTests(unittest.TestCase):
    def test_sampled_profiles_do_not_contain_target_scores(self):
        requests = sample_generation_requests(total=12, seed=7)

        self.assertEqual(len(requests), 12)
        for request in requests:
            payload = request.profile.to_dict()
            self.assertNotIn("target_scores", payload)
            self.assertIn(request.project_domain, request.profile.likely_domains)
            self.assertTrue(request.question_id.startswith("q_"))


if __name__ == "__main__":
    unittest.main()
