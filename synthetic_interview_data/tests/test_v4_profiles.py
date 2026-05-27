import unittest

from src.config_loader import load_profiles
from src.schemas import ASPECTS, ProfileSpec
from src.targeted_generation import profile_for_v4_weak_aspect, v4_weak_profile


class V4ProfilesTests(unittest.TestCase):
    def test_all_v4_profiles_load_with_constraints_and_without_target_scores(self):
        profiles = load_profiles()["profiles"]
        expected = {
            "team_only_no_personal_ownership",
            "observer_no_personal_task",
            "rambling_unclear_answer",
            "fragmented_missing_context_answer",
            "severely_unclear_but_realistic_answer",
            "no_outcome_answer",
            "unfinished_project_no_result",
            "irrelevant_non_software_project",
            "non_software_team_activity",
            "generic_school_assignment_no_coding",
            "project_manager_observer_role",
            "presentation_only_project",
            "learning_reflection_without_project",
            "design_only_no_implementation",
            "irrelevant_volunteer_project",
            "vague_tools_no_explanation",
            "polished_but_shallow_answer",
        }

        self.assertTrue(expected.issubset(profiles))
        for profile_id in expected:
            payload = profiles[profile_id]
            self.assertNotIn("target_scores", payload)
            self.assertIn("expected_score_constraints", payload)
            profile = ProfileSpec.from_dict({"profile_id": profile_id, **payload})
            self.assertTrue(profile.expected_score_constraints)
            for aspect in profile.expected_score_constraints:
                self.assertIn(aspect, ASPECTS)

    def test_v4_targeted_weak_builders_map_to_intended_profiles(self):
        self.assertEqual(profile_for_v4_weak_aspect("personal_contribution", seed=1).profile_id, "team_only_no_personal_ownership")
        self.assertIn(profile_for_v4_weak_aspect("clarity", seed=2).profile_id, {"rambling_unclear_answer", "fragmented_missing_context_answer"})
        self.assertIn(profile_for_v4_weak_aspect("impact", seed=3).profile_id, {"no_outcome_answer", "unfinished_project_no_result"})
        self.assertIn(
            profile_for_v4_weak_aspect("role_relevance", seed=4).profile_id,
            {
                "non_software_team_activity",
                "generic_school_assignment_no_coding",
                "project_manager_observer_role",
                "presentation_only_project",
                "learning_reflection_without_project",
                "design_only_no_implementation",
                "irrelevant_volunteer_project",
            },
        )
        self.assertIn(profile_for_v4_weak_aspect("technical_depth", seed=5).profile_id, {"vague_tools_no_explanation", "polished_but_shallow_answer"})
        self.assertEqual(v4_weak_profile("polished_but_shallow_answer").expected_score_constraints["technical_depth"]["max"], 2)

    def test_v4_low_role_relevance_profiles_are_strict(self):
        for profile_id in {
            "non_software_team_activity",
            "generic_school_assignment_no_coding",
            "project_manager_observer_role",
            "presentation_only_project",
            "learning_reflection_without_project",
            "design_only_no_implementation",
            "irrelevant_volunteer_project",
        }:
            profile = v4_weak_profile(profile_id)
            self.assertEqual(profile.expected_score_constraints["role_relevance"]["max"], 2)
            self.assertEqual(profile.expected_score_constraints["role_relevance"]["strict"], 1)


if __name__ == "__main__":
    unittest.main()
