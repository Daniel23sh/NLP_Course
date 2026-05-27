import unittest

from src.distribution_analysis import find_underrepresented_bands, score_distribution
from src.schemas import ASPECTS
from tests.test_validator import make_record


class DistributionAnalysisTests(unittest.TestCase):
    def test_identifies_underrepresented_score_bands(self):
        record = make_record({aspect: 3 for aspect in ASPECTS})
        record.final_scores = {aspect: 3 for aspect in ASPECTS}

        distribution = score_distribution([record])
        missing = find_underrepresented_bands([record], min_per_score=1)

        self.assertEqual(distribution["clarity"][3], 1)
        self.assertIn(1, missing["clarity"])
        self.assertIn(5, missing["impact"])


if __name__ == "__main__":
    unittest.main()
