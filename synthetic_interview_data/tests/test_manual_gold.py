import tempfile
import unittest
from pathlib import Path

from src.manual_gold import apply_manual_gold, load_manual_gold
from src.schemas import ASPECTS
from tests.test_validator import make_record


class ManualGoldTests(unittest.TestCase):
    def test_verified_manual_gold_overrides_final_scores(self):
        record = make_record({aspect: 3 for aspect in ASPECTS})
        record.validation.final_status = "accepted"
        record.final_scores = {aspect: 3 for aspect in ASPECTS}
        manual_scores = {aspect: 4 for aspect in ASPECTS}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manual.jsonl"
            path.write_text(
                '{"example_id":"ex_test","manual_scores":'
                + str(manual_scores).replace("'", '"')
                + ',"manual_notes":"checked","reviewer_id":"r1","review_status":"verified"}\n',
                encoding="utf-8",
            )

            manual = load_manual_gold(path)
            updated = apply_manual_gold([record], manual)[0]

        self.assertEqual(updated.final_scores, manual_scores)
        self.assertEqual(updated.metadata["manual_review_status"], "verified")


if __name__ == "__main__":
    unittest.main()
