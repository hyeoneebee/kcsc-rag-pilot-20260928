import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('audit', Path(__file__).resolve().parents[1] / 'tools/verify_handoff.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def test_alternative_and_required_groups(self):
        self.assertEqual(audit.score([2, 4, 8], [[1, 2], [8]], 2), {'complete': 0, 'hit': 1, 'recall': .5})
        self.assertEqual(audit.score([2, 4, 8], [[1, 2], [8]], 3)['complete'], 1)

    def test_missing_all(self):
        self.assertEqual(audit.score([1, 2], [[3], [4]], 2), {'complete': 0, 'hit': 0, 'recall': 0})

    def test_invalid_gold_and_ranking(self):
        for groups in ([], [[]]):
            with self.assertRaises(ValueError):
                audit.score([1], groups, 1)
        with self.assertRaises(ValueError):
            audit.score([1, 1], [[1]], 1)

    def test_transition_denominators(self):
        self.assertEqual(audit.transitions([1, 1, 0, 0], [0, 1, 1, 1]),
                         {'recovered': 2, 'regressed': 1, 'recovery_rate': 1, 'regression_rate': .5})
        self.assertIsNone(audit.transitions([1], [1])['recovery_rate'])
        self.assertIsNone(audit.transitions([0], [0])['regression_rate'])

    def test_stage_distinction(self):
        self.assertEqual(audit.stage([1, 2, 3], [[1]], 1), 'SUCCESS')
        self.assertEqual(audit.stage([1, 2, 3], [[1], [3]], 1), 'RANK_ONLY')
        self.assertEqual(audit.stage([1, 2, 3], [[1], [9]], 1), 'CANDIDATE_ABSENT')
        self.assertEqual(audit.stage([1, 2, 3], [[1], [3], [9]], 1), 'MIXED')

    def test_percentiles(self):
        self.assertEqual(audit.percentile([4, 1, 3, 2], .5), 2.5)
        self.assertEqual(audit.percentile([7], .95), 7)

    def test_summary_mismatch_fails(self):
        with self.assertRaises(AssertionError):
            audit.equal({'count': 46}, {'count': 47}, 'deliberately wrong result')


if __name__ == '__main__':
    unittest.main()
