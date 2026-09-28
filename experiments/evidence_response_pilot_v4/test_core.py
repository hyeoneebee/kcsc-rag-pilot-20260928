import unittest
from core import evaluate, paired_counts, preserve_top3, stage

class CoreTests(unittest.TestCase):
    def test_alternative_group(self):
        self.assertEqual(evaluate([8, 5, 2], [[5, 6], [2, 3]], 3)['complete'], 1)
        self.assertEqual(evaluate([8, 5, 2], [[5, 6], [2, 3]], 2)['recall'], .5)

    def test_empty_gold_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([1], [], 1)

    def test_preserve_and_deduplicate(self):
        self.assertEqual(preserve_top3([1, 2, 3, 4], [4, 3, 2, 1]), [1, 2, 3, 4])

    def test_preserve_does_not_add_new_candidate(self):
        base = list(range(100))
        ranked = list(reversed(range(50)))
        result = preserve_top3(base, ranked)
        self.assertEqual(set(result), set(ranked))
        self.assertEqual(len(result), 50)

    def test_missing_stage(self):
        self.assertEqual(stage([1, 2, 3], [[1], [3]], 2), 'RANK_ONLY')
        self.assertEqual(stage([1, 2, 3], [[1], [9]], 2), 'CANDIDATE_ABSENT')
        self.assertEqual(stage([1, 2, 3], [[1], [3], [9]], 2), 'MIXED')

    def test_transition_conservation(self):
        result = paired_counts([1, 1, 0, 0], [0, 1, 1, 1])
        self.assertEqual(result['recovered'], 2)
        self.assertEqual(result['regressed'], 1)

    def test_cutoff_monotonic(self):
        values = [evaluate(list(range(20)), [[3], [12]], k) for k in [5, 10, 15]]
        self.assertEqual([v['complete'] for v in values], [0, 0, 1])

if __name__ == '__main__':
    unittest.main()
