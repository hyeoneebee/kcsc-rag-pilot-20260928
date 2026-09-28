"""Small auditable transforms; no question type or gold used by a response."""
import math

ACTIONS = ('STOP', 'RERANK', 'REFERENCE', 'SPLIT', 'HYBRID_RERANK', 'KEEP3')
CUTOFFS = (5, 10, 15)

def preserve_top3(base_ids, reranked_ids):
    return list(dict.fromkeys(list(base_ids[:3]) + list(reranked_ids)))[:50]

def evaluate(ids, groups, k):
    if not groups or any(not group for group in groups):
        raise ValueError('Nonempty evidence groups required')
    present = set(ids[:k])
    found = [bool(present.intersection(group)) for group in groups]
    return {'complete': int(all(found)), 'hit': int(any(found)), 'recall': sum(found) / len(found)}

def stage(ids, groups, k):
    missing = [g for g in groups if not set(ids[:k]).intersection(g)]
    if not missing:
        return 'SUCCESS'
    absent = [g for g in missing if not set(ids[:100]).intersection(g)]
    if not absent:
        return 'RANK_ONLY'
    return 'CANDIDATE_ABSENT' if len(absent) == len(missing) else 'MIXED'

def paired_counts(before, after):
    if len(before) != len(after):
        raise ValueError('Mismatched pairs')
    recovered = sum(not a and b for a, b in zip(before, after))
    regressed = sum(a and not b for a, b in zip(before, after))
    assert recovered - regressed == sum(after) - sum(before)
    return {'recovered': recovered, 'regressed': regressed,
            'recovery_rate': recovered / (len(before) - sum(before)) if len(before) > sum(before) else None,
            'regression_rate': regressed / sum(before) if sum(before) else None}

def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    x = (len(values) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)
