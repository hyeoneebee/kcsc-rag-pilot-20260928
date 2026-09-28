"""Independent, offline audit of public artifacts; no experiment-core imports."""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
V4 = 'experiments/evidence_response_pilot_v4/'
V3 = 'experiments/response_selection_pilot_v3/'
ACTIONS = ('STOP', 'RERANK', 'REFERENCE', 'SPLIT', 'HYBRID_RERANK', 'KEEP3')
CUTOFFS = (5, 10, 15)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(ranking, groups, k):
    if not groups or any(not group for group in groups):
        raise ValueError('Empty gold group')
    if k < 1 or len(ranking) != len(set(ranking)):
        raise ValueError('Invalid ranking/cutoff')
    positions = {sid: pos for pos, sid in enumerate(ranking, 1)}
    hits = sum(any(positions.get(sid, math.inf) <= k for sid in group) for group in groups)
    return {'complete': int(hits == len(groups)), 'hit': int(hits > 0), 'recall': hits / len(groups)}


def stage(ranking, groups, k):
    missing = [g for g in groups if not score(ranking, [g], k)['complete']]
    absent = [g for g in missing if not score(ranking, [g], 100)['complete']]
    return 'SUCCESS' if not missing else 'RANK_ONLY' if not absent else 'CANDIDATE_ABSENT' if len(absent) == len(missing) else 'MIXED'


def transitions(before, after):
    if len(before) != len(after):
        raise ValueError('Different question counts')
    rec = sum(a == 0 and b == 1 for a, b in zip(before, after))
    reg = sum(a == 1 and b == 0 for a, b in zip(before, after))
    assert rec - reg == sum(after) - sum(before)
    return dict(recovered=rec, regressed=reg,
                recovery_rate=rec / (len(before) - sum(before)) if len(before) != sum(before) else None,
                regression_rate=reg / sum(before) if sum(before) else None)


def percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low = int(position)
    high = math.ceil(position)
    return ordered[low] * (high - position) + ordered[high] * (position - low) if high != low else ordered[low]


def equal(actual, expected, label):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), ('keys', label)
        for key in expected:
            equal(actual[key], expected[key], f'{label}/{key}')
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12), (label, actual, expected)
    else:
        assert actual == expected, (label, actual, expected)


def verify(root):
    read = lambda p: json.loads((root / p).read_text())
    rows = lambda p: [json.loads(x) for x in (root / p).read_text().splitlines()]
    checks = {}
    listed = set()
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        expected, rel = line.split('  ', 1)
        assert rel not in listed and not Path(rel).is_absolute() and '..' not in Path(rel).parts
        assert digest(root / rel) == expected, ('package checksum', rel)
        listed.add(rel)
    checks['package_checksums'] = len(listed)
    source = read('provenance/export_manifest.json')
    source_ix = {e['source_path']: e for e in source['files']}
    for e in source['files']:
        assert digest(root / e['package_path']) == e['package_sha256']
        assert (e['source_sha256'] == e['package_sha256']) == e['byte_identical']
    original_sha = lambda p: source_ix[p]['source_sha256']
    checks['documented_source_to_public_transformations'] = len(source_ix)
    qlist = rows(V4 + 'questions.jsonl')
    qs = {q['question_id']: q for q in qlist}
    assert len(qs) == len(qlist) == 48
    assert sorted(Counter(q['primary_type'] for q in qlist).values()) == [12] * 4
    assert all('text_search' not in e for q in qlist for e in q['answer_evidence'])
    assert all('bridge_text' not in q.get('reference_validation', {}) for q in qlist)
    checks['questions_balanced_and_verbatim_evidence_omitted'] = True
    run = V4 + 'run_20260928/'
    manifest = read(run + 'manifest.json')
    dm = read(V4 + 'dataset_manifest.json')
    oldmanifest = read(V3 + 'run_v1/manifest.json')
    assert original_sha(V4 + 'questions.jsonl') == manifest['dataset_sha256'] == dm['question_sha256']
    assert original_sha(V3 + 'questions.v2.jsonl') == dm['source_question_sha256']
    assert original_sha(V3 + 'questions.v1.jsonl') == oldmanifest['question_sha256']
    assert original_sha(V3 + 'run_v1/manifest.json') == manifest['source_run_manifest_sha256']
    assert digest(root / (V3 + 'run_v1/actions.jsonl')) == manifest['source_actions_sha256']
    assert digest(root / (V4 + 'PROTOCOL.md')) == manifest['protocol_sha256']
    for name, sha in manifest['new_code_hashes'].items():
        assert digest(root / (V4 + name)) == sha
    for name, sha in manifest['source_code_hashes'].items():
        assert digest(root / name) == sha
    analysis = read(run + 'analysis_manifest.json')
    assert digest(root / (V4 + 'analyze.py')) == analysis['analysis_script_sha256']
    assert digest(root / (V4 + 'core.py')) == analysis['metric_code_sha256']
    assert analysis['input_questions_sha256'] == original_sha(V4 + 'questions.jsonl')
    checks['original_and_export_hashes_distinguished_and_code_frozen'] = True
    data = rows(run + 'actions.jsonl')
    assert digest(root / (run + 'actions.jsonl')) == manifest['actions_sha256'] == analysis['input_actions_sha256']
    ix = {(r['question_id'], r['action'], r['repeat']): r for r in data}
    expected_keys = {(qid, action, rep) for qid in qs for action in ACTIONS for rep in (0, 1)}
    assert set(ix) == expected_keys and len(data) == len(ix) == 576
    assert Counter(r['execution_origin'] for r in data) == {'REUSED_20260927': 360, 'NEW_QUERY_20260928': 120, 'NEW_TRANSFORM_20260928': 96}
    old = rows(V3 + 'run_v1/actions.jsonl')
    oldq = {q['question_id']: q for q in rows(V3 + 'questions.v1.jsonl')}
    for qid in manifest['reused_queries']:
        assert qs[qid]['question'] == oldq[qid]['question']
        assert qs[qid]['evidence_groups'] == oldq[qid]['evidence_groups']
    checks['complete_576_record_matrix_and_36_query_reuse_contract'] = True
    with (root / (run + 'question_action_metrics.csv')).open() as f:
        csv_rows = list(csv.DictReader(f))
    ci = {(r['question_id'], r['action'], int(r['k'])): r for r in csv_rows}
    assert len(ci) == len(csv_rows) == 864
    assert set(ci) == {(qid, a, k) for qid in qs for a in ACTIONS for k in CUTOFFS}
    score_count = reused = 0
    for (qid, action, rep), r in ix.items():
        assert r['query_sha256'] == hashlib.sha256(qs[qid]['question'].encode()).hexdigest()
        assert len(r['ranked_ids']) == (100 if action == 'STOP' else 50)
        assert len(r['ranked_ids']) == len(set(r['ranked_ids']))
        assert r['ranked_ids'][:15] == ix[qid, action, 1 - rep]['ranked_ids'][:15]
        equal(r['pipeline_seconds'], r['base_seconds'] + r['action_seconds'], 'component timing')
        assert r['detail'].get('extra_dense_queries', 0) <= 2
        assert r['detail'].get('rerank_pairs', 0) <= 50
        assert all(len(e['candidates']) <= 20 for e in r['detail'].get('reference_edges', []))
        if r['execution_origin'] == 'REUSED_20260927':
            raw = old[r['source_line'] - 1]
            assert hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest() == r['source_row_sha256']
            for field in ('question_id', 'action', 'repeat', 'ranked_ids', 'base_seconds', 'action_seconds', 'detail', 'base_hits'):
                assert raw[field] == r[field], ('reused record changed', qid, action, field)
            reused += 1
        elif r['execution_origin'] == 'NEW_QUERY_20260928':
            assert qid in manifest['new_queries'] and action != 'KEEP3'
        if action == 'KEEP3':
            keep = ix[qid, 'STOP', rep]['ranked_ids'][:3]
            rerank = ix[qid, 'RERANK', rep]['ranked_ids']
            assert r['ranked_ids'] == (keep + [x for x in rerank if x not in keep])[:50]
            assert set(r['ranked_ids']) == set(rerank)
        for k in CUTOFFS:
            actual = score(r['ranked_ids'], qs[qid]['evidence_groups'], k)
            expected = ci[qid, action, k]
            for key, val in actual.items():
                equal(val, float(expected[key]), f'{qid}/{action}/{rep}/{k}/{key}')
            equal(mean(ix[qid, action, j]['pipeline_seconds'] for j in (0, 1)), float(expected['mean_seconds']), 'CSV mean time')
            equal(mean(ix[qid, action, j]['action_seconds'] for j in (0, 1)), float(expected['mean_action_seconds']), 'CSV action time')
            assert expected['base_stage'] == stage(ix[qid, 'STOP', 0]['ranked_ids'], qs[qid]['evidence_groups'], k)
            score_count += 1
    assert reused == 360 and score_count == 1728
    checks.update(reused_records_identical=reused, repeated_top15_pairs=288, independently_scored_repeat_cutoffs=score_count, complete_csv_rows=864)

    def aggregate(ids, action, k):
        values = [score(ix[qid, action, 0]['ranked_ids'], qs[qid]['evidence_groups'], k) for qid in ids]
        after = [v['complete'] for v in values]
        before = [score(ix[qid, 'STOP', 0]['ranked_ids'], qs[qid]['evidence_groups'], k)['complete'] for qid in ids]
        rr = [score(ix[qid, 'RERANK', 0]['ranked_ids'], qs[qid]['evidence_groups'], k)['complete'] for qid in ids]
        times = [mean(ix[qid, action, r]['pipeline_seconds'] for r in (0, 1)) for qid in ids]
        base_times = [mean(ix[qid, 'STOP', r]['pipeline_seconds'] for r in (0, 1)) for qid in ids]
        return dict(n=len(ids), complete_count=sum(after), complete_rate=mean(after),
                    hit_rate=mean(v['hit'] for v in values), evidence_recall=mean(v['recall'] for v in values),
                    mean_seconds=mean(times), p50_seconds=percentile(times, .5), p95_seconds=percentile(times, .95),
                    mean_extra_dense_queries=mean(ix[qid, action, 0]['detail'].get('extra_dense_queries', 0) for qid in ids),
                    mean_lexical_queries=int(action == 'HYBRID_RERANK'),
                    mean_rerank_pairs=mean(ix[qid, action, 0]['detail'].get('rerank_pairs', 0) for qid in ids),
                    mean_delta_vs_stop_seconds=mean(a-b for a, b in zip(times, base_times)),
                    vs_stop=transitions(before, after), vs_rerank=transitions(rr, after))

    summary = read(run + 'summary.json')
    assert summary['dataset_n'] == 48 and summary['cutoffs'] == list(CUTOFFS)
    for action in ACTIONS:
        for k in CUTOFFS:
            equal(aggregate(list(qs), action, k), summary['by_action'][action][str(k)], f'aggregate/{action}/{k}')
            for kind, group in summary['by_type'].items():
                ids = [qid for qid, q in qs.items() if q['primary_type'] == kind]
                equal(aggregate(ids, action, k), group[action][str(k)], f'type/{kind}/{action}/{k}')
        for origin, group in summary['by_origin'].items():
            ids = [qid for qid in qs if ix[qid, 'STOP', 0]['execution_origin'] == origin]
            equal(aggregate(ids, action, 10), group[action], f'origin/{origin}/{action}')
        for kind, group in summary['by_baseline_stage_at10'].items():
            ids = [qid for qid in qs if stage(ix[qid, 'STOP', 0]['ranked_ids'], qs[qid]['evidence_groups'], 10) == kind]
            equal(aggregate(ids, action, 10), group[action], f'stage/{kind}/{action}')
    for k in CUTOFFS:
        counts = dict(Counter(stage(ix[qid, 'STOP', 0]['ranked_ids'], qs[qid]['evidence_groups'], k) for qid in qs))
        equal(counts, summary['baseline_stages'][str(k)], f'stages@{k}')
    complete = lambda qid, action, k: score(ix[qid, action, 0]['ranked_ids'], qs[qid]['evidence_groups'], k)['complete']
    assert summary['all_actions_success_at10'] == [qid for qid in qs if all(complete(qid, a, 10) for a in ACTIONS)]
    assert summary['all_actions_fail_at10'] == [qid for qid in qs if not any(complete(qid, a, 10) for a in ACTIONS)]
    assert summary['reference_expansion_triggered'] == [qid for qid in qs if ix[qid, 'REFERENCE', 0]['detail']['reference_edges']]
    assert summary['split_triggered'] == [qid for qid in qs if ix[qid, 'SPLIT', 0]['detail']['extra_dense_queries']]
    assert 520106 not in ix['REF-05', 'RERANK', 0]['detail']['candidate_ids']
    assert 520106 in ix['REF-05', 'REFERENCE', 0]['detail']['candidate_ids']
    assert complete('REF-05', 'REFERENCE', 10) == 0 and complete('REF-05', 'REFERENCE', 15) == 1
    assert max(manifest['query_token_lengths'].values()) <= 128
    checks['all_summary_groups_times_transitions_stages_and_key_trace_match'] = True
    checks['full_db_or_model_inference_required_for_this_audit'] = False
    return dict(status='PASS', dataset='evidence_response_pilot_v4_20260928',
                checks=checks, independently_rerun_inference=False, human_validation=False,
                not_rechecked=['Original corpus text and all returned section existence',
                               'Reference code strings in omitted original passages',
                               'Semantic completeness of gold and expert validity',
                               'GPU inference, full 73-test source repository suite, bibliographic accuracy'],
                original_verification_record=run + 'verification.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-report', type=Path)
    args = parser.parse_args()
    result = verify(ROOT)
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.write_report:
        args.write_report.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
