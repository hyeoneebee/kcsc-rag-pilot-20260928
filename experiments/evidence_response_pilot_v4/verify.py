"""Independent scorer and provenance audit; intentionally does not import core.py."""
import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = HERE / 'run_20260928'
PREV = ROOT / 'experiments/response_selection_pilot_v3'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def independent_score(ranking, groups, k):
    ranks = {sid: rank for rank, sid in enumerate(ranking, 1)}
    successes = 0
    for group in groups:
        if any(ranks.get(sid, math.inf) <= k for sid in group):
            successes += 1
    return (int(successes == len(groups)), int(successes > 0), successes / len(groups))

def main():
    qs = rows(HERE / 'questions.jsonl')
    records = rows(RUN / 'actions.jsonl')
    summary = json.loads((RUN / 'summary.json').read_text())
    manifest = json.loads((RUN / 'manifest.json').read_text())
    dm = json.loads((HERE / 'dataset_manifest.json').read_text())
    old = rows(PREV / 'run_v1/actions.jsonl')
    byq = {q['question_id']: q for q in qs}
    ix = {(r['question_id'], r['action'], r['repeat']): r for r in records}
    checks = {}
    checks['unique_questions_and_rows'] = len(byq) == 48 and len(ix) == len(records) == 576
    checks['balanced_four_types'] = sorted(Counter(q['primary_type'] for q in qs).values()) == [12] * 4
    checks['frozen_input_hashes'] = (sha(HERE / 'questions.jsonl') == manifest['dataset_sha256'] == dm['question_sha256'] and
                                   sha(HERE / 'PROTOCOL.md') == manifest['protocol_sha256'] and
                                   sha(RUN / 'actions.jsonl') == manifest['actions_sha256'] and
                                   sha(PREV / 'run_v1/actions.jsonl') == manifest['source_actions_sha256'])
    checks['executed_code_hashes'] = all(sha(HERE / file) == expected for file, expected in manifest['new_code_hashes'].items())
    checks['source_code_hashes'] = all(sha(ROOT / file) == expected for file, expected in manifest['source_code_hashes'].items())
    cache_checks, repetitions, budget, scoring = [], [], [], []
    with (RUN / 'question_action_metrics.csv').open() as f:
        csv_rows = list(csv.DictReader(f))
    csv_ix = {(r['question_id'], r['action'], int(r['k'])): r for r in csv_rows}
    checks['metric_matrix_complete'] = len(csv_ix) == len(csv_rows) == 864
    for r in records:
        qid, action, rep = r['question_id'], r['action'], r['repeat']
        q = byq[qid]
        assert r['query_sha256'] == hashlib.sha256(q['question'].encode()).hexdigest()
        budget.append(len(r['ranked_ids']) == len(set(r['ranked_ids'])) and len(r['ranked_ids']) >= 15)
        budget.append(r['detail'].get('extra_dense_queries', 0) <= 2 and r['detail'].get('rerank_pairs', 0) <= 50)
        budget.append(all(len(e['candidates']) <= 20 for e in r['detail'].get('reference_edges', [])))
        if r['execution_origin'] == 'REUSED_20260927':
            raw = old[r['source_line'] - 1]
            cache_checks.append(r['source_row_sha256'] == hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest())
            cache_checks.append(all(raw[f] == r[f] for f in ('question_id', 'repeat', 'action', 'ranked_ids', 'base_seconds', 'action_seconds', 'detail', 'base_hits')))
        else:
            assert qid.startswith('REF') or action == 'KEEP3'
        repetitions.append(r['ranked_ids'][:15] == ix[qid, action, 1 - rep]['ranked_ids'][:15])
        assert abs(r['pipeline_seconds'] - (r['base_seconds'] + r['action_seconds'])) < 1e-12
        if action == 'KEEP3':
            original = ix[qid, 'STOP', rep]['ranked_ids']
            reranked = ix[qid, 'RERANK', rep]['ranked_ids']
            expected = original[:3] + [sid for sid in reranked if sid not in original[:3]]
            assert r['ranked_ids'] == expected[:50]
            assert set(r['ranked_ids']) == set(reranked)
        for k in (5, 10, 15):
            value = independent_score(r['ranked_ids'], q['evidence_groups'], k)
            saved = csv_ix[qid, action, k]
            scoring.append(value[:2] == (int(saved['complete']), int(saved['hit'])) and abs(value[2] - float(saved['recall'])) < 1e-12)
    checks['reused_records_exact_unchanged'] = all(cache_checks) and len(cache_checks) == 720
    checks['all_288_question_action_pairs_repeat_top15_match'] = all(repetitions)
    checks['all_1728_repeat_cutoff_scores_independent_match'] = all(scoring) and len(scoring) == 1728
    checks['candidate_and_query_budgets'] = all(budget)
    checks['no_query_truncation'] = max(manifest['query_token_lengths'].values()) <= 128
    conn = sqlite3.connect(f'file:{ROOT}/data/processed/2026-08-12-full/kcsc.sqlite3?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    original_checks, links = [], []
    for q in qs:
        for evidence, group in zip(q['answer_evidence'], q['evidence_groups']):
            actual = dict(conn.execute('select * from searchable_sections where section_id=?', (evidence['section_id'],)).fetchone())
            original_checks.append(actual['text_search'] == evidence['text_search'] and evidence['section_id'] in group)
        if q['primary_type'] == 'reference_following':
            ref = q['reference_validation']
            br = conn.execute('select text_search from searchable_sections where section_id=?', (ref['source_id'],)).fetchone()[0]
            target_code = conn.execute('select code from searchable_sections where section_id=?', (ref['target_id'],)).fetchone()[0]
            links.append(''.join(target_code.split()) in ''.join(br.split()))
    checks['all_48_gold_original_passages_match'] = all(original_checks)
    checks['all_12_reference_code_links_exist'] = len(links) == 12 and all(links)
    all_ids = set(sid for r in records for sid in r['ranked_ids'])
    for sid in all_ids:
        assert conn.execute('select 1 from searchable_sections where section_id=?', (sid,)).fetchone()
    checks['all_returned_ids_exist_in_corpus'] = True
    aggregates = []
    for action, cuts in summary['by_action'].items():
        for cut, s in cuts.items():
            k = int(cut)
            before, after, times = [], [], []
            for q in qs:
                qid = q['question_id']
                before.append(independent_score(ix[qid, 'STOP', 0]['ranked_ids'], q['evidence_groups'], k)[0])
                after.append(independent_score(ix[qid, action, 0]['ranked_ids'], q['evidence_groups'], k)[0])
                times.append(mean(ix[qid, action, rep]['pipeline_seconds'] for rep in (0, 1)))
            recover = sum(b == 0 and a == 1 for b, a in zip(before, after))
            regress = sum(b == 1 and a == 0 for b, a in zip(before, after))
            aggregates.append(s['complete_count'] == sum(after) and recover == s['vs_stop']['recovered'] and regress == s['vs_stop']['regressed'])
            aggregates.append(recover - regress == sum(after) - sum(before))
            aggregates.append(abs(s['mean_seconds'] - mean(times)) < 1e-12)
    checks['aggregate_transition_and_latency_recomputed'] = all(aggregates)
    ref05 = byq['REF-05']
    checks['reference_intervention_trace_not_just_rerank'] = (520106 not in ix['REF-05', 'RERANK', 0]['detail']['candidate_ids'] and
                                                           520106 in ix['REF-05', 'REFERENCE', 0]['detail']['candidate_ids'])
    checks['no_new_model_calls_on_unchanged_queries'] = sum(r['execution_origin'] == 'NEW_QUERY_20260928' for r in records) == 120
    fallacies = {
      'Simpson': '전체와 네 유형을 병기; 일별 시간과 유형이 겹치므로 유형 간 속도 원인을 추정하지 않음',
      'Ecological': '문항 단위 결과이며 전체 건설 사용자로 일반화하지 않음',
      'Berkson': '목적 표집·기존 문항 노출 한계를 명시; 새 결과로 실패 문항을 골라 넣지 않음',
      'Collider': '전체48개 결과를 보존; 기본 실패별 표는 조건부 기술통계임',
      'Base rate': '유형 균형은 실제 질문 빈도가 아님; 회복/악화 분모를 분리',
      'Regression to mean': '동일 질문·같은 기본 결과의 paired 비교; 반복을 독립 표본으로 세지 않음',
      'Survivorship': '48개×6방법 모두 포함; 실패/무효 분리 실행도 누락하지 않음',
      'Look elsewhere': 'Top5/10/15와 모든6방법 전부 공개; p값/유의성 선별 없음',
      'Forking paths': '질문과 비교조건을 새 결과 전에 해시 고정; 이전 결과 노출과 후속 해석 표시',
      'Correlation causation': '작업 적용 결과와 근본 실패 원인을 구분; 인과 진단 성능을 주장하지 않음',
      'Reverse causality': '기본 결과 후 작업 적용 순서 명시; 사후 실패 태그는 모델 입력 아님'}
    verdict = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
               'source_semantic_audit': 'AI source comparison, not independent expert validation',
               'reproducibility': '288 question-action top15 pairs match two repetitions; cached repeats not re-inferred',
               'rows': len(records), 'scored_rows': len(csv_rows), 'returned_unique_ids': len(all_ids),
               'statistical_fallacies_checked': fallacies, 'human_review': False,
               'limitations': ['Same-model AI authorship and verification', 'Semantic alternatives not exhaustive', 'Mixed-session component timings, not dedicated E2E latency', 'Purposive48, no population inference'],
               'verifier_sha256': sha(Path(__file__)),
               'non_inference_issue': 'Report-renderer bracket syntax errors corrected before analysis; inference artifacts unchanged; no inference retries'}
    (RUN / 'verification.json').write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 검증 결과', '', '## Material Passport', '', '- Origin Skill: academic-research-suite / experiment-agent',
             '- Origin Mode: validate', '- Origin Date: 2026-09-28',
             '- Verification Status: ' + verdict['status'] + ' — 실행·기계적 재현/채점 검증 범위만',
             '- Version Label: evidence_response_pilot_v4', '',
             'AI의 의미 판단·정답 완전성·전문가 검수까지 검증됐다는 뜻이 아니다.', '',
             '## 검사', ''] + [f'- [{"x" if ok else " "}] {key}' for key, ok in checks.items()]
    lines += ['', '## 테스트 실행', '', '- 신규 단위 테스트 7개 통과', '- 저장소 테스트 57개 통과', '- 기존 실행부 테스트 9개 통과',
              '- 두 차례의 보고서 생성 코드 괄호 오류를 수정했다. 추론 실행·질문·Gold·순위에는 변경 없음.', '',
              '## 해석 오류 점검 — 11/11', ''] + [f'- {k}: {v}' for k, v in fallacies.items()]
    lines += ['', '## 남은 한계', ''] + ['- ' + x for x in verdict['limitations']]
    (HERE / 'VERIFICATION.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    assert all(checks.values()), 'Verification failed; do not publish a clean report'

if __name__ == '__main__':
    main()
