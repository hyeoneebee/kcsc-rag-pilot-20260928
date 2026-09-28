"""Descriptive paired analysis. No policy fitting, no post-hoc query selection."""
import csv
import html
import json
import sqlite3
from collections import Counter
from pathlib import Path
from statistics import mean
from core import ACTIONS, CUTOFFS, evaluate, paired_counts, percentile, stage
from build_dataset import digest

HERE = Path(__file__).resolve().parent
RUN = HERE / 'run_20260928'
LABELS = {'STOP': '의미 검색 그대로', 'RERANK': '의미 검색+재정렬', 'REFERENCE': '참조 보충+재정렬',
          'SPLIT': '질문 분리+재정렬', 'HYBRID_RERANK': '결합 검색+재정렬', 'KEEP3': '재정렬+기존 상위3 보존'}
TYPES = {'direct_single_clause': '단일 근거', 'applicability_near_miss': '적용 조건 구분',
         'reference_following': '참조 관계', 'multi_evidence': '복수 근거'}
STAGES = {'SUCCESS': '기본 검색 성공', 'RANK_ONLY': '후보는 있지만 순위 부족',
          'CANDIDATE_ABSENT': '필수 근거가 후보100에 없음', 'MIXED': '순위 부족+후보 미확보'}

def read(path):
    return [json.loads(x) for x in path.read_text().splitlines()]

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])

def main():
    questions = read(HERE / 'questions.jsonl')
    records = read(RUN / 'actions.jsonl')
    manifest = json.loads((RUN / 'manifest.json').read_text())
    assert manifest['status'] == 'COMPLETED_AI_EXPLORATORY'
    ix = {(r['question_id'], r['action'], r['repeat']): r for r in records}
    assert len(ix) == len(records) == 48 * 6 * 2
    cases, flat = [], []
    for q in questions:
        qid, groups = q['question_id'], q['evidence_groups']
        base = ix[qid, 'STOP', 0]['ranked_ids']
        c = {'id': qid, 'question': q['question'], 'type': q['primary_type'], 'groups': groups,
             'origin': ix[qid, 'STOP', 0]['execution_origin'],
             'base_stage': {str(k): stage(base, groups, k) for k in CUTOFFS},
             'base_group_ranks': [min([base.index(s) + 1 for s in group if s in base], default=None) for group in groups],
             'actions': {}}
        for action in ACTIONS:
            two = [ix[qid, action, rep] for rep in range(2)]
            assert two[0]['ranked_ids'][:15] == two[1]['ranked_ids'][:15], (qid, action)
            results = {str(k): evaluate(two[0]['ranked_ids'], groups, k) for k in CUTOFFS}
            stats = {'metrics': results, 'mean_seconds': mean(r['pipeline_seconds'] for r in two),
                     'mean_action_seconds': mean(r['action_seconds'] for r in two),
                     'extra_dense_queries': two[0]['detail'].get('extra_dense_queries', 0),
                     'lexical_queries': int(action == 'HYBRID_RERANK'),
                     'rerank_pairs': two[0]['detail'].get('rerank_pairs', 0),
                     'reference_edges': two[0]['detail'].get('reference_edges', []),
                     'subqueries': two[0]['detail'].get('subqueries', []),
                     'top15': two[0]['ranked_ids'][:15]}
            c['actions'][action] = stats
            for k in CUTOFFS:
                flat.append({'question_id': qid, 'type': q['primary_type'], 'action': action, 'k': k,
                             **results[str(k)], 'base_stage': c['base_stage'][str(k)],
                             'base_complete': evaluate(base, groups, k)['complete'],
                             'mean_seconds': stats['mean_seconds'], 'mean_action_seconds': stats['mean_action_seconds'],
                             'extra_dense_queries': stats['extra_dense_queries'], 'lexical_queries': stats['lexical_queries'],
                             'rerank_pairs': stats['rerank_pairs'], 'base_origin': c['origin']})
        cases.append(c)
    def aggregate(subset, action, k):
        stats = [c['actions'][action] for c in subset]
        metrics = [s['metrics'][str(k)] for s in stats]
        times = [s['mean_seconds'] for s in stats]
        before = [c['actions']['STOP']['metrics'][str(k)]['complete'] for c in subset]
        after = [m['complete'] for m in metrics]
        rr = [c['actions']['RERANK']['metrics'][str(k)]['complete'] for c in subset]
        return {'n': len(subset), 'complete_count': sum(after), 'complete_rate': mean(after),
                'hit_rate': mean(m['hit'] for m in metrics), 'evidence_recall': mean(m['recall'] for m in metrics),
                'mean_seconds': mean(times), 'p50_seconds': percentile(times, .5), 'p95_seconds': percentile(times, .95),
                'mean_extra_dense_queries': mean(s['extra_dense_queries'] for s in stats),
                'mean_lexical_queries': mean(s['lexical_queries'] for s in stats),
                'mean_rerank_pairs': mean(s['rerank_pairs'] for s in stats),
                'mean_delta_vs_stop_seconds': mean(s['mean_seconds'] - c['actions']['STOP']['mean_seconds'] for c, s in zip(subset, stats)),
                'vs_stop': paired_counts(before, after), 'vs_rerank': paired_counts(rr, after)}
    summary = {'dataset_n': 48, 'cutoffs': list(CUTOFFS),
               'by_action': {a: {str(k): aggregate(cases, a, k) for k in CUTOFFS} for a in ACTIONS},
               'by_type': {kind: {a: {str(k): aggregate([c for c in cases if c['type'] == kind], a, k) for k in CUTOFFS} for a in ACTIONS} for kind in TYPES},
               'by_origin': {origin: {a: aggregate([c for c in cases if c['origin'] == origin], a, 10) for a in ACTIONS} for origin in sorted({c['origin'] for c in cases})},
               'baseline_stages': {str(k): dict(Counter(c['base_stage'][str(k)] for c in cases)) for k in CUTOFFS},
               'by_baseline_stage_at10': {s: {a: aggregate([c for c in cases if c['base_stage']['10'] == s], a, 10) for a in ACTIONS} for s in STAGES if any(c['base_stage']['10'] == s for c in cases)},
               'all_actions_success_at10': [c['id'] for c in cases if all(c['actions'][a]['metrics']['10']['complete'] for a in ACTIONS)],
               'all_actions_fail_at10': [c['id'] for c in cases if not any(c['actions'][a]['metrics']['10']['complete'] for a in ACTIONS)],
               'n_with_any_success_at10': sum(any(c['actions'][a]['metrics']['10']['complete'] for a in ACTIONS) for c in cases),
               'reference_expansion_triggered': [c['id'] for c in cases if c['actions']['REFERENCE']['reference_edges']],
               'split_triggered': [c['id'] for c in cases if c['actions']['SPLIT']['extra_dense_queries']],
               'stats_scope': '48 purposive questions; descriptive paired comparisons; repeats are not independent samples',
               'timing_scope': manifest['timing_scope']}
    summary['reranker_input_gap_at10'] = {
        'missing_but_all_in_top50': [c['id'] for c in cases if c['base_stage']['10'] != 'SUCCESS' and all(r is not None and r <= 50 for r in c['base_group_ranks'])],
        'some_required_outside_top50': [c['id'] for c in cases if c['base_stage']['10'] != 'SUCCESS' and any(r is None or r > 50 for r in c['base_group_ranks'])],
        'scope': 'Post-run localization of the predeclared top100/50/10 pipeline; not a new intervention'}
    (RUN / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (RUN / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2) + '\n')
    with (RUN / 'question_action_metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = ['# 전체 코퍼스 근거 확보·최소 대응 파일럿 결과', '',
             '2026-09-28 · AI 작성·원문 대조 기반 탐색 · 새 사람 검수 없음 · 선택·학습·Agent 실행 없음', '',
             '## 1. 무엇을 비교했나?', '',
             '동결된 전체 코퍼스의 48문항을 네 유형 각 12개로 비교했다. 철도에 할당량이나 제외 조건을 두지 않았다. '
             '36문항은 일치 검사를 통과한 이전 결과를 재사용했고, 참조 질문 12개를 수정·교체해 새로 검색했다. '
             '추가 작업은 실패 질문에만 배정하지 않고 모든 질문에 동일하게 적용했다.', '',
             '참조 유형은 명시적으로 연결된 조항의 서로 다른 답변 내용을 요구한다. 특정 검색 경로를 반드시 사용해야 성공인 것은 아니다. '
             'REF-12는 보강토옹벽 자체의 95% 기준과 일반 쌓기의 시설물 기준 우선 규정을 묻도록 고쳤다.', '',
             '## 2. Top-5 / Top-10 / Top-15', '',
             '성공 = 상위 k개 안에 필수 근거 묶음을 모두 확보. 생성 답변 정확도가 아니다.', '',
             table(['방법', '@5', '@10', '@15', '@10 Hit Rate', '@10 Evidence Recall'],
                   [[LABELS[a]] + [f'{summary["by_action"][a][str(k)]["complete_count"]}/48' for k in CUTOFFS] +
                    [f'{summary["by_action"][a]["10"]["hit_rate"]:.1%}', f'{summary["by_action"][a]["10"]["evidence_recall"]:.1%}'] for a in ACTIONS]), '',
             '## 3. 회복과 악화를 함께 보기 — Top-10', '',
             table(['방법', 'Dense 실패→성공', 'Dense 성공→실패', '회복률', '악화율', 'Rerank 대비 회복/악화'],
                   [[LABELS[a], summary['by_action'][a]['10']['vs_stop']['recovered'], summary['by_action'][a]['10']['vs_stop']['regressed'],
                     f'{summary["by_action"][a]["10"]["vs_stop"]["recovery_rate"]:.1%}' if summary['by_action'][a]['10']['vs_stop']['recovery_rate'] is not None else '해당 없음',
                     f'{summary["by_action"][a]["10"]["vs_stop"]["regression_rate"]:.1%}' if summary['by_action'][a]['10']['vs_stop']['regression_rate'] is not None else '해당 없음',
                     f'{summary["by_action"][a]["10"]["vs_rerank"]["recovered"]}/{summary["by_action"][a]["10"]["vs_rerank"]["regressed"]}'] for a in ACTIONS]), '',
             '참조 보충·질문 분리는 재정렬을 포함한다. 따라서 Dense와의 차이 전체를 참조·분리 자체의 효과로 해석하지 않는다. '
             '같은 재정렬만 한 RERANK와의 비교를 함께 봐야 한다.', '',
             '## 4. 유형별 성공 문항 수', '']
    for k in CUTOFFS:
        lines += [f'### Top-{k}', '', table(['유형 / 각12개'] + [LABELS[a] for a in ACTIONS],
                 [[TYPES[t]] + [summary['by_type'][t][a][str(k)]['complete_count'] for a in ACTIONS] for t in TYPES]), '']
    lines += ['## 5. 기본 검색에서 부족했던 지점', '',
              table(['상태'] + [f'Top-{k}' for k in CUTOFFS], [[STAGES[s]] + [summary['baseline_stages'][str(k)].get(s, 0) for k in CUTOFFS] for s in STAGES]), '',
              '후보100에 없는 것은 전체 코퍼스에 없다는 뜻이 아니다. 순위 부족은 원문을 못 찾은 문제가 아니라 최종 k 안에 들지 못한 상태다. '
              '질문 유형은 요구하는 근거 구조이고, 이 표는 실제 검색 결과의 부족 지점이다.', '',
              f'재정렬 입력50개를 기준으로 더 나누면, 기본 미충족 중 {len(summary["reranker_input_gap_at10"]["missing_but_all_in_top50"])}개는 모든 근거가 입력50 안에 있고, '
              f'{len(summary["reranker_input_gap_at10"]["some_required_outside_top50"])}개는 필수 근거가 입력50 밖에 있다. '
              'REF-05의 승인 절차 근거는 Dense86위여서 일반 재정렬에 들어가지 않는다. 이는 top100 이후 누락과 다른 후보 절단 문제다.', '',
              table(['기본 상태@10', '문항 수'] + [LABELS[a] for a in ACTIONS[1:]],
                    [[STAGES[s], next(iter(v.values()))['n']] + [f'{v[a]["vs_stop"]["recovered"]} 회복 / {v[a]["vs_stop"]["regressed"]} 악화' for a in ACTIONS[1:]]
                     for s, v in summary['by_baseline_stage_at10'].items()]), '',
              '## 6. 시간과 작업량', '',
              '**시간은 구간별 측정값을 합친 검색 파이프라인 시간이다.** 질문 임베딩·Dense·해당 추가 작업을 포함하며 선택용 특징 계산은 제외했다. '
              '독립적인 실제 요청의 end-to-end 지연시간 측정과 다르다. 36문항은 9/27, 수정12문항은 9/28 실행으로, '
              '같은 모델·라이브러리·GPU·드라이버를 확인했지만 실행일/장비부하 차이는 남는다. 표의 정밀한 시간 차이를 일반화하지 않는다.', '',
              table(['방법', '평균 ms', 'P50 ms', 'P95 ms', 'Dense 대비 추가 ms', '추가 Dense/질문', 'FTS/질문', '재정렬 문서/질문'],
                    [[LABELS[a]] + [f'{summary["by_action"][a]["10"][key]*1000:.2f}' for key in ('mean_seconds','p50_seconds','p95_seconds','mean_delta_vs_stop_seconds')] +
                     [f'{summary["by_action"][a]["10"][key]:.2f}' for key in ('mean_extra_dense_queries','mean_lexical_queries','mean_rerank_pairs')] for a in ACTIONS]), '',
              '각 질문의 반복2회 평균을 구한 뒤 48문항 간 평균/P50/P95를 계산했다. 반복 실행을 문항 수에 더하지 않았다. '
              '외부 API 호출은 0회지만 로컬 연산 비용이 0원이라는 뜻은 아니다. GPU 전용 연산시간·전력·금액은 측정하지 않았다. '
              '모델 로딩·인덱스 구축·AI 문항 작성 비용은 질문당 실행 시간에 포함되지 않는다.', '',
              '### 재사용과 신규 실행을 분리한 시간 — 평균 ms', '',
              table(['구분'] + [LABELS[a] for a in ACTIONS], [[o] + [f'{v[a]["mean_seconds"]*1000:.2f}' for a in ACTIONS] for o, v in summary['by_origin'].items()]), '',
              '## 7. 사례별 변화 — Top-10', '',
              '아래에는 기본 실패뿐 아니라 추가 작업으로 약화된 질문도 포함한다. 전체48개×6방법×3cutoff는 CSV에 있다.', '',
              table(['ID', '유형', 'Dense 근거순위'] + list(ACTIONS),
                    [[c['id'], TYPES[c['type']], str(c['base_group_ranks'])] + ['성공' if c['actions'][a]['metrics']['10']['complete'] else '미충족' for a in ACTIONS]
                     for c in cases if not all(c['actions'][a]['metrics']['10']['complete'] for a in ACTIONS)]), '',
              f'모든 방법 성공: {len(summary["all_actions_success_at10"])}문항. 모든 방법 미충족: {summary["all_actions_fail_at10"]}.', '',
              '## 8. 해석 범위와 다음 작업', '',
              '- 결과는 원문을 바탕으로 AI가 작성하고 같은 AI가 검토한 탐색 결과다. 전문가 정답 검수나 독립 검증으로 표현하지 않는다.',
              '- 이전 결과에 노출된 문항, 유형별 균형 표집, 출처를 명시한 질문, 중복 문서군이 있다. 전체 사용자 질문 성공률로 확대하지 않는다.',
              '- 동등 근거는 같은 코드·동일 본문만 자동 인정했다. 의미가 같은 다른 조항의 정답 누락 가능성은 남는다.',
              '- 참조 보충은 상위5에서 명시적으로 파싱되는 링크만 1단계 확인한다. 이 구현의 결과가 참조 추적 전체의 가능성은 아니다.',
              '- 질문 분리는 접속표현 기반 규칙이다. 질문 분리가 작동하지 않은 문항도 제외하지 않고 그대로 보고한다.',
              '- 어떤 작업의 성공도 단일한 실패 원인의 증명이 아니다. 성공·악화를 함께 보고하고 원인 설명은 가설로 남긴다.',
              '- 다음: 검증 보고서 확인 후 이 표와 대표 사례를 논문 방법·결과에 사용한다. 선택 실행·학습 실행 및 더 넓은 평가와 독립적인 지연시간 측정은 future work다.', '',
              '## 재현 파일', '',
              '- questions.jsonl / question_changes.json / dataset_manifest.json: 질문·근거·변경 이력',
              '- PROTOCOL.md: 이번 실행 전 비교 조건',
              '- run_20260928/actions.jsonl / manifest.json: 새 실행과 재사용 기록',
              '- run_20260928/question_action_metrics.csv / summary.json / cases.json: 전체 결과',
              '- VERIFICATION.md / run_20260928/verification.json: 별도 채점 및 검증',
              '- EVIDENCE_REVIEW.html: 질문·원문·방법별 top15를 함께 확인', '']
    (HERE / 'RESULTS.md').write_text('\n'.join(lines))
    # A self-contained audit view; rankings carry IDs, not an invented interpretation.
    h = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>전체 코퍼스 파일럿 검증</title>',
         '<style>body{font:15px/1.6 sans-serif;margin:24px;max-width:1300px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px;vertical-align:top}pre{white-space:pre-wrap;background:#f6f6f6;padding:10px}.gold{background:#d9efda}summary{cursor:pointer}article{border-top:2px solid #777;margin-top:28px}</style>',
         '<h1>전체 코퍼스 파일럿 — 근거와 결과 대조</h1><p>AI 작성·검증, 새 사람 검수 없음. 문서 ID는 동결 코퍼스 section_id. Top15에서 녹색은 현재 인정한 필수 근거이다. 답변 생성 평가는 하지 않았다.</p>']
    byq = {q['question_id']: q for q in questions}
    candidate_ids = sorted({sid for c in cases for a in ACTIONS for sid in c['actions'][a]['top15']})
    conn = sqlite3.connect(f'file:{HERE.parents[1]}/data/processed/2026-08-12-full/kcsc.sqlite3?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    passages = {sid: dict(conn.execute('select section_id,code,document_name,title,label,text_search from searchable_sections where section_id=?', (sid,)).fetchone()) for sid in candidate_ids}
    (RUN / 'review_passages.json').write_text(json.dumps(passages, ensure_ascii=False, indent=2) + '\n')
    for c in cases:
        q = byq[c['id']]
        gold = set(sum(q['evidence_groups'], []))
        h += [f'<article id="{c["id"]}"><h2>{c["id"]} · {TYPES[c["type"]]}</h2><p>{html.escape(q["question"])}</p>']
        for i, e in enumerate(q['answer_evidence'], 1):
            h += [f'<details open><summary>필수 근거{i} — {e["section_id"]} · {html.escape(e["code"])} · {html.escape(e["title"])}</summary><pre>{html.escape(e["text_search"])}</pre></details>']
        if q.get('reference_validation'):
            h += [f'<p>참조 연결 확인용 원문(연결만을 이유로 필수 채점하지 않음): {html.escape(q["reference_validation"]["bridge_text"])}</p>']
        h += ['<table><tr><th>순위</th>' + ''.join(f'<th>{LABELS[a]}</th>' for a in ACTIONS) + '</tr>']
        for rank in range(15):
            h += [f'<tr><td>{rank + 1}</td>' + ''.join(f'<td class="{"gold" if c["actions"][a]["top15"][rank] in gold else ""}"><a href="#sid-{c["actions"][a]["top15"][rank]}">{c["actions"][a]["top15"][rank]}</a></td>' for a in ACTIONS) + '</tr>']
        h += ['</table></article>']
    h += ['<h1>검색된 원문 — ID를 눌러 이동, 펼쳐서 확인</h1>']
    for sid, e in passages.items():
        h += [f'<div id="sid-{sid}"><details><summary>{sid} · {html.escape(e["code"])} · {html.escape(e["document_name"])} · {html.escape(e["title"])}</summary><pre>{html.escape(e["text_search"])}</pre></details></div>']
    h += ['</html>']
    (HERE / 'EVIDENCE_REVIEW.html').write_text('\n'.join(h))
    (RUN / 'analysis_manifest.json').write_text(json.dumps({'analysis_script_sha256': digest(Path(__file__)), 'metric_code_sha256': digest(HERE / 'core.py'), 'input_actions_sha256': digest(RUN / 'actions.jsonl'), 'input_questions_sha256': digest(HERE / 'questions.jsonl'), 'csv_rows': len(flat)}, indent=2) + '\n')
    print(json.dumps({'by_action_at10': {a: summary['by_action'][a]['10'] for a in ACTIONS}, 'baseline_stages': summary['baseline_stages'], 'all_fail': summary['all_actions_fail_at10']}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
