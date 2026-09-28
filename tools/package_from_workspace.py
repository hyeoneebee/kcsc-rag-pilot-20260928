"""Allowlisted handoff export. Never copies a database, weights, credentials or .git."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V4 = Path('experiments/evidence_response_pilot_v4')
V3 = Path('experiments/response_selection_pilot_v3')
OLD = Path('experiments/retrieval_failure_discovery_v1')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path, help='Original kcsc-collector checkout')
    parser.add_argument('--refresh-export', action='store_true', help='Refresh only this script\'s allowlisted exports after an export-policy change')
    args = parser.parse_args()
    source = args.source.resolve()
    entries = []

    def export(rel, destination=None, transform=None, note=None):
        rel = Path(rel)
        dest = Path(destination) if destination else rel
        before = (source / rel).read_bytes()
        after = transform(before) if transform else before
        path = ROOT / dest
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != after and not args.refresh_export:
            raise RuntimeError(f'Refusing to overwrite changed artifact: {dest}')
        path.write_bytes(after)
        entries.append(dict(source_path=rel.as_posix(), package_path=dest.as_posix(),
                            source_sha256=sha(before), package_sha256=sha(after),
                            byte_identical=before == after, change=note))

    def redact_gpu(data):
        obj = json.loads(data)
        if 'nvidia_smi' in obj:
            lines = obj['nvidia_smi'].splitlines()
            boundary = next(i for i, line in enumerate(lines) if '| Processes:' in line)
            obj['nvidia_smi'] = '\n'.join(lines[:boundary]) + '\n[Process names and PIDs omitted in handoff copy.]\n'
        return (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode()

    def redact_local_docs(data):
        text = data.decode().replace(str(source) + '/', '')
        text = '\n'.join(line for line in text.splitlines() if 'app.notion.com/' not in line) + '\n'
        return text.encode()

    def public_questions(data):
        rows = []
        for line in data.decode().splitlines():
            q = json.loads(line)
            bodies = [e.get('text_search', '') for e in q.get('answer_evidence', [])]
            if any(body and body in q.get('answer_draft', '') for body in bodies):
                q['answer_draft'] = '[원문을 이어붙인 답변 초안은 공개본에서 제외. 근거 ID·코드·절 메타데이터로 확인.]'
            for evidence in q.get('answer_evidence', []):
                evidence.pop('text_search', None)
            q.get('reference_validation', {}).pop('bridge_text', None)
            rows.append(json.dumps(q, ensure_ascii=False))
        return ('\n'.join(rows) + '\n').encode()

    for name in ('PROTOCOL.md', 'RESULTS.md', 'DISCUSSION.md', 'VERIFICATION.md',
                 'question_changes.json',
                 'dataset_manifest.json', 'build_dataset.py', 'core.py',
                 'run_incremental.py', 'test_core.py', 'analyze.py', 'verify.py'):
        export(V4 / name)
    export(V4 / 'questions.jsonl', transform=public_questions,
           note='Public copy: text_search, reference bridge_text and source-concatenated answer_draft omitted. AI paraphrases retained. Questions, gold groups and metadata unchanged.')
    export(V4 / 'NOTION_REPORT.md', transform=redact_local_docs,
           note='Local checkout prefix and private Notion navigation removed; result content retained.')
    for name in ('actions.jsonl', 'analysis_manifest.json', 'cases.json',
                 'question_action_metrics.csv', 'summary.json', 'verification.json'):
        export(V4 / 'run_20260928' / name)
    export(V4 / 'run_20260928/manifest.json', transform=redact_gpu,
           note='Only nvidia_smi process listing removed; original digest retained here.')
    # Earlier artifacts are provenance and imported implementation only, NOT current results.
    for name in ('run_local.py', 'selection_core.py', 'run_v1/actions.jsonl'):
        export(V3 / name)
    for name in ('build_dataset.py', 'authored_questions.json', 'dataset_manifest.json'):
        export(V3 / name)
    export(V3 / 'prepare_candidates.py', transform=lambda b: re.sub(
        r"Path\('[^']*/reviewed_questions\.jsonl'\)",
        "(ROOT/'provenance/previous_reviewed_questions_NOT_DISTRIBUTED.jsonl')", b.decode()).encode(),
        note='Personal path for older reviewed-QA input replaced with explicit non-distributed dependency. Historical authoring only.')
    def strip_pool_text(data):
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items() if k != 'text_search'}
            if isinstance(obj, list):
                return [clean(v) for v in obj]
            return obj
        return (json.dumps(clean(json.loads(data)), ensure_ascii=False, indent=2) + '\n').encode()
    export(V3 / 'authoring_pool.json', transform=strip_pool_text,
           note='Historical candidate-pool metadata only; all verbatim text_search omitted.')
    for name in ('questions.v1.jsonl', 'questions.v2.jsonl'):
        export(V3 / name, transform=public_questions,
               note='Public provenance copy: verbatim evidence text and source-concatenated answer drafts omitted; question text and gold groups unchanged. Not current evaluation data.')
    export(V3 / 'run_v1/manifest.json', transform=redact_gpu,
           note='Only nvidia_smi process listing removed; original digest retained here.')
    for name in ('scripts/run_baselines.py', 'scripts/retrieval_config.py', 'retrieval_config_v1.json',
                 'runs/preflight_v1/run_manifest.json', 'runs/cache/7e2408a65f4cdef444db/state.json'):
        export(OLD / name)
    for name in ('src/__init__.py', 'src/build.py', 'src/html_text.py', 'src/integrity.py'):
        export(name)
    export('data/processed/2026-08-12-full/build-report.json',
           'provenance/corpus_build_report.json', transform=redact_local_docs,
           note='Absolute local checkout prefix removed from sourceSnapshot.')
    export('experiments/cost_efficiency_pilot_v2/LITERATURE_MATRIX.md',
           'background/LITERATURE_REVIEW_20260923.md',
           transform=lambda b: b.decode().replace(
               '실험 구성과 실제 로컬 점검 결과는 [RESEARCH_AND_DESIGN.md](RESEARCH_AND_DESIGN.md)에 기록했다.',
               '이 문서는 9/23의 문헌 검토 기록이다. 이번 실험 조건은 ../docs/EXPERIMENT_CONDITIONS.md를 따른다.').encode(),
           note='Replaced old design navigation with current handoff navigation; literature text unchanged.')
    provenance = dict(
        package_version='paper-handoff-v4-20260928-public',
        source_repository='https://github.com/hyeoneebee/kcsc-collector',
        source_git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip(),
        source_status_note='v4 and v3 experiment artifacts were untracked at export; Git HEAD alone does not identify them. Use per-file SHA-256 below.',
        transformations_are_not_new_experiments=True,
        excluded=['entire corpus and raw API snapshots', 'embedding arrays', 'model weights',
                  'verbatim evidence text_search and reference bridge_text in QA',
                  'answer_draft fields that were concatenations of original evidence passages',
                  'EVIDENCE_REVIEW.html and review_passages.json with bulk source excerpts',
                  'credentials, environment files and .git', 'Notion publication metadata and publishing script',
                  'old policy training results and railway-quota pilot results'],
        files=entries)
    (ROOT / 'provenance').mkdir(exist_ok=True)
    (ROOT / 'provenance/export_manifest.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    qa = [json.loads(line) for line in (ROOT / V4 / 'questions.jsonl').read_text().splitlines()]
    lines = ['# 질문과 정답 근거 목록 — 공개본', '',
             '질문과 AI가 요약한 답변 초안은 공개하고, 기준 원문 및 원문을 이어붙인 답변 초안은 제외했다. 남아 있는 초안도 전문가 검증된 정답 문장이 아니다.',
             '숫자는 동결 DB 안의 section_id이다. 근거 그룹 안의 ID는 대체 허용이고, 서로 다른 그룹은 모두 필요하다.', '']
    for q in qa:
        lines += [f'## {q["question_id"]} / {q["primary_type"]}', '', q['question'], '',
                  '**답변 초안:** ' + q.get('answer_draft', ''), '',
                  '**허용 근거 그룹:** `' + json.dumps(q['evidence_groups']) + '`', '',
                  '| 대표 절 ID | 기준 코드 | 문서명 | 절 제목·label |', '|---|---|---|---|']
        for e in q['answer_evidence']:
            fields = (str(e['section_id']), e['code'], e['document_name'], str(e['title']) + ' / ' + str(e.get('label') or ''))
            lines.append('| ' + ' | '.join(x.replace('|', '\\|').replace('\n', ' ') for x in fields) + ' |')
        lines.append('')
    (ROOT / 'docs').mkdir(exist_ok=True)
    (ROOT / 'docs/QUESTION_INDEX.md').write_text('\n'.join(lines).rstrip() + '\n')
    run = json.loads((ROOT / V4 / 'run_20260928/manifest.json').read_text())
    config = run['source_config']
    effective = dict(
        status='HANDOFF_ANNOTATION_OF_EXECUTED_V4_NOT_A_NEW_RUN',
        evidence=['v4/PROTOCOL.md', 'v4/run_incremental.py', 'v3/run_local.py', 'v4/run_20260928/manifest.json'],
        dataset_version='evidence_response_pilot_v4_20260928', n_questions=48,
        seed=20260927, repeats=2, actions=run['actions'], evaluation_cutoffs=[5, 10, 15],
        primary_metric='complete_evidence_success_at_10',
        corpus=dict(snapshot='2026-08-12-full', documents=3520, sections=557027,
                    database_sha256=json.loads((ROOT / V4 / 'dataset_manifest.json').read_text())['database_sha256']),
        input=config['input'], lexical=config['lexical'], dense=config['dense'], hybrid=config['hybrid'],
        reranker={**config['reranker'], 'batch_size': 4, 'input_top_k': 50, 'output_top_k': 50},
        reference=dict(source_top_k=5, max_code_matches_per_source=3, targets_per_match=20, hops=1, merged_candidates=50),
        split=dict(max_extra_queries=2, per_query_top_k=100, merged_candidates=50, method='frozen connective regex'),
        keep3=dict(preserve_initial=3, candidate_limit=50, extra_model_calls=0),
        runtime_recorded=run['runtime_versions'], gpu='NVIDIA GeForce RTX 3090', driver='595.84',
        timing_scope=run['timing_scope'], external_api_calls=0, measured_currency_cost=None,
        warning='source_config/effective_config inside the original run manifest retain old evaluation fields; see docs/EXPERIMENT_CONDITIONS.md')
    (ROOT / 'provenance/effective_settings.json').write_text(json.dumps(effective, ensure_ascii=False, indent=2) + '\n')
    print(f'Exported {len(entries)} allowlisted artifacts; {sum(not x["byte_identical"] for x in entries)} documented transformations.')


if __name__ == '__main__':
    main()
