"""Reuse verified old inference; execute only changed queries and new transforms."""
import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREV = ROOT / 'experiments/response_selection_pilot_v3'
from build_dataset import digest
from core import ACTIONS, preserve_top3
sys.path.insert(0, str(PREV))

def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines()]

def query_hash(query):
    return hashlib.sha256(query.encode()).hexdigest()

def validate_reuse():
    previous = json.loads((PREV / 'run_v1/manifest.json').read_text())
    assert digest(PREV / 'questions.v1.jsonl') == previous['question_sha256']
    for name, sha in previous['code_hashes'].items():
        assert digest(ROOT / name) == sha, ('changed source code', name)
    manifest = json.loads((HERE / 'dataset_manifest.json').read_text())
    assert digest(HERE / 'questions.jsonl') == manifest['question_sha256']
    frozen = json.loads((ROOT / 'experiments/retrieval_failure_discovery_v1/runs/preflight_v1/run_manifest.json').read_text())
    assert manifest['database_sha256'] == frozen['data']['database_sha256']
    assert digest(ROOT / 'data/processed/2026-08-12-full/kcsc.sqlite3') == manifest['database_sha256']
    old_q = {q['question_id']: q for q in read_jsonl(PREV / 'questions.v1.jsonl')}
    questions = read_jsonl(HERE / 'questions.jsonl')
    unchanged = [q['question_id'] for q in questions if q['question'] == old_q[q['question_id']]['question']]
    assert len(unchanged) == 36
    for q in questions:
        if q['question_id'] in unchanged:
            assert q['evidence_groups'] == old_q[q['question_id']]['evidence_groups']
    rows = read_jsonl(PREV / 'run_v1/actions.jsonl')
    index = {(r['question_id'], r['repeat'], r['action']): (i + 1, r) for i, r in enumerate(rows)}
    assert len(index) == len(rows) == 576
    for qid in unchanged:
        for name in ACTIONS[:-1]:
            a, b = index[qid, 0, name][1], index[qid, 1, name][1]
            assert a['ranked_ids'][:15] == b['ranked_ids'][:15], (qid, name)
    return previous, questions, unchanged, index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='run_20260928')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    previous, questions, unchanged, cache = validate_reuse()
    import numpy as np
    import torch
    assert np.__version__ == previous['runtime_versions']['numpy']
    assert torch.__version__ == previous['runtime_versions']['torch']
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip()
    assert 'NVIDIA GeForce RTX 3090' in gpu and '595.84' in gpu
    if args.check_only:
        print(json.dumps({'reuse_queries': len(unchanged), 'new_queries': len(questions) - len(unchanged),
                          'reused_action_records': len(unchanged) * 5 * 2, 'status': 'REUSE_CONTRACT_PASS'}, ensure_ascii=False))
        return
    out = HERE / args.out
    if out.exists():
        raise ValueError(f'Refusing to overwrite {out}')
    out.mkdir()
    # Import unchanged Engine only after preflight. It receives question text, not QA metadata.
    from run_local import Engine
    started = time.perf_counter()
    stamp = lambda: datetime.now(timezone.utc).isoformat()
    manifest = {'status': 'RUNNING', 'started_at': stamp(), 'seed': 20260927,
                'dataset_sha256': digest(HERE / 'questions.jsonl'), 'protocol_sha256': digest(HERE / 'PROTOCOL.md'),
                'source_run_manifest_sha256': digest(PREV / 'run_v1/manifest.json'),
                'source_actions_sha256': digest(PREV / 'run_v1/actions.jsonl'),
                'source_config': previous['config'], 'runtime_versions': previous['runtime_versions'],
                'source_code_hashes': previous['code_hashes'], 'source_runtime_overrides': previous['effective_runtime_overrides_postrun_annotation'],
                'new_code_hashes': {p.name: digest(p) for p in HERE.glob('*.py')},
                'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True),
                'repeats': 2, 'actions': ACTIONS, 'reused_queries': unchanged,
                'new_queries': [q['question_id'] for q in questions if q['question_id'] not in unchanged],
                'cache_match': 'query exact, model/config/code/corpus hashes; same torch/numpy/GPU/driver',
                'timing_scope': 'query embedding+dense+one response; component-summed wall time, excludes unused selector features, cache I/O, evaluation; old and new sessions reported separately',
                'external_api_calls': 0, 'currency_cost': None, 'energy_cost': None,
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    def save_manifest():
        (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    save_manifest()
    records = []
    with (out / 'actions.jsonl').open('x') as handle:
        def emit(row):
            assert len(row['ranked_ids']) == len(set(row['ranked_ids']))
            records.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
        for q in questions:
            if q['question_id'] not in unchanged:
                continue
            for rep in range(2):
                for name in ACTIONS[:-1]:
                    line, raw = cache[q['question_id'], rep, name]
                    row = copy.deepcopy(raw)
                    row.pop('features', None)
                    row['legacy_pipeline_seconds'] = row.pop('pipeline_seconds')
                    row['unused_selector_feature_seconds'] = row.pop('feature_seconds')
                    row['pipeline_seconds'] = row['base_seconds'] + row['action_seconds']
                    row.update(query_sha256=query_hash(q['question']), execution_origin='REUSED_20260927',
                               source_line=line, source_row_sha256=hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest())
                    emit(row)
        print(f'REUSED {len(records)} inference records; no repeated model call', flush=True)
        engine = Engine()
        manifest['query_token_lengths'] = {q['question_id']: len(engine.tokenizer(q['question'])['input_ids']) for q in questions}
        assert max(manifest['query_token_lengths'].values()) <= 128, 'A question would be truncated'
        manifest['setup_seconds'] = time.perf_counter() - started
        save_manifest()
        warm = '콘크리트 공사의 품질 관리 기준은 무엇인가?'
        wh, ws = engine.dense(warm)
        for name in ACTIONS[:-1]:
            engine.action(name, warm, wh, ws)
        engine.sync()
        new = [q for q in questions if q['question_id'] not in unchanged]
        for rep in range(2):
            order = list(new)
            random.Random(20260927 + rep).shuffle(order)
            for number, q in enumerate(order, 1):
                engine.sync()
                t = time.perf_counter()
                hits, scores = engine.dense(q['question'])
                engine.sync()
                base_seconds = time.perf_counter() - t
                actions = list(ACTIONS[:-1])
                random.Random(f'{rep}-{q["question_id"]}').shuffle(actions)
                for name in actions:
                    engine.sync()
                    t = time.perf_counter()
                    ids, detail = engine.action(name, q['question'], hits, scores)
                    engine.sync()
                    duration = time.perf_counter() - t
                    emit({'question_id': q['question_id'], 'query_sha256': query_hash(q['question']),
                          'repeat': rep, 'action': name, 'ranked_ids': ids,
                          'base_seconds': base_seconds, 'action_seconds': duration,
                          'pipeline_seconds': base_seconds + duration, 'detail': detail,
                          'base_hits': hits if name == 'STOP' else None,
                          'execution_origin': 'NEW_QUERY_20260928'})
                print(f'NEW repeat {rep + 1}/2 query {number}/{len(new)} {q["question_id"]}', flush=True)
        index = {(r['question_id'], r['repeat'], r['action']): r for r in records}
        for q in questions:
            for rep in range(2):
                base = index[q['question_id'], rep, 'STOP']
                ranked = index[q['question_id'], rep, 'RERANK']
                t = time.perf_counter()
                ids = preserve_top3(base['ranked_ids'], ranked['ranked_ids'])
                duration = time.perf_counter() - t
                emit({'question_id': q['question_id'], 'query_sha256': query_hash(q['question']),
                      'repeat': rep, 'action': 'KEEP3', 'ranked_ids': ids,
                      'base_seconds': base['base_seconds'], 'action_seconds': ranked['action_seconds'] + duration,
                      'pipeline_seconds': base['base_seconds'] + ranked['action_seconds'] + duration,
                      'detail': {**ranked['detail'], 'preserve_top': 3, 'transform_seconds': duration},
                      'base_hits': None, 'execution_origin': 'NEW_TRANSFORM_20260928',
                      'parent_execution_origin': ranked['execution_origin']})
        manifest.update(completed_at=stamp(), status='COMPLETED_AI_EXPLORATORY', wall_seconds=time.perf_counter() - started,
                        reused_inference_records=360, new_inference_records=120, new_transform_records=96,
                        total_records=len(records), peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                        actions_sha256=digest(out / 'actions.jsonl'), effective_config=engine.config)
        assert len(records) == 576
        save_manifest()
    print('COMPLETED; next: independent scoring, provenance checks, case inspection', flush=True)

if __name__ == '__main__':
    main()
