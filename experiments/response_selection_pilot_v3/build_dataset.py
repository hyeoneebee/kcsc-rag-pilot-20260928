"""Attach immutable corpus evidence and authoring provenance to AI-written drafts."""
import hashlib
import html
import json
import sqlite3
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def main():
    c=sqlite3.connect(f'file:{ROOT}/data/processed/2026-08-12-full/kcsc.sqlite3?mode=ro',uri=True)
    c.row_factory=sqlite3.Row
    source=json.loads((HERE/'authored_questions.json').read_text())
    data=[]
    for kind,items in source.items():
        for i,item in enumerate(items,1):
            question,ids=item[:2]
            evidence=[dict(c.execute('select * from searchable_sections where section_id=?',(sid,)).fetchone()) for sid in ids]
            # Do not silently accept similar wording from different authorities/conditions.
            groups=[]
            for e in evidence:
                groups.append([r[0] for r in c.execute('select section_id from searchable_sections where code=? and text_search=?',(e['code'],e['text_search']))])
            data.append(dict(question_id=f'{kind[:3].upper()}-{i:02d}',question=question,primary_type=kind,
                secondary_tags=['multi_evidence'] if kind=='reference_following' else [],
                answer_status='AI_DRAFT_NOT_HUMAN_REVIEWED',answer_evidence=evidence,
                answer_draft='\n'.join(e['text_search'] for e in evidence),evidence_groups=groups,
                group_codes=sorted(set(e['code'] for e in evidence)),
                distractor_ids=item[2] if len(item)>2 else [],review_note=item[3] if len(item)>3 else '',
                authoring_method='Codex current conversation, source-conditioned manual drafting; no external generation API',
                sampling='deterministic candidate pool + purposive author selection before retrieval scores',
                alternative_evidence_status='exact same-code/text duplicates only; semantic alternatives need pooled human review'))
    assert len(data)==48 and all(v==12 for v in Counter(q['primary_type'] for q in data).values())
    payload=''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in data)
    path=HERE/'questions.v1.jsonl'
    if path.exists() and path.read_text()!=payload:raise RuntimeError('Frozen dataset exists; version changes explicitly')
    path.write_text(payload)
    manifest={'question_sha256':hashlib.sha256(payload.encode()).hexdigest(),'n':len(data),
        'primary_types':dict(Counter(q['primary_type'] for q in data)),
        'source_families':dict(Counter(q['answer_evidence'][0]['code'].split()[0] for q in data)),
        'status':'AI_DRAFT_EXPLORATORY_ONLY','generation_cost_usd':None,
        'generation_cost_note':'Conversation drafting cost not separately metered; runtime external API calls=0',
        'old_exposure_exclusion':'Excluded current reviewed gold document codes in candidate pool; not guaranteed independent of all historical drafts',
        'sampling_bias':['source-conditioned questions','balanced capability quotas','text-only','many questions name a standards family','purposive selection, not random corpus QA sample']}
    (HERE/'dataset_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    blocks=['<!doctype html><meta charset="utf-8"><title>새 QA 48문항 검토</title><style>body{max-width:1000px;margin:30px auto;font:16px/1.6 sans-serif}article{border-top:2px solid #bbb;padding:20px}pre{white-space:pre-wrap;background:#f5f5f5;padding:12px}textarea{width:95%;height:70px}</style><h1>새 QA 48문항 — AI 초안</h1><p>답은 원문 발췌 초안입니다. 정답 충분성·동등 근거·질문 유형을 검토해 주세요. 입력란은 자동 저장되지 않습니다. 인쇄/PDF 또는 별도 문서로 의견을 저장하세요.</p>']
    for q in data:
        blocks.append(f'<article id="{q["question_id"]}"><h2>{q["question_id"]} · {q["primary_type"]}</h2><p>{html.escape(q["question"])}</p>')
        for e in q['answer_evidence']:
            blocks.append(f'<h3>근거 {e["section_id"]} · {html.escape(e["code"])} · {html.escape(e["document_name"])} · {html.escape(e["title"])} {html.escape(e["label"] or "")}</h3><pre>{html.escape(e["text_search"])}</pre>')
        for sid in q['distractor_ids']:
            e=dict(c.execute('select * from searchable_sections where section_id=?',(sid,)).fetchone())
            blocks.append(f'<details><summary>혼동 후보 {sid} · {html.escape(e["code"])}</summary><pre>{html.escape(e["text_search"])}</pre></details>')
        blocks.append(f'<p>{html.escape(q["review_note"])}</p><p>□ 질문 자연스러움 □ 유형 적합 □ 필수 근거 충분 □ 동등 근거 확인</p><textarea placeholder="검토 의견 — 자동 저장되지 않음"></textarea></article>')
    (HERE/'QA_REVIEW.html').write_text('\n'.join(blocks))
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
