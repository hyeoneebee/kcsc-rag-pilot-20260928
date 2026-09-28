"""Versioned AI-authored pilot; never overwrites earlier questions or results."""
import copy
import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREV = ROOT / 'experiments/response_selection_pilot_v3'
DB = ROOT / 'data/processed/2026-08-12-full/kcsc.sqlite3'

# Each item states two distinct answer requirements, not a required retrieval tool.
# Reference edges are evaluation metadata and must never be passed to retrieval.
REFS = [
    ('유리섬유 강화 폴리머 보강근을 사용하는 보가 없는 2방향 슬래브에서 허용하는 이음 방식은 무엇이며, 인접한 경간 길이가 다르면 부모멘트 보강근의 최소 연장은 어느 경간을 기준으로 하는가?',
     [10069, 10562], 10069,
     ['B급 인장 겹침이음 또는 기계적이음', '긴 경간'],
     ['B급 인장 겹침이음', '긴 경간']),
    ('아스팔트 콘크리트 중간층의 현장배합 허용오차는 중간층 자체의 기준을 적용하는가? 시험비빔과 시험포장 결과를 검토한 뒤 공사감독자와 협의하여 결정할 항목도 설명하라.',
     [381803, 381435], 381803,
     ['중간층 현장배합 허용오차는 중간층의 표 3.4-1 적용', '입도·아스팔트 함량·혼합시간·믹서 배출온도 협의'],
     ['중간층의 현장배합 허용오차', '공사감독자와 협의']),
    ('하상세굴 위험이 있는 제외지의 취수관로에서 보강해야 하는 범위는 어디이며, 밑다짐은 최대유속 때 상류에서 굴러오는 돌에 대해 어떤 성능을 갖춰야 하는가?',
     [245429, 246173], 245429,
     ['취수관 주위와 하상을 보강', '돌에 저항할 자중과 강한 상호 연결'],
     ['취수관의 주위와 하상', '상호 연결체']),
    ('터널 세그먼트의 내부 현장타설라이닝을 시공하기 전에 필요한 사전처리는 무엇이며, 거푸집 구조는 어떤 타설 조건과 하중을 고려해야 하는가?',
     [168919, 168818], 168919,
     ['세그먼트 방수·청소·이음볼트 확인', '타설량·길이·속도를 고려하고 콘크리트 압력에 저항'],
     ['이음볼트의 확인', '타설속도']),
    ('유도등과 유도표지 시공의 일반사항에 적용되는 화재안전기준의 두 종류는 무엇이며, 현장 여건 때문에 도면대로 시공하기 어렵다면 어떤 절차를 거쳐야 하는가?',
     [533730, 520106], 533730,
     ['화재안전 성능기준과 화재안전 기술기준', '공사감독자 협의와 변경 승인 후 시공'],
     ['화재안전 기술기준', '변경 승인']),
    ('비접착식 콘크리트 덧씌우기 포장에서도 접착강도 품질시험을 실시해야 하는가? 슬럼프의 허용편차와 하루 최소 측정 횟수도 설명하라.',
     [237160, 237073], 237160,
     ['접착강도 품질시험 제외', '설계슬럼프 ±10mm, 하루 3회 이상'],
     ['접착강도 품질시험은 실시하지 않는다', '1일 3회']),
    ('건축물 전기설비의 배터리랙과 배터리는 지진에 대비해 어떻게 고정·보호해야 하며, 관련 건축물 내진설계의 성능수준은 어떤 네 단계로 구분되는가?',
     [46692, 71395], 46692,
     ['랙 파손·변형 방지 및 배터리 고정장치', '기능수행·즉시복구·인명보호·붕괴방지'],
     ['고정장치', '즉시복구']),
    ('바닥 공사용 목재류의 수종·등급·치수는 무엇에 따르고 견본은 누구의 승인을 받아야 하며, 내장 마감 목재의 함수율 상한과 필요시 적용하는 더 낮은 상한은 얼마인가?',
     [221168, 211447], 221168,
     ['공사시방서에 따르고 견본은 담당원 승인', '함수율 15% 이하, 필요시 12% 이하'],
     ['견본을 미리 제출', '12%']),
    ('건축물 강구조공사의 치수정밀도 검사는 무엇을 측정해 무엇을 확인하는 검사이며, 관리허용차와 한계허용차는 각각 어떤 판단 기준인가?',
     [210741, 209805], 210741,
     ['제품치수 측정으로 소정의 치수정밀도 확보 확인', '관리목표치와 초과 불가 합격판정 기준값'],
     ['제품치수를 측정', '관리목표치']),
    ('건축물 강구조공사에 용접재를 납품받을 때 확인할 송장과 포장 사항은 무엇이며, KS 규격품의 재료시험을 생략할 수 있는 서류 조건은 무엇인가?',
     [210135, 209789], 210135,
     ['납품송장과 제조사 포장상태 확인', '규격증명서와 시험성적서 첨부 시 재료시험 생략 가능'],
     ['납품송장', '규격증명서 및 시험성적서']),
    ('소규모 콘크리트중력댐의 퇴사압 산정에서 퇴사 깊이는 몇 년간의 퇴사량을 기본으로 하며, 댐의 안정성을 위해 함께 고려해야 하는 힘의 종류는 무엇인가?',
     [108040, 106601], 108040,
     ['50년간 퇴사량 기준', '자중·정수압·동수압·풍하중·온도하중·양압력·파압·빙압·퇴사압·지진관성력'],
     ['50년간', '지진관성력']),
    ('보강토옹벽 뒤채움흙에 일반 노체부의 90% 다짐률을 그대로 적용해도 되는가? 옹벽 뒤채움의 다짐률과 시험 방법, 일반 쌓기 기준과 시설물별 다짐 기준이 다를 때 우선할 기준을 설명하라.',
     [146760, 140973], 146802,
     ['최대건조밀도 D·E 방법의 95% 이상', '시설물별 기준이 정해져 있으면 해당 시설물 기준 우선'],
     ['95%', '해당 시설물 기준']),
]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def dump_frozen(path, obj, jsonl=False):
    text = (''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in obj) if jsonl
            else json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    if path.exists() and path.read_text() != text:
        raise ValueError(f'Frozen artifact differs: {path}')
    path.write_text(text)

def main():
    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    prev = [json.loads(x) for x in (PREV / 'questions.v2.jsonl').read_text().splitlines()]
    out, changes = [], []
    for old in prev:
        q = copy.deepcopy(old)
        q['dataset_version'] = 'evidence_response_pilot_v4_20260928'
        q['provenance'] = {'prior_dataset': 'response_selection_pilot_v3', 'prior_id': q['question_id']}
        q['review_status'] = 'AI_SOURCE_CHECKED_NOT_INDEPENDENT_HUMAN_VALIDATED'
        if q['primary_type'] == 'reference_following':
            query, ids, bridge, units, anchors = REFS[int(q['question_id'][-2:]) - 1]
            evidence = [dict(conn.execute('select * from searchable_sections where section_id=?', (i,)).fetchone()) for i in ids]
            br = dict(conn.execute('select * from searchable_sections where section_id=?', (bridge,)).fetchone())
            assert re.sub(r'\s+', '', evidence[-1]['code']) in re.sub(r'\s+', '', br['text_search'])
            assert all(a in e['text_search'] for a, e in zip(anchors, evidence))
            q.update(question=query, answer_evidence=evidence,
                     evidence_groups=[[r[0] for r in conn.execute('select section_id from searchable_sections where code=? and text_search=?', (e['code'], e['text_search']))] for e in evidence],
                     answer_draft='; '.join(units), answer_status='AI_DRAFT_NOT_HUMAN_REVIEWED',
                     required_answer_units=units, evidence_anchor_checks=anchors,
                     group_codes=sorted({e['code'] for e in evidence}), distractor_ids=[],
                     reference_validation={'source_id': bridge, 'target_id': ids[-1], 'bridge_text': br['text_search'],
                       'two_distinct_answer_requirements': True,
                       'source_only_insufficient': 'Missing target-specific requirement: ' + units[-1],
                       'target_only_insufficient': 'Missing source/local requirement: ' + units[0],
                       'bridge_is_mandatory_gold': bridge in ids,
                       'retrieval_path_required': False},
                     review_note='Source and target contain distinct requested facts. Tool path is not scored. REF-12 corrects local 95% vs generic 90% conflict.',
                     secondary_tags=['multi_evidence'] + (['applicability_near_miss'] if q['question_id'] in ('REF-06','REF-12') else []))
            q.pop('human_review', None)
            changes.append({'id': q['question_id'], 'old_question': old['question'], 'new_question': query,
                            'old_groups': old['evidence_groups'], 'new_groups': q['evidence_groups'],
                            'reason': q['review_note']})
        for e, group in zip(q['answer_evidence'], q['evidence_groups']):
            actual = dict(conn.execute('select * from searchable_sections where section_id=?', (e['section_id'],)).fetchone())
            assert actual['text_search'] == e['text_search'] and e['section_id'] in group
            for sid in group:
                alt = conn.execute('select code,text_search from searchable_sections where section_id=?', (sid,)).fetchone()
                assert alt['code'] == e['code'] and alt['text_search'] == e['text_search']
        assert not re.search(r'추적|재검색|재정렬|출발.?조항|참조.*찾|기준까지 찾아', q['question'])
        out.append(q)
    counts = Counter(q['primary_type'] for q in out)
    assert len(out) == 48 and len(counts) == 4 and max(counts.values()) - min(counts.values()) <= 2
    assert len({q['question_id'] for q in out}) == len(out)
    dump_frozen(HERE / 'questions.jsonl', out, True)
    dump_frozen(HERE / 'question_changes.json', changes)
    manifest = dict(dataset_version=out[0]['dataset_version'], n=48, primary_types=dict(counts),
                    source_families=dict(Counter(q['answer_evidence'][0]['code'].split()[0] for q in out)),
                    source_question_sha256=digest(PREV / 'questions.v2.jsonl'), question_sha256=digest(HERE / 'questions.jsonl'),
                    database_sha256=digest(DB), revised_questions=12, unchanged_queries=36,
                    selection='All-corpus purposive capability-balanced pilot. No railway quota or exclusion. Retained prior whole-corpus candidate set; no selection on current outcomes.',
                    reference_definition='Distinct requested facts across explicitly linked provisions, not proof that a particular tool path is necessary.',
                    human_validation='No new human review, per user instruction. Prior MUL-12 approval is provenance, not dataset-wide approval.',
                    limitations=['AI author and AI verifier are not independent', 'Existing non-reference questions previously exposed', 'Same-code exact-text alternatives only; semantic incompleteness remains', '48 purposive questions do not estimate real-user prevalence', 'Text-section retrieval only; no generated-answer scoring'])
    dump_frozen(HERE / 'dataset_manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
