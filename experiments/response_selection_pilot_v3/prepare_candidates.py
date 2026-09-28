"""Deterministic authoring pool. Does not run retrieval or inspect model scores."""
import hashlib
import json
import re
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DB = ROOT / 'data/processed/2026-08-12-full/kcsc.sqlite3'
CODE = re.compile(r'\b(KRACS|KRCCS|KWCS|LHCS|NHCS|SMCS|EXCS|KDS|KCS)\s*(\d{2})\s*(\d{2})\s*(\d{2})(?!\d)')

def main():
    c = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    c.row_factory = sqlite3.Row
    old = [json.loads(x) for x in (ROOT/'provenance/previous_reviewed_questions_NOT_DISTRIBUTED.jsonl').read_text().splitlines()]
    oldids = {int(i) for q in old for key in ('answer_groups','connection_groups') for g in q.get('evidence_rule',{}).get(key,[]) for i in g}
    excluded = {r[0] for r in c.execute('select distinct code from searchable_sections where section_id in (%s)' % ','.join('?'*len(oldids)), list(oldids))}
    rows = [dict(r) for r in c.execute('''select ss.*,s.has_image,s.has_table,s.ordinal from searchable_sections ss
       join sections s using(section_id) where length(ss.text_search) between 70 and 420
       and s.has_image=0 and s.has_table=0''')]
    bydoc = {}
    for r in rows:
        if r['code'] not in excluded and not re.search(r'관련법규|참고 기준|참고기준|관련기준|관련 기준|용어의 정의',r['title']):
            bydoc.setdefault(r['document_id'],[]).append(r)
    families = ['KDS','KCS','EXCS','KRACS','KRCCS','KWCS','LHCS','NHCS','SMCS']
    pool = {}
    for fam in families:
        docs = [v for v in bydoc.values() if v[0]['code'].split()[0]==fam and len(v)>=4]
        docs.sort(key=lambda v:hashlib.sha256(('20260927'+str(v[0]['document_id'])).encode()).hexdigest())
        pool[fam] = [r for v in docs[:3] for r in v[:8]]
    refs=[]
    for r in rows:
        if r['code'] in excluded or not re.search(r'따른다|따라야|준용',r['text_search']): continue
        matches=list(CODE.finditer(r['text_search']))
        if len(matches)!=1: continue
        m=matches[0]; code=m[1]+' '+''.join(m.groups()[1:])
        if code==r['code'] or code in excluded: continue
        sec=re.match(r'\s*\(?\s*(\d+\.\d+(?:\.\d+)*)',r['text_search'][m.end():])
        if not sec:continue
        target=[dict(x) for x in c.execute('''select ss.* from searchable_sections ss join sections s using(section_id)
            where ss.code=? and (ss.title like ? or ss.label=?) and length(ss.text_search) between 65 and 420
            and s.has_image=0 and s.has_table=0 order by ss.section_id''',(code,sec[1]+' %',sec[1]))]
        if target:refs.append({'source':r,'target':target[:3],'reference_clause':sec[1]})
    refs.sort(key=lambda r:hashlib.sha256(str(r['source']['section_id']).encode()).hexdigest())
    out={'seed':20260927,'excluded_current_gold_codes':sorted(excluded),'pool':pool,'references':refs[:45]}
    (HERE/'authoring_pool.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print('pool', {k:len(v) for k,v in pool.items()},'reference_candidates',len(refs))

if __name__=='__main__':main()
