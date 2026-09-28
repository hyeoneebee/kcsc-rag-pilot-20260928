"""Deployment-side inputs: query, first-search scores and metadata. Never gold."""
import re
import numpy as np

ACTIONS=('STOP','RERANK','REFERENCE','SPLIT','SCOPE')
FAMILY=re.compile(r'\b(KRACS|KRCCS|KWCS|LHCS|NHCS|SMCS|EXCS|KDS|KCS)\b')
REF_WORDS=re.compile(r'따르는 기준|따르는 재료|준용하는 기준|참조|추적')
SPLITTER=re.compile(r'(?:이며|하며|으며|하고),?\s+')
CODE=re.compile(r'\b(KRACS|KRCCS|KWCS|LHCS|NHCS|SMCS|EXCS|KDS|KCS)\s*((?:\d{2}\s*){3,5})(?!\d)')

def split_query(question):
    parts=SPLITTER.split(question,maxsplit=1)
    if len(parts)<2:return []
    prefix=' '.join(question.split()[:3])
    return [parts[0]+'?',prefix+' '+parts[1]]

def fixed_rule(question):
    if REF_WORDS.search(question):return 'REFERENCE'
    if split_query(question):return 'SPLIT'
    if FAMILY.search(question):return 'SCOPE'
    return 'RERANK'

def features(question,hits,metadata):
    scores=[h['score'] for h in hits]
    top=[metadata[h['section_id']] for h in hits[:10]]
    family=FAMILY.search(question)
    return [len(question),len(question.split()),int(bool(REF_WORDS.search(question))),
        int(bool(split_query(question))),int(bool(family)),
        scores[0],scores[0]-scores[1],scores[0]-scores[9],float(np.std(scores[:10])),
        len(set(r['document_id'] for r in top))/10,
        sum(bool(CODE.search(r['text_search'])) for r in top)/10,
        sum(r['code'].split()[0]==family[1] for r in top)/10 if family else 0]

def metrics(ids,groups,k):
    got=set(ids[:k]); flags=[bool(got.intersection(g)) for g in groups]
    if not flags:raise ValueError('Answerable evidence groups required')
    return {'complete':int(all(flags)),'hit':int(any(flags)),'recall':sum(flags)/len(flags)}

def merge_round_robin(rankings,limit):
    out=[]; seen=set()
    for i in range(max(map(len,rankings),default=0)):
        for ranking in rankings:
            if i<len(ranking) and ranking[i] not in seen:
                out.append(ranking[i]);seen.add(ranking[i])
                if len(out)==limit:return out
    return out
