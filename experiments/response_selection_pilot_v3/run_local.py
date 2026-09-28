"""Local, offline-only action matrix. Gold is not an Engine input.

Run: .venv-rag/bin/python experiments/response_selection_pilot_v3/run_local.py
Results are component-timed counterfactual action executions, not a multi-agent run.
"""
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'experiments/retrieval_failure_discovery_v1'
sys.path.insert(0,str(OLD/'scripts'))
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from run_baselines import database_connection,fetch_sections,load_section_ids,load_dense_model,encode_questions,stable_top_k,lexical_search,sha256_file
from retrieval_config import assemble_passage,reciprocal_rank_fusion
from selection_core import ACTIONS,CODE,FAMILY,features,split_query,merge_round_robin

class Engine:
    def __init__(self):
        self.config=json.loads((OLD/'retrieval_config_v1.json').read_text())
        self.config['dense']['batch_size']=1
        self.conn=database_connection(ROOT/'data/processed/2026-08-12-full/kcsc.sqlite3')
        self.ids=load_section_ids(self.conn)
        cache=OLD/'runs/cache/7e2408a65f4cdef444db'
        state=json.loads((cache/'state.json').read_text())
        assert hashlib.sha256(self.ids.tobytes()).hexdigest()==state['section_ids_sha256']
        assert sha256_file(cache/'passages.f32.npy')==state['embedding_file_sha256']
        preflight=json.loads((OLD/'runs/preflight_v1/run_manifest.json').read_text())
        assert sha256_file(ROOT/'data/processed/2026-08-12-full/kcsc.sqlite3')==preflight['data']['database_sha256']
        torch.set_num_threads(4)
        torch.manual_seed(20260927)
        torch.backends.cuda.matmul.allow_tf32=False
        self.device=torch.device('cuda:0')
        self.tokenizer,self.model=load_dense_model(self.config['dense'],self.device)
        self.corpus=torch.tensor(np.load(cache/'passages.f32.npy'),device=self.device,dtype=torch.float32)
        cfg=self.config['reranker']
        self.rtokenizer=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'],local_files_only=True)
        self.rmodel=AutoModelForSequenceClassification.from_pretrained(cfg['model'],revision=cfg['revision'],local_files_only=True,dtype=torch.float16).to(self.device).eval()
        meta=list(self.conn.execute('select section_id,code,title,label from searchable_sections order by section_id'))
        self.codes=np.asarray([r['code'] for r in meta])
        self.bycode={}
        for r in meta:self.bycode.setdefault(r['code'],[]).append((r['section_id'],r['title'],r['label']))

    def sync(self):torch.cuda.synchronize()

    def dense(self,question):
        vector=encode_questions([{'question':question}],self.tokenizer,self.model,self.device,self.config['dense'])
        with torch.inference_mode():
            scores=(torch.from_numpy(vector).to(self.device) @ self.corpus.T).cpu().numpy()[0]
        return stable_top_k(scores,self.ids,100),scores

    def rerank(self,question,ids):
        ids=list(dict.fromkeys(ids))[:50]
        meta=fetch_sections(self.conn,ids)
        scores=[]
        for start in range(0,len(ids),4):
            part=ids[start:start+4]
            inputs=self.rtokenizer([question]*len(part),[assemble_passage(meta[i]) for i in part],
                padding=True,truncation='only_second',max_length=1536,return_tensors='pt')
            with torch.inference_mode():
                scores.extend(self.rmodel(**{k:v.to(self.device) for k,v in inputs.items()}).logits.view(-1).float().cpu().tolist())
        return [int(i) for i,_ in sorted(zip(ids,scores),key=lambda x:(-x[1],x[0]))]

    def action(self,name,question,hits,scores):
        base=[h['section_id'] for h in hits]
        detail={'extra_dense_queries':0,'rerank_pairs':0}
        if name=='STOP':return base,detail
        candidates=base[:50]
        if name=='SPLIT':
            queries=split_query(question)
            branches=[]
            for query in queries:
                rows,_=self.dense(query)
                branches.append([r['section_id'] for r in rows])
            candidates=merge_round_robin([base]+branches,50)
            detail.update(extra_dense_queries=len(queries),subqueries=queries)
        elif name=='REFERENCE':
            metadata=fetch_sections(self.conn,base[:5]); added=[]; edges=[]
            for sid in base[:5]:
                text=metadata[sid]['text_search']
                for match in list(CODE.finditer(text))[:3]:
                    code=match[1]+' '+re.sub(r'\s+','',match[2])
                    sec=re.match(r'\s*\(?\s*(\d+\.\d+(?:\.\d+)*)',text[match.end():])
                    if sec is None:continue
                    target=[i for i,title,label in self.bycode.get(code,[]) if re.match(re.escape(sec[1])+r'(?:\s|\.|$)',title) or label==sec[1]]
                    # Corpus-only operation: no gold, no authored reference graph.
                    added.extend(target[:20]);edges.append({'source':sid,'code':code,'clause':sec[1],'candidates':target[:20]})
            candidates=merge_round_robin([base,added],50)
            detail['reference_edges']=edges
        elif name=='SCOPE':
            match=CODE.search(question);fam=FAMILY.search(question)
            if match:
                code=match[1]+' '+re.sub(r'\s+','',match[2]);mask=self.codes==code
            elif fam:mask=np.char.startswith(self.codes,fam[1]+' ')
            else:mask=np.ones(len(self.ids),dtype=bool)
            scoped=stable_top_k(scores[mask],self.ids[mask],50)
            candidates=[r['section_id'] for r in scoped] or candidates
        elif name=='HYBRID_RERANK':
            _,lex=lexical_search(self.conn,question,100)
            fusion=reciprocal_rank_fusion([[r['section_id'] for r in lex],base],k0=60,limit=100)
            candidates=[r['section_id'] for r in fusion[:50]]
            detail['lexical_top100']=[r['section_id'] for r in lex]
        detail['rerank_pairs']=len(candidates)
        detail['candidate_ids']=candidates
        return self.rerank(question,candidates),detail

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repeats',type=int,default=2);parser.add_argument('--out',default='run_v1')
    args=parser.parse_args()
    run=HERE/args.out
    if run.exists():raise RuntimeError(f'Preserve existing run: {run}')
    run.mkdir()
    started=time.perf_counter()
    questions=[json.loads(x) for x in (HERE/'questions.v1.jsonl').read_text().splitlines()]
    manifest={'started_at':datetime.now(timezone.utc).isoformat(),'seed':20260927,'repeats':args.repeats,
        'question_sha256':sha256_file(HERE/'questions.v1.jsonl'),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'code_hashes':{str(p.relative_to(ROOT)):sha256_file(p) for p in [Path(__file__),HERE/'selection_core.py',OLD/'scripts/run_baselines.py',OLD/'scripts/retrieval_config.py']},
        'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),
        'runtime_versions':{'torch':torch.__version__,'numpy':np.__version__},
        'cost':{'external_api_calls':0,'external_api_cost_usd':0,'electricity_hardware_cost_usd':None},
        'timing_scope':'resident-model query embedding + dense + features + one action; component-summed counterfactual, excludes evaluation and other actions'}
    (run/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    engine=Engine();manifest['setup_seconds']=time.perf_counter()-started
    print(f'SETUP {manifest["setup_seconds"]:.2f}s',flush=True)
    # Warm all paths on a non-evaluation query; scores are not evaluated.
    warm='콘크리트 공사의 품질 관리 기준은 무엇인가?'
    wh,ws=engine.dense(warm)
    for name in (*ACTIONS,'HYBRID_RERANK'):engine.action(name,warm,wh,ws)
    engine.sync();manifest['warmup_seconds']=time.perf_counter()-started-manifest['setup_seconds']
    all_names=(*ACTIONS,'HYBRID_RERANK')
    with (run/'actions.jsonl').open('w') as f:
        for repeat in range(args.repeats):
            order=list(questions);random.Random(20260927+repeat).shuffle(order)
            for n,q in enumerate(order,1):
                query=q['question'];engine.sync();t=time.perf_counter()
                hits,scores=engine.dense(query);engine.sync();base_seconds=time.perf_counter()-t
                t=time.perf_counter();meta=fetch_sections(engine.conn,[h['section_id'] for h in hits[:10]])
                x=features(query,hits,meta);feature_seconds=time.perf_counter()-t
                actions=list(all_names);random.Random(f'{repeat}-{q["question_id"]}').shuffle(actions)
                for name in actions:
                    engine.sync();t=time.perf_counter();ids,detail=engine.action(name,query,hits,scores);engine.sync()
                    elapsed=time.perf_counter()-t
                    row={'question_id':q['question_id'],'repeat':repeat,'action':name,'ranked_ids':ids,
                         'base_seconds':base_seconds,'feature_seconds':feature_seconds,'action_seconds':elapsed,
                         'pipeline_seconds':base_seconds+feature_seconds+elapsed,'features':x,'detail':detail,
                         'base_hits':hits if name=='STOP' else None}
                    f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
                print(f'REPEAT {repeat+1}/{args.repeats} QUESTION {n}/48 {q["question_id"]}',flush=True)
    manifest.update(completed_at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),status='COMPLETED_AI_DRAFT_EXPLORATORY',config=engine.config)
    (run/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETED',flush=True)

if __name__=='__main__':main()
