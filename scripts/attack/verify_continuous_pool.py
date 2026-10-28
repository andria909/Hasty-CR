"""Independently recompute frozen600-game schedule, counters and standings."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path


from scripts.attack.evaluation_execution import execution_for,check_row_execution,check_duel_execution


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(collected,prepared):
    root=Path(collected);prepared=Path(prepared)
    receipt=json.loads((root/'collection_receipt.json').read_text())
    cfg=json.loads((prepared/'preparation.json').read_text())
    p=json.loads((prepared/'dataset/protocol.json').read_text())
    if not receipt.get('complete')or receipt.get('remote_status')!='COMPLETE'or receipt['kernel']!=cfg['kernel']:
        raise ValueError('Terminal matching600-game receipt required')
    for name,meta in receipt['files'].items():
        path=root/name
        if path.is_symlink()or not path.resolve().is_relative_to(root.resolve())or sha(path)!=meta['sha256']:
            raise ValueError('Collected file differs')
    result=json.loads((root/'result.json').read_text())
    if result.get('status')!='complete'or result['protocol_sha256']!=sha(prepared/'dataset/protocol.json'):
        raise ValueError('Protocol/status changed')
    if 'execution' in p and result.get('execution')!=execution_for(p):raise ValueError('Result execution differs')
    resumed_games=0;resume_path=prepared/'dataset/resume_manifest.json'
    if resume_path.exists():
        if sha(resume_path)!=cfg['config']['pins']['resume_manifest.json']:raise ValueError('Resume manifest changed')
        resume=json.loads(resume_path.read_text())
        if result.get('resumed_parts_sha256')!=resume['files']:raise ValueError('Resumed part identity differs')
        for name,digest in resume['files'].items():
            if sha(root/name)!=digest:raise ValueError('Resumed output changed')
            resumed_games+=len(json.loads((root/name).read_text())['rows'])
        if result.get('resumed_games')!=resumed_games or result.get('computed_games')!=600-resumed_games:
            raise ValueError('Resumed/computed game accounting differs')
    models=p['models'];ids=[m['id']for m in models]
    required={m+'_test.json'for m in ids}|{a+'__'+b+'_duels.json'for a,b in itertools.combinations(ids,2)}
    if set(result['files'])!=required:raise ValueError('Evaluation file coverage differs')
    if any(sha(root/n)!=digest for n,digest in result['files'].items()):raise ValueError('Internal evaluation digest differs')
    tests={};standings={m:Counter()for m in ids};matrix={m:{}for m in ids}
    schedule={(f'{t}::{f}/heldout_random',s)for t in('attack','defense')for f in('hog','giant','air','golem','swarm','bridge')for s in p['test_seeds']}
    total_test=0;total_duels=0
    for model in models:
        report=json.loads((root/(model['id']+'_test.json')).read_text());rows=report['rows']
        if report['model']!=model or len(rows)!=len(schedule)or {(r['scenario_id'],r['seed'])for r in rows}!=schedule:
            raise ValueError('Changed/missing/duplicate heldout context')
        counts=Counter(r['outcome']for r in rows)
        if set(counts)-{'win','loss','draw'}or dict(counts)!=report['counts']:raise ValueError('Heldout outcomes differ')
        if abs(report['win_rate']-counts['win']/len(rows))>1e-12:raise ValueError('Heldout win rate differs')
        cards=Counter()
        for row in rows:
            if 'execution' in p:check_row_execution(row,execution_for(p))
            if row['split']!='test':raise ValueError('Heldout row is not test')
            card_counts=row['accepted_card_counts']
            if any(type(v)is not int or v<0 for v in card_counts.values()):raise ValueError('Invalid card counts')
            cards.update(card_counts)
        if any(abs(value-cards[card]/len(rows))>1e-12 for card,value in report['mean_card_plays'].items()):
            raise ValueError('Accepted card averages differ')
        overflow=sum(r['elixir_overflow']for r in rows)/len(rows)
        if abs(overflow-report['mean_elixir_overflow'])>1e-12:raise ValueError('Overflow average differs')
        tests[model['id']]=dict(games=len(rows),counts=dict(counts),win_rate=counts['win']/len(rows),
            mean_card_plays=report['mean_card_plays'],mean_elixir_overflow=overflow)
        total_test+=len(rows)
    for a,b in itertools.combinations(ids,2):
        report=json.loads((root/(a+'__'+b+'_duels.json')).read_text());rows=report['rows']
        contexts={(s,seat)for s in p['duel_seeds']for seat in(1,-1)}
        if report['pair']!=[a,b]or len(rows)!=len(contexts)or {(r['seed'],r['candidate_side'])for r in rows}!=contexts:
            raise ValueError('Changed/missing/duplicate paired duel')
        if 'execution' in p:
            for row in rows:check_duel_execution(row,execution_for(p))
        counts=Counter(r['outcome']for r in rows)
        if set(counts)-{'win','loss','draw'}or dict(counts)!=report['counts']:raise ValueError('Duel outcome counts differ')
        reverse=Counter(win=counts['loss'],loss=counts['win'],draw=counts['draw'])
        standings[a].update(counts);standings[b].update(reverse)
        matrix[a][b]=dict(counts);matrix[b][a]=dict(reverse);total_duels+=len(rows)
    if(total_test,total_duels)!=(480,120):raise ValueError('Incomplete600-game budget')
    table={}
    for name,counts in standings.items():
        games=sum(counts.values())
        table[name]=dict(games=games,wins=counts['win'],losses=counts['loss'],draws=counts['draw'],
            win_rate=counts['win']/games,score=(counts['win']+.5*counts['draw'])/games)
    executors=sorted({m['training_executor']for m in models if 'training_executor'in m})
    return dict(schema='verified-continuous-frozen-pool-v1',kernel=cfg['kernel'],models=models,
        training_cohort=p.get('training_cohort','control'),training_executors=executors,
        resumed_games=resumed_games,computed_games=600-resumed_games,
        total_games=600,tests=tests,standings=table,matrix=matrix,quality_accepted=False,
        limits=['Synthetic random opponents and four-model tournament only',
            'Training executors are recorded per model; all evaluation uses corrected slot execution',
            'Not live validation; no automatic promotion'],test_used_for_selection=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--collected',required=True);parser.add_argument('--prepared',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();report=verify(a.collected,a.prepared);Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['standings']))
