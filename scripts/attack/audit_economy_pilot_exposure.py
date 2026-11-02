"""Count observed training episodes and validation changes in verified PPO runs."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.kaggle.local_artifact_transfer import file_sha


def audit(config_path):
    config_path=Path(config_path)
    config=json.loads(config_path.read_text());rows=[];sources={}
    for study in config['studies']:
        root=Path(study['root']);proof_path=Path(study['proof'])
        proof=json.loads(proof_path.read_text());collected=root/'collected'
        receipt=json.loads((collected/'collection_receipt.json').read_text())
        if (proof['pilot_steps_verified']!=819200 or not receipt['complete'] or
                receipt['kernel']!=proof['kernel'] or receipt['remote_status']!='COMPLETE'):
            raise ValueError('Verified complete pilot required')
        sources[str(proof_path)]=file_sha(proof_path)
        sources[str(collected/'collection_receipt.json')]=file_sha(collected/'collection_receipt.json')
        for model in proof['pilot_models']:
            path=collected/model['id']/'metrics.jsonl';relative=str(path.relative_to(collected))
            if file_sha(path)!=receipt['files'][relative]['sha256']:
                raise ValueError('Training measurements changed')
            metrics=[json.loads(line)for line in path.read_text().splitlines()]
            if [r['update']for r in metrics]!=list(range(1,101)) or any(not r['actual_cuda']for r in metrics):
                raise ValueError('100 actual CUDA updates required')
            actor=[r for r in metrics if not r['optimizer_controls']['value_only_warmup']]
            warm=[r for r in metrics if r['optimizer_controls']['value_only_warmup']]
            episodes=[e for r in actor for e in r['episodes']]
            counts=Counter((e['family_id'],e['task'])for e in episodes)
            train_wins=Counter((e['family_id'],e['task'])for e in episodes if e['outcome']=='win')
            validations=[]
            for r in metrics:
                if 'validation'not in r:continue
                records=r['validation']['records']
                validations.append(dict(update=r['update'],wins=sum(e['outcome']=='win'for e in records),
                    defense_wins=sum(e['outcome']=='win'and e['task']=='defense'for e in records),
                    families={f:sum(e['outcome']=='win'and e['family_id']==f for e in records)
                              for f in ('air','bridge','giant','golem','hog','swarm')}))
            if len(warm)!=20 or len(actor)!=80 or [r['update']for r in validations]!=[50,100]:
                raise ValueError('Expected warmup and validation schedule differs')
            rows.append(dict(kernel=proof['kernel'],id=model['id'],metrics_sha256=file_sha(path),
                actor_updates=len(actor),warmup_updates=len(warm),actor_episode_completions=len(episodes),
                family_tasks=[dict(family=f,task=t,episodes=n,wins=train_wins[f,t])for(f,t),n in sorted(counts.items())],
                validation_checkpoints=validations,
                win_change_50_to_100=validations[1]['wins']-validations[0]['wins'],
                defense_change_50_to_100=validations[1]['defense_wins']-validations[0]['defense_wins']))
    if len(rows)!=24:raise ValueError('All24 pilots required')
    return dict(schema='economy-pilot-exposure-audit-v1',rows=rows,sources=sources,
        config_sha256=file_sha(config_path),source_sha256=file_sha(__file__),
        actor_episode_range=[min(r['actor_episode_completions']for r in rows),max(r['actor_episode_completions']for r in rows)],
        air_defense_episode_range=[min(x['episodes']for r in rows for x in r['family_tasks']if x['family']=='air'and x['task']=='defense'),
                                   max(x['episodes']for r in rows for x in r['family_tasks']if x['family']=='air'and x['task']=='defense')],
        validation_win_direction=dict(improved=sum(r['win_change_50_to_100']>0 for r in rows),
            unchanged=sum(r['win_change_50_to_100']==0 for r in rows),worse=sum(r['win_change_50_to_100']<0 for r in rows)),
        limitations=['Episodes counted by completion update; some started during critic warmup',
            'Training episodes use changing sampled policies; counts are exposure, not evaluation win rates',
            '50/100 validation contexts are reused development data; no causal proof that a larger budget will help',
            'A family combines different pushes; number of episodes is not number of distinct tactical decisions'],
        remote_queries=False,training_changed=False,live_games=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=audit(a.config)
    with Path(a.output).open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k]for k in ('actor_episode_range','air_defense_episode_range','validation_win_direction')}))
