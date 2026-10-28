"""Paired TRAIN games isolate the extra post-acceptance slot lock, no training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sim.continuous_push import ContinuousPushEnv,DECKS
from scripts.gamestate.model_bridge import MemoryPolicy


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def game(network,config,seed,family,cooldown):
    profile=dict(config.get('continuous_profile')or{})
    profile.update(action_cooldown_ms=cooldown,decision_interval_ms=250,cannon_center_only=0,tower_damage_all=0)
    env=ContinuousPushEnv(split='train',seed=seed,action_delay_ms=1350,
        geometry=config['geometry'],coordinate_mode=config['coordinate_mode'],continuous_profile=profile)
    obs,info=env.reset(seed=seed,scenario_id=f'defense::{family}/push')
    counts=Counter();elapsed=Counter();wait_mass=[]
    with torch.inference_mode():
        while not env.done:
            mask=torch.from_numpy(info['action_mask'])
            logits,_=network(torch.from_numpy(obs['planes'])[None],torch.from_numpy(obs['scalars'])[None])
            logits=logits[0].masked_fill(~mask,-torch.inf)
            if not torch.isfinite(logits[mask]).all():raise ValueError('Invalid policy output')
            action=int(logits.argmax());playable=bool(mask[1:].any())
            counts['decisions']+=1
            counts['playable']+=playable
            counts['voluntary_wait']+=playable and action==0
            if playable:wait_mass.append(float(logits.softmax(-1)[0]))
            before=env.match.elapsed_ms;low=float(obs['scalars'][0])*10<4
            obs,_,_,_,info=env.step(action)
            elapsed['total_ms']+=env.match.elapsed_ms-before
            if low:elapsed['available_below4_ms']+=env.match.elapsed_ms-before
    return dict(seed=seed,family=family,cooldown_ms=cooldown,action_delay_ms=1350,
        decision_interval_ms=250,profile=profile,decisions=dict(counts),timing=dict(elapsed),
        mean_wait_probability_when_playable=float(np.mean(wait_mass))if wait_mass else None,
        metrics=info['metrics'])


def run(models,output,seed):
    torch.set_num_threads(1);root=Path(output);root.mkdir(exist_ok=False)
    spec=dict(seed=seed,families=list(DECKS),cooldowns_ms=[2100,0],models=[dict(id=n,path=p,sha256=sha(p))for n,p in models],
        source_sha256={p:sha(p)for p in ['sim/continuous_push.py','sim/engine.py','sim/match.py',__file__]},
        scope='TRAIN development only; fresh current physics, fixed1350ms/250ms; isolate post-acceptance lock. No checkpoint selection, no live games, no training.')
    (root/'protocol.json').write_text(json.dumps(spec,indent=2)+'\n');rows=[]
    for name,path in models:
        blob=torch.load(path,map_location='cpu',weights_only=False);config=blob['config'];policy=MemoryPolicy(Path(path))
        if not policy.joint:raise ValueError('Joint model required')
        for family in DECKS:
            for cooldown in spec['cooldowns_ms']:
                row=dict(model=name,**game(policy.network,config,seed,family,cooldown));rows.append(row)
                (root/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
                print(json.dumps(dict(completed=len(rows),total=len(models)*len(DECKS)*2,model=name,family=family,cooldown=cooldown,outcome=row['metrics']['outcome'])),flush=True)
    summaries=[]
    for name,_ in models:
        for cooldown in spec['cooldowns_ms']:
            rs=[r for r in rows if r['model']==name and r['cooldown_ms']==cooldown];cards=Counter()
            for r in rs:cards.update(r['metrics']['accepted_card_counts'])
            time=sum(r['timing']['total_ms']for r in rs)
            summaries.append(dict(model=name,cooldown_ms=cooldown,games=len(rs),wins=sum(r['metrics']['outcome']=='win'for r in rs),
                available_below4_time_fraction=sum(r['timing'].get('available_below4_ms',0)for r in rs)/time,
                cards_per_game={c:v/len(rs)for c,v in sorted(cards.items())},
                mean_own_fraction_loss=float(np.mean([r['metrics']['own_fraction_loss']for r in rs])),
                voluntary_wait=sum(r['decisions']['voluntary_wait']for r in rs),playable=sum(r['decisions']['playable']for r in rs)))
    report=dict(protocol=spec,rows=rows,summaries=summaries)
    (root/'result.json').write_text(json.dumps(report,indent=2)+'\n');return summaries


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',action='append',required=True,help='label=checkpoint');p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=8137);a=p.parse_args()
    print(json.dumps(run([s.split('=',1)for s in a.model],a.output,a.seed)))
