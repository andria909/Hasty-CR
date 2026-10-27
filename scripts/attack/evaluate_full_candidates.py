"""Run a frozen two-opponent test and all-pairs tournament on2CUDA devices.

Only bound, verified full-budget candidates and the precommitted baselines.
No optimizer, reward tuning, checkpoint search or live gameplay.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
from pathlib import Path
import re
import threading

import numpy as np
import torch
from scripts.attack.evaluate_continuous_pool import choose,duel,load,sha
from sim import arena
from sim.continuous_push import ContinuousPushEnv,PushCase
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26
from sim.runner import resolve_deck
from sim.tactical_rewards import fireball_outcomes


def validate_protocol(protocol,weights):
    weights=Path(weights)
    blueprint_path=weights/'evaluation_blueprint.json'
    if protocol.get('schema')!='bound-full-candidate-evaluation-v1' or sha(blueprint_path)!=protocol.get('blueprint_sha256'):
        raise ValueError('Bound precommitted blueprint required')
    blueprint=json.loads(blueprint_path.read_text())
    models=protocol['models'];names=[m['id']for m in models]
    if len(set(names))!=len(names)or any(not re.fullmatch(r'[a-z][a-z0-9]*(?:_[a-z0-9]+)*',n)for n in names):
        raise ValueError('Unique safe model identifiers required')
    candidates=[m for m in models if m['role']=='candidate']
    baselines=[m for m in models if m['role']=='baseline']
    if len(candidates)not in blueprint['allowed_candidate_counts']or len(candidates)+len(baselines)!=len(models):
        raise ValueError('Two to four completed candidates required')
    if {m['id']for m in baselines}!={m['id']for m in blueprint['baselines']}:
        raise ValueError('Every precommitted baseline required')
    for entry in models:
        if entry['file']!=entry['id']+'.pt' or sha(weights/entry['file'])!=entry['sha256']:
            raise ValueError('Frozen evaluation weight changed')
    for entry in baselines:
        expected=next(m for m in blueprint['baselines']if m['id']==entry['id'])
        if (entry['sha256']!=expected['checkpoint_sha256']or entry['mode']!=expected['mode']or entry['scalars']!=expected['scalars']):
            raise ValueError('Baseline identity or input contract differs')
    for entry in candidates:
        if entry.get('full_steps')!=4096000 or entry['scalars']!=309 or entry['mode']!='joint_enemy_presence_v1':
            raise ValueError('Candidate full budget or public input contract differs')
        proof_name=entry['full_proof_file']
        if Path(proof_name).name!=proof_name or not proof_name.endswith('.json'):
            raise ValueError('Unsafe candidate proof path')
        if sha(weights/proof_name)!=entry['full_proof_sha256']:
            raise ValueError('Candidate verified proof changed')
        proof=json.loads((weights/proof_name).read_text())
        original=next((m for m in proof.get('models',[])if m['id']==entry['origin_model_id']),None)
        if proof.get('full_steps_verified')!=8192000 or original is None or original['checkpoint_sha256']!=entry['sha256']:
            raise ValueError('Candidate does not match a verified full checkpoint')
    return blueprint


def schedule(protocol,blueprint):
    models=protocol['models'];jobs=[]
    for entry in models:
        for test in blueprint['test_sets']:
            jobs.append(dict(kind='test',model=entry['id'],test=test['id'],games=test['games_per_model'],
                file=f"test__{test['id']}__{entry['id']}.json"))
    for a,b in itertools.combinations(models,2):
        jobs.append(dict(kind='duel',pair=[a['id'],b['id']],games=blueprint['tournament']['games_per_pair'],
            file=f"duel__{a['id']}__{b['id']}.json"))
    total=sum(j['games']for j in jobs)
    expected=blueprint['budgets_by_candidate_count'][str(len(models)-3)]['total_games']
    if total!=expected:raise ValueError('Frozen evaluation budget differs')
    return jobs


def test_game(network,entry,test,task,family,seed,device):
    env=ContinuousPushEnv(split='test',action_delay_ms=225,max_ms=310000)
    case=PushCase(task,family,test['style'],'test');env.scenarios=[case]
    env.reset(seed=seed,scenario_id=case.scenario_id)
    for _ in range(1242):
        reserved=sum(env.cards[c[1]].cost*1000 for c in env.pending)
        action=choose(network,env.match,1,entry['mode'],env._mask(),device,reserved)
        _,_,done,truncated,info=env.step(action)
        if done or truncated:
            row=dict(info['metrics'],seed=seed,test_set=test['id'])
            if row['outcome']not in('win','loss','draw'):raise ValueError('Nonterminal test outcome')
            return row
    raise RuntimeError('Finite evaluation game did not finish')


def public_duel_metrics(match):
    result={}
    for side in(1,-1):
        own={name:tower.hp_fraction for name,tower in match.towers[side].items()}
        if side==-1:own=dict(own,left=own['right'],right=own['left'])
        plays=[p for p in match.play_log if p['side']==side]
        def lane(play):
            x=play['x']if side==1 else arena.WIDTH-play['x']
            return 'left'if x<arena.CENTRE_X else'right'
        result[str(side)]=dict(own_tower_fraction_loss={k:1-v for k,v in own.items()},
            accepted_card_counts=dict(Counter(p['card']for p in plays)),
            accepted_card_lane_counts={name:dict(Counter(p['card']for p in plays if lane(p)==name))for name in('left','right')},
            **fireball_outcomes(match.battle.spell_cast_log,side=side),
            missed_fireballs=sum(e.get('caster_side')==side and e.get('spell')=='fireball'and e.get('resolved',False)and not e.get('hits')for e in match.battle.spell_cast_log))
    return result


def summarize_test(rows):
    counts=Counter(r['outcome']for r in rows)
    if not rows or set(counts)-{'win','loss','draw'}:raise ValueError('Known test outcomes required')
    def mean(key):return float(np.mean([r[key]for r in rows]))
    return dict(games=len(rows),counts=dict(counts),points=(counts['win']+.5*counts['draw'])/len(rows),
        mean_card_plays={card:float(np.mean([r['accepted_card_counts'].get(card,0)for r in rows]))for card in DECK_26},
        mean_card_lane_plays={lane:{card:float(np.mean([r['accepted_card_lane_counts'].get(lane,{}).get(card,0)for r in rows]))for card in DECK_26}for lane in('left','right')},
        mean_tower_loss={lane:float(np.mean([r['own_tower_fraction_loss'][lane]for r in rows]))for lane in('left','right','king')},
        mean_elixir_overflow=mean('elixir_overflow'),mean_empty_fireballs=mean('missed_fireballs'),
        mean_cluster_fireballs=mean('cluster_fireballs'),mean_rejected_commands=mean('illegal_actions'))


def evaluate(protocol_path,weights,output):
    protocol_path,weights,output=Path(protocol_path),Path(weights),Path(output)
    protocol=json.loads(protocol_path.read_text());blueprint=validate_protocol(protocol,weights)
    jobs=schedule(protocol,blueprint)
    if not torch.cuda.is_available()or torch.cuda.device_count()<2:raise RuntimeError('Two CUDA devices required; no CPU fallback')
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    models={m['id']:m for m in protocol['models']};tests={t['id']:t for t in blueprint['test_sets']}
    schedules=[[],[]];loads=[0,0]
    for job in sorted(jobs,key=lambda j:-j['games']):
        index=min(range(2),key=lambda i:loads[i]);schedules[index].append(job);loads[index]+=job['games']
    lock=threading.Lock();completed=0;total=sum(loads)
    def finished_game():
        nonlocal completed
        with lock:
            completed+=1
            print(json.dumps(dict(event='heldout_game_progress',completed=completed,total=total)),flush=True)
    def worker(index):
        device=f'cuda:{index}';nets={name:load(entry,weights,device)for name,entry in models.items()}
        cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
        for job in schedules[index]:
            rows=[]
            if job['kind']=='test':
                entry=models[job['model']];test=tests[job['test']]
                for task in blueprint['tasks']:
                    for family in blueprint['families']:
                        for seed in test['seeds']:
                            rows.append(test_game(nets[entry['id']],entry,test,task,family,seed,device));finished_game()
                report=dict(kind='test',model=entry,test_set=test['id'],summary=summarize_test(rows),rows=rows,
                    by_task_family={f'{task}/{family}':summarize_test([r for r in rows if r['task']==task and r['family_id']==family])for task in blueprint['tasks']for family in blueprint['families']})
            else:
                a,b=(models[name]for name in job['pair'])
                for seed in blueprint['tournament']['seeds']:
                    for seat in blueprint['tournament']['seats']:
                        row=duel((nets[a['id']],a['mode']),(nets[b['id']],b['mode']),device,seed,seat,cards,spells,on_complete=public_duel_metrics)
                        if row['outcome']not in('win','loss','draw'):raise ValueError('Nonterminal duel outcome')
                        rows.append(row);finished_game()
                report=dict(kind='duel',pair=job['pair'],games=len(rows),counts=dict(Counter(r['outcome']for r in rows)),rows=rows)
            if len(rows)!=job['games']:raise ValueError('Incomplete scheduled evaluation part')
            with(output/job['file']).open('x')as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(event='CUDA_READY',gpus=[torch.cuda.get_device_name(i)for i in(0,1)])),flush=True)
    with ThreadPoolExecutor(2)as pool:
        futures=[pool.submit(worker,i)for i in(0,1)]
        for future in futures:future.result()
    if completed!=total:raise ValueError('Evaluation game budget missing')
    result=dict(schema='full-candidate-independent-evaluation-v1',status='complete',
        protocol_sha256=sha(protocol_path),blueprint_sha256=protocol['blueprint_sha256'],models=list(models.values()),
        total_games=total,files={j['file']:sha(output/j['file'])for j in jobs},
        test_used_for_selection=False,quality_accepted=False,live_checkpoint_changed=False)
    with(output/'result.json').open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True)
    parser.add_argument('--weights',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();evaluate(args.protocol,args.weights,args.output)
