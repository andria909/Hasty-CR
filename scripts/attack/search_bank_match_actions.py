"""Paired action improvement from saved full TRAIN games, with paid reinforcements.

Screen a finite cell set, then finish each selected branch with the same actor.
This estimates improvement under that continuation, not an optimal expert.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import pickle

import torch

from scripts.attack.collect_bank_match_states import candidate
from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.cannon_placement import CENTRAL_CELLS
from sim.train_attack_ppo import initialize_attack_policy

DELAYS=(1350,2025)
MUSK_CELLS=tuple((x,y)for x in (3,6,8,9,11,14)for y in (20,23,26))


def action_candidates(env):
    mask=env._mask();rows=[dict(action=0,card='wait',cell=None)]
    for slot,card in enumerate(env.match.players[1].hand):
        cells=CENTRAL_CELLS if card=='cannon'else MUSK_CELLS if card=='musketeer'else ()
        for x,y in cells:
            action=1+slot*576+y*18+x
            if mask[action]:rows.append(dict(action=action,card=card,cell=[x,y]))
    return rows


def actor_action(network,env):
    obs=env._observe();mask=torch.as_tensor(env._mask())
    with torch.inference_mode():
        logits,_=network(torch.as_tensor(obs['planes'])[None],torch.as_tensor(obs['scalars'])[None])
        if not torch.isfinite(logits[0,mask]).all():raise ValueError('Nonfinite actor logits')
        return int(logits[0].masked_fill(~mask,-torch.inf).argmax())


def branch(template,network,first,delay,horizon_ms=None):
    env=deepcopy(template);env.action_delay_ms=delay
    if env.pending or not env._mask()[first['action']]:raise ValueError('Legal initial command without pending orders required')
    start=env.match.elapsed_ms;before=env._metrics();own_commands=len(env.actions);enemy_commands=len(env.enemy_actions)
    start_hp={k:t.hitpoints for k,t in env.match.towers[1].items()}
    start_enemy_hp=sum(t.hitpoints for t in env.match.towers[-1].values())
    env.step(first['action'])
    while not env.done and (horizon_ms is None or env.match.elapsed_ms<start+horizon_ms):
        env.step(actor_action(network,env))
    metrics=env._metrics()
    # A command near a terminal event might never execute. It is not a valid label.
    physical_cell=([17-first['cell'][0],first['cell'][1]]if env.reflected else first['cell']) if first['action']else None
    accepted=not first['action']or any(c['card']==first['card'] and [c['x'],c['y']]==physical_cell
        and c['at_ms']==start+((delay+49)//50)*50 for c in env.actions[own_commands:])
    if env.illegal!=before['illegal_actions']:raise ValueError('Branch generated rejected or illegal commands')
    return dict(**first,delay_ms=delay,terminal=env.done,first_command_accepted=accepted,
        start_ms=start,end_ms=env.match.elapsed_ms,elapsed_ms=env.match.elapsed_ms-start,
        own_damage=sum(start_hp[k]-t.hitpoints for k,t in env.match.towers[1].items()),
        enemy_damage=start_enemy_hp-sum(t.hitpoints for t in env.match.towers[-1].values()),
        spent_elixir=metrics['elixir_spent']-before['elixir_spent'],
        final_elixir=env.match.players[1].elixir/1000,
        remaining_enemy_hp=sum(e.hitpoints for e in env.match.battle.entities.values()
            if e.alive and e.side==-1 and not e.is_tower),
        own_commands=env.actions[own_commands:],enemy_commands=env.enemy_actions[enemy_commands:],metrics=metrics)


def screen_rank(row):
    return (row['own_damage'],-row['enemy_damage'],row['remaining_enemy_hp'],row['spent_elixir'],row['action'])


def quality(row):
    if not row['terminal']:raise ValueError('Full match result required')
    return ({'win':2,'draw':1,'loss':0}[row['metrics']['outcome']],-row['own_damage'])


def robust_labels(finals):
    """Only strict outcome/own-HP improvement over WAIT at BOTH measured delays."""
    base={r['delay_ms']:r for r in finals if r['action']==0}
    if set(base)!=set(DELAYS):raise ValueError('Both WAIT controls required')
    groups={}
    for row in finals:
        key=(row['action'],row['delay_ms'])
        if key in groups:raise ValueError('Duplicate action/delay branch')
        groups[key]=row
    accepted=[]
    for action in sorted({r['action']for r in finals}-{0}):
        pair=[groups.get((action,d))for d in DELAYS]
        if any(r is None for r in pair):raise ValueError('Every finalist needs both delays')
        if all(r['first_command_accepted'] and quality(r)>quality(base[r['delay_ms']])for r in pair):accepted.append(action)
    return accepted


def search_context(root,manifest,record,network):
    path=root/record['snapshot']
    if file_sha(path)!=record['sha256']:raise ValueError('Snapshot digest differs')
    env=pickle.loads(path.read_bytes())  # Only verified locally generated snapshots.
    if (env.split!='train'or env.seed!=record['episode_seed']or not candidate(env,0)or
            actor_action(network,env)!=0):raise ValueError('Pinned TRAIN actor/WAIT state required')
    choices=action_candidates(env);screen=[];selected={0}
    for delay in DELAYS:
        rows=[branch(env,network,a,delay,15000)for a in choices];screen.extend(rows)
        # Keep both card types even if the other looks stronger at 15 seconds.
        for card in ('cannon','musketeer'):
            selected.update(r['action']for r in sorted((r for r in rows if r['card']==card),key=screen_rank)[:2])
    finals=[]
    for delay in DELAYS:
        for action in choices:
            if action['action'] in selected:finals.append(branch(env,network,action,delay))
    baseline=next(r for r in finals if r['action']==0 and r['delay_ms']==1350)
    original=next(g['metrics']for g in manifest['games']if g['episode_seed']==record['episode_seed'])
    if baseline['metrics']!=original:raise ValueError('WAIT continuation does not reproduce original complete game')
    return dict(state=record,screen=screen,finals=finals,baseline_reproduced_exactly=True,
        successful_actions=robust_labels(finals),scope='One initial action, original learned policy thereafter; finite candidate cells')


def search(root,output,indices=None):
    torch.set_num_threads(1);root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    path=root/'manifest.json';m=json.loads(path.read_text());pins=physics_pins();source=file_sha(__file__)
    if m['schema']!='bank-full-match-candidate-states-v1'or m['split']!='train'or m['physics_pins']!=pins:
        raise ValueError('Current TRAIN snapshot collection required')
    if file_sha(root/'driver.py')!=m['source_sha256']or file_sha(m['weights'])!=m['weights_sha256']:
        raise ValueError('Pinned collection source and checkpoint required')
    selected=list(range(len(m['records'])))if indices is None else list(indices)
    if not selected or len(set(selected))!=len(selected)or any(i<0 or i>=len(m['records'])for i in selected):
        raise ValueError('Unique in-range snapshot indices required')
    net,_,_=initialize_attack_policy(torch.load(m['weights'],map_location='cpu',weights_only=False),device='cpu',
        num_planes=20,num_scalars=309,observation_contract='joint_public_presence_continuous_slots_v2')
    net.eval();rows=[]
    protocol=dict(schema='bank-match-action-search-v1',root=str(root),manifest_sha256=file_sha(path),
        weights_sha256=m['weights_sha256'],physics_pins=pins,source_sha256=source,indices=selected,
        delays_ms=list(DELAYS),screen_ms=15000,musketeer_cells=MUSK_CELLS,cannon_cells=CENTRAL_CELLS,
        continuation='Original deterministic policy to actual game termination; paid opponent continues',
        acceptance='Strict lexicographic outcome/own-tower-HP improvement over WAIT at both delays',
        full_training_accepted=False,live_games=False)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (output/'driver.py').write_bytes(Path(__file__).read_bytes())
    for index in selected:
        row=search_context(root,m,m['records'][index],net);row['index']=index
        target=output/f'context_{index:04d}.json';target.write_text(json.dumps(row,indent=2)+'\n')
        rows.append(dict(index=index,path=str(target),sha256=file_sha(target),fold=row['state']['fold'],
            successful_actions=row['successful_actions'],screen_branches=len(row['screen']),full_branches=len(row['finals'])))
        print(json.dumps(dict(event='COUNTERFACTUAL_PROGRESS',complete=len(rows),target=len(selected),
            improved_states=sum(bool(r['successful_actions'])for r in rows))),flush=True)
    if pins!=physics_pins()or source!=file_sha(__file__):raise ValueError('Search source changed during execution')
    result=dict(protocol=protocol,contexts=rows,complete=True,labels_ready=False,
        scope='Action-improvement candidates; independent replay and teacher assembly still required')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--indices',type=int,nargs='+');a=p.parse_args();search(a.root,a.output,a.indices)
