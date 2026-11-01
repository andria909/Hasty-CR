"""Capture replayable TRAIN states where a policy can defend but chooses WAIT.

These are candidate counterfactual search states, NOT successful action labels.
Snapshots include simulator internals only for replay; model inputs stay public.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import pickle

import numpy as np
import torch

from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.continuous_push import DECKS
from sim.train_attack_ppo import initialize_attack_policy, make_training_env


def candidate(env, action):
    if action!=0 or env.pending or env.match.elapsed_ms<5000:return False
    if not any(e.alive and e.side==-1 and not e.is_tower for e in env.match.battle.entities.values()):return False
    mask=env._mask()
    return any(card in ('cannon','musketeer')and mask[1+slot*576:1+(slot+1)*576].any()
        for slot,card in enumerate(env.match.players[1].hand))


def checked_snapshot(env):
    """Serialize locally trusted state and verify a one-second paired replay."""
    payload=pickle.dumps(env,protocol=pickle.HIGHEST_PROTOCOL)
    restored=pickle.loads(payload);reference=deepcopy(env)
    for _ in range(5):
        a,b=reference._observe(),restored._observe()
        if (any(not np.array_equal(a[k],b[k])for k in ('planes','scalars'))or
                not np.array_equal(reference._mask(),restored._mask())or reference._metrics()!=restored._metrics()):
            raise ValueError('Snapshot roundtrip changed observation, legality or metrics')
        if reference.done:break
        reference.step(0);restored.step(0)
    return payload


def collect(weights, config, output):
    torch.set_num_threads(1);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    c=json.loads(Path(config).read_text())
    keys=('curriculum','continuous_profile','validation_selection','continuous_focus','geometry','coordinate_mode',
        'max_ms','tower_support_weight','spell_waste_weight','crown_only_weight','dormant_king_weight')
    options={k:c[k]for k in keys};options['action_delay_ms']=1350
    p=options['continuous_profile']
    if (options['curriculum']!='continuous_v1'or p['action_cooldown_ms']!=0 or
            p['decision_interval_ms']!=250 or any(p[k]!=1 for k in
            ('cannon_center_only','source_radius_reach','hog_sight_clipping'))):
        raise ValueError('Current execution contract required')
    pins=physics_pins();source=file_sha(__file__);weight_sha=file_sha(weights)
    net,_,_=initialize_attack_policy(torch.load(weights,map_location='cpu',weights_only=False),device='cpu',
        num_planes=20,num_scalars=309,observation_contract='joint_public_presence_continuous_slots_v2')
    net.eval();records=[];games=[]
    for fold in (352,353,354):
        for task in ('attack','defense'):
            for family in sorted(DECKS):
                # Fresh TRAIN RNGs, disjoint from the full validation experiment.
                seed=202611100+len(games)
                style='push'if fold%2==0 else'random'
                scenario=f'{task}::{family}/{style}'
                env=make_training_env(task_mode='joint',split='train',seed=seed,**options)
                obs,info=env.reset(seed=seed,scenario_id=scenario);captured=set()
                while not env.done:
                    with torch.inference_mode():
                        logits,_=net(torch.as_tensor(obs['planes'])[None],torch.as_tensor(obs['scalars'])[None])
                        mask=torch.as_tensor(info['action_mask'])
                        if not torch.isfinite(logits[0,mask]).all():raise ValueError('Nonfinite actor')
                        action=int(logits[0].masked_fill(~mask,-torch.inf).argmax())
                    phase=0 if env.match.elapsed_ms<45000 else 1 if env.match.elapsed_ms<120000 else 2
                    if phase not in captured and candidate(env,action):
                        name=f'state_{len(records):04d}.pickle';(output/name).write_bytes(checked_snapshot(env))
                        records.append(dict(snapshot=name,sha256=file_sha(output/name),fold=fold,episode_seed=seed,
                            split='train',scenario=scenario,phase=phase,at_ms=env.match.elapsed_ms,
                            action=action,hand=list(env.match.players[1].hand),elixir=env.match.players[1].elixir/1000,
                            pending_count=len(env.pending),replay_roundtrip_verified=True,action_label_accepted=False))
                        captured.add(phase)
                    obs,_,_,_,info=env.step(action)
                games.append(dict(fold=fold,episode_seed=seed,scenario=scenario,metrics=info['metrics']))
                (output/'progress.json').write_text(json.dumps(dict(games=len(games),target_games=36,states=len(records)))+'\n')
                print(json.dumps(dict(event='TRAIN_STATES_PROGRESS',games=len(games),target_games=36,states=len(records))),flush=True)
    if pins!=physics_pins()or source!=file_sha(__file__)or weight_sha!=file_sha(weights):
        raise ValueError('Source or weights changed during collection')
    (output/'driver.py').write_bytes(Path(__file__).read_bytes())
    result=dict(schema='bank-full-match-candidate-states-v1',split='train',weights=str(weights),
        weights_sha256=weight_sha,config_sha256=file_sha(config),options=options,physics_pins=pins,
        source_sha256=source,records=records,games=games,folds=[352,353,354],
        heldout_fold=354,labels_ready=False,quality_accepted=False,
        scope='Unlabelled WAIT states from paid full TRAIN games; no pending commands; bounded 3 states/game. '
        'Next: compare alternative paid continuations and keep whole fold354 outside optimizer data. '
        'Do not load pickle snapshots from untrusted sources.')
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('weights','config','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();collect(a.weights,a.config,a.output)
