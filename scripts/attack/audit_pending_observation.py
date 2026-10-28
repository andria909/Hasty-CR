"""Measure demonstration collisions and a pending-position observability gap."""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.attack.audit_air_label_latency import opening
from scripts.attack.search_late_air_defense import rollout
from scripts.attack.sequence_physics import require_current_physics,physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.continuous_push import ContinuousPushEnv
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26
from sim.runner import resolve_deck


def audit(data):
    data=Path(data);manifest=json.loads((data/'manifest.json').read_text());require_current_physics(manifest)
    if file_sha(data/'examples.npz')!=manifest['data_sha256']:raise ValueError('Dataset changed')
    with np.load(data/'examples.npz')as raw:arrays={k:raw[k]for k in raw.files}
    groups=defaultdict(list);heldout=max(manifest['seeds'])
    for record in manifest['records']:
        index=record['array_index'];digest=hashlib.sha256()
        for key in ('planes','scalars','masks'):digest.update(arrays[key][index].tobytes())
        groups[digest.hexdigest()].append(record)
    conflicts=[];cross=[]
    for key,records in groups.items():
        train=[r for r in records if r['seed']!=heldout];hold=[r for r in records if r['seed']==heldout]
        if train and hold:cross.append(key)
        if len(train)>1 and not np.logical_and.reduce([arrays['success_masks'][r['array_index']]for r in train]).any():
            conflicts.append([r['array_index']for r in train])
    counterexample=pending_counterexample()
    return dict(schema='pending-position-observability-audit-v1',physics_pins=physics_pins(),source_sha256=file_sha(__file__),
        dataset=str(data),data_sha256=manifest['data_sha256'],frames=manifest['examples'],unique_inputs=len(groups),
        train_conflict_groups=conflicts,cross_split_identical_inputs=len(cross),
        counterexample=counterexample,
        interpretation='Pending target position is not identifiable from current observation and legal mask; same next command can have different consequences',
        limitations=['Not proof different optimal next actions are required','Not an estimate of frequency or live loss attribution',
                    'No input/model change and no quality promotion'])


def pending_counterexample():
    record=dict(seed=3800057,lane='left',support=False,initial_own_elixir=10.,reaction_ms=1500,decision_ms=2000)
    cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)));initial,hp=opening(record,cards,spells)
    observations=[];outcomes=[]
    for cell in ([1,18],[8,28]):
        env=object.__new__(ContinuousPushEnv);env.match=deepcopy(initial);env.cards=cards;env.spells=spells
        env.profile={'cannon_center_only':1}
        env.slot_ready={};env.reflected=False;slot=env.match.players[1].hand.index('musketeer')
        env.pending=[(initial.elapsed_ms+1350,'musketeer',slot,*cell)]
        for _ in range(5):env.match.step(50)
        observations.append((env._observe(),env._mask()))
        row=rollout(initial,hp,cards,spells,'left',dict(card='musketeer',cell=cell),
                    dict(card='electro_spirit',offset_ms=250,variant=0))
        outcomes.append(dict(pending_cell=cell,**row))
    same={k:np.array_equal(observations[0][0][k],observations[1][0][k])for k in ('planes','scalars')}
    same['mask']=np.array_equal(observations[0][1],observations[1][1])
    return dict(opening=record,observed_ms=initial.elapsed_ms+250,input_arrays_equal=same,outcomes=outcomes)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    with Path(a.output).open('x')as f:json.dump(audit(a.data),f,indent=2);f.write('\n')
