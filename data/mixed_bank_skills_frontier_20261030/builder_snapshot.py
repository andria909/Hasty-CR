"""Mix reproduced bank plans, paid air support and fresh spell skills.

Simulation seeds and split groups are separate: native spell seeds are assigned
to three declared folds before training. No validation match data is consumed.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scripts.attack.build_bank_counterfactuals import opening
from scripts.attack.build_late_air_sequences import replay_match
from scripts.attack.sequence_physics import require_current_physics, physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.adapter import grid_to_point


def support_source(root):
    root=Path(root);prep=json.loads((root/'preparation.json').read_text())
    receipt=json.loads((root/'collected/collection_receipt.json').read_text())
    result=json.loads((root/'collected/result.json').read_text())
    if not prep.get('support') or not receipt['complete'] or receipt['kernel']!=prep['kernel']:
        raise ValueError('Complete paid-support search required')
    if result['schema']!='bank-central-cannon-search-v1' or not result['support'] or result['contexts']!=36:
        raise ValueError('Wrong support search')
    for name in ('contexts.jsonl','result.json'):
        if file_sha(root/'collected'/name)!=receipt['files']['bank_cannon_search/'+name]['sha256']:
            raise ValueError('Support receipt changed')
    if file_sha(root/'collected/contexts.jsonl')!=result['raw_sha256'] or file_sha(root/'job/train.py')!=prep['bootstrap_sha256']:
        raise ValueError('Support source changed')
    config=ast.literal_eval(ast.parse((root/'job/train.py').read_text()).body[-1].value.args[0])
    if file_sha('scripts/attack/build_bank_counterfactuals.py')!=config['pins']['bank_builder.py']:
        raise ValueError('Support opening changed')
    for name,digest in config['extensions'].items():
        if name.startswith('sim/') and name!='sim/train_attack_ppo.py' and file_sha(name)!=digest:
            raise ValueError('Support physics changed: '+name)
    return [json.loads(line)for line in (root/'collected/contexts.jsonl').read_text().splitlines()]


def support_opening(row):
    env=opening(row['seed'],row['lane'],row['bank'],15)
    enemy=env.match.players[-1];before=enemy.elixir
    if 'baby_dragon'not in enemy.hand or not env.match.play_card(-1,'baby_dragon',grid_to_point(3 if row['lane']=='left'else 14,12,1)):
        raise ValueError('Support must be legal and paid')
    if before-enemy.elixir!=4000:raise ValueError('Support payment differs')
    own=env.match.players[1];own.hand=['cannon','electro_spirit','zap','musketeer'if row['hand']=='musketeer'else'fireball']
    own.queue=[c for c in own.deck if c not in own.hand]
    return env


def build(bank,support,fireball,output):
    bank,fireball=Path(bank),Path(fireball)
    b=json.loads((bank/'manifest.json').read_text());f=json.loads((fireball/'manifest.json').read_text())
    for path,m in ((bank,b),(fireball,f)):
        require_current_physics(m)
        if m['split']!='train' or file_sha(path/'examples.npz')!=m['data_sha256']:
            raise ValueError('Pinned current TRAIN data required')
    if b['schema']!='bank-safe-sequence-labels-v1' or b['seeds']!=[352,353,354]:
        raise ValueError('Three-fold bank sequences required')
    if (f.get('draft_only') or f['command_delay_ms']!=1350 or len(f['seeds'])!=3 or
            f.get('cannon_placement_mode')!='central_defense_v1' or
            f.get('empty_board_forbidden_spells')!=['fireball','zap'] or
            f['builder_sha256']!=file_sha('scripts/attack/build_native_fireball_examples.py')):
        raise ValueError('Fresh central-Cannon spell labels and both-spell restraint required')
    with np.load(bank/'examples.npz')as a:arrays={k:list(a[k])for k in a.files}
    records=deepcopy(b['records'])
    for r in records:
        r.update(context='bank:'+str(r['context']),simulation_seed=r['seed'])
    audit=[]
    def append(state,record):
        if state.get('pending_count',0) or state.get('reserved_elixir',0):
            raise ValueError('Public slot augmentation must not hide pending commands')
        index=len(records)
        for key in arrays:arrays[key].append(state[key])
        records.append(dict(record,accepted=True,array_index=index,pending_count=0,reserved_elixir=0))
    for index,row in enumerate(support_source(support)):
        safe=[r for r in row['trials']if r['valid'] and r['clear_without_damage'] and r['commands']]
        chosen=min(safe,key=lambda r:(r['elixir_spent'],json.dumps(r['commands'],sort_keys=True)))if safe else None
        audit.append(dict(seed=row['seed'],lane=row['lane'],bank=row['bank'],hand=row['hand'],accepted=chosen is not None,plan=chosen))
        if chosen is None:continue
        env=support_opening(row);hp={k:t.hitpoints for k,t in env.match.towers[1].items()}
        states=replay_match(env.match,hp,chosen,env.cards,env.spells)
        # Replay every command, but do not train ambiguous WAIT while a queued
        # card's position is absent from the public309 observation contract.
        for state in states:
            if state['pending_count']:continue
            append(state,dict(seed=row['seed'],simulation_seed=row['seed'],context='support:'+str(index),
                skill='air_support_defense',lane=row['lane'],decision_ms=state['decision_ms'],action=state['action']))
    mapping=dict(zip(sorted(f['seeds']),[352,353,354]))
    with np.load(fireball/'examples.npz')as data:
        for r in f['records']:
            if not r['accepted']:continue
            i=r['array_index'];state={key:data[key][i]for key in arrays}
            skill='spell_restraint'if r['skill']=='fireball_restraint'else'fireball_group'
            append(state,dict(seed=mapping[r['seed']],simulation_seed=r['seed'],context='spell:'+str(i),skill=skill,
                action=None if skill=='spell_restraint'else int(np.flatnonzero(state['success_masks'])[0])))
    required={'safe_bank_sequence','air_support_defense','fireball_group','spell_restraint'}
    for seed in (352,353,354):
        if {r['skill']for r in records if r['seed']==seed}!=required:
            raise ValueError('Every whole fold must cover all four skills')
    identities={}
    for r in records:
        i=r['array_index'];digest=hashlib.sha256(b''.join(arrays[k][i].tobytes()for k in ('planes','scalars','masks'))).hexdigest()
        if digest in identities and identities[digest]!=r['seed']:raise ValueError('Input leakage across folds')
        identities[digest]=r['seed'];r['input_sha256']=digest
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'examples.npz',**{k:np.stack(v)for k,v in arrays.items()})
    manifest=dict(schema='mixed-bank-public-sequences-v1',split='train',seeds=[352,353,354],records=records,
        examples=len(records),physics_pins=physics_pins(),command_delay_ms=1350,
        slot_augmentation_contract='public309_no_pending_busy_slots_v1',
        source_hashes=dict(bank=b['data_sha256'],fireball=f['data_sha256'],support=file_sha(Path(support)/'collected/contexts.jsonl')),
        builder_sha256=file_sha(__file__),data_sha256=file_sha(output/'examples.npz'),
        support_audit=audit,spell_fold_mapping=mapping,
        grouping='seed is split fold; simulation_seed is the actual RNG seed; maximum whole fold held out',
        skills=sorted(required),sampling='Equal skills and contexts; bounded WAIT share only within wait/deploy sequences',
        full_game_training_ready=False,quality_accepted=False,
        limitations=['Synthetic isolated paid threats, not second-wave resource planning',
            'Mixed-skill demonstrations, not proof each chosen action dominates every legal alternative',
            'Pending states omitted from labels, all underlying commands and outcomes still replayed',
            'Urgent controls and two separated paid waves remain before the general teacher PPO study'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('bank','support','fireball','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();m=build(a.bank,a.support,a.fireball,a.output);print(json.dumps(dict(examples=m['examples'],skills=m['skills'])))
