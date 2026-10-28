"""Rebuild empty-board targets to forbid both Fireball and Zap without changing inputs."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from scripts.attack.sequence_physics import require_current_physics
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.continuous_push import DECKS
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26
from sim.eval_defense_duel import mechanical_mask,policy_observation,JOINT_MODE
from sim.match import Match
from sim.runner import resolve_deck


def build(data,output):
    data,output=Path(data),Path(output)
    original=json.loads((data/'manifest.json').read_text());require_current_physics(original)
    if original['schema']!='native-fireball-development-examples-v1' or original['split']!='train' or file_sha(data/'examples.npz')!=original['data_sha256']:
        raise ValueError('Pinned native development examples required')
    with np.load(data/'examples.npz',allow_pickle=False)as source:arrays={k:source[k].copy()for k in source.files}
    cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    result=deepcopy(original);count=changed=0;removed=0
    for row in result['records']:
        if not row['accepted'] or row['skill']!='fireball_restraint':continue
        m=Match(cards=cards,spells=spells,decks=(list(DECK_26),list(DECKS['hog'])),seed=row['seed'],geometry='native_anchors_candidate_v1')
        m.players[1].elixir=round(row['initial_own_elixir']*1000)
        while m.elapsed_ms<row['age_ms']:m.step(50)
        if any(e.alive and e.side==-1 and not e.is_tower for e in m.battle.entities.values()):
            raise ValueError('Restraint scene has a non-tower target')
        if any(t.hitpoints!=t.max_hitpoints for ts in m.towers.values()for t in ts.values()):
            raise ValueError('Do not suppress potential tower finish examples')
        i=row['array_index'];obs=policy_observation(m,1,JOINT_MODE);mask=mechanical_mask(m,cards,spells,1)
        if not all(np.array_equal(arrays[k][i],v)for k,v in [('planes',obs['planes']),('scalars',obs['scalars']),('masks',mask)]):
            raise ValueError('Stored restraint observation does not reproduce')
        success=mask.copy()
        for slot,card in enumerate(m.players[1].hand):
            if card in ('fireball','zap'):success[1+slot*576:1+(slot+1)*576]=False
        if not success.any() or np.any(success&~arrays['success_masks'][i]):
            raise ValueError('New restraint may only narrow old successful actions')
        n=int((arrays['success_masks'][i]&~success).sum());removed+=n;changed+=bool(n);count+=1
        arrays['success_masks'][i]=success
        row['forbidden_empty_spells']=['fireball','zap']
    if not changed:raise ValueError('No Zap restraint coverage')
    output.mkdir(parents=True,exist_ok=False);np.savez_compressed(output/'examples.npz',**arrays)
    result.update(data_sha256=file_sha(output/'examples.npz'),used_in_training=False,
        restraint_version='full_health_empty_board_no_fireball_or_zap_v1',
        restraint_builder_sha256=file_sha(__file__),source_data=str(data),source_data_sha256=original['data_sha256'],
        restraint_audit=dict(reproduced_scenes=count,changed_scenes=changed,removed_zap_targets=removed,
                             observations_unchanged=True,damage_labels_unchanged=True,heldout_groups_unchanged=True))
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=build(a.data,a.output);print(json.dumps(r['restraint_audit']))
