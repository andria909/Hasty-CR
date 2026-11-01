"""Append replayed on-policy action improvements to the mixed TRAIN curriculum."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from scripts.attack.sequence_physics import require_current_physics,physics_pins
from scripts.attack.sequence_sampling import mixed_skill_probabilities
from scripts.kaggle.local_artifact_transfer import file_sha

SKILL='onpolicy_defense_response'
ARRAYS=('planes','scalars','masks','success_masks')


def replayed_example(collection,reference,receipt):
    root=Path(collection);m=json.loads((root/'manifest.json').read_text())
    context=Path(reference['context_path'])
    if file_sha(context)!=reference['context_sha256']:raise ValueError('Counterfactual context changed')
    row=json.loads(context.read_text());state=m['records'][reference['index']]
    if row['state']!=state or row['successful_actions']!=reference['successful_actions']:
        raise ValueError('Accepted action identity differs')
    r=json.loads(Path(receipt).read_text())
    if (r.get('schema')!='bank-match-context-independent-replay-v1'or
            r['context_sha256']!=reference['context_sha256']or r['index']!=reference['index']or
            r['fold']!=state['fold']or r['source_sha256']!=file_sha('scripts/attack/replay_bank_match_context.py')):
        raise ValueError('Independent current replay required')
    wanted={0,*reference['successful_actions']};pairs={(a,d)for a in wanted for d in (1350,2025)}
    actual={(x['action'],x['delay_ms']):x for x in r['replayed']}
    if set(actual)!=pairs or len(actual)!=len(r['replayed']):raise ValueError('Replay branches incomplete')
    for trial in row['finals']:
        key=(trial['action'],trial['delay_ms'])
        if key not in pairs:continue
        check=actual[key]
        if (not check.get('all_terminal_metrics_exact')or check['own_damage']!=trial['own_damage']or
                check['outcome']!=trial['metrics']['outcome']):raise ValueError('Replayed outcomes differ')
    path=root/state['snapshot']
    if file_sha(path)!=state['sha256']:raise ValueError('Saved state changed')
    env=pickle.loads(path.read_bytes())  # Locally generated, hash-checked snapshots only.
    if env.split!='train'or env.pending or env.seed!=state['episode_seed']:
        raise ValueError('Public no-pending TRAIN state required')
    obs=env._observe();mask=env._mask();success=np.zeros_like(mask)
    if not reference['successful_actions']:raise ValueError('Nonempty demonstrated improvement required')
    for action in reference['successful_actions']:
        if not 0<action<len(mask)or not mask[action]:raise ValueError('Demonstrated action is not legal')
        success[action]=True
    sample=dict(planes=obs['planes'],scalars=obs['scalars'],masks=mask,success_masks=success)
    record=dict(seed=state['fold'],simulation_seed=state['episode_seed'],context='match:'+state['snapshot'],
        skill=SKILL,action=min(reference['successful_actions']),accepted=True,pending_count=0,reserved_elixir=0,
        decision_ms=state['at_ms'],scenario=state['scenario'],improved_action_count=int(success.sum()),
        source_context_sha256=reference['context_sha256'],replay_sha256=file_sha(receipt),
        scope='Improved outcome/own HP under original-policy continuation at both delays; not zero-damage guarantee')
    return sample,record


def assemble(base,verified,replays,output):
    base,verified,replays,output=map(Path,(base,verified,replays,output))
    b=json.loads((base/'manifest.json').read_text());proof=json.loads(verified.read_text())
    require_current_physics(b);require_current_physics(proof)
    if (b['schema']!='mixed-bank-public-sequences-v1'or b['split']!='train'or
            b['seeds']!=[352,353,354]or file_sha(base/'examples.npz')!=b['data_sha256']):
        raise ValueError('Pinned three-fold mixed base required')
    collection=Path(proof['root']);m=json.loads((collection/'manifest.json').read_text())
    if (proof.get('schema')!='verified-bank-match-search-v1'or
            proof['contexts_verified']!=len(m['records'])or
            proof['collection_sha256']!=file_sha(collection/'manifest.json')):
        raise ValueError('Complete verified search required')
    # Recompute the all-context verification rather than trust a hand-edited summary.
    from scripts.attack.verify_bank_match_search import verify
    output.mkdir(parents=True,exist_ok=False)
    fresh=verify([r['root']for r in proof['shards']],output/'search_verified.json')
    if fresh!=proof:raise ValueError('Search proof differs from source evidence')
    with np.load(base/'examples.npz',allow_pickle=False)as archive:
        arrays={key:list(archive[key])for key in ARRAYS}
    records=deepcopy(b['records']);additions=[]
    for reference in proof['accepted']:
        receipt=replays/f"context_{reference['index']:04d}.json"
        sample,record=replayed_example(collection,reference,receipt)
        record['array_index']=len(arrays['planes'])
        for key in ARRAYS:arrays[key].append(sample[key])
        records.append(record);additions.append(record)
    if {r['seed']for r in additions}!={352,353,354}:
        raise ValueError('New skill needs training folds and heldout fold; no train-only acceptance')
    identities={}
    for record in records:
        i=record['array_index'];digest=hashlib.sha256(b''.join(arrays[k][i].tobytes()for k in ('planes','scalars','masks'))).hexdigest()
        if digest in identities and identities[digest]!=record['seed']:raise ValueError('Public inputs leak across folds')
        identities[digest]=record['seed'];record['input_sha256']=digest
    train=[r['array_index']for r in records if r['seed']!=354]
    probabilities=mixed_skill_probabilities(records,train,.5)
    mass={skill:float(sum(probabilities[j]for j,i in enumerate(train)if records[i]['skill']==skill))
        for skill in {r['skill']for r in records}}
    np.savez_compressed(output/'examples.npz',**{k:np.stack(v)for k,v in arrays.items()})
    manifest=deepcopy(b)
    manifest.update(records=records,examples=len(records),skills=sorted(mass),physics_pins=physics_pins(),
        data_sha256=file_sha(output/'examples.npz'),builder_sha256=file_sha(__file__),
        base_data=str(base),base_manifest_sha256=file_sha(base/'manifest.json'),
        correction_search_sha256=file_sha(verified),correction_examples=len(additions),
        correction_counts_by_fold={str(f):sum(r['seed']==f for r in additions)for f in (352,353,354)},
        sampling_skill_mass=mass,full_game_training_ready=False,quality_accepted=False,
        correction_scope='Local first-action improvement in full TRAIN games under fixed actor continuation; '
        'not an expert full trajectory. Equal skill mass is a new treatment; paired full validation required.')
    manifest['limitations'].append('On-policy corrections are limited to Cannon/Musk from no-pending WAIT states; no general planning guarantee')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('base','verified','replays','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();r=assemble(a.base,a.verified,a.replays,a.output);print(json.dumps(dict(examples=r['examples'],corrections=r['correction_examples'])))
