"""Add paired immediate-defense witnesses to the bank-preservation curriculum."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.attack.build_bank_counterfactuals import opening,rollout,label,score


def augment(source,output):
    source,output=Path(source),Path(output)
    base=json.loads((source/'manifest.json').read_text())
    if hashlib.sha256((source/'examples.npz').read_bytes()).hexdigest()!=base['data_sha256']:
        raise ValueError('Base data changed')
    if hashlib.sha256(Path('scripts/attack/build_bank_counterfactuals.py').read_bytes()).hexdigest()!=base['builder_sha256']:
        raise ValueError('Base builder changed')
    output.mkdir(parents=True,exist_ok=False)
    with np.load(source/'examples.npz',allow_pickle=False)as a:data={k:list(a[k])for k in a.files}
    records=deepcopy(base['records']);search=[]
    for r in records:r['context']='bank-'+r['lane']
    for seed in base['seeds']:
        for lane in ('left','right'):
            x=3 if lane=='left'else 14;found=False
            # Search at the engine's50ms resolution:250ms spacing can skip
            # the narrow interval where one policy decision changes a hit.
            for delay in dict.fromkeys([1750,4750,*range(0,8001,50)]):
                env=opening(seed,lane,4000,15)
                for _ in range(delay//50):env.match.step(50)
                env.match.players[1].elixir=4000;env._previous=env._metrics()
                action=1+2*576+20*18+x
                wait=rollout(env,0,(x,20));deploy=rollout(env,action,(x,20))
                probe=dict(seed=seed,lane=lane,delay_ms=delay,wait=score(wait),deploy=score(deploy))
                search.append(probe)
                if score(deploy)>=score(wait):continue
                accepted,success,outcomes=label(env,lane)
                if not accepted or success[0]:continue
                obs=env._observe();index=len(data['planes'])
                for k,v in dict(planes=obs['planes'],scalars=obs['scalars'],masks=env._mask(),success_masks=success).items():data[k].append(v)
                records.append(dict(seed=seed,lane=lane,bank=4000,enemy_y=15,reaction_extra_ms=delay,
                    accepted=True,array_index=index,skill='bank_urgent_deploy',context='bank-'+lane,
                    action=int(np.flatnonzero(success)[0]),alternatives=outcomes))
                found=True;print(json.dumps(dict(seed=seed,lane=lane,delay_ms=delay,deploy_label_verified=True)),flush=True);break
            if not found:raise ValueError('No urgency witness; do not train an always-WAIT teacher')
    np.savez_compressed(output/'examples.npz',**{k:np.stack(v)for k,v in data.items()})
    identities={}
    for r in records:
        if not r['accepted']:continue
        i=r['array_index'];digest=hashlib.sha256(b''.join(data[k][i].tobytes()for k in ('planes','scalars','masks'))).hexdigest()
        if digest in identities and identities[digest]!=r['seed']:raise ValueError('Input leakage across seeds')
        identities[digest]=r['seed'];r['input_sha256']=digest
    base.update(records=records,examples=len(data['planes']),
        base_data_sha256=base['data_sha256'],augmentation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256((output/'examples.npz').read_bytes()).hexdigest(),urgency_search=search,
        sampling_groups='context=lane curriculum group, not a single trajectory; equal WAIT/deploy training mass')
    (output/'manifest.json').write_text(json.dumps(base,indent=2)+'\n')
    return base


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=augment(a.source,a.output);print(json.dumps(dict(examples=r['examples'])))
