"""Distinguish WAIT, alternative cards, and nearby placements on correction labels."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
import torch

from sim.env import ClashEnv
from sim.continuous_push import CONTRACT
from sim.train_attack_ppo import initialize_attack_policy
from scripts.kaggle.local_artifact_transfer import file_sha


def classify(choice,successful):
    if choice in successful:return dict(category='accepted_label',distance_tiles=0)
    if choice==0:return dict(category='wait',distance_tiles=None)
    decoded=ClashEnv.decode(choice)
    labels=[ClashEnv.decode(a)for a in successful]
    if decoded[0]=='ability':return dict(category='different_card',distance_tiles=None)
    same=[p for p in labels if p is not None and p[0]==decoded[0]]
    if not same:return dict(category='different_card',distance_tiles=None)
    distance=min(math.hypot(p[1]-decoded[1],p[2]-decoded[2])for p in same)
    return dict(category='same_card_other_cell',distance_tiles=distance)


def diagnose(verified,data,output):
    verified,data=Path(verified),Path(data);proof=json.loads(verified.read_text())
    if proof.get('schema')!='verified-bank-correction-pair-v1'or not proof['actual_cuda_verified']:
        raise ValueError('Independently verified CUDA cohort required')
    manifest=json.loads((data/'manifest.json').read_text())
    if file_sha(data/'examples.npz')!=proof['common_data_sha256'] or manifest['data_sha256']!=proof['common_data_sha256']:
        raise ValueError('Different correction data')
    selected=[r for r in manifest['records']if r.get('skill')=='onpolicy_defense_response'and r['accepted']]
    ids=[r['array_index']for r in selected];heldout=max(r['seed']for r in selected)
    with np.load(data/'examples.npz',allow_pickle=False)as z:
        planes=torch.tensor(z['planes'][ids]);scalars=torch.tensor(z['scalars'][ids])
        masks=torch.tensor(z['masks'][ids],dtype=torch.bool);success=z['success_masks'][ids].astype(bool)
    torch.set_num_threads(1);models=[]
    for candidate in proof['runs']:
        path=Path(candidate['checkpoint'])
        if file_sha(path)!=candidate['checkpoint_sha256']:raise ValueError('Changed checkpoint')
        blob=torch.load(path,map_location='cpu',weights_only=False)
        net,_,_=initialize_attack_policy(blob,device='cpu',num_planes=20,num_scalars=309,observation_contract=CONTRACT)
        net.eval()
        with torch.inference_mode():
            logits,_=net(planes,scalars);probs=logits.masked_fill(~masks,-torch.inf).softmax(-1)
            choices=probs.argmax(-1).tolist()
        rows=[]
        for i,(r,choice)in enumerate(zip(selected,choices)):
            actions=np.flatnonzero(success[i]).tolist();kind=classify(choice,actions)
            rows.append(dict(array_index=r['array_index'],context=r['context'],seed=r['seed'],
                choice=choice,successful_actions=actions,wait_probability=float(probs[i,0]),
                **kind))
        parts={}
        for split,is_heldout in (('train',False),('local_heldout',True)):
            subset=[r for r in rows if (r['seed']==heldout)==is_heldout];counts=Counter(r['category']for r in subset)
            reported=candidate['common_data_score']['by_skill']['onpolicy_defense_response'][split]
            if reported['successful_choices']!=counts['accepted_label']or reported['examples']!=len(subset):
                raise ValueError('Choice decoding disagrees with independent scoring')
            parts[split]=dict(examples=len(subset),categories=dict(counts),
                same_card_within_one_tile=sum(r['category']=='same_card_other_cell'and r['distance_tiles']<=1 for r in subset),
                mean_wait_probability=sum(r['wait_probability']for r in subset)/len(subset))
        models.append(dict(id=candidate['id'],checkpoint_sha256=candidate['checkpoint_sha256'],splits=parts,rows=rows))
    report=dict(schema='verified-correction-choice-diagnostic-v1',verified_sha256=file_sha(verified),
        data_sha256=manifest['data_sha256'],models=models,full_training_accepted=False,
        scope='Distances to searched labels, not proof that every alternative loses; not full-game win rate')
    Path(output).write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('verified','data','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();r=diagnose(a.verified,a.data,a.output)
    print(json.dumps([dict(id=x['id'],splits=x['splits'])for x in r['models']],indent=2))
