"""Diagnostic permutations of a local holdout; never training or model selection."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from sim.hand_slot_augmentation import permute
from sim.continuous_push import CONTRACT
from sim.train_attack_ppo import initialize_attack_policy


def audit(data,checkpoints):
    data=Path(data);manifest=json.loads((data/'air_manifest.json').read_text())
    raw=data/'air_examples.npz'
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if manifest['split']!='train' or sha(raw)!=manifest['data_sha256']:
        raise ValueError('Verified local TRAIN fixtures required')
    records=[r for r in manifest['records'] if r['accepted']]
    heldout=max(r['seed'] for r in records)
    rows=[r for r in records if r['seed']==heldout]
    permutations=list(itertools.permutations(range(4)))
    torch.set_num_threads(1)
    with np.load(raw,allow_pickle=False) as arrays:
        planes=[];scalars=[];masks=[];labels=[]
        for row in rows:
            i=row['array_index']
            for order in permutations:
                s,m=permute(arrays['scalars'][i],arrays['masks'][i],order)
                _,label=permute(arrays['scalars'][i],arrays['success_masks'][i],order)
                planes.append(arrays['planes'][i]);scalars.append(s);masks.append(m);labels.append(label)
    planes=torch.as_tensor(np.stack(planes));scalars=torch.as_tensor(np.stack(scalars))
    masks=torch.as_tensor(np.stack(masks));labels=torch.as_tensor(np.stack(labels))
    result=[]
    for checkpoint in checkpoints:
        blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
        net,_,_=initialize_attack_policy(blob,device='cpu',num_scalars=309,observation_contract=CONTRACT)
        net.eval();correct=[]
        with torch.inference_mode():
            for start in range(0,len(planes),64):
                logits,_=net(planes[start:start+64],scalars[start:start+64])
                mask=masks[start:start+64]
                if not torch.isfinite(logits[mask]).all():raise ValueError('Nonfinite legal logits')
                choice=logits.masked_fill(~mask,-torch.inf).argmax(-1)
                correct.extend(labels[start:start+64].gather(1,choice[:,None]).flatten().tolist())
        correct=np.asarray(correct).reshape(len(rows),24)
        stats={}
        for skill in sorted({r['skill'] for r in rows}):
            selected=correct[[i for i,r in enumerate(rows) if r['skill']==skill]]
            stats[skill]=dict(states=len(selected),original_correct=int(selected[:,0].sum()),
                permutation_mean_accuracy=float(selected.mean()),any_permutation_correct=int(selected.any(1).sum()),
                every_permutation_correct=int(selected.all(1).sum()))
        result.append(dict(checkpoint=str(checkpoint),sha256=sha(checkpoint),by_skill=stats))
    return dict(schema='local-holdout-slot-sensitivity-v1',data_sha256=sha(raw),heldout_seed=heldout,
        models=result,scope='Diagnostic only: oracle-any is not an implementable policy or win rate. Same board, permuted hand and matching action labels. No labels added to training.',quality_accepted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--weights',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=audit(a.data,a.weights);Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
