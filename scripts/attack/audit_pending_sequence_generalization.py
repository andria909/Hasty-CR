"""Heldout sequence slot diagnostic; never trains or chooses a live policy."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from sim.pending_slot_augmentation import permute_pending
from sim.pending_inputs import CONTRACT
from sim.train_attack_ppo import initialize_attack_policy


def audit(data,root,output):
    data,root,output=map(Path,(data,root,output));manifest=json.loads((data/'manifest.json').read_text())
    if hashlib.sha256((data/'examples.npz').read_bytes()).hexdigest()!=manifest['data_sha256']:raise ValueError('Data changed')
    arrays=np.load(data/'examples.npz');heldout=max(manifest['seeds'])
    records=[r for r in manifest['records']if r['accepted']and r['seed']==heldout]
    orders=list(itertools.permutations(range(4)));planes=[];scalars=[];legal=[];targets=[]
    for r in records:
        i=r['array_index'];obs=dict(planes=arrays['planes'][i],scalars=arrays['scalars'][i])
        for order in orders:
            changed,mask=permute_pending(obs,arrays['masks'][i],order)
            _,success=permute_pending(obs,arrays['success_masks'][i],order)
            planes.append(changed['planes']);scalars.append(changed['scalars']);legal.append(mask);targets.append(success)
    torch.set_num_threads(1)
    planes=torch.tensor(np.array(planes));scalars=torch.tensor(np.array(scalars));legal=torch.tensor(np.array(legal));targets=torch.tensor(np.array(targets))
    rows=[]
    for profile in ('bc800','replay500'):
        name=f'pilot_{profile}_s252';selected=json.loads((root/(name+'_summary.json')).read_text())
        for phase,weights in [('before',root/(name+'_warmstart/policy.pt')),('after',root/selected['checkpoint'])]:
            blob=torch.load(weights,map_location='cpu',weights_only=False)
            net,_,_=initialize_attack_policy(blob,num_planes=24,num_scalars=313,observation_contract=CONTRACT);net.eval();correct=[]
            with torch.inference_mode():
                for start in range(0,len(planes),32):
                    logits,_=net(planes[start:start+32],scalars[start:start+32])
                    choices=logits.masked_fill(~legal[start:start+32],-torch.inf).argmax(-1)
                    correct.extend(targets[start:start+32].gather(1,choices[:,None]).flatten().tolist())
            matrix=np.array(correct).reshape(len(records),24)
            groups={}
            for kind,ids in [('all',list(range(len(records)))),('wait',[i for i,r in enumerate(records)if r['action']==0]),('deploy',[i for i,r in enumerate(records)if r['action']!=0])]:
                m=matrix[ids];groups[kind]=dict(states=len(ids),original_correct=int(m[:,0].sum()),
                    any_permutation_correct=int(m.any(1).sum()),mean_permutation_accuracy=float(m.mean()))
            rows.append(dict(profile=profile,phase=phase,weight_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),groups=groups))
    result=dict(data_sha256=manifest['data_sha256'],heldout_seed=heldout,rows=rows,quality_accepted=False,
        scope='One seed252 per profile; correlated168heldoutframes, diagnostic permutations only; no full-game quality claim')
    output.write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    for arg in ('data','root','output'):p.add_argument('--'+arg,required=True)
    a=p.parse_args();r=audit(a.data,a.root,a.output);print(json.dumps(r))
