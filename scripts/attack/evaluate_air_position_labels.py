"""Read-only local action-label diagnostic before/after PPO.

The examples are TRAIN fixtures with a grouped local holdout. This is neither
full-game validation nor an independent test win rate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sim.continuous_push import CONTRACT
from sim.train_attack_ppo import initialize_attack_policy


def score_checkpoint(checkpoint,data):
    checkpoint=Path(checkpoint);data=Path(data)
    flat=(data/'air_examples.npz').exists()
    if flat and (data/'examples.npz').exists():raise ValueError('Ambiguous fixture files')
    manifest=json.loads((data/('air_manifest.json'if flat else'manifest.json')).read_text())
    raw=data/('air_examples.npz'if flat else'examples.npz')
    digest=hashlib.sha256(raw.read_bytes()).hexdigest()
    if manifest['split']!='train'or digest!=manifest['data_sha256']:raise ValueError('Pinned TRAIN fixtures required')
    arrays=np.load(raw,allow_pickle=False);groups={r['array_index']:r['seed']for r in manifest['records']if r['accepted']}
    if set(groups)!=set(range(len(arrays['planes'])))or len(set(groups.values()))<3:raise ValueError('Incomplete fixture group map')
    heldout=max(groups.values());torch.set_num_threads(1)
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
    state=blob['state_dict']
    planes,scalars,contract=20,309,CONTRACT
    if manifest.get('pending_inputs'):
        from sim.pending_inputs import CONTRACT as pending_contract
        if manifest.get('schema')!='late-air-pending-sequence-labels-v1' or manifest.get('observation_contract')!=pending_contract:
            raise ValueError('Pending diagnostic contract differs')
        planes,scalars,contract=24,313,pending_contract
    source_shape=(state['conv.0.weight'].shape[1],state['scalar.0.weight'].shape[1])
    if source_shape not in ((planes,scalars),(20,309)):
        raise ValueError('Diagnostic weight input dimensions differ')
    if arrays['planes'].shape[1:]!=(planes,32,18) or arrays['scalars'].shape!=(len(groups),scalars):
        raise ValueError('Diagnostic data input dimensions differ')
    net,_,_=initialize_attack_policy(blob,device='cpu',num_planes=planes,num_scalars=scalars,observation_contract=contract);net.eval()
    mask=torch.as_tensor(arrays['masks'],dtype=torch.bool);success=torch.as_tensor(arrays['success_masks'],dtype=torch.bool)
    if (success&~mask).any()or not success.any(-1).all():raise ValueError('Illegal or empty successful action set')
    with torch.inference_mode():
        logits,_=net(torch.as_tensor(arrays['planes']),torch.as_tensor(arrays['scalars']))
        if not torch.isfinite(logits[mask]).all():raise ValueError('Nonfinite legal scores')
        choices=logits.masked_fill(~mask,-torch.inf).argmax(-1)
        correct=success.gather(1,choices[:,None]).flatten()
        log_mass=torch.logsumexp(logits.double().masked_fill(~success,-torch.inf),-1)-torch.logsumexp(logits.double().masked_fill(~mask,-torch.inf),-1)
    def summarize(ids):
        return dict(examples=len(ids),successful_choices=int(correct[ids].sum()),
            fraction=float(correct[ids].float().mean()) if ids else None,mean_log_successful_probability=float(log_mass[ids].mean()) if ids else None)
    skills={r['array_index']:r.get('skill','air_defense')for r in manifest['records']if r['accepted']}
    return dict(scorer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),data_sha256=digest,
        heldout_seed=heldout,train=summarize([i for i in groups if groups[i]!=heldout]),
        local_heldout=summarize([i for i in groups if groups[i]==heldout]),
        by_skill={skill:dict(train=summarize([i for i in groups if groups[i]!=heldout and skills[i]==skill]),
            local_heldout=summarize([i for i in groups if groups[i]==heldout and skills[i]==skill]))
            for skill in sorted(set(skills.values()))},
        by_decision_type=({kind:dict(train=summarize([r['array_index']for r in manifest['records'] if r['accepted'] and r['seed']!=heldout and r.get('action')is not None and (r['action']==0)==wait]),
            local_heldout=summarize([r['array_index']for r in manifest['records'] if r['accepted'] and r['seed']==heldout and r.get('action')is not None and (r['action']==0)==wait]))
            for kind,wait in [('wait',True),('deploy',False)]} if any(r.get('action')is not None for r in manifest['records']if r['accepted']) else None),
        scope='Grouped local TRAIN-fixture action diagnostic; not game wins or acceptance')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--weights',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=dict(schema='air-local-label-checkpoint-diagnostic-v1',models=[score_checkpoint(w,a.data)for w in a.weights],quality_accepted=False)
    Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
