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
    if state['conv.0.weight'].shape[1]!=20 or state['scalar.0.weight'].shape[1]!=309:
        raise ValueError('Diagnostic requires20-plane309-scalar weights')
    net,_,_=initialize_attack_policy(blob,device='cpu',num_scalars=309,observation_contract=CONTRACT);net.eval()
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
            fraction=float(correct[ids].float().mean()),mean_log_successful_probability=float(log_mass[ids].mean()))
    return dict(scorer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),data_sha256=digest,
        heldout_seed=heldout,train=summarize([i for i in groups if groups[i]!=heldout]),
        local_heldout=summarize([i for i in groups if groups[i]==heldout]),
        scope='Grouped local TRAIN-fixture action diagnostic; not game wins or acceptance')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--weights',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=dict(schema='air-local-label-checkpoint-diagnostic-v1',models=[score_checkpoint(w,a.data)for w in a.weights],quality_accepted=False)
    Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
