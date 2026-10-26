"""Small supervised warm start from successful TRAIN-only local actions.

Local heldout groups measure this narrow exercise only. Full-game validation
and PPO remain required; no checkpoint is promoted by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sim.continuous_push import CONTRACT
from sim.train_attack_ppo import initialize_attack_policy


def successful_set_loss(logits,legal,success):
    if logits.shape!=legal.shape or legal.shape!=success.shape or logits.ndim!=2:
        raise ValueError('Action tensor shapes differ')
    if legal.dtype!=torch.bool or success.dtype!=torch.bool or not success.any(-1).all()or (success&~legal).any():
        raise ValueError('Every example needs successful legal actions')
    if not torch.isfinite(logits[legal]).all():raise ValueError('Nonfinite legal logits')
    return (torch.logsumexp(logits.masked_fill(~legal,-torch.inf),-1)-
        torch.logsumexp(logits.masked_fill(~success,-torch.inf),-1)).mean()


def train(data,initial,output,device='cuda:0',updates=200,kl_weight=.1,seed=112):
    if updates<1 or not 0<=kl_weight<=10:raise ValueError('Invalid supervised training controls')
    if device.startswith('cuda')and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    if device=='cpu'and updates>5:raise ValueError('CPU limited to five preflight updates')
    torch.set_num_threads(1);torch.manual_seed(seed)
    data=Path(data);manifest=json.loads((data/'manifest.json').read_text());raw=data/'examples.npz'
    digest=hashlib.sha256(raw.read_bytes()).hexdigest()
    if manifest['split']!='train'or digest!=manifest['data_sha256']:raise ValueError('Pinned TRAIN examples required')
    arrays=np.load(raw,allow_pickle=False)
    planes=torch.as_tensor(arrays['planes'],device=device,dtype=torch.float32)
    scalars=torch.as_tensor(arrays['scalars'],device=device,dtype=torch.float32)
    masks=torch.as_tensor(arrays['masks'],device=device,dtype=torch.bool)
    success=torch.as_tensor(arrays['success_masks'],device=device,dtype=torch.bool)
    if planes.shape[1:]!=(20,32,18)or scalars.shape!=(len(planes),309):raise ValueError('Public observation contract differs')
    groups={r['array_index']:r['seed']for r in manifest['records']if r['accepted']}
    if set(groups)!=set(range(len(planes)))or len(set(groups.values()))<3:raise ValueError('At least three complete seed groups required')
    heldout_seed=max(groups.values())
    train_ids=torch.tensor([i for i in groups if groups[i]!=heldout_seed],device=device)
    heldout_ids=torch.tensor([i for i in groups if groups[i]==heldout_seed],device=device)
    blob=torch.load(initial,map_location='cpu',weights_only=False)
    net,architecture,_=initialize_attack_policy(blob,device=device,num_scalars=309,observation_contract=CONTRACT)
    net.eval()
    with torch.no_grad():reference_logits,reference_value=net(planes,scalars)
    reference_logp=reference_logits.masked_fill(~masks,-torch.inf).log_softmax(-1).masked_fill(~masks,0)
    reference_probs=reference_logits.masked_fill(~masks,-torch.inf).softmax(-1)
    def measure(ids):
        with torch.no_grad():
            logits,_=net(planes[ids],scalars[ids]);probs=logits.masked_fill(~masks[ids],-torch.inf).softmax(-1)
            choice=probs.argmax(-1)
            log_mass=torch.logsumexp(logits.masked_fill(~success[ids],-torch.inf),-1)-torch.logsumexp(logits.masked_fill(~masks[ids],-torch.inf),-1)
            return dict(examples=len(ids),successful_greedy_fraction=float(success[ids].gather(1,choice[:,None]).float().mean()),
                mean_successful_probability=float((probs*success[ids]).sum(-1).mean()),mean_log_successful_probability=float(log_mass.mean()))
    baseline=dict(train=measure(train_ids),local_heldout=measure(heldout_ids))
    optimizer=torch.optim.Adam(net.parameters(),lr=5e-5);history=[]
    for update in range(1,updates+1):
        ids=train_ids[torch.randperm(len(train_ids),device=device)[:32]]
        logits,value=net(planes[ids],scalars[ids]);imitate=successful_set_loss(logits,masks[ids],success[ids])
        logp=logits.masked_fill(~masks[ids],-torch.inf).log_softmax(-1).masked_fill(~masks[ids],0)
        kl=(reference_probs[ids]*(reference_logp[ids]-logp)).sum(-1).mean()
        value_anchor=(value-reference_value[ids]).square().mean()
        loss=imitate+kl_weight*kl+.1*value_anchor
        if not torch.isfinite(loss):raise ValueError('Nonfinite warmstart objective')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),.5);optimizer.step()
        history.append(dict(update=update,loss=float(loss.detach()),imitation=float(imitate.detach()),kl=float(kl.detach())))
    if not all(torch.isfinite(t).all()for t in net.state_dict().values()):raise ValueError('Nonfinite warmstart weights')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='air-local-position-warmstart-v1',supervised_updates=updates,environment_steps=0,
        actual_cuda=torch.cuda.get_device_name(device)if device.startswith('cuda')else None,heldout_seed=heldout_seed,
        train_seeds=sorted(set(groups.values())-{heldout_seed}),baseline=baseline,
        final=dict(train=measure(train_ids),local_heldout=measure(heldout_ids)),data_sha256=digest,
        initial_sha256=hashlib.sha256(Path(initial).read_bytes()).hexdigest(),kl_weight=kl_weight,
        scope='Local action labels, not full-game validation/test; checkpoint for PPO initialization only',quality_accepted=False)
    config=dict(blob.get('config',{}));config.update(architecture=architecture,observation_contract=CONTRACT,num_scalars=309,num_planes=20,
        supervised_warmstart=result)
    torch.save(dict(state_dict=net.state_dict(),config=config),output/'policy.pt')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');(output/'updates.json').write_text(json.dumps(history,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--init',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--updates',type=int,default=200);p.add_argument('--kl-weight',type=float,default=.1);p.add_argument('--seed',type=int,default=112)
    a=p.parse_args();print(json.dumps(train(a.data,a.init,a.output,a.device,a.updates,a.kl_weight,a.seed)))
