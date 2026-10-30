"""Small supervised warm start from successful TRAIN-only local actions.

Local heldout groups measure this narrow exercise only. Full-game validation
and PPO remain required; no checkpoint is promoted by this script.
"""
import argparse
from copy import deepcopy
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


def zero_actor_heads(network):
    """Remove the action prior, retaining encoder/trunk/value and RNG state."""
    names=('placement','card_bias','slot_query','coordinate','hold','abilities')
    if any(not hasattr(network,name)for name in names):raise ValueError('Unsupported actor-head architecture')
    with torch.no_grad():
        for name in names:
            module=getattr(network,name)
            if module is not None:
                for parameter in module.parameters():parameter.zero_()


def train(data,initial,output,device='cuda:0',updates=200,kl_weight=.1,seed=112,sequence_wait_fraction=None,slot_augmentation=False,pending_inputs=False,reset_actor_heads=False):
    if slot_augmentation and sequence_wait_fraction is not None and not pending_inputs:
        path=Path(data)/'manifest.json'
        contract=json.loads(path.read_text()) if path.is_file() else {}
        if (contract.get('schema')not in ('bank-safe-sequence-labels-v1','mixed-bank-public-sequences-v1') or
                contract.get('slot_augmentation_contract')!='public309_no_pending_busy_slots_v1' or
                not contract.get('records') or
                any(r.get('pending_count')!=0 or r.get('reserved_elixir')!=0 for r in contract['records']if r['accepted'])):
            raise ValueError('Pending sequence slots require a separate permutation contract')
    if updates<1 or not 0<=kl_weight<=10:raise ValueError('Invalid supervised training controls')
    if device.startswith('cuda')and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    if device=='cpu'and updates>5:raise ValueError('CPU limited to five preflight updates')
    torch.set_num_threads(1);torch.manual_seed(seed)
    data=Path(data);manifest=json.loads((data/'manifest.json').read_text());raw=data/'examples.npz'
    digest=hashlib.sha256(raw.read_bytes()).hexdigest()
    if manifest['split']!='train'or digest!=manifest['data_sha256']:raise ValueError('Pinned TRAIN examples required')
    if manifest.get('schema')in ('bank-safe-sequence-labels-v1','mixed-bank-public-sequences-v1'):
        from scripts.attack.sequence_physics import require_current_physics
        require_current_physics(manifest)
        if manifest.get('command_delay_ms')!=1350:raise ValueError('Bank teacher requires measured command delay')
    contract=CONTRACT;num_planes=20;num_scalars=309
    if pending_inputs:
        from sim.pending_inputs import CONTRACT as pending_contract
        from scripts.attack.sequence_physics import require_current_physics
        if (manifest.get('schema')!='late-air-pending-sequence-labels-v1' or
                manifest.get('observation_contract')!=pending_contract or not manifest.get('pending_inputs')):
            raise ValueError('Explicit pending sequence dataset required')
        require_current_physics(manifest)
        for name in ('sim/pending_inputs.py','sim/continuous_pending.py'):
            if manifest.get('pending_source_sha256',{}).get(name)!=hashlib.sha256(Path(name).read_bytes()).hexdigest():
                raise ValueError('Pending observation source differs; regenerate data')
        contract=pending_contract;num_planes=24;num_scalars=313
    elif manifest.get('pending_inputs'):
        raise ValueError('Pending dataset requires explicit pending-inputs option')
    arrays=np.load(raw,allow_pickle=False)
    planes=torch.as_tensor(arrays['planes'],device=device,dtype=torch.float32)
    scalars=torch.as_tensor(arrays['scalars'],device=device,dtype=torch.float32)
    masks=torch.as_tensor(arrays['masks'],device=device,dtype=torch.bool)
    success=torch.as_tensor(arrays['success_masks'],device=device,dtype=torch.bool)
    if planes.shape[1:]!=(num_planes,32,18)or scalars.shape!=(len(planes),num_scalars):raise ValueError('Public observation contract differs')
    groups={r['array_index']:r['seed']for r in manifest['records']if r['accepted']}
    if set(groups)!=set(range(len(planes)))or len(set(groups.values()))<3:raise ValueError('At least three complete seed groups required')
    heldout_seed=max(groups.values())
    train_ids=torch.tensor([i for i in groups if groups[i]!=heldout_seed],device=device)
    sampling=None
    if sequence_wait_fraction is not None:
        from scripts.attack.sequence_sampling import sequence_probabilities,mixed_skill_probabilities
        sampler=mixed_skill_probabilities if manifest['schema']=='mixed-bank-public-sequences-v1'else sequence_probabilities
        sampling=torch.as_tensor(sampler(manifest['records'],train_ids.cpu().tolist(),sequence_wait_fraction),device=device,dtype=torch.float64)
    heldout_ids=torch.tensor([i for i in groups if groups[i]==heldout_seed],device=device)
    blob=torch.load(initial,map_location='cpu',weights_only=False)
    net,architecture,_=initialize_attack_policy(blob,device=device,num_scalars=num_scalars,num_planes=num_planes,observation_contract=contract)
    if reset_actor_heads:zero_actor_heads(net)
    net.eval()
    reference_net=deepcopy(net).eval() if slot_augmentation else None
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
    original_measure=measure
    action_by_index={r['array_index']:r.get('action') for r in manifest['records'] if r['accepted']}
    skill_by_index={r['array_index']:r.get('skill') for r in manifest['records'] if r['accepted']}
    def measure(ids):
        result=original_measure(ids)
        if any(action_by_index[i] is not None for i in ids.cpu().tolist()):
            result['by_decision_type']={}
            for kind,wait in [('wait',True),('deploy',False)]:
                subset=torch.tensor([i for i in ids.cpu().tolist() if action_by_index[i] is not None and (action_by_index[i]==0)==wait],device=device,dtype=torch.long)
                result['by_decision_type'][kind]=original_measure(subset) if len(subset) else dict(examples=0)
        skills={skill_by_index[i]for i in ids.cpu().tolist()}
        if None not in skills:
            result['by_skill']={skill:original_measure(torch.tensor([i for i in ids.cpu().tolist()
                if skill_by_index[i]==skill],device=device,dtype=torch.long))for skill in sorted(skills)}
        return result
    def input_identity(p,s,m):return hashlib.sha256(p.tobytes()+s.tobytes()+m.tobytes()).digest()
    heldout_inputs={input_identity(arrays['planes'][i],arrays['scalars'][i],arrays['masks'][i]) for i in heldout_ids.cpu().tolist()} if slot_augmentation else set()
    if heldout_inputs and any(input_identity(arrays['planes'][i],arrays['scalars'][i],arrays['masks'][i])in heldout_inputs for i in train_ids.cpu().tolist()):
        raise ValueError('TRAIN inputs overlap heldout before augmentation')
    excluded_permutations=0
    baseline=dict(train=measure(train_ids),local_heldout=measure(heldout_ids))
    optimizer=torch.optim.Adam(net.parameters(),lr=5e-5);history=[]
    for update in range(1,updates+1):
        ids=train_ids[torch.randperm(len(train_ids),device=device)[:32]] if sampling is None else train_ids[torch.multinomial(sampling,32,replacement=True)]
        batch_planes=planes[ids]
        batch_scalars,batch_masks,batch_success=scalars[ids],masks[ids],success[ids]
        ref_logp,ref_probs,ref_value=reference_logp[ids],reference_probs[ids],reference_value[ids]
        if slot_augmentation:
            from sim.hand_slot_augmentation import permute
            packed=[]
            for i in ids.cpu().tolist():
                order=torch.randperm(4).tolist()
                if pending_inputs:
                    from sim.pending_slot_augmentation import permute_pending
                    obs=dict(planes=arrays['planes'][i],scalars=arrays['scalars'][i])
                    changed,mask=permute_pending(obs,arrays['masks'][i],order)
                    _,label=permute_pending(obs,arrays['success_masks'][i],order)
                    if input_identity(changed['planes'],changed['scalars'],mask) in heldout_inputs:
                        changed,mask,label=obs,arrays['masks'][i],arrays['success_masks'][i]
                        excluded_permutations+=1
                    scalar,plane=changed['scalars'],changed['planes']
                else:
                    scalar,mask=permute(arrays['scalars'][i],arrays['masks'][i],order)
                    _,label=permute(arrays['scalars'][i],arrays['success_masks'][i],order)
                    plane=arrays['planes'][i]
                    if input_identity(plane,scalar,mask) in heldout_inputs:
                        scalar,mask,label=arrays['scalars'][i],arrays['masks'][i],arrays['success_masks'][i]
                        excluded_permutations+=1
                packed.append((plane,scalar,mask,label))
            batch_planes=torch.as_tensor(np.stack([x[0]for x in packed]),device=device,dtype=torch.float32)
            batch_scalars=torch.as_tensor(np.stack([x[1]for x in packed]),device=device,dtype=torch.float32)
            batch_masks=torch.as_tensor(np.stack([x[2]for x in packed]),device=device,dtype=torch.bool)
            batch_success=torch.as_tensor(np.stack([x[3]for x in packed]),device=device,dtype=torch.bool)
            # Anchor to the parent on exactly the transformed observation.
            with torch.no_grad():
                ref_logits,ref_value=reference_net(batch_planes,batch_scalars)
                ref_logp=ref_logits.masked_fill(~batch_masks,-torch.inf).log_softmax(-1).masked_fill(~batch_masks,0)
                ref_probs=ref_logits.masked_fill(~batch_masks,-torch.inf).softmax(-1)
        logits,value=net(batch_planes,batch_scalars);imitate=successful_set_loss(logits,batch_masks,batch_success)
        logp=logits.masked_fill(~batch_masks,-torch.inf).log_softmax(-1).masked_fill(~batch_masks,0)
        kl=(ref_probs*(ref_logp-logp)).sum(-1).mean()
        value_anchor=(value-ref_value).square().mean()
        loss=imitate+kl_weight*kl+.1*value_anchor
        if not torch.isfinite(loss):raise ValueError('Nonfinite warmstart objective')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),.5);optimizer.step()
        history.append(dict(update=update,loss=float(loss.detach()),imitation=float(imitate.detach()),kl=float(kl.detach())))
    if not all(torch.isfinite(t).all()for t in net.state_dict().values()):raise ValueError('Nonfinite warmstart weights')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='air-local-position-warmstart-v1',supervised_updates=updates,environment_steps=0,seed=seed,
        actual_cuda=torch.cuda.get_device_name(device)if device.startswith('cuda')else None,heldout_seed=heldout_seed,
        train_seeds=sorted(set(groups.values())-{heldout_seed}),baseline=baseline,
        final=dict(train=measure(train_ids),local_heldout=measure(heldout_ids)),data_sha256=digest,
        initial_sha256=hashlib.sha256(Path(initial).read_bytes()).hexdigest(),kl_weight=kl_weight,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sequence_wait_fraction=sequence_wait_fraction,slot_augmentation=slot_augmentation,pending_inputs=pending_inputs,
        reset_actor_heads=reset_actor_heads,reference_anchor='post_initialization_policy',
        heldout_overlap_permutations_excluded=excluded_permutations,
        sampling=('uniform_without_replacement' if sampling is None else
                  'equal_skill_context_mass_bounded_wait_with_replacement'if manifest['schema']=='mixed-bank-public-sequences-v1'else
                  'equal_sequence_mass_bounded_wait_with_replacement'),
        scope='Local action labels, not full-game validation/test; checkpoint for PPO initialization only',quality_accepted=False)
    config=dict(blob.get('config',{}));config.update(architecture=architecture,observation_contract=contract,num_scalars=num_scalars,num_planes=num_planes,
        supervised_warmstart=result)
    torch.save(dict(state_dict=net.state_dict(),config=config),output/'policy.pt')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');(output/'updates.json').write_text(json.dumps(history,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--init',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--updates',type=int,default=200);p.add_argument('--kl-weight',type=float,default=.1);p.add_argument('--seed',type=int,default=112)
    p.add_argument('--sequence-wait-fraction',type=float)
    p.add_argument('--slot-augmentation',action='store_true')
    p.add_argument('--pending-inputs',action='store_true')
    p.add_argument('--reset-actor-heads',action='store_true')
    a=p.parse_args();print(json.dumps(train(a.data,a.init,a.output,a.device,a.updates,a.kl_weight,a.seed,a.sequence_wait_fraction,a.slot_augmentation,a.pending_inputs,a.reset_actor_heads)))
