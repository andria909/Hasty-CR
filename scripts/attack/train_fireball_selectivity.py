"""Second, explicitly new refinement phase using TRAIN-only waste evidence.

Starts at the frozen first Fireball candidate. Positive impact demonstrations,
empty full-HP control pairs and original-policy rehearsal are all retained.
The first candidate's running tests cannot influence this declared protocol.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from scripts.attack.train_fireball_refinement import (PARENT_SHA,demonstration_data,
    rehearsal_data,evaluate)
from sim.fireball_decision_pairs import decision_pair,selectivity_loss,CONTRACT
from sim.train_attack_ppo import initialize_attack_policy
from sim.train_defense_ppo import tensor_observations

INIT_SHA='042963aa53cec93b88326dbed62670806828000296ea8e2197b9a38775fa5c04'
METHOD='impact_verified_BC_with_selectivity_and_rehearsal'


def negative_pairs(count,split):
    offset=860000 if split=='train' else 960000
    pairs=[decision_pair(offset+i,('left','right')[i%2],split)for i in range(count)]
    return ([p['negative']for p in pairs],np.array([p['action_mask']for p in pairs]),
            np.array([p['fireball_mask']for p in pairs]),[p['metadata']for p in pairs])


def negative_metric(network,planes,scalars,legal,fireball):
    with torch.no_grad():
        logits,_=network(planes,scalars)
        legal_logits=logits.masked_fill(~legal,-1e9)
        actions=legal_logits.argmax(-1)
        is_fireball=fireball.gather(1,actions[:,None]).float().mean()
        probability=(legal_logits.softmax(-1)*fireball).sum(-1).mean()
    return dict(episodes=len(actions),first_action_fireball_rate=float(is_fireball),mean_fireball_probability=float(probability))


def train(args):
    torch.set_num_threads(2);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    for path,digest in ((args.parent,PARENT_SHA),(args.init,INIT_SHA)):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Pinned policy SHA differs')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU training fallback')
    device='cuda:0';output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    teacher,architecture,_=initialize_attack_policy(torch.load(args.parent,map_location='cpu',weights_only=False),device=device,num_scalars=309)
    network,_,_=initialize_attack_policy(torch.load(args.init,map_location='cpu',weights_only=False),device=device,num_scalars=309)
    teacher.eval().requires_grad_(False)
    config=dict(method=METHOD,parent_sha256=PARENT_SHA,initial_checkpoint_sha256=INIT_SHA,
        architecture=architecture,num_scalars=309,observation_contract='joint_attack_defense_timer_task_v1',
        control_contract=CONTRACT,epochs=40,lr=3e-5,positive_weight=1.,negative_weight=1.,retention_weight=2.,
        demonstrations=3072,negative_pairs=1024,rehearsal=1536,validation_positive=96,validation_negative=96,
        min_rehearsal_agreement=.90,negative_cast_margin=.05,selection_negative_penalty=.25,
        test_used_for_selection=False,protocol_fixed_before_first_candidate_test_results=True)
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(dict(event='CUDA_READY',gpu=torch.cuda.get_device_name(0),method=METHOD,init_sha256=INIT_SHA)),flush=True)
    obs,masks,labels,audit=demonstration_data(config['demonstrations'])
    negative_obs,nm,nf,metadata=negative_pairs(config['negative_pairs'],'train')
    validation,vm,vf,valmeta=negative_pairs(config['validation_negative'],'validation')
    (output/'control_contexts.json').write_text(json.dumps(dict(train=metadata,validation=valmeta),indent=2)+'\n')
    (output/'demonstration_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    replay,replay_masks,replay_probs=rehearsal_data(teacher,device,config['rehearsal'])
    planes,scalars=tensor_observations(obs,device);mask=torch.as_tensor(masks,device=device);labels=torch.as_tensor(labels,device=device)
    np_,ns=tensor_observations(negative_obs,device);nm=torch.as_tensor(nm,device=device);nf=torch.as_tensor(nf,device=device)
    vp,vs=tensor_observations(validation,device);vm=torch.as_tensor(vm,device=device);vf=torch.as_tensor(vf,device=device)
    rp,rs=tensor_observations(replay,device);rm=torch.as_tensor(replay_masks,device=device)
    parent_negative=negative_metric(teacher,vp,vs,vm,vf)
    initial_positive=evaluate(network,device,'validation',96)
    initial_negative=negative_metric(network,vp,vs,vm,vf)
    (output/'baseline_validation.json').write_text(json.dumps(dict(initial_positive=initial_positive,
        initial_negative=initial_negative,parent_negative=parent_negative),indent=2)+'\n')
    optimizer=torch.optim.Adam(network.parameters(),lr=config['lr']);best=-float('inf');best_epoch=None
    with (output/'metrics.jsonl').open('x')as log:
        for epoch in range(1,41):
            network.train();losses=[]
            for batch in torch.randperm(len(labels),device=device).split(128):
                logits,_=network(planes[batch],scalars[batch])
                positive=F.cross_entropy(logits.masked_fill(~mask[batch],-1e9),labels[batch])
                idx=torch.randint(len(negative_obs),(len(batch),),device=device)
                neg,_=network(np_[idx],ns[idx]);penalty=selectivity_loss(neg,nm[idx],nf[idx])
                idx=torch.randint(len(replay),(len(batch),),device=device)
                prior,_=network(rp[idx],rs[idx])
                retention=F.kl_div(F.log_softmax(prior.masked_fill(~rm[idx],-1e9),-1),replay_probs[idx],reduction='batchmean')
                loss=positive+penalty+2*retention
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),.5);optimizer.step()
                losses.append(float(loss.detach()))
            row=dict(epoch=epoch,percent=100*epoch/40,loss=sum(losses)/len(losses))
            if epoch%5==0:
                network.eval();positive=evaluate(network,device,'validation',96)
                negative_validation=negative_metric(network,vp,vs,vm,vf)
                with torch.no_grad():
                    current,_=network(rp,rs)
                    agreement=float((current.masked_fill(~rm,-1e9).argmax(-1)==replay_probs.argmax(-1)).float().mean())
                score=positive['successes']/96-.25*negative_validation['first_action_fireball_rate']
                eligible=(agreement>=.90 and negative_validation['first_action_fireball_rate']<=parent_negative['first_action_fireball_rate']+.05)
                row.update(validation=positive,negative_validation=negative_validation,rehearsal_action_agreement=agreement,selection_score=score,eligible=eligible)
                blob=dict(state_dict=network.state_dict(),config=config,epoch=epoch,parent_sha256=PARENT_SHA,
                    initial_checkpoint_sha256=INIT_SHA,validation=positive,negative_validation=negative_validation,
                    rehearsal_action_agreement=agreement,selection_score=score)
                torch.save(blob,output/f'epoch_{epoch}.pt')
                if eligible and score>best:
                    best=score;best_epoch=epoch;torch.save(blob,output/'candidate.pt')
            log.write(json.dumps(row)+'\n');log.flush()
            print(json.dumps({k:v for k,v in row.items()if k not in ('validation',)},default=str),flush=True)
    (output/'result.json').write_text(json.dumps(dict(status='complete',candidate_available=best_epoch is not None,
        best_epoch=best_epoch,best_score=best if best_epoch else None,quality_accepted=False,
        live_checkpoint_changed=False),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','init','output'):p.add_argument('--'+name,required=True)
    train(p.parse_args())
