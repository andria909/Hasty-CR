"""Refine the selected joint policy with impact-verified Fireball demonstrations.

No prior benchmark test data enters training or checkpoint selection. This is
supervised fine-tuning with frozen-policy rehearsal, not a new PPO run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from sim.fireball_curriculum import FireballEnv,CONTRACT
from sim.train_attack_ppo import initialize_attack_policy
from sim.train_defense_ppo import tensor_observations
from sim.train_ppo import masked_distribution

KINDS=('tower_finish','building_finish','tower_unit')
PARENT_SHA='9d6ac7d1d41ff00df6a9c74354fa8447793e0be8d69f723b27ff43b23c4c73e8'


def contexts(split,count):
    offset={'train':820000,'validation':920000,'test':1020000}[split]
    for i in range(count):
        yield KINDS[i%3],('left','right')[(i//3)%2],offset+i


def demonstration_data(count):
    observations=[];masks=[];actions=[];rejected=[];coverage={}
    envs={(k,l):FireballEnv(kind=k,lane=l) for k in KINDS for l in ('left','right')}
    for index,(kind,lane,seed) in enumerate(contexts('train',count)):
        env=envs[kind,lane];obs,info=env.reset(seed=seed)
        first=env.reference_action()
        episode=[]
        for step in range(30):
            chosen=first if step==0 else 0
            episode.append((obs,info['action_mask'].copy(),chosen))
            obs,_,done,truncated,info=env.step(chosen)
            if done or truncated:break
        if not info['metrics']['fireball_success']:
            rejected.append(dict(kind=kind,lane=lane,seed=seed))
        else:
            # Retain the decision and one real in-flight WAIT, avoiding a
            # large imbalance of repeated no-op labels from flight duration.
            for obs,mask,chosen in episode[:2]:
                observations.append(obs);masks.append(mask);actions.append(chosen)
            coverage[kind]=coverage.get(kind,0)+1
        if index%128==0:print(json.dumps(dict(phase='demonstrations',completed=index,total=count)),flush=True)
    if any(coverage.get(kind,0)<count//6 for kind in KINDS):
        raise RuntimeError('Insufficient impact-verified coverage')
    return observations,np.asarray(masks),np.asarray(actions),dict(accepted=coverage,rejected=rejected)


def rehearsal_data(teacher,device,count):
    from sim.strategy_env import StrategyEnv
    env=StrategyEnv(split='train',geometry='native_anchors_candidate_v1')
    observations=[];masks=[];probabilities=[]
    cases=env.scenarios
    # Uniform reproducible mix of old train tasks, never old held-out cases.
    indices=list(range(len(cases)));random.Random(42).shuffle(indices)
    with torch.no_grad():
        for episode,index in enumerate(indices):
            obs,info=env.reset(seed=830000+episode,scenario_id=cases[index].scenario_id)
            for step in range(32):
                logits,_=teacher(*tensor_observations([obs],device))
                mask=torch.as_tensor(info['action_mask'][None],device=device)
                dist=masked_distribution(logits,mask)
                observations.append(obs);masks.append(info['action_mask'].copy())
                probabilities.append(dist.probs[0].cpu())
                obs,_,done,truncated,info=env.step(int(dist.probs.argmax()))
                if done or truncated or len(observations)>=count:break
            if len(observations)>=count:break
    return observations,np.asarray(masks),torch.stack(probabilities).to(device)


def evaluate(network,device,split,count):
    rows=[];network.eval()
    envs={(k,l):FireballEnv(kind=k,lane=l,split=split) for k in KINDS for l in ('left','right')}
    with torch.no_grad():
        for kind,lane,seed in contexts(split,count):
            env=envs[kind,lane];obs,info=env.reset(seed=seed)
            for step in range(30):
                logits,_=network(*tensor_observations([obs],device))
                dist=masked_distribution(logits,torch.as_tensor(info['action_mask'][None],device=device))
                obs,_,done,truncated,info=env.step(int(dist.probs.argmax()))
                if done or truncated:break
            rows.append(dict(kind=kind,lane=lane,seed=seed,success=bool(info['metrics']['fireball_success']),
                             reason=info['terminal_reason']))
    return dict(episodes=len(rows),successes=sum(r['success'] for r in rows),
                by_kind={k:dict(successes=sum(r['success'] for r in rows if r['kind']==k),
                                episodes=sum(r['kind']==k for r in rows)) for k in KINDS},rows=rows)


def train(args):
    torch.set_num_threads(2);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if hashlib.sha256(Path(args.init).read_bytes()).hexdigest()!=PARENT_SHA:
        raise ValueError('Wrong parent: require selected joint economy_soft checkpoint')
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        raise RuntimeError('CUDA unavailable; refusing silent CPU training')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    device=torch.device(args.device)
    blob=torch.load(args.init,map_location='cpu',weights_only=False)
    teacher,architecture,_=initialize_attack_policy(blob,device=device,num_scalars=309)
    network,_,_=initialize_attack_policy(blob,device=device,num_scalars=309)
    teacher.eval()
    for p in teacher.parameters():p.requires_grad_(False)
    print(json.dumps(dict(phase='ready',device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                          parent_sha256=PARENT_SHA,contract=CONTRACT,method='impact_verified_BC_with_policy_rehearsal')),flush=True)
    baseline=evaluate(teacher,device,'validation',args.eval_episodes)
    (output/'baseline_validation.json').write_text(json.dumps(baseline,indent=2)+'\n')
    obs,masks,labels,audit=demonstration_data(args.demonstrations)
    (output/'demonstration_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    replay,replay_masks,replay_probs=rehearsal_data(teacher,device,args.rehearsal)
    planes,scalars=tensor_observations(obs,device)
    mask=torch.as_tensor(masks,device=device);labels=torch.as_tensor(labels,device=device)
    rp,rs=tensor_observations(replay,device);rm=torch.as_tensor(replay_masks,device=device)
    optimizer=torch.optim.Adam(network.parameters(),lr=args.lr)
    best=-1;best_epoch=0
    config=dict(vars(args),parent_sha256=PARENT_SHA,method='impact_verified_BC_with_policy_rehearsal',
                observation_contract='joint_attack_defense_timer_task_v1',num_scalars=309,
                architecture=architecture,fireball_contract=CONTRACT,training_phase='weights_only_finetune',
                test_used_for_selection=False)
    with (output/'metrics.jsonl').open('x') as log:
        for epoch in range(1,args.epochs+1):
            network.train();losses=[]
            for batch in torch.randperm(len(labels),device=device).split(args.batch_size):
                logits,_=network(planes[batch],scalars[batch])
                bc=F.cross_entropy(logits.masked_fill(~mask[batch],-1e9),labels[batch])
                idx=torch.randint(len(replay),(len(batch),),device=device)
                prior,_=network(rp[idx],rs[idx])
                logp=F.log_softmax(prior.masked_fill(~rm[idx],-1e9),dim=-1)
                retention=F.kl_div(logp,replay_probs[idx],reduction='batchmean')
                loss=bc+args.retention*retention
                if not torch.isfinite(loss):raise ValueError('Nonfinite refinement loss')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),.5);optimizer.step()
                losses.append(float(loss.detach()))
            row=dict(epoch=epoch,percent=100*epoch/args.epochs,loss=sum(losses)/len(losses))
            if epoch%args.eval_every==0 or epoch==args.epochs:
                result=evaluate(network,device,'validation',args.eval_episodes)
                row['validation']=result
                with torch.no_grad():
                    current,_=network(rp,rs)
                    agreement=float((current.masked_fill(~rm,-1e9).argmax(-1)==replay_probs.argmax(-1)).float().mean())
                row['rehearsal_action_agreement']=agreement
                # Keep all measured candidates. This proxy is not evidence of
                # preserved match win rate; full regression evaluation follows.
                score=result['successes']/result['episodes']
                checkpoint=dict(state_dict=network.state_dict(),config=config,epoch=epoch,
                                parent_sha256=PARENT_SHA,validation=result,rehearsal_action_agreement=agreement)
                torch.save(checkpoint,output/f'epoch_{epoch}.pt')
                if score>best and agreement>=args.min_agreement:
                    best=score;best_epoch=epoch;torch.save(checkpoint,output/'candidate.pt')
            log.write(json.dumps(row)+'\n');log.flush()
            print(json.dumps({k:v for k,v in row.items() if k!='validation'},default=str),flush=True)
    result=dict(status='complete',best_epoch=best_epoch,best_validation=best,
                candidate_available=(output/'candidate.pt').is_file(),baseline_validation=baseline['by_kind'],
                parent_preserved=True,live_checkpoint_changed=False,regression_evaluation_required=True)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0')
    for name,default in [('demonstrations',3072),('rehearsal',1536),('epochs',40),('batch-size',128),('eval-every',5),('eval-episodes',96)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--retention',type=float,default=1.)
    p.add_argument('--min-agreement',type=float,default=.90)
    return p


if __name__=='__main__':train(parser().parse_args())
