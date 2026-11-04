"""TRAIN next-rollout actor/critic gradient probe under frozen training sources.

Each minibatch starts at the same saved weights/Adam state. These are CPU
counterfactual updates, not a GPU replay, new candidate, or game-quality test.
"""
import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def sha(path):
    with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()


def vectors(loss,parameters):
    values=torch.autograd.grad(loss,[p for _,p in parameters],retain_graph=True,allow_unused=True)
    return [torch.zeros_like(p)if g is None else g.detach()for (_,p),g in zip(parameters,values)]


def gradient_summary(parameters,parts):
    result={}
    for group in('all','shared','actor_heads'):
        ids=[i for i,(name,_)in enumerate(parameters)if group=='all' or
             (name.startswith(('conv.','scalar.','trunk.'))if group=='shared'else
              not name.startswith(('conv.','scalar.','trunk.','value.')))]
        flat={name:torch.cat([v[i].flatten()for i in ids])for name,v in parts.items()}
        actor=flat['policy']+flat['entropy']+flat['balance'];value=flat['value'];total=actor+value
        cosine=float(torch.dot(actor,value)/(actor.norm()*value.norm()))if actor.norm()*value.norm()>1e-12 else None
        result[group]=dict(component_norms={k:float(v.norm())for k,v in flat.items()},
            actor_objective_norm=float(actor.norm()),total_norm=float(total.norm()),
            actor_value_cosine=cosine,
            value_to_actor_norm_ratio=float(value.norm()/actor.norm())if actor.norm()>1e-12 else None,
            total_direction_actor_dot=float(torch.dot(total,actor)))
    return result


def worker(weights,expected,output,probe_seed):
    from sim import train_attack_ppo as train
    from sim.card_balance import card_balance_loss
    torch.set_num_threads(1);weights=Path(weights);output=Path(output)
    if sha(weights)!=expected:raise ValueError('Checkpoint changed')
    blob=torch.load(weights,map_location='cpu',weights_only=False);c=blob['config']
    fingerprint=train.code_fingerprint(**{k:c[k]for k in inspect.signature(train.code_fingerprint).parameters if k in c})
    if fingerprint!=c['source_code_sha256']:raise ValueError('Run under the exact frozen training source')
    if c.get('policy_reduction','all')!='all' or c.get('advantage_normalization','all')!='all':
        raise ValueError('This audit targets the legacy all-step objective')
    if c.get('supervised_replay_weight',0)or c.get('spatial_entropy_weight',1)!=1:
        raise ValueError('Unsupported extra loss terms')
    if c['value_coef']!=.5 or blob['step']<=c['value_warmup_steps']:
        raise ValueError('Expected post-warmup baseline critic coefficient .5')
    output.mkdir(parents=True,exist_ok=False)
    net,_,_=train.initialize_attack_policy(blob,device='cpu',num_planes=c['num_planes'],
        num_scalars=c['num_scalars'],observation_contract=c['observation_contract'])
    net.train();parameters=list(net.named_parameters())
    initial={k:v.clone()for k,v in net.state_dict().items()}
    options={k:c[k]for k in('curriculum','continuous_profile','validation_selection','continuous_focus',
        'geometry','coordinate_mode','max_ms','action_delay_ms','deadline_scale','visible_fraction','initial_hand_mode',
        'tower_support_weight','spell_waste_weight','crown_only_weight','dormant_king_weight','tactical_start_fraction')}
    defense,attack=train.reward_objects(c['reward_profile']);options.update(defense_rewards=defense,attack_rewards=attack)
    envs=[train.make_training_env(task_mode=c['task_mode'],split='train',seed=c['seed']+i,**options)for i in range(c['envs'])]
    sampler=train.training_sampler(envs[0],seed=c['seed'],initial_hand_mode=c['initial_hand_mode'],task_mode=c['task_mode'])
    for env in envs:env.reset(**sampler.next())
    sampler.load_state_dict(blob['sampler'])
    if sampler.catalog_sha256!=c['catalog_sha256']:raise ValueError('TRAIN catalog differs')
    for env,state in zip(envs,blob['environment_states']):env.restore(state)
    observations=copy.deepcopy(blob['observations']);infos=copy.deepcopy(blob['infos'])
    for env,obs,info in zip(envs,observations,infos):
        actual=env._observe()
        if any(not np.array_equal(actual[k],obs[k])for k in('planes','scalars'))or not np.array_equal(env._mask(),info['action_mask']):
            raise ValueError('Restored TRAIN observation/mask differs')
    # CPU action RNG is deliberately new; CUDA generator sequence is not claimed.
    torch.manual_seed(probe_seed)
    rows=[];completed=[]
    for _ in range(c['rollout']):
        planes,scalars=train.tensor_observations(observations,'cpu');masks=train.action_masks(infos,'cpu')
        with torch.no_grad():
            logits,value=net(planes,scalars);logits=train.tempered_logits(logits,c['policy_temperature'])
            dist=train.training_distribution(logits,masks,c['card_exploration_epsilon'])
            actions=dist.sample();old_log=dist.log_prob(actions)
        obs,rewards,terminals,truncations,new_infos=map(list,zip(*[
            env.step(int(a))for env,a in zip(envs,actions)]))
        for info in new_infos:train.validate_transition(info)
        with torch.no_grad():_,next_value=net(*train.tensor_observations(obs,'cpu'))
        rows.append((planes,scalars,masks,actions,old_log,value.reshape(-1),torch.tensor(rewards),
                     next_value.reshape(-1),torch.tensor(terminals),torch.tensor(truncations)))
        observations,infos=obs,new_infos
        for i,(terminal,truncated)in enumerate(zip(terminals,truncations)):
            if terminal or truncated:
                completed.append(new_infos[i]['metrics'])
                observations[i],infos[i]=envs[i].reset(**sampler.next())
    planes,scalars,masks,actions,old_log,values,rewards,next_values,terminals,truncations=[torch.stack(x)for x in zip(*rows)]
    advantages,returns=train.generalized_advantage(rewards,values,next_values,terminals,truncations,c['gamma'],c['gae_lambda'])
    advantages=train.normalize_advantages(advantages).flatten()
    planes,scalars,masks=[x.flatten(0,1)for x in(planes,scalars,masks)]
    actions,old_log,returns=[x.flatten()for x in(actions,old_log,returns)]
    def losses(indices):
        logits,value=net(planes[indices],scalars[indices]);logits=train.tempered_logits(logits,c['policy_temperature'])
        dist=train.training_distribution(logits,masks[indices],c['card_exploration_epsilon'])
        ratio=(dist.log_prob(actions[indices])-old_log[indices]).exp()
        policy=-torch.minimum(ratio*advantages[indices],ratio.clamp(1-c['clip'],1+c['clip'])*advantages[indices]).mean()
        return dict(policy=policy,value=c['value_coef']*(value.reshape(-1)-returns[indices]).square().mean(),
            entropy=-c['entropy_coef']*dist.entropy().mean(),
            balance=c['card_balance_weight']*card_balance_loss(logits,scalars[indices],masks[indices],c['card_balance_target']))
    batches=[]
    for batch,indices in enumerate(torch.randperm(len(actions)).split(c['minibatch_size'])):
        net.load_state_dict(initial);terms=losses(indices);parts={k:vectors(v,parameters)for k,v in terms.items()}
        total=vectors(sum(terms.values()),parameters)
        if any(not torch.allclose(sum(parts[k][i]for k in parts),g,atol=2e-5,rtol=2e-5)for i,g in enumerate(total)):
            raise ValueError('Loss gradients do not add to original objective')
        before={k:float(v.detach())for k,v in terms.items()};virtual=[]
        for mode,value_coef in(('full',.5),('value_01',.1),('value_005',.05),('actor_only',0.)):
            net.load_state_dict(initial)
            optimizer=torch.optim.Adam(net.parameters(),lr=c['lr']);optimizer.load_state_dict(copy.deepcopy(blob['optimiser']))
            optimizer.zero_grad(set_to_none=True)
            for i,(name,p)in enumerate(parameters):
                p.grad=(total[i].clone()if mode=='full'else None if value_coef==0 and name.startswith('value.')else
                        (sum(parts[k][i]for k in('policy','entropy','balance'))+value_coef/c['value_coef']*parts['value'][i]).clone())
            raw_norm=torch.nn.utils.clip_grad_norm_(net.parameters(),c['max_grad_norm'],error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():after={k:float(v)for k,v in losses(indices).items()}
            deltas={k:after[k]-before[k]for k in before}
            virtual.append(dict(mode=mode,value_coef=value_coef,raw_norm=float(raw_norm),
                clip_scale=min(1.,c['max_grad_norm']/(float(raw_norm)+1e-6)),loss_change=deltas,
                actor_objective_change=sum(deltas[k]for k in('policy','entropy','balance')),
                shared_parameter_delta_norm=float(torch.cat([(p.detach()-initial[name]).flatten()
                    for name,p in parameters if name.startswith(('conv.','scalar.','trunk.'))]).norm())))
        row=dict(batch=batch,samples=len(indices),choice_samples=int((masks[indices].sum(-1)>1).sum()),
            loss_before=before,gradients=gradient_summary(parameters,parts),virtual_steps=virtual)
        batches.append(row)
        print(json.dumps(dict(batch=batch,choice_samples=row['choice_samples'],
            shared_value_to_actor=row['gradients']['shared']['value_to_actor_norm_ratio'],
            actor_changes=[r['actor_objective_change']for r in virtual])),flush=True)
    net.load_state_dict(initial)
    if sha(weights)!=expected:raise ValueError('Source weights changed during audit')
    result=dict(schema='frozen-ppo-objective-gradients-v1',checkpoint=str(weights),checkpoint_sha256=expected,
        checkpoint_update=blob['update'],training_seed=c['seed'],probe_seed=probe_seed,
        frozen_source_sha256=fingerprint,diagnostic_sha256=sha(__file__),environment_steps=len(actions),
        virtual_optimizer_steps=4*len(batches),persistent_optimizer_updates=0,models_saved=0,
        completed_train_episodes=completed,batches=batches,quality_accepted=False,
        scope='One CPU-sampled next TRAIN rollout; independent first-epoch minibatches at saved weights/Adam moments. Not GPU exact replay, sequential PPO epochs, or candidate win rate.')
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def audit(collected,project,output):
    collected=Path(collected).absolute();project=Path(project).absolute();output=Path(output).absolute()
    receipt=json.loads((collected/'collection_receipt.json').read_text())
    if not receipt.get('complete')or receipt['remote_status']!='COMPLETE':raise ValueError('Verified terminal artifacts required')
    cases=[]
    for seed in(462,463):
        summary=collected/f'pilot_control03_s{seed}_summary.json'
        if sha(summary)!=receipt['files'][summary.name]['sha256']:raise ValueError('Summary receipt differs')
        selected=json.loads(summary.read_text())
        for kind,name in(('selected',selected['checkpoint']),('latest',f'pilot_control03_s{seed}/update_000400.pt')):
            if sha(collected/name)!=receipt['files'][name]['sha256']:raise ValueError('Checkpoint receipt differs')
            cases.append(dict(seed=seed,kind=kind,checkpoint=str(collected/name),sha256=receipt['files'][name]['sha256']))
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(cases=cases,frozen_project=str(project),receipt_sha256=sha(collected/'collection_receipt.json'),
        probe_seed=2026110401,driver_sha256=sha(__file__),no_test_data=True,live_games=False)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (output/'driver.py').write_bytes(Path(__file__).read_bytes())
    results=[]
    for case in cases:
        target=output/f"s{case['seed']}_{case['kind']}"
        with target.with_suffix('.log').open('x')as log:
            subprocess.run([sys.executable,str(Path(__file__).absolute()),'--weights',case['checkpoint'],
                '--expected-sha',case['sha256'],'--probe-seed',str(protocol['probe_seed']),'--output',str(target)],
                cwd=project,env=dict(os.environ,PYTHONPATH=str(project)),stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((target/'result.json').read_text());results.append(dict(kind=case['kind'],**result))
        print(json.dumps(dict(seed=case['seed'],kind=case['kind'],batches=len(result['batches']),complete=True)),flush=True)
    if sha(__file__)!=protocol['driver_sha256']:raise ValueError('Diagnostic source changed')
    result=dict(protocol=protocol,complete=True,models=results,quality_accepted=False)
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collected');p.add_argument('--project')
    p.add_argument('--weights');p.add_argument('--expected-sha');p.add_argument('--probe-seed',type=int,default=2026110401)
    p.add_argument('--output',required=True);a=p.parse_args()
    if a.weights:worker(a.weights,a.expected_sha,a.output,a.probe_seed)
    else:audit(a.collected,a.project,a.output)
