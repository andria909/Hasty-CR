"""Bounded CPU GRPO algorithm checks; no real games or long training."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sim.train_attack_grpo import (group_advantages,trajectory_objective,rollout,
    tensors,resume_config_differences,scale_deadlines,validate_episode_info)
from sim.attack_sampler import AttackScenarioCycle
from sim.attack_scenarios import build_catalog


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=torch.nn.Linear(2,3)
    def forward(self,planes,scalars):
        return self.actor(scalars),scalars[:,0]*0


class TinyAttack:
    def __init__(self):self.contexts=[]
    def reset(self,seed=None,scenario_id=None):
        self.contexts.append((seed,scenario_id));self.count=0
        return self.obs(),self.info()
    def obs(self):return dict(planes=np.zeros((1,1,1),np.float32),scalars=np.array([1.,.5],np.float32))
    def info(self):return dict(action_mask=np.array([True,True,False]),terminal_reason='',metrics={})
    def step(self,action):
        assert action!=2
        self.count+=1
        return self.obs(),float(action),self.count==2,False,self.info()


def test_real_cpu_gradient_group_uses_same_context_and_frozen_reference():
    torch.manual_seed(2)
    net=TinyPolicy();ref=deepcopy(net).requires_grad_(False)
    before=deepcopy(net.state_dict());reference=deepcopy(ref.state_dict())
    env=TinyAttack();rng=torch.Generator().manual_seed(13)
    group=[rollout(env,net,'cpu',rng,42,scenario_id='same-hidden-window')for _ in range(8)]
    assert env.contexts==[(42,'same-hidden-window')]*8
    advantage,degenerate=group_advantages([t['return']for t in group])
    assert not degenerate
    optimizer=torch.optim.Adam(net.parameters(),lr=.01);optimizer.zero_grad()
    for trajectory,adv in zip(group,advantage):
        p,s=tensors(trajectory['observations'],'cpu')
        logits,_=net(p,s)
        with torch.no_grad():rl,_=ref(p,s)
        loss,kl,entropy=trajectory_objective(logits,rl,torch.tensor(np.stack(trajectory['masks'])),
            torch.tensor(trajectory['actions']),torch.tensor(trajectory['old_logp']),adv)
        assert torch.isfinite(loss+kl+entropy)
        (loss/8).backward()
    assert all(torch.isfinite(p.grad).all()for p in net.parameters())
    optimizer.step()
    assert any(not torch.equal(before[k],v)for k,v in net.state_dict().items())
    assert all(torch.equal(reference[k],v)for k,v in ref.state_dict().items())
    assert all(p.grad is None for p in ref.parameters())


def test_degenerate_and_technical_timeout_fail_closed():
    advantages,degenerate=group_advantages([2,2,2])
    assert degenerate and not advantages.any()
    for info in ({'terminal_reason':'technical_timeout'},
                 {'metrics':{'terminal_reason':'invalid_script_trigger_timeout','scenario_id':'bad'}}):
        with pytest.raises(ValueError):validate_episode_info(info)
    validate_episode_info({'terminal_reason':'goal_window_missed'})


def test_resume_contract_rejects_hidden_timer_hand_shape_or_source_change():
    for key in ('initial_hand_mode','deadline_scale','visible_fraction','num_scalars',
                'observation_contract','environment_code_sha256'):
        old={key:'before'};new={key:'after'}
        assert key in resume_config_differences(old,new)


def test_sampler_resume_and_full_mixed_cycle_without_validation():
    scenarios=[s for s in build_catalog()[0]if s.split=='train']
    a=AttackScenarioCycle(scenarios,seed=71)
    first=[a.next()for _ in range(7)]
    state=a.state_dict();expected=[a.next()for _ in range(11)]
    b=AttackScenarioCycle(scenarios,seed=71);b.load_state_dict(state)
    assert [b.next()for _ in range(11)]==expected
    c=AttackScenarioCycle(scenarios,seed=71)
    cycle=[c.next()for _ in scenarios]
    assert {x['scenario_id']for x in cycle}=={s.scenario_id for s in scenarios}
    assert len({x['seed']for x in cycle})==1
    with pytest.raises(ValueError):AttackScenarioCycle([replace(scenarios[0],split='test')])


def test_scaled_deadlines_are_explicit_and_bounded():
    row=build_catalog()[0][0];env=SimpleNamespace(scenarios=[row])
    scale_deadlines(env,2)
    assert env.scenarios[0].deadline_ms_range==tuple(x*2 for x in row.deadline_ms_range)
    assert row.deadline_ms_range!=env.scenarios[0].deadline_ms_range
    for value in (0,float('nan'),1000):
        with pytest.raises(ValueError):scale_deadlines(SimpleNamespace(scenarios=[row]),value)


def test_boundary_checkpoint_restores_policy_reference_optimizer_sampler_rng_and_counters():
    import random
    from sim.train_attack_grpo import restore_training_state
    net=TinyPolicy();ref=deepcopy(net);optimizer=torch.optim.Adam(net.parameters())
    rows=[s for s in build_catalog()[0]if s.split=='train']
    sampler=AttackScenarioCycle(rows,seed=3);sampler.next()
    generator=torch.Generator().manual_seed(19)
    blob=dict(resume_supported=True,checkpoint_phase='boundary',resume_schema='attack-grpo-boundary-v1',
        config={},state_dict=deepcopy(net.state_dict()),reference_state_dict=deepcopy(ref.state_dict()),
        optimizer=optimizer.state_dict(),sampler=sampler.state_dict(),torch_rng=torch.get_rng_state(),
        cuda_rng=[],action_rng=generator.get_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
        updates=7,environment_steps=39,degenerate_groups=2,actual_environment_steps={'train':39,'validation':6},
        best_validation=1.5)
    expected=torch.rand(3,generator=generator)
    expected_context=sampler.next()
    target=TinyPolicy();target_ref=TinyPolicy();target_optimizer=torch.optim.Adam(target.parameters())
    target_sampler=AttackScenarioCycle(rows,seed=3)
    restored=restore_training_state(blob,target,target_ref,target_optimizer,target_sampler,generator,{})
    assert restored['completed']==7 and restored['steps']==39
    assert restored['actual_steps']=={'train':39,'validation':6}
    assert torch.equal(torch.rand(3,generator=generator),expected)
    assert target_sampler.next()==expected_context
    assert all(torch.equal(v,target.state_dict()[k])for k,v in net.state_dict().items())
    for changes in ({'checkpoint_phase':'collection'},{'resume_schema':'defense-grpo-boundary-v1'}):
        with pytest.raises(ValueError):restore_training_state(dict(blob,**changes),target,target_ref,target_optimizer,target_sampler,generator,{})
    partial=dict(blob);partial.pop('reference_state_dict')
    with pytest.raises(ValueError,match='Incomplete'):
        restore_training_state(partial,target,target_ref,target_optimizer,target_sampler,generator,{})


def test_trajectory_weight_is_unchanged_when_identical_steps_repeat():
    logits=torch.tensor([[.2,-.3,9.]],requires_grad=True)
    reference=torch.zeros_like(logits)
    mask=torch.tensor([[True,True,False]])
    action=torch.tensor([0]);old=torch.log_softmax(logits.detach()[:,:2],-1)[:,0]
    first=trajectory_objective(logits,reference,mask,action,old,torch.tensor(1.))[0]
    repeated=trajectory_objective(logits.repeat(7,1),reference.repeat(7,1),mask.repeat(7,1),
        action.repeat(7),old.repeat(7),torch.tensor(1.))[0]
    assert torch.allclose(first,repeated)


def test_joint_validation_balanced_fixed_seeds_no_test_or_training_selection():
    from sim.train_attack_grpo import validation_contexts,validation_summary
    rows=[SimpleNamespace(scenario_id=f'{task}::{i}',split='validation')for task in ('attack','defense')for i in range(3)]
    contexts=validation_contexts(rows,32,'joint')
    assert len(contexts)==64
    assert contexts==validation_contexts(list(reversed(rows)),32,'joint')
    for task in ('attack','defense'):
        subset=[c for c in contexts if c['task']==task]
        assert [c['seed']for c in subset]==list(range(900000,900032))
        assert all(c['scenario_id'].startswith(task+'::')for c in subset)
    summary=validation_summary([{'return':2.,'metrics':{'task':'attack'}},
                                {'return':8.,'metrics':{'task':'defense'}}],'joint')
    assert summary['mean_return']==5.
    assert validation_summary([{'return':2.,'metrics':{'task':'attack'}}],'joint')['mean_return']is None
    with pytest.raises(ValueError):validation_contexts([replace(build_catalog()[0][0],split='test')],1,'joint')
    assert 'task_mode'in resume_config_differences({'task_mode':'attack'},{'task_mode':'joint'})


def test_actual_joint309_observations_shared_policy_real_finite_gradient():
    from sim.strategy_env import StrategyEnv,OBSERVATION_CONTRACT
    from sim.train_attack_ppo import initialize_attack_policy,load_attack_state
    torch.set_num_threads(1)
    env=StrategyEnv(seed=42,split='train',deadline_scale=3.)
    network,_,_=initialize_attack_policy(None,device='cpu',num_scalars=309,
                                         observation_contract=OBSERVATION_CONTRACT)
    reference=deepcopy(network).requires_grad_(False)
    optimizer=torch.optim.Adam(network.parameters(),lr=1e-4)
    losses=[]
    for task in ('attack','defense'):
        scenario=next(s for s in env.scenarios if s.task==task)
        obs,info=env.reset(seed=42,scenario_id=scenario.scenario_id)
        assert obs['scalars'].shape==(309,)
        assert obs['scalars'][-1]==float(task=='attack')
        if task=='defense':assert not obs['scalars'][-3:-1].any()
        legal=int(np.flatnonzero(info['action_mask'][1:])[0]+1)
        samples=[]
        for action in (0,legal):
            observation,info=env.reset(seed=42,scenario_id=scenario.scenario_id)
            _,reward,_,_,result=env.step(action)
            assert result['metrics']['task']==task
            samples.append((observation,info,action,reward))
        advantages,_=group_advantages([s[3]for s in samples])
        # Entropy also gives a real finite gradient if this tick defers both actions.
        for (observation,info,action,reward),adv in zip(samples,advantages):
            p,s=tensors([observation],'cpu');logits,_=network(p,s)
            with torch.no_grad():ref_logits,_=reference(p,s)
            mask=torch.as_tensor(info['action_mask'],dtype=torch.bool)[None]
            old=logits.detach().masked_fill(~mask,-torch.inf).log_softmax(-1)[:,action]
            losses.append(trajectory_objective(logits,ref_logits,mask,torch.tensor([action]),old,adv,
                                                entropy_coef=.01)[0]/4)
    optimizer.zero_grad();sum(losses).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in network.parameters())
    assert all(torch.isfinite(p.grad).all()for p in network.parameters()if p.grad is not None)
    optimizer.step()
    checkpoint=dict(state_dict=network.state_dict(),config={'observation_contract':OBSERVATION_CONTRACT})
    load_attack_state(network,checkpoint,exact=True,observation_contract=OBSERVATION_CONTRACT)
    with pytest.raises(ValueError):load_attack_state(network,checkpoint,exact=True,observation_contract='wrong')
