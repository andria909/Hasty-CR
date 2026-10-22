import numpy as np
import pytest
import torch

from sim.train_defense_grpo import (group_advantages, trajectory_objective,
                                    checked_distribution, rollout)


def test_group_advantages_zero_variance_and_standardization():
    a, degenerate = group_advantages([4,4,4])
    assert degenerate and torch.equal(a,torch.zeros(3))
    a, degenerate = group_advantages([0,1,2])
    assert not degenerate
    assert a.mean().item() == pytest.approx(0)
    assert a.std(unbiased=False).item() == pytest.approx(1)
    with pytest.raises(ValueError):
        group_advantages([float('nan'),1])


def test_exact_kl_legal_support_and_reference_frozen():
    logits = torch.tensor([[0.,1.,100.],[1.,0.,-100.]],requires_grad=True)
    reference = torch.zeros_like(logits,requires_grad=True)
    mask = torch.tensor([[True,True,False],[True,True,False]])
    actions = torch.tensor([1,0])
    old = checked_distribution(logits.detach(),mask).log_prob(actions)
    loss,kl,entropy = trajectory_objective(logits,reference,mask,actions,old,1.)
    assert torch.isfinite(loss) and kl > 0 and entropy > 0
    loss.backward()
    assert reference.grad is None
    assert torch.equal(logits.grad[:,2],torch.zeros(2))
    with pytest.raises(ValueError):
        trajectory_objective(logits,reference,mask,torch.tensor([2,0]),old,1.)
    with pytest.raises(ValueError):
        checked_distribution(logits,torch.zeros_like(mask))


def test_repeating_steps_does_not_increase_trajectory_weight():
    logits = torch.tensor([[0.,1.]],requires_grad=True)
    mask = torch.ones_like(logits,dtype=torch.bool)
    old = checked_distribution(logits.detach(),mask).log_prob(torch.tensor([1]))
    one = trajectory_objective(logits,torch.zeros_like(logits),mask,torch.tensor([1]),old,0.7)[0]
    many = trajectory_objective(logits.repeat(9,1),torch.zeros(9,2),mask.repeat(9,1),
                                torch.ones(9,dtype=torch.long),old.repeat(9),0.7)[0]
    assert one.item() == pytest.approx(many.item())


def test_sampling_rng_is_separate_from_environment_seed():
    class Env:
        def reset(self,seed=None,scenario_id=None):
            self.seed=seed; self.n=0
            return self.obs(),{'action_mask':np.array([True,True])}
        def obs(self):
            return {'planes':np.zeros((1,1,1),np.float32),'scalars':np.zeros(1,np.float32)}
        def step(self,action):
            self.n+=1
            return self.obs(),float(action),self.n==5,False,{'action_mask':np.ones(2,bool),'metrics':{'seed':self.seed}}
    class Network(torch.nn.Module):
        def forward(self,p,s):
            return torch.zeros(len(p),2),torch.zeros(len(p))
    generator = torch.Generator().manual_seed(50)
    env=Env()
    first=rollout(env,Network(),'cpu',generator,123)
    second=rollout(env,Network(),'cpu',generator,123)
    assert first['metrics']==second['metrics']=={'seed':123}
    assert first['actions'] != second['actions']
    repeat=rollout(env,Network(),'cpu',torch.Generator().manual_seed(50),123)
    assert first['actions']==repeat['actions']


def test_invalid_script_fails_closed_before_training():
    from sim.train_defense_grpo import validate_episode_info
    for info in (
        {'terminal_reason':'invalid_script_play_timeout','metrics':{'scenario_id':'s42'}},
        {'metrics':{'terminal_reason':'invalid_script_trigger_timeout','scenario_id':'s42'}},
        {'terminal_reason':'cleared','metrics':{'terminal_reason':'invalid_script_hidden','scenario_id':'s42'}},
    ):
        with pytest.raises(ValueError,match='s42'):
            validate_episode_info(info)
    validate_episode_info({'terminal_reason':'timeout','metrics':{'scenario_id':'s42'}})
    validate_episode_info({'metrics':{'terminal_reason':'cleared'}})


def test_scenario_cycle_keeps_real_hand_until_all_attacks_complete():
    from sim.defense_sampler import ScenarioCycle
    from sim.defense_env import DefenseEnv
    from sim.defense_scenarios import select_scenarios
    scenarios=select_scenarios('train',difficulty='simple')[:3]
    sampler=ScenarioCycle([s.scenario_id for s in scenarios],seed=42)
    env=DefenseEnv(split='train',difficulty='simple')
    hands=[]
    contexts=[]
    for _ in range(len(scenarios)):
        context=sampler.next()
        contexts.append(context)
        env.reset(**context)
        hands.append(tuple(env.match.players[1].hand))
    assert len({c['scenario_id'] for c in contexts})==len(scenarios)
    assert {c['seed'] for c in contexts}=={42}
    assert len(set(hands))==1
    next_context=sampler.next()
    assert next_context['seed']==43
    env.reset(**next_context)
    assert tuple(env.match.players[1].hand)!=hands[0]
    # All members of a GRPO group get exactly the same public initial state.
    obs_a,info_a=env.reset(**next_context)
    obs_b,info_b=env.reset(**next_context)
    for key in obs_a:
        np.testing.assert_array_equal(obs_a[key],obs_b[key])
    np.testing.assert_array_equal(info_a['action_mask'],info_b['action_mask'])


def test_boundary_resume_restores_rng_sampler_and_counters_without_training():
    import copy
    import random
    from sim.defense_sampler import ScenarioCycle
    from sim.train_defense_grpo import restore_training_state,RESUME_KEYS
    network=torch.nn.Linear(2,2)
    reference=copy.deepcopy(network).requires_grad_(False)
    optimizer=torch.optim.Adam(network.parameters(),lr=3e-5)
    sampler=ScenarioCycle(['a','b'],seed=42)
    sampler.next()
    generator=torch.Generator().manual_seed(19)
    config={key:None for key in RESUME_KEYS}
    config['seed']=42
    blob=dict(resume_supported=True,checkpoint_phase='boundary',
        resume_schema='defense-grpo-boundary-v1',config=config,
        state_dict=copy.deepcopy(network.state_dict()),
        reference_state_dict=copy.deepcopy(reference.state_dict()),
        optimizer=copy.deepcopy(optimizer.state_dict()),sampler=sampler.state_dict(),
        torch_rng=torch.get_rng_state(),cuda_rng=[],action_rng=generator.get_state(),
        numpy_rng=np.random.get_state(),python_rng=random.getstate(),updates=3,
        environment_steps=123,degenerate_groups=1,actual_environment_steps={'train':123,'validation':4},
        best_validation=5.)
    expected=(torch.rand(2),torch.rand(2,generator=generator),np.random.rand(2),random.random(),sampler.next())
    with torch.no_grad():
        network.weight.zero_()
    restored=restore_training_state(blob,network,reference,optimizer,sampler,generator,config)
    actual=(torch.rand(2),torch.rand(2,generator=generator),np.random.rand(2),random.random(),sampler.next())
    torch.testing.assert_close(expected[0],actual[0],rtol=0,atol=0)
    torch.testing.assert_close(expected[1],actual[1],rtol=0,atol=0)
    np.testing.assert_array_equal(expected[2],actual[2])
    assert expected[3:]==actual[3:]
    assert restored==dict(completed=3,steps=123,degenerate_count=1,actual_steps={'train':123,'validation':4},best_validation=5.)
    torch.testing.assert_close(network.weight,blob['state_dict']['weight'])
    with pytest.raises(ValueError,match='configuration mismatch'):
        restore_training_state(blob,network,reference,optimizer,sampler,generator,dict(config,seed=99))
    for phase in ('collection','update','validation'):
        with pytest.raises(ValueError,match='unfinished'):
            restore_training_state(dict(blob,checkpoint_phase=phase,resume_supported=False),
                network,reference,optimizer,sampler,generator,config)
