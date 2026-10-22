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
