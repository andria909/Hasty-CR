import pytest
import torch

from scripts.attack.audit_ppo_objective_gradients import vectors,gradient_summary


def test_component_gradients_reconstruct_joint_loss_and_critic_has_no_actor_head_gradient():
    shared=torch.tensor([1.,2.],requires_grad=True)
    actor=torch.tensor([2.],requires_grad=True)
    critic=torch.tensor([3.],requires_grad=True)
    parameters=[('trunk.weight',shared),('hold.weight',actor),('value.weight',critic)]
    terms=dict(policy=shared.sum()*actor.sum(),value=(shared.sum()*critic.sum()-1).square(),
               entropy=-.01*(shared.sum()+actor.sum()),balance=.03*(shared.sum()-actor.sum()).square())
    parts={k:vectors(loss,parameters)for k,loss in terms.items()}
    total=vectors(sum(terms.values()),parameters)
    assert all(torch.allclose(sum(parts[k][i]for k in parts),g)for i,g in enumerate(total))
    report=gradient_summary(parameters,parts)
    assert report['actor_heads']['component_norms']['value']==0
    assert report['shared']['value_to_actor_norm_ratio']>1
    assert report['shared']['actor_value_cosine']==pytest.approx(1.)


def test_opposing_shared_gradients_are_detected_even_with_large_critic_head():
    parameters=[('trunk.weight',torch.zeros(2)),('hold.weight',torch.zeros(1)),('value.weight',torch.zeros(1))]
    zero=[torch.zeros_like(p)for _,p in parameters]
    parts=dict(policy=[torch.tensor([1.,0.]),torch.tensor([1.]),torch.tensor([0.])],
        value=[torch.tensor([-2.,0.]),torch.tensor([0.]),torch.tensor([100.])],entropy=zero,balance=zero)
    report=gradient_summary(parameters,parts)
    assert report['shared']['value_to_actor_norm_ratio']==2
    assert report['shared']['actor_value_cosine']==-1
    assert report['shared']['total_direction_actor_dot']<0
