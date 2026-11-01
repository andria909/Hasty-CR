import pytest
import torch

from sim.group_entropy import weighted_entropy,entropy_components
from sim.train_ppo import masked_distribution


def test_control_matches_original_entropy_and_gradients_exactly():
    torch.manual_seed(14)
    logits=torch.randn(3,2306,requires_grad=True);mask=torch.rand(3,2306)>.35;mask[:,0]=True
    dist=masked_distribution(logits,mask)
    old=dist.entropy();new=weighted_entropy(dist,1.)
    assert torch.equal(old,new)
    a=torch.autograd.grad(old.sum(),logits,retain_graph=True)[0]
    b=torch.autograd.grad(new.sum(),logits)[0]
    assert torch.equal(a,b)


def test_group_entropy_separates_cell_uncertainty_without_changing_probabilities():
    logits=torch.zeros(2,2306,requires_grad=True);mask=torch.zeros_like(logits,dtype=torch.bool)
    mask[:,0]=True;mask[0,1:577]=True;mask[1,1]=True
    dist=masked_distribution(logits,mask);original=dist.probs.detach().clone()
    full,group=entropy_components(dist)
    assert full[0]>group[0]
    assert full[1].item()==pytest.approx(group[1].item())
    for weight in (0.,.1,.3,1.):
        objective=weighted_entropy(dist,weight)
        assert torch.allclose(objective,group+weight*(full-group))
        gradient=torch.autograd.grad(objective.sum(),logits,retain_graph=True)[0]
        assert torch.isfinite(gradient).all()
        assert (gradient[~mask]==0).all()
        assert torch.equal(original,dist.probs)


@pytest.mark.parametrize('weight', [-.1,1.1,float('nan'),True])
def test_invalid_entropy_weight_rejected(weight):
    with pytest.raises(ValueError):weighted_entropy(torch.distributions.Categorical(logits=torch.zeros(1,2306)),weight)
