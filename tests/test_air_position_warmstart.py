import pytest
import torch
from scripts.attack.train_air_position_warmstart import successful_set_loss


def test_successful_set_gradient_survives_probability_underflow():
    logits=torch.tensor([[200.,0.,-10.]],requires_grad=True)
    legal=torch.tensor([[True,True,False]]);success=torch.tensor([[False,True,False]])
    assert float(logits.masked_fill(~legal,-torch.inf).softmax(-1)[0,1].detach())==0
    loss=successful_set_loss(logits,legal,success);loss.backward()
    assert float(loss.detach())==pytest.approx(200.)
    assert logits.grad[0,1]<-.99 and logits.grad[0,0]>.99
    assert logits.grad[0,2]==0


def test_success_labels_cannot_include_illegal_actions():
    with pytest.raises(ValueError,match='legal'):
        successful_set_loss(torch.zeros(1,3),torch.tensor([[True,False,False]]),torch.tensor([[False,True,False]]))


def test_any_successful_cell_is_acceptable_not_one_arbitrary_position():
    logits=torch.tensor([[0.,0.,0.]],requires_grad=True)
    legal=torch.tensor([[True,True,True]]);success=torch.tensor([[False,True,True]])
    loss=successful_set_loss(logits,legal,success)
    assert float(loss.detach())==pytest.approx(-__import__('math').log(2/3))
    loss.backward();assert logits.grad[0,1]==logits.grad[0,2]<0
