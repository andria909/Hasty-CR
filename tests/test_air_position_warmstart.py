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


def test_pending_sequence_training_updates_new_inputs_and_preserves_split(tmp_path):
    from pathlib import Path
    from scripts.attack.train_air_position_warmstart import train
    from sim.train_attack_ppo import initialize_attack_policy
    from sim.continuous_push import CONTRACT
    net,architecture,_=initialize_attack_policy(num_scalars=309,observation_contract=CONTRACT)
    initial=tmp_path/'initial.pt'
    torch.save(dict(state_dict=net.state_dict(),config=dict(architecture=architecture,observation_contract=CONTRACT)),initial)
    data=Path('data/late_air_pending_motionfix_20261028')
    result=train(data,initial,tmp_path/'trained','cpu',1,0.,112,.25,pending_inputs=True)
    checkpoint=torch.load(tmp_path/'trained/policy.pt',map_location='cpu',weights_only=False)
    assert checkpoint['config']['num_planes']==24 and checkpoint['config']['num_scalars']==313
    assert checkpoint['state_dict']['conv.0.weight'][:,20:].abs().sum()>0
    assert checkpoint['state_dict']['scalar.0.weight'][:,309:].abs().sum()>0
    assert result['heldout_seed'] not in result['train_seeds']
    assert result['pending_inputs'] and result['environment_steps']==0
    with pytest.raises(ValueError,match='explicit pending'):
        train(data,initial,tmp_path/'wrong','cpu',1)


def test_pending_slot_augmentation_requires_matching_contract(tmp_path):
    from scripts.attack.train_air_position_warmstart import train
    with pytest.raises(ValueError,match='Pending slot permutation'):
        train(tmp_path,tmp_path,tmp_path,slot_augmentation=True,pending_inputs=True)
