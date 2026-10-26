import pytest
import torch
from sim.train_attack_ppo import tempered_logits
from sim.train_ppo import masked_distribution


def test_identity_and_greedy_order_are_preserved():
    scores=torch.tensor([[200.,0.,-100.]])
    assert tempered_logits(scores,1.)is scores
    for t in(10.,30.,50.):assert torch.equal(tempered_logits(scores,t).argmax(-1),scores.argmax(-1))


def test_rollout_and_recomputed_ppo_distribution_match_at_same_temperature():
    scores=torch.tensor([[200.,0.,-10.]],requires_grad=True);legal=torch.tensor([[True,True,False]])
    old=masked_distribution(tempered_logits(scores.detach(),30),legal)
    actions=torch.tensor([1]);logp=old.log_prob(actions)
    current=masked_distribution(tempered_logits(scores,30),legal)
    ratio=(current.log_prob(actions)-logp).exp()
    assert float(ratio.detach())==pytest.approx(1.)
    (-current.log_prob(actions).mean()).backward();assert torch.isfinite(scores.grad).all()
    assert old.probs[0,1]>0 and old.probs[0,2]==0
    wrong=masked_distribution(scores.detach(),legal)
    assert float((wrong.log_prob(actions)-logp).exp())<.001


@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf'),101])
def test_invalid_temperatures_fail_before_training(value):
    with pytest.raises(ValueError):tempered_logits(torch.empty(0),value)


def test_actual_update_records_must_match_declared_temperature(tmp_path):
    import json
    from scripts.attack.verify_continuous_focus import metric_boundaries
    path=tmp_path/'metrics.jsonl'
    row=dict(update=1,step=1024,target_environment_steps=1024,actual_cuda='Tesla T4',percent=100,policy_temperature=1.)
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='temperature'):metric_boundaries(path,1024,30.)
    row['policy_temperature']=30.;path.write_text(json.dumps(row)+'\n')
    assert metric_boundaries(path,1024,30.)==[]
