import pytest
import torch

from sim.policy_choice import choices,normalized_advantages,reduce_actor
from sim.train_defense_ppo import generalized_advantage,normalize_advantages


def test_all_mode_matches_legacy_values_and_gradients_exactly():
    torch.manual_seed(7)
    terms=torch.randn(64,dtype=torch.float64,requires_grad=True)
    choice=torch.arange(64)%7==0
    old=terms.mean();new=reduce_actor(terms,choice,'all')
    assert torch.equal(old,new)
    assert torch.equal(torch.autograd.grad(old,terms,retain_graph=True)[0],
                       torch.autograd.grad(new,terms)[0])
    assert torch.equal(normalized_advantages(terms.detach(),choice,'all'),normalize_advantages(terms.detach()))


def test_forced_state_rewards_still_reach_earlier_choices_and_value_targets():
    # Reward arrives after the only available action became WAIT; never remove
    # that elapsed step or its reward from GAE/critic to normalize actor losses.
    reward=torch.tensor([[0.],[10.]])
    value=torch.zeros_like(reward)
    terminal=torch.tensor([[False],[True]])
    advantage,returns=generalized_advantage(reward,value,value,terminal,torch.zeros_like(terminal),.9,.8)
    mask=torch.tensor([[[True,True]],[[True,False]]])
    original_returns=returns.clone()
    normalized=normalized_advantages(advantage,choices(mask),'choice')
    assert advantage[0].item()==pytest.approx(7.2)
    assert normalized[0].item()==pytest.approx(7.2)  # singleton retains signal
    assert normalized[1].item()==0
    assert torch.equal(returns,original_returns)
    assert returns[1].item()==10


def test_choice_normalization_ignores_forced_advantage_outliers():
    advantage=torch.tensor([1.,3.,100000.,-999999.])
    selected=torch.tensor([True,True,False,False])
    assert torch.equal(normalized_advantages(advantage,selected,'choice'),torch.tensor([-1.,1.,0.,0.]))


def test_empty_choice_minibatch_has_zero_finite_actor_loss_and_gradient():
    terms=torch.tensor([2.,-4.,6.],requires_grad=True)
    selected=torch.zeros(3,dtype=torch.bool)
    loss=reduce_actor(terms,selected,'choice')
    assert loss.item()==0
    loss.backward()
    assert torch.equal(terms.grad,torch.zeros(3))
    assert torch.equal(normalized_advantages(terms.detach(),selected,'choice'),torch.zeros(3))


def test_choice_mean_uses_actual_choices_and_preserves_their_relative_weight():
    terms=torch.tensor([2.,-4.,6.],requires_grad=True)
    selected=torch.tensor([True,False,True])
    loss=reduce_actor(terms,selected,'choice');loss.backward()
    assert loss.item()==4
    assert torch.equal(terms.grad,torch.tensor([.5,0.,.5]))


@pytest.mark.parametrize('mode',['bad',None,True])
def test_unknown_mode_cannot_silently_select_a_different_objective(mode):
    with pytest.raises(ValueError):reduce_actor(torch.ones(2),torch.ones(2,dtype=torch.bool),mode)


@pytest.mark.parametrize('name',['policy_choice.py','train_defense_ppo.py'])
def test_resume_fingerprint_binds_reduction_and_gae_normalization_sources(monkeypatch,name):
    from pathlib import Path
    from sim.train_attack_ppo import code_fingerprint
    original=Path.read_bytes
    baseline=code_fingerprint()
    monkeypatch.setattr(Path,'read_bytes',lambda path:original(path)+(b'\n# changed'if path.name==name else b''))
    assert code_fingerprint()!=baseline


def test_explicit_all_defaults_keep_old_control_replicates_in_the_same_pool():
    from scripts.attack.select_pilot_pool import economy_identity
    preparation=dict(study_variant='credit_assignment_v1',config=dict(models=[dict(id='parent',sha256='same')]))
    old=dict(id='old',initial_id='parent',gae_lambda=.95,card_balance_weight=.03)
    explicit=dict(old,id='new',policy_reduction='all',advantage_normalization='all')
    assert economy_identity(preparation,old)==economy_identity(preparation,explicit)
    for key in ('policy_reduction','advantage_normalization'):
        changed=dict(explicit);changed[key]='choice'
        assert economy_identity(preparation,old)!=economy_identity(preparation,changed)


def test_real_no_choice_rollout_keeps_critic_learning_without_nan(monkeypatch,tmp_path):
    import json
    import math
    import sim.train_attack_ppo as trainer
    from sim.continuous_push import ContinuousPushEnv
    torch.set_num_threads(1)
    reset=ContinuousPushEnv.reset
    def empty_mana(self,*args,**kwargs):
        reset(self,*args,**kwargs)
        # A valid low-mana state; normal regeneration and all simulation steps
        # continue. In this one-second rollout no card becomes affordable.
        self.match.players[1].elixir=0
        self._previous=self._metrics()
        return self._observe(),self._info()
    monkeypatch.setattr(ContinuousPushEnv,'reset',empty_mana)
    initial={};build=trainer.build_attack_network
    def capture(*args,**kwargs):
        net=build(*args,**kwargs)
        initial.update({k:v.detach().clone()for k,v in net.state_dict().items()if k.startswith('value')})
        return net
    monkeypatch.setattr(trainer,'build_attack_network',capture)
    args=trainer.parser().parse_args(['--output',str(tmp_path/'forced'),'--device','cpu',
        '--task-mode','joint','--curriculum','continuous_v1','--policy-reduction','choice',
        '--advantage-normalization','choice','--updates','1','--envs','1','--rollout','4',
        '--epochs','1','--minibatch-size','4','--eval-every','0','--target-environment-steps','4'])
    root=trainer.train(args)
    row=json.loads((root/'metrics.jsonl').read_text())
    d=row['policy_decisions']
    assert d['choice_steps']==0 and d['forced_steps']==4 and d['no_choice_minibatches']==1
    assert d['optimizer_samples']==4 and d['choice_samples']==0 and d['actor_updates_enabled']
    assert math.isfinite(row['loss']) and row['gradient_norm']>0
    blob=torch.load(root/'update_000001.pt',map_location='cpu',weights_only=False)
    assert any(not torch.equal(initial[k],blob['state_dict'][k])for k in initial)
