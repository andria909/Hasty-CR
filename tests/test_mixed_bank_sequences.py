import numpy as np
import pytest
from scripts.attack.sequence_sampling import mixed_skill_probabilities
from scripts.attack.build_mixed_bank_sequences import support_opening
from scripts.attack.search_late_air_defense import rollout
from scripts.attack.build_late_air_sequences import replay_match


def test_mixed_sampling_preserves_skill_mass_and_keeps_choice_distinct():
    rows=[]
    def add(skill,context,action,seed=352):
        rows.append(dict(array_index=len(rows),accepted=True,skill=skill,context=context,action=action,seed=seed))
    for _ in range(6):add('bank','one',0)
    add('bank','one',17)
    add('air','a',50);add('air','b',60)
    add('fireball','one',70);add('restraint','one',None)
    ids=list(range(len(rows)));add('heldout_only','hidden',0,seed=354)
    p=mixed_skill_probabilities(rows,ids,.5)
    assert np.isclose(p.sum(),1)
    for skill in ('bank','air','fireball','restraint'):
        assert np.isclose(sum(v for v,r in zip(p,rows)if r['skill']==skill),.25)
    assert np.isclose(sum(v for v,r in zip(p,rows)if r['action']==0),.125)
    assert np.isclose(p[-1],.25)
    broken=[dict(array_index=0,accepted=True,skill='bank',context='one',action=0,seed=352)]
    with pytest.raises(ValueError,match='always-WAIT'):mixed_skill_probabilities(broken,[0])


def test_paid_air_support_musketeer_plan_reproduces_without_hidden_pending():
    env=support_opening(dict(seed=352,lane='left',bank=6000,hand='musketeer'))
    assert env.match.players[-1].elixir_spent==9
    hp={k:t.hitpoints for k,t in env.match.towers[1].items()}
    plans=[rollout(env.match,hp,env.cards,env.spells,'left',
                   dict(card='musketeer',cell=cell))for cell in ([3,20],[6,22],[3,24])]
    plan=next(p for p in plans if p['valid'] and p['clear_without_damage'])
    states=replay_match(env.match,hp,plan,env.cards,env.spells)
    assert len(states)==1 and states[0]['pending_count']==0
    assert (states[0]['action']-1)//576==3


def test_zero_actor_head_keeps_features_value_and_random_state():
    import torch
    from copy import deepcopy
    from sim.train_attack_ppo import initialize_attack_policy
    from scripts.attack.train_air_position_warmstart import zero_actor_heads
    torch.set_num_threads(1)
    net,_,_=initialize_attack_policy(num_scalars=309)
    old=deepcopy(net.state_dict());planes=torch.rand(2,20,32,18);scalars=torch.rand(2,309)
    with torch.no_grad():_,value=net(planes,scalars)
    rng=torch.get_rng_state();zero_actor_heads(net)
    assert torch.equal(rng,torch.get_rng_state())
    with torch.no_grad():logits,after_value=net(planes,scalars)
    assert torch.count_nonzero(logits)==0
    torch.testing.assert_close(value,after_value,rtol=0,atol=0)
    for key,tensor in net.state_dict().items():
        if key.startswith(('conv.','scalar.','trunk.','value.')):
            torch.testing.assert_close(tensor,old[key],rtol=0,atol=0)


def test_always_wait_policy_cannot_pass_deployment_or_spell_skill(tmp_path):
    import torch,json
    from pathlib import Path
    from sim.train_attack_ppo import initialize_attack_policy
    from scripts.attack.train_air_position_warmstart import zero_actor_heads
    from scripts.attack.evaluate_air_position_labels import score_checkpoint
    net,architecture,_=initialize_attack_policy(num_scalars=309)
    zero_actor_heads(net)
    policy=tmp_path/'wait.pt'
    torch.save(dict(state_dict=net.state_dict(),config=dict(architecture=architecture,
        observation_contract='joint_public_presence_continuous_slots_v2')),policy)
    data=Path('data/mixed_bank_skills_frontier_20261030')
    score=score_checkpoint(policy,data)
    m=json.loads((data/'manifest.json').read_text())
    expected=sum(r['action']is not None and r['action']!=0 and r['seed']==354 for r in m['records'])
    assert score['by_decision_type']['deploy']['local_heldout']['examples']==expected
    assert score['by_decision_type']['deploy']['local_heldout']['successful_choices']==0
    assert score['by_skill']['fireball_group']['local_heldout']['fraction']==0
    assert score['by_skill']['spell_restraint']['local_heldout']['fraction']==1


def test_support_labels_accept_safe_alternatives_but_require_executed_prefix():
    from scripts.attack.build_mixed_bank_sequences import safe_support_actions
    def command(x,t=0,slot=0):
        return dict(cell=[x,20],requested_ms=t,slot=slot)
    def plan(commands,safe=True):
        return dict(commands=commands,valid=True,clear_without_damage=safe)
    first=command(3);second=command(7,250,1)
    chosen=plan([first,second]);mask=np.ones(2321,dtype=bool)
    action=1+576+20*18+7
    alternatives=[chosen,plan([first,command(8,250,1)]),
        plan([command(4),command(9,250,1)]),plan([first,command(10,250,1)],False)]
    success=safe_support_actions(dict(decision_ms=250,action=action),chosen,alternatives,mask)
    assert np.flatnonzero(success).tolist()==[action,action+1]


def test_spell_restraint_traits_cover_both_spells_every_slot_each_fold():
    import json
    from pathlib import Path
    from scripts.attack.build_spell_restraint_coverage import coverage
    root=Path('data/mixed_bank_skills_margin_safe_spells_20261030')
    manifest=json.loads((root/'manifest.json').read_text())
    with np.load(root/'examples.npz',allow_pickle=False)as arrays:
        traits=coverage(manifest,arrays)
        assert all(v['slots']==[0,1,2,3]for fold in traits.values()for v in fold.values())
        assert not np.any(arrays['success_masks']&~arrays['masks'])
        identities={}
        for row in manifest['records']:
            digest=row['input_sha256']
            assert digest not in identities or identities[digest]==row['seed']
            identities[digest]=row['seed']
    assert manifest['bank_skill_safety_margin_ms']==750
    assert manifest['bank_skill_validated_delays_ms']==[1350,2025]
    assert manifest['full_game_training_ready']is False
