from scripts.attack.build_bank_counterfactuals import opening,rollout,score


def test_paired_rollout_keeps_original_state_and_uses_paid_delayed_command():
    env=opening(352,'left',3000,15)
    start=env.match.elapsed_ms
    mana=env.match.players[1].elixir
    result=rollout(env,0,(3,20))
    assert env.match.elapsed_ms==start and env.match.players[1].elixir==mana
    assert env.actions==[] and env.pending==[]
    assert len(result['commands'])==1 and result['spent_elixir']==4
    assert result['commands'][0]['at_ms']>=start+1350
    assert result['end_ms']>=start+45000


def test_wait_is_not_labelled_correct_when_immediate_defense_avoids_damage():
    from scripts.attack.build_bank_counterfactuals import label
    env=opening(352,'left',4000,15)
    for _ in range(1750//50):env.match.step(50)
    env.match.players[1].elixir=4000;env._previous=env._metrics()
    accepted,success,rows=label(env,'left')
    assert accepted and not success[0]
    assert min(map(score,rows))<score(next(r for r in rows if r['first_action']==0))
    assert not (success&~env._mask()).any()


def test_frozen_teacher_has_both_decisions_in_each_group_and_balanced_training_sampler():
    from scripts.kaggle.prepare_bank_teacher_pilot import validate_data
    from scripts.attack.sequence_sampling import sequence_probabilities
    manifest,groups=validate_data('data/bank_counterfactual_20261029_v4')
    assert all(r['wait_only'] and r['deploy_only']for r in groups.values())
    heldout=max(manifest['seeds'])
    rows=[r for r in manifest['records']if r['accepted'] and r['seed']!=heldout]
    ids=[r['array_index']for r in rows]
    probabilities=sequence_probabilities(manifest['records'],ids,.5)
    assert abs(sum(p for p,r in zip(probabilities,rows)if r['action']==0)-.5)<1e-9
    assert not any(r['seed']==heldout for r in rows)
    states={}
    for row in manifest['records']:
        if row['accepted']:
            assert row['input_sha256'] not in states or states[row['input_sha256']]==row['seed']
            states[row['input_sha256']]=row['seed']


def test_omitted_legal_cannon_refutes_restricted_wait_label():
    from scripts.attack.build_bank_counterfactuals import label
    env=opening(352,'left',3000,15)
    _,success,rows=label(env,'left')
    action=1+env.match.players[1].hand.index('cannon')*576+18*18+7
    assert success[0] and env._mask()[action]
    cannon=rollout(env,action,(3,20))
    assert cannon['own_damage']==0
    assert score(cannon)<min(map(score,rows))


def test_unrepaired_teacher_cannot_be_prepared_for_more_training(tmp_path):
    import pytest
    from scripts.kaggle.prepare_bank_teacher_pilot import prepare
    with pytest.raises(ValueError,match='Cannon-first'):
        prepare(tmp_path/'new','data/bank_counterfactual_20261029_v4')
    assert not(tmp_path/'new').exists()
