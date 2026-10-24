import pytest

from sim.fireball_curriculum import FireballEnv,FAMILIES


@pytest.mark.parametrize('kind',['tower_finish','building_finish','tower_unit'])
@pytest.mark.parametrize('lane',['left','right'])
def test_reference_cast_meets_task_with_actual_impacts(kind,lane):
    env=FireballEnv(kind=kind,lane=lane)
    for seed in (42,43,44):
        obs,_=env.reset(seed=seed)
        assert len(obs['scalars'])==309
        assert obs['scalars'][-3:].tolist()==[0.,0.,float(kind=='tower_finish')]
        action=env.reference_action()
        assert action
        for step in range(30):
            _,reward,done,truncated,info=env.step(action if step==0 else 0)
            if done or truncated:break
        assert done and not truncated
        assert info['metrics']['fireball_success']
        assert reward>0


def test_wrong_spell_and_wait_do_not_satisfy_fireball_task():
    env=FireballEnv(kind='tower_finish')
    env.reset(seed=42)
    # A tower depleted by any non-Fireball source is never task success.
    env.match.towers[-1]['left'].hitpoints=0
    for _ in range(30):
        _,reward,done,truncated,info=env.step(0)
        if done or truncated:break
    assert done and not info['metrics']['fireball_success'] and reward<0


def test_split_families_do_not_overlap():
    assert not set(FAMILIES['train'])&set(FAMILIES['validation'])
    assert not set(FAMILIES['train'])&set(FAMILIES['test'])
    assert not set(FAMILIES['validation'])&set(FAMILIES['test'])


@pytest.mark.parametrize('kind',['tower_finish','building_finish','tower_unit'])
def test_fireball_task_feature_matches_actual_full_match_deployment(kind):
    from sim.eval_defense_duel import joint_attack_task
    env=FireballEnv(kind=kind);obs,_=env.reset(seed=42)
    assert float(obs['scalars'][-1])==float(joint_attack_task(env.match,1))
    for uid in env.practice_uids:
        entity=env.match.battle.entities[uid]
        if not entity.is_tower:entity.hitpoints=0
    assert float(env._observe()['scalars'][-1])==float(joint_attack_task(env.match,1))
