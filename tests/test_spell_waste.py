from copy import deepcopy
from sim.continuous_push import ContinuousPushEnv
from sim.arena import Point
from sim.spell_waste import SpellWasteTracker


def match_with(card):
    env=ContinuousPushEnv()
    for seed in range(100):
        env.reset(seed=seed)
        if card in env.match.players[1].hand:
            env.match.players[1].elixir=10000
            return env.match
    raise AssertionError('No hand')


def test_paid_empty_zap_and_rejected_cast_no_duplicate():
    match=match_with('zap');tracker=SpellWasteTracker(match).attach()
    assert match.play_card(1,'zap',Point(9000,16000))
    assert tracker.metrics()['empty_spell_elixir']==2
    assert tracker.metrics()['empty_spell_counts']=={'zap':1}
    assert not match.play_card(1,'zap',Point(9000,16000))
    assert tracker.metrics()['empty_spell_elixir']==2
    copied=deepcopy(tracker)
    assert copied.metrics()==tracker.metrics()
    assert len(copied.match.battle.spell_cast_log)==1


def test_zap_tower_hit_preserves_physics_and_is_not_empty():
    match=match_with('zap');control=deepcopy(match);tracker=SpellWasteTracker(match).attach()
    at=match.towers[-1]['left'].pos
    assert match.play_card(1,'zap',at) and control.play_card(1,'zap',at)
    assert tracker.metrics()['empty_spell_elixir']==0
    assert match.towers[-1]['left'].hitpoints==control.towers[-1]['left'].hitpoints
    assert match.players[1].elixir==control.players[1].elixir
    assert match.battle.spell_cast_log[0]['hits']


def test_flying_fireball_not_penalized_until_resolved():
    match=match_with('fireball');tracker=SpellWasteTracker(match).attach()
    assert match.play_card(1,'fireball',Point(9000,16000))
    assert tracker.metrics()['empty_spell_elixir']==0
    assert tracker.metrics()['unresolved_spell_counts']=={'fireball':1}
    for _ in range(100):match.step(50)
    assert tracker.metrics()['empty_spell_elixir']==4
    assert tracker.metrics()['empty_spell_counts']=={'fireball':1}


def test_zap_contact_with_shield_and_reset_is_not_empty():
    from sim.entities import make_unit
    match=match_with('zap');tracker=SpellWasteTracker(match).attach()
    enemy=make_unit(0,match.cards['knight'].unit,-1,Point(9000,16000),0)
    enemy.shield_hitpoints=10000
    enemy.ramp_target_uid=123;enemy.ramp_started_ms=500
    match.battle.add(enemy)
    hp=enemy.hitpoints
    assert match.play_card(1,'zap',enemy.pos)
    assert enemy.hitpoints==hp
    assert enemy.shield_hitpoints<10000
    assert enemy.ramp_target_uid is None
    assert tracker.metrics()['empty_spell_elixir']==0


def test_environment_charges_once_and_resumes_exactly():
    from sim.continuous_spell_waste import SpellWasteContinuousPushEnv
    from sim.env import ClashEnv
    import pytest
    control=SpellWasteContinuousPushEnv(action_delay_ms=0,spell_waste_weight=0.)
    penalized=SpellWasteContinuousPushEnv(action_delay_ms=0,spell_waste_weight=.5,
        continuous_profile={'missed_fireball_penalty':9.})
    for seed in range(100):
        control.reset(seed=seed);penalized.reset(seed=seed)
        if 'zap' in control.match.players[1].hand:break
    slot=control.match.players[1].hand.index('zap');action=ClashEnv.encode(slot,8,15)
    assert penalized.profile['missed_fireball_penalty']==0
    snapshot=penalized.snapshot()
    baseline=control.step(action);result=penalized.step(action)
    assert result[4]['empty_spell_step_penalty']==1
    assert result[1]==pytest.approx(baseline[1]-1)
    assert penalized.step(0)[4]['empty_spell_step_penalty']==0
    penalized.restore(snapshot)
    replay=penalized.step(action)
    assert replay[1:4]==result[1:4]
    assert replay[4]['metrics']==result[4]['metrics']
