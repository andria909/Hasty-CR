from scripts.attack.probe_full_elixir_opening import prepare_state,STUCK_HAND
from sim.env import DECK_26


def test_controlled_idle_probe_has_real_hand_and_reproduces_lost_income():
    env=prepare_state(101100,'reported_hand');player=env.match.players[1]
    assert tuple(player.hand)==STUCK_HAND
    assert sorted(player.hand+player.queue)==sorted(DECK_26)
    assert len(env._observe()['scalars'])==309
    for _ in range(60):env.step(0)
    metrics=env._metrics()
    assert env.match.elapsed_ms==15000 and env._mask()[1:].any()
    assert player.elixir==10000 and metrics['elixir_overflow']>5
    assert not env.enemy_actions and not env.actions and metrics['enemy_fraction_damage']==0


def test_empty_fireball_penalized_once_only_after_real_impact():
    control=prepare_state(101100,'reported_hand');penalty=prepare_state(101100,'reported_hand')
    penalty.profile['missed_fireball_penalty']=.15
    penalty._previous=penalty._metrics()
    # Slot1 Fireball at(15,14), far from every enemy tower, no enemy units.
    action=1+576+14*18+15
    returns=[0.,0.];first_impact=None
    for i in range(20):
        a=action if i==0 else 0
        left=control.step(a);right=penalty.step(a)
        returns[0]+=left[1];returns[1]+=right[1]
        assert (left[0]['scalars']==right[0]['scalars']).all()
        count=right[4]['metrics']['missed_fireballs']
        if not count:assert left[1]==right[1]
        elif first_impact is None:first_impact=i
        else:assert left[1]==right[1]
    assert first_impact is not None and first_impact>0
    assert penalty._metrics()['missed_fireballs']==1
    assert abs((returns[1]-returns[0])+.6)<1e-9


def test_pending_and_zero_damage_hits_are_not_empty_impacts():
    env=prepare_state(101100,'reported_hand')
    log=env.match.battle.spell_cast_log
    log.append(dict(caster_side=1,spell='fireball',resolved=False,hits=[]))
    assert env._metrics()['missed_fireballs']==0
    log[0].update(resolved=True,hits=[{'damage':0}])
    assert env._metrics()['missed_fireballs']==0
