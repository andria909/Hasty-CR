import math
import numpy as np
import pytest

from sim.continuous_push import ContinuousPushEnv
from sim.kite_geometry import close_kite_cell
from sim.projectile_calibration import projectile_speed


def test_overflow_counts_income_lost_at_cap_without_creating_mana():
    env=ContinuousPushEnv();env.reset(seed=4);m=env.match
    m.players[1].elixir=10000
    m._regen(2800)
    assert m.players[1].elixir==10000
    assert m.players[1].natural_elixir_overflow_milli==1000
    m.players[1].elixir=8500;m._regen(2800)
    assert m.players[1].elixir==9500
    assert m.players[1].natural_elixir_overflow_milli==1000


def test_random_pushes_pay_from_real_hand_and_cycle():
    e=ContinuousPushEnv();o,i=e.reset(seed=7,scenario_id='defense::giant/random')
    assert len(o['scalars'])==309 and list(o['scalars'][-3:-1])==[0,0]
    for _ in range(80):e.step(0)
    assert len(e.enemy_actions)>=3
    for a in e.enemy_actions:
        assert a['elixir_after']>=0
        assert a['elixir_before']-a['elixir_after']==e.cards[a['card']].cost*1000
    assert len(e.match.players[-1].hand)==4
    assert len(e.match.players[-1].queue)==4
    assert not e.done


def test_delayed_command_locks_only_its_slot_and_reserves_mana():
    e=ContinuousPushEnv(action_delay_ms=750,continuous_profile={'action_cooldown_ms':2100,'decision_interval_ms':500})
    e.reset(seed=3)
    e.match.players[1].elixir=10000
    a=int(np.flatnonzero(e._mask()[1:])[0]+1);slot=(a-1)//576
    e.step(a);assert e.pending and not e.actions
    assert not e._mask()[1+slot*576:1+(slot+1)*576].any()
    assert e._mask()[1:].any()
    other=int(np.flatnonzero(e._mask()[1:])[0]+1)
    assert (other-1)//576!=slot
    e.step(other);assert len(e.actions)==1 and e.pending
    assert not e._mask()[1+slot*576:1+(slot+1)*576].any()
    e.step(0);assert len(e.actions)==2
    while e.match.elapsed_ms<3000:e.step(0)
    assert e.match.elapsed_ms>=3000


def test_snapshot_replays_opponent_and_reward_exactly():
    e=ContinuousPushEnv();e.reset(seed=2);state=e.snapshot()
    first=e.step(0);events=list(e.enemy_actions)
    e.restore(state);second=e.step(0)
    assert first[1:4]==second[1:4] and e.enemy_actions==events
    np.testing.assert_array_equal(first[0]['scalars'],second[0]['scalars'])


def test_close_kite_and_measured_projectile_conversion():
    for x in (3,14):
        kx,ky=close_kite_cell(x,18)
        assert math.hypot(kx-x,ky-18)<3
    assert projectile_speed('MusketeerProjectile',1000)==20000
    assert projectile_speed('unknown',1000)==16666
    assert projectile_speed('RocketSpell',350)==7000


@pytest.mark.parametrize('x',[3,14])
def test_knight_acquires_close_bait_in_actual_combat(x):
    from sim.entities import make_unit
    from sim.adapter import grid_to_point
    e=ContinuousPushEnv();e.reset(seed=2);m=e.match
    threat=m.battle.add(make_unit(0,e.cards['knight'].unit,-1,grid_to_point(x,18,1),0))
    kx,ky=close_kite_cell(x,18)
    bait=m.battle.add(make_unit(0,e.cards['knight'].unit,1,grid_to_point(kx,ky,1),0))
    for _ in range(30):m.step()
    assert threat.target_uid==bait.uid


def test_snapshot_is_serializable_for_gpu_checkpoint(tmp_path):
    import torch
    e=ContinuousPushEnv();e.reset(seed=2);e.step(0)
    path=tmp_path/'state.pt';torch.save(e.snapshot(),path)
    restored=torch.load(path,map_location='cpu',weights_only=False)
    e.restore(restored)
    assert e.step(0)[4]['metrics']['elapsed_ms']==500


def test_no_positive_leak_reward_or_infinite_match_configuration():
    with pytest.raises(ValueError):ContinuousPushEnv(continuous_profile={'overflow_penalty':-1})
    with pytest.raises(ValueError):ContinuousPushEnv(max_ms=10**9)
