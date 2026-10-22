"""Adversarial event-boundary regressions; no training or live-game input."""
from dataclasses import replace

import pytest

from sim.defense_env import DefenseEnv
from sim.defense_scenarios import Scenario,ScheduledPlay


def make_env():
    scenario=Scenario('boundary/knight','boundary','simple','train','left',
                      (ScheduledPlay('knight'),))
    env=DefenseEnv(scenario=scenario,action_delay_ms=0)
    env.reset(seed=7)
    return env


def remove_bodies(env):
    battle=env.match.battle
    source=next(u for u in battle.entities.values() if not u.is_tower)
    for uid,u in list(battle.entities.items()):
        if not u.is_tower:
            del battle.entities[uid]
    return source


@pytest.mark.parametrize('side',[1,-1])
def test_king_damage_unwinds_later_subphase_and_preserves_metrics(side):
    env=make_env()
    king=env.match.towers[side]['king']
    princess=env.match.towers[1]['right']
    before=princess.hitpoints
    def lethal_first_phase():
        king.take_damage(king.hitpoints)
    def forbidden_later_phase():
        princess.take_damage(111)
    env.match.battle._resolve_spell_impacts=lethal_first_phase
    env.match.battle._resolve_projectiles=forbidden_later_phase
    _,_,terminal,truncated,info=env.step(0)
    assert terminal and not truncated
    assert princess.hitpoints==before
    assert info['metrics']['own_tower_hp_loss']['right']==0
    assert info['terminal_reason']==('own_king_destroyed' if side==1 else 'enemy_king_destroyed')


def test_instant_own_spell_kill_keeps_elixir_cost_and_stops_clock():
    env=make_env()
    king=env.match.towers[-1]['king']
    king.hitpoints=1
    player=env.match.players[1]
    player.hand[0]='zap'
    x=round(king.pos.x/1000)
    y=round(king.pos.y/1000)
    before=env.match.elapsed_ms
    action=1+y*18+x
    _,reward,terminal,truncated,info=env.step(action)
    assert terminal and not truncated
    assert info['terminal_reason']=='enemy_king_destroyed'
    assert env.match.elapsed_ms==before
    assert info['metrics']['elixir_spent']==env._cards['zap'].cost
    assert reward>0


def test_remaining_tower_buff_prevents_normal_completion():
    env=make_env()
    remove_bodies(env)
    tower=env.match.towers[1]['left']
    tower.buff_until_ms=5000
    tower.buff_hit_speed_pct=-100
    env._check_boundaries()
    assert env.phase=='settle'
    assert not env._done, 'An active tower effect has not finished'
    assert env._action_mask().sum()==1
    env.match.battle.now_ms=5000
    env._check_boundaries()
    assert env._done and env.terminal_reason=='cleared'


def test_rolling_spell_with_future_troops_keeps_defense_open():
    env=make_env()
    source=remove_bodies(env)
    spec=replace(env._spells['the_log'],spawn_character='Skeleton',spawn_count=1)
    # Actual engine queue layout: spec, head, side, travelled, hits, captures.
    env.match.battle.rolling_spells.append([spec,source.pos,-1,0,set(),set()])
    env._check_boundaries()
    assert env._has_pending_hostile_spawns()
    assert env.phase=='defense'
    assert not env._done
    assert env._action_mask().sum()>1
