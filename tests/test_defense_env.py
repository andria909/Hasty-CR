import pickle

import numpy as np
import pytest

from sim import arena
from sim.defense_env import DefenseEnv
from sim.defense_scenarios import Scenario,ScheduledPlay,build_catalog
from sim.entities import make_unit


def env_for(plays=None,**kwargs):
    scenario=Scenario('test/push','test','simple','train','left',
                      tuple(plays or [ScheduledPlay('giant')]))
    env=DefenseEnv(scenario=scenario,**kwargs)
    env.reset(seed=7)
    return env


def test_family_split_never_leaks_lane_or_setup():
    scenarios,_=build_catalog()
    assert scenarios
    families={}
    for scenario in scenarios:
        families.setdefault(scenario.family_id,set()).add(scenario.split)
    assert all(len(splits)==1 for splits in families.values())
    assert {s.split for s in scenarios}=={'train','validation','test'}


def test_ten_elixir_and_mechanical_mask_allow_actual_cards():
    env=env_for()
    assert env.match.players[1].elixir==10000
    assert env._action_mask().sum()>100
    for slot,card in enumerate(env.match.players[1].hand):
        assert env._action_mask()[1+slot*576:1+(slot+1)*576].any()


def test_king_death_aborts_remaining_events_in_same_tick():
    env=env_for()
    king=env.match.towers[1]['king']
    princess=env.match.towers[1]['left']
    before=princess.hitpoints
    def effects():
        king.take_damage(king.hitpoints)
        princess.take_damage(123)  # must never execute
    env.match.battle._resolve_spell_impacts=effects
    _,_,terminated,truncated,info=env.step(0)
    assert terminated and not truncated
    assert info['terminal_reason']=='own_king_destroyed'
    assert princess.hitpoints==before
    stopped=env.match.elapsed_ms
    with pytest.raises(RuntimeError):env.step(0)
    assert env.match.elapsed_ms==stopped


def test_manual_enemy_king_death_ends_without_waiting_for_living_units():
    env=env_for()
    env.match.towers[-1]['king'].hitpoints=0
    _,_,terminated,truncated,info=env.step(0)
    assert terminated and not truncated
    assert info['terminal_reason']=='enemy_king_destroyed'
    assert env.match.elapsed_ms==0


def test_cutoff_blocks_cards_while_existing_projectiles_finish():
    env=env_for()
    for unit in env.match.battle.entities.values():
        if not unit.is_tower:unit.hitpoints=0
    source=env.match.towers[1]['left']
    target=env.match.towers[-1]['left']
    env.match.battle.in_flight.append([5000,source,target.uid,1])
    env._check_boundaries()
    assert env.phase=='settle' and not env._done
    assert env._action_mask().sum()==1


def test_scheduled_gap_does_not_close_attack():
    env=env_for([ScheduledPlay('giant'),ScheduledPlay('knight',at_ms=5000)])
    for unit in env.match.battle.entities.values():
        if not unit.is_tower:unit.hitpoints=0
    env._check_boundaries()
    assert env.phase=='defense' and env._action_mask().sum()>1


def test_pending_hostile_summons_prevent_premature_cutoff():
    env=env_for()
    source=next(u for u in env.match.battle.entities.values()if not u.is_tower)
    source.hitpoints=0
    spec=env._cards['skeletons'].unit
    env.match.battle.ability_summons.append([5000,source,spec,0,100])
    env._check_boundaries()
    assert env.phase=='defense'
    assert env._action_mask().sum()>1


def test_instant_enemy_spell_yields_one_terminal_transition_and_damage_reward():
    from dataclasses import replace
    env=env_for([ScheduledPlay('zap',position='tower')])
    env._spells['zap']=replace(env._spells['zap'],target_buff='',buff_time_ms=0)
    env.reset(seed=7)
    assert env.terminal_reason=='cleared'
    _,reward,terminal,truncated,info=env.step(0)
    assert terminal and not truncated and reward<0
    assert info['metrics']['own_tower_hp_loss']['left']>0
    with pytest.raises(RuntimeError):env.step(0)


def test_public_zap_stun_is_active_and_must_expire_before_clear():
    env=env_for([ScheduledPlay('zap',position='tower')])
    assert env.match.towers[1]['left'].buff_until_ms==500
    assert env.phase=='settle' and not env._done
    _,reward,terminal,truncated,_=env.step(0)
    assert terminal and not truncated and reward<0
    assert env.match.elapsed_ms==500


def test_snapshot_roundtrip_preserves_pending_action_and_rng():
    env=env_for(action_delay_ms=750)
    action=int(np.flatnonzero(env._action_mask())[1])
    env.step(action)
    assert env._pending is not None
    state=pickle.loads(pickle.dumps(env.snapshot()))
    first=env.step(0)
    env.restore(state)
    second=env.step(0)
    for key in ('planes','scalars'):
        np.testing.assert_array_equal(first[0][key],second[0][key])
    assert first[1:4]==second[1:4]
    assert first[4]['metrics']==second[4]['metrics']


def test_delay_is_not_clipped_to_decision_cadence():
    env=env_for(action_delay_ms=750)
    action=int(np.flatnonzero(env._action_mask())[1])
    env.step(action)
    assert env.match.players[1].elixir_spent==0
    env.step(0)
    assert env.match.players[1].elixir_spent>0
    assert env._actions[0]['at_ms']==750


def test_timeout_is_a_separate_truncation():
    env=env_for(max_ms=550)
    env.step(0)
    _,_,terminated,truncated,info=env.step(0)
    assert not terminated and truncated
    assert info['terminal_reason']=='timeout'
    assert info['metrics']['elapsed_ms']==550


def test_medium_and_complex_catalog_include_three_and_four_card_pushes():
    scenarios, _ = build_catalog()
    for difficulty, sizes in [('medium', {2, 3, 4}), ('complex', {3, 4})]:
        selected = [s for s in scenarios if s.difficulty == difficulty and not s.family_id.startswith('building:')]
        assert {len(s.plays) for s in selected} == sizes
        for family in {s.family_id for s in selected}:
            variants = [s for s in selected if s.family_id == family]
            assert {s.lane for s in variants} == {'left', 'right'}
            if difficulty == 'complex':
                assert {s.elixir_multiplier for s in variants} == {1, 2, 3}
            else:
                assert {s.setup for s in variants} == {'bridge', 'back', 'corner'}
    assert len({s.scenario_id for s in scenarios}) == len(scenarios)


def test_building_spell_guesses_are_fixed_mirrored_and_do_not_track_defenders():
    from sim.defense_scenarios import BUILDING_GUESSES
    from sim.arena import Point
    scenarios, _ = build_catalog()
    for lane in ('left', 'right'):
        scenario=next(s for s in scenarios if s.family_id=='building:fireball:hog_rider' and s.lane==lane)
        env=DefenseEnv(scenario=scenario,split=scenario.split)
        env.reset(seed=42)
        for guess,(x,y) in BUILDING_GUESSES.items():
            event=ScheduledPlay('fireball',position='building_guess:'+guess)
            target=env._target(event)
            assert target==Point(x if lane=='left' else 18000-x,y)
            env.match.towers[1][lane].pos=Point(1000,25000)
            assert env._target(event)==target


def test_bridge_trigger_is_remembered_after_escort_dies():
    env=env_for([ScheduledPlay('knight'),ScheduledPlay('archers',near_bridge_after='knight')])
    unit=next(u for u in env.match.battle.entities.values() if u.side==-1 and not u.is_tower)
    unit.pos=arena.Point(3500,12000)
    env.match.players[-1].elixir=0
    env._process_enemy()
    assert 'knight' in env._bridge_ready
    unit.hitpoints=0
    env.match.players[-1].elixir=10000
    env._process_enemy()
    assert env._script_index==2


def test_building_target_spell_does_not_exempt_attack_from_no_damage_gate():
    scenarios,_=build_catalog()
    building=[s for s in scenarios if s.family_id.startswith('building:')]
    assert building and all(not s.unavoidable_tower_damage for s in building)
    tower_spell=next(s for s in scenarios if s.family_id=='solo:fireball')
    assert tower_spell.unavoidable_tower_damage


def test_projectile_diagnostic_history_does_not_prevent_settling():
    env=env_for()
    for entity in env.match.battle.entities.values():
        if not entity.is_tower:entity.hitpoints=0
    env.match.battle.unmodelled_projectiles.append({'arrival_ms':100,'source_uid':999})
    env._check_boundaries()
    assert env._done and env.terminal_reason=='cleared'
    assert 'unmodelled_projectiles' in env._metrics()['coverage_flags']


def test_fragment_geometry_limitation_is_visible_in_episode_metrics():
    scenario=Scenario('fragment-audit','solo:firecracker','simple','train','left',
                      (ScheduledPlay('firecracker'),))
    env=DefenseEnv(scenario=scenario,split='train')
    _,info=env.reset(seed=42)
    assert any(flag.startswith('approximate_unit_mechanics:firecracker:')
               for flag in info['metrics']['coverage_flags'])


def test_unused_pickup_does_not_count_as_troop_but_pending_shot_still_blocks_end():
    env=env_for()
    spec=env.match.battle.unit_lookup('SantaPresent')
    assert spec.object_role=='pickup_effect_carrier'
    pickup=env.match.battle.add(make_unit(0,spec,-1,arena.Point(4000,10000),0))
    for entity in env.match.battle.entities.values():
        if not entity.is_tower and entity.uid!=pickup.uid:entity.hitpoints=0
    env.match.battle.in_flight.append([1000,pickup,env.match.towers[1]['left'].uid,0])
    env._check_boundaries()
    assert env.phase=='settle' and not env._done
    env.match.battle.in_flight.clear()
    env._check_boundaries()
    assert env._done and env.terminal_reason=='cleared' and pickup.alive


def test_default_timeout_allows_solo_princess_to_finish_real_sustained_damage():
    scenario=Scenario('princess-duration','solo:princess','simple','train','left',
                      (ScheduledPlay('princess',position='back'),))
    env=DefenseEnv(scenario=scenario,split='train')
    env.reset(seed=42)
    for _ in range(480):
        _,_,terminal,truncated,info=env.step(0)
        if terminal or truncated:break
    assert terminal and not truncated
    assert info['terminal_reason']=='own_king_destroyed'
    assert 180000<info['metrics']['elapsed_ms']<240000


def test_collector_scenario_cycles_and_pays_for_a_legal_opening_card(monkeypatch):
    from sim.defense_env import DefenseMatch
    scenario=next(s for s in build_catalog()[0]if s.family_id=='solo:elixir_collector')
    seen=[]
    original=DefenseMatch.play_card
    def record(match,side,card,point):
        if side==-1:seen.append((card,tuple(match.players[side].hand)))
        return original(match,side,card,point)
    monkeypatch.setattr(DefenseMatch,'play_card',record)
    env=DefenseEnv(scenario=scenario,split=scenario.split)
    _,info=env.reset(seed=42)
    assert [card for card,_ in seen]==['skeletons','elixir_collector']
    assert 'elixir_collector' not in seen[0][1]
    assert 'elixir_collector' in seen[1][1]
    assert info['metrics']['enemy_elixir_spent']==7
    assert env.match.players[-1].elixir==3000
