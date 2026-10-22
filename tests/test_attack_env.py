"""Real attack mechanics, tiny finite simulator exercises; no model/training."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from sim import arena
from sim.attack_env import AttackEnv, OBSERVATION_CONTRACT
from sim.env import NUM_SCALARS


def scenario(**kwargs):
    fields=dict(scenario_id='attack-test',family_id='attack-test',split='train',
        goal_kind='spell_finish',lane='left',target_lane='left',initial_elixir=10,
        hand_seed=42,initial_hand=('fireball','zap','hog_rider','knight'),enemy_initial_elixir=10,
        tower_hp=(),initial_units=(),enemy_plays=(),deadline_ms_range=(8000,14000),
        deadline_visibility='hidden',geometry='legacy_v1',opportunity_start_ms=0)
    fields.update(kwargs)
    return NS(**fields)


def action(env,card,x=3,y=7):
    return 1+env.match.players[1].hand.index(card)*576+y*18+x


def run(env,first=0,bound=100):
    for index in range(bound):
        obs,reward,terminal,truncated,info=env.step(first if index==0 else 0)
        if terminal or truncated:return obs,reward,terminal,truncated,info
    raise AssertionError('Finite test bound exceeded')


@pytest.mark.parametrize('card',['fireball','zap'])
def test_actual_spell_kills_target_with_real_crown_damage(card):
    setup=NS(side=-1,lane='left',hp=1)
    env=AttackEnv(scenario=scenario(tower_hp=(setup,)),action_delay_ms=0)
    env.reset(seed=42)
    result=run(env,action(env,card))
    assert result[2] and not result[3]
    assert result[4]['terminal_reason']=='goal_success'
    assert not env.match.towers[-1]['left'].alive
    assert result[4]['metrics']['successful_spell_finish']
    assert any(hit.get('killed') and hit.get('target_uid')==env.match.towers[-1]['left'].uid
               for event in env.match.battle.spell_cast_log for hit in event.get('hits',()))


def test_no_enemy_units_does_not_settle_own_attack_and_wrong_spell_cannot_hog_goal():
    env=AttackEnv(scenario=scenario(goal_kind='hog_tower_damage'),action_delay_ms=0)
    _,initial=env.reset(seed=42)
    assert initial['action_mask'][action(env,'hog_rider',3,16)]
    _,_,terminal,truncated,info=env.step(action(env,'fireball'))
    assert not terminal and not truncated
    while env.match.elapsed_ms<4000:
        _,_,terminal,truncated,info=env.step(0)
    assert env.match.towers[-1]['left'].hitpoints<env.match.towers[-1]['left'].max_hitpoints
    assert not terminal and not truncated
    assert info['metrics']['hog_target_tower_damage']==0
    assert info['action_mask'][action(env,'hog_rider',3,16)]


def test_actual_hog_hit_success_and_other_unit_damage_does_not_count():
    for card,success in [('hog_rider',True),('knight',False)]:
        unit=NS(card=card,side=1,x=3500,y=10000,deployed_ms_ago=1000)
        env=AttackEnv(scenario=scenario(goal_kind='hog_tower_damage',initial_units=(unit,),
                                        deadline_ms_range=(8000,8000)))
        env.reset(seed=42)
        result=run(env)
        assert (result[4]['terminal_reason']=='goal_success')==success
        assert (result[4]['metrics']['hog_target_tower_damage']>0)==success


def test_hidden_deadlines_do_not_leak_through_clock_board_mask_or_reward():
    envs=[AttackEnv(scenario=scenario(deadline_ms_range=interval))for interval in
          [(5000,7000),(12000,14000)]]
    results=[env.reset(seed=99)for env in envs]
    assert envs[0]._deadline_ms!=envs[1]._deadline_ms
    for _ in range(5):
        for key in ('planes','scalars'):
            assert results[0][0][key].tobytes()==results[1][0][key].tobytes()
        assert results[0][1]['action_mask'].tobytes()==results[1][1]['action_mask'].tobytes()
        assert results[0][0]['scalars'][-2:].tolist()==[0.,0.]
        a,b=[env.step(0)for env in envs]
        assert a[1]==b[1]
        results=[(a[0],a[4]),(b[0],b[4])]


def test_visible_flag_and_independent_seed_pair_reproducibility():
    s=scenario(deadline_visibility='mixed')
    envs=[AttackEnv(scenario=s,visible_fraction=.5)for _ in range(2)]
    for seed in range(5):
        a,b=[env.reset(seed=seed)for env in envs]
        assert envs[0]._deadline_ms==envs[1]._deadline_ms
        assert envs[0]._timer_visible==envs[1]._timer_visible
        assert a[0]['scalars'].tobytes()==b[0]['scalars'].tobytes()
        assert len(a[0]['scalars'])==NUM_SCALARS+2
        assert a[1]['metrics']['observation_contract']==OBSERVATION_CONTRACT
    visible=AttackEnv(scenario=scenario(deadline_visibility='visible',deadline_ms_range=(500,500)))
    obs,_=visible.reset(seed=1)
    assert obs['scalars'][-1]==1 and obs['scalars'][-2]>0
    obs,_,terminal,_,_=visible.step(0)
    assert terminal and obs['scalars'][-2:].tolist()==[0.,1.]


def test_deadline_hard_cutoff_does_not_credit_spell_still_inflight():
    env=AttackEnv(scenario=scenario(tower_hp=(NS(side=-1,lane='left',hp=1),),
                                    deadline_ms_range=(500,500)),action_delay_ms=0)
    env.reset(seed=42)
    result=run(env,action(env,'fireball'))
    assert result[2] and not result[3]
    assert result[4]['terminal_reason']=='goal_window_missed'
    assert env.match.towers[-1]['left'].alive
    assert env.match.battle.spell_impacts
    assert result[4]['metrics']['unresolved_spells']==1
    assert not result[4]['metrics']['successful_spell_finish']
    with pytest.raises(RuntimeError):env.step(0)


def test_king_priority_and_technical_timeout_distinct_from_goal_window():
    for side,reason in [(1,'own_king_destroyed'),(-1,'enemy_king_destroyed')]:
        env=AttackEnv(scenario=scenario(tower_hp=(NS(side=side,lane='king',hp=0),)))
        env.reset(seed=42)
        result=env.step(0)
        assert result[2] and not result[3] and result[4]['terminal_reason']==reason
    env=AttackEnv(scenario=scenario(deadline_ms_range=(10000,10000)),max_ms=500)
    env.reset(seed=42)
    result=env.step(0)
    assert result[3] and not result[2]
    assert result[4]['terminal_reason']=='technical_timeout'


def test_auto_goal_public_tie_and_destroyed_princess_rules():
    env=AttackEnv(scenario=scenario(goal_kind='hog_tower_damage',target_lane='auto'))
    env.reset(seed=42)
    towers=env.match.towers[-1]
    assert env._goal_target_uids()=={towers['left'].uid,towers['right'].uid}
    towers['left'].hitpoints-=1
    assert env._goal_target_uids()=={towers['left'].uid}
    towers['left'].hitpoints=0
    assert env._goal_target_uids()=={towers['king'].uid}


def test_snapshot_roundtrip_and_wrong_contract_rejected():
    env=AttackEnv(scenario=scenario())
    env.reset(seed=42)
    snapshot=env.snapshot()
    first=env.step(0)
    env.restore(snapshot)
    second=env.step(0)
    assert first[0]['scalars'].tobytes()==second[0]['scalars'].tobytes()
    assert first[1:4]==second[1:4]
    other=AttackEnv(scenario=scenario(),coordinate_mode='legacy_training')
    with pytest.raises(ValueError):other.restore(snapshot)


def test_catalog_visibility_siblings_have_identical_public_state_and_paired_deadline():
    from sim.attack_scenarios import build_catalog
    scenarios,_=build_catalog()
    groups={}
    for s in scenarios:groups.setdefault(s.paired_case_id,[]).append(s)
    pair=next(values for values in groups.values()if len(values)==2)
    envs=[AttackEnv(scenario=s,split=s.split)for s in pair]
    observations=[env.reset(seed=42)[0]for env in envs]
    assert envs[0]._deadline_ms==envs[1]._deadline_ms
    assert envs[0]._timer_visible!=envs[1]._timer_visible
    assert observations[0]['planes'].tobytes()==observations[1]['planes'].tobytes()
    assert observations[0]['scalars'][:-2].tobytes()==observations[1]['scalars'][:-2].tobytes()
    assert envs[0]._action_mask().tobytes()==envs[1]._action_mask().tobytes()


def test_actual_king_lethal_stops_attack_before_pending_finisher():
    unit=NS(card='knight',side=-1,x=9500,y=26500,deployed_ms_ago=1000)
    env=AttackEnv(scenario=scenario(tower_hp=(NS(side=1,lane='king',hp=1),
                                            NS(side=-1,lane='left',hp=1)),initial_units=(unit,)),
                  action_delay_ms=0)
    env.reset(seed=42)
    result=run(env,action(env,'fireball'))
    assert result[4]['terminal_reason']=='own_king_destroyed'
    assert result[2] and not result[3]
    assert not env.match.towers[1]['king'].alive
    assert env.match.towers[-1]['left'].alive


def test_reactive_building_waits_for_visible_hog_then_uses_declared_legal_position():
    event=NS(at_ms=0,card='cannon',x=9000,y=12500,position_kind='reactive_visible',
             trigger_card='hog_rider',visibility_delay_ms=300)
    env=AttackEnv(scenario=scenario(goal_kind='hog_tower_damage',enemy_plays=(event,)),
                  action_delay_ms=0)
    env.reset(seed=42)
    env.step(0)
    assert env._script_index==0
    env.step(action(env,'hog_rider',3,16))
    assert env._script_index==1
    cannon=next(e for e in env.match.battle.entities.values()if e.name=='cannon' and e.side==-1)
    assert cannon.pos==arena.Point(9000,12500)


def test_actual_fireball_then_unneeded_zap_counts_redundancy_before_resolution():
    env=AttackEnv(scenario=scenario(tower_hp=(NS(side=-1,lane='left',hp=100),)),action_delay_ms=0)
    env.reset(seed=42)
    first=env.step(action(env,'fireball'))
    assert not first[2] and env.match.battle.spell_impacts
    second=env.step(action(env,'zap'))
    assert second[4]['metrics']['redundant_spells']==1
    evidence=second[4]['metrics']['redundant_spell_evidence'][0]
    assert evidence['card']=='zap'
    assert evidence['evidence'][0]['pending_impacts'][0]['spell']=='fireball'
    assert evidence['evidence'][0]['pending_impacts'][0]['damage']>=100
    final=run(env)
    assert final[4]['terminal_reason']=='goal_success'


def test_second_fireball_redundancy_with_actual_already_flying_spell():
    from sim.spells import cast_spell
    env=AttackEnv(scenario=scenario(tower_hp=(NS(side=-1,lane='left',hp=100),)),action_delay_ms=0)
    env.reset(seed=42)
    target=env.match.towers[-1]['left']
    cast_spell(env.match.battle,env._spells['fireball'],target.pos,1,
               origin=env.match.towers[1]['king'].pos)
    result=env.step(action(env,'fireball'))
    assert result[4]['metrics']['redundant_spells']==1
    assert result[4]['metrics']['missed_resolved_spells']==0


def test_pending_lethal_projection_respects_radius_deadline_and_uncertain_rules():
    from sim.spells import cast_spell
    env=AttackEnv(scenario=scenario(tower_hp=(NS(side=-1,lane='left',hp=100),)),action_delay_ms=0)
    env.reset(seed=42)
    target=env.match.towers[-1]['left']
    cast_spell(env.match.battle,env._spells['fireball'],target.pos,1,
               origin=env.match.towers[1]['king'].pos)
    assert env._pending_lethal_evidence('zap',target.pos)
    assert not env._pending_lethal_evidence('zap',arena.Point(14500,7500))
    env._deadline_ms=1
    assert not env._pending_lethal_evidence('zap',target.pos)
    env._deadline_ms=14000
    target.invulnerable_until_ms=100
    assert not env._pending_lethal_evidence('zap',target.pos)
    target.invulnerable_until_ms=0
    env._spell_audit['fireball']={'limitations':['unknown damage semantics']}
    assert not env._pending_lethal_evidence('zap',target.pos)


def test_anticipatory_play_has_separate_lead_and_nonnegative_latency():
    env=AttackEnv(scenario=scenario(opportunity_start_ms=2000),action_delay_ms=0)
    env.reset(seed=42)
    metrics=env.step(action(env,'fireball'))[4]['metrics']
    assert metrics['time_to_first_goal_card_ms']==0
    assert metrics['anticipatory_goal_card_ms']==2000
    assert metrics['first_goal_card_play_ms']==0


@pytest.mark.parametrize('card',['zap','fireball'])
def test_real_enemy_king_spell_lethal_rewards_win_but_does_not_guess_missing_hit(card):
    env=AttackEnv(scenario=scenario(target_lane='king',tower_hp=(NS(side=-1,lane='king',hp=1),)),
                  action_delay_ms=0)
    env.reset(seed=42)
    result=run(env,action(env,card,9,3))
    metrics=result[4]['metrics']
    assert result[2] and not result[3]
    assert metrics['terminal_reason']=='enemy_king_destroyed'
    assert metrics['attack_success'] and result[1]>0
    assert not metrics['goal_success']  # distinct terminal reason retained
    assert metrics['successful_spell_finish'] is None
    assert metrics['spell_finish_attribution']=='king_terminal_interrupted_hit_ledger'
    assert not metrics['hog_damage_ledger_complete']
    now=env.match.elapsed_ms
    with pytest.raises(RuntimeError):env.step(0)
    assert env.match.elapsed_ms==now
