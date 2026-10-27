from copy import deepcopy
from types import SimpleNamespace as NS
import pytest
from sim.tactical_rewards import fireball_outcomes
from sim.continuous_push import ContinuousPushEnv,DECKS
from sim.match import Match
from sim.env import DECK_26
from sim.adapter import grid_to_point


def cast(hits,resolved=True):
    return dict(spell='fireball',caster_side=1,side=1,resolved=resolved,hits=hits)


def hit(value=1000,damage=500,hp=670,tower=False):
    return dict(target_side=-1,damage=damage,max_hp=hp,elixir_milli=value,is_tower=tower)


def test_group_reward_uses_damage_fraction_and_paid_value_not_headcount():
    barb=fireball_outcomes([cast([hit()for _ in range(5)])])
    skeleton=fireball_outcomes([cast([hit(value=1000/15,damage=81,hp=81)for _ in range(15)])])
    assert barb['fireball_exchange_value']>0
    assert skeleton['fireball_exchange_value']<0
    assert barb['cluster_fireballs']==1
    assert fireball_outcomes([cast([],False)])['resolved_fireballs']==0
    assert fireball_outcomes([cast([])])['fireball_exchange_value']==0
    reflected=cast([hit()for _ in range(5)]);reflected['side']=-1
    assert fireball_outcomes([reflected])['fireball_exchange_value']==0


def test_tower_chip_and_zero_damage_do_not_farm_positive_reward():
    assert fireball_outcomes([cast([hit(tower=True)])])['fireball_exchange_value']==0
    assert fireball_outcomes([cast([hit(damage=0)])])['fireball_exchange_value']==0
    assert fireball_outcomes([cast([hit(value=100000,damage=100000,hp=1)])])['fireball_exchange_value']==6


def test_actual_paid_barbarians_and_fireball_are_scored_after_impact_only():
    e=ContinuousPushEnv();e.reset(seed=2)
    enemy=list(DECKS['hog']);enemy[3]='barbarians'
    m=next(Match(cards=e.cards,spells=e.spells,decks=(list(DECK_26),enemy),seed=s,geometry=e.geometry)
           for s in range(100)if 'fireball'in Match(cards=e.cards,spells=e.spells,decks=(list(DECK_26),enemy),seed=s,geometry=e.geometry).players[1].hand
           and 'barbarians'in Match(cards=e.cards,spells=e.spells,decks=(list(DECK_26),enemy),seed=s,geometry=e.geometry).players[-1].hand)
    m.players[1].elixir=10000;m.players[-1].elixir=10000
    assert m.play_card(-1,'barbarians',grid_to_point(14,18,-1))
    for _ in range(30):m.step(50)
    assert m.play_card(1,'fireball',grid_to_point(3,13,1))
    log=m.battle.spell_cast_log
    assert fireball_outcomes(log)['resolved_fireballs']==0
    for _ in range(100):
        m.step(50)
        if fireball_outcomes(log)['resolved_fireballs']:break
    result=fireball_outcomes(log)
    assert result['fireball_unit_hits']>=3
    assert result['fireball_exchange_value']>0
    assert m.players[-1].elixir_spent==5 and m.players[1].elixir_spent==4
    assert fireball_outcomes(log)==result


def test_reserve_potential_telescopes_to_no_hoarding_terminal_bonus():
    e=ContinuousPushEnv(continuous_profile={'reserve_potential_weight':.1,'shaping_gamma':.999})
    e.reset(seed=2);initial=e._previous['reserve_potential'];e.next_enemy=10**9
    e.max_ms=500
    e.step(0);e.step(0)
    assert e.done
    assert e._reserve_shaping_discounted_return==pytest.approx(-.1*initial)


def test_barbarians_augmentation_is_train_only_and_paid_from_normal_hand():
    e=ContinuousPushEnv(continuous_profile={'barbarian_train_fraction':1.})
    e.reset(seed=2,scenario_id='defense::hog/random')
    assert 'barbarians'in e.match.players[-1].hand+e.match.players[-1].queue
    for split in ('validation','test'):
        e=ContinuousPushEnv(split=split,continuous_profile={'barbarian_train_fraction':1.});e.reset(seed=2)
        assert 'barbarians'not in e.match.players[-1].hand+e.match.players[-1].queue
