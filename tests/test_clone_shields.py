"""Clone preserves shield presence and cannot become a healed full-HP troop."""
import pytest

from sim.arena import Point
from sim.engine import Battle
from sim.entities import make_unit
from sim.public_gamedata import load_public_gamedata
from sim.public_scenario_spells import load_public_scenario_spells
from sim.spells import cast_spell


@pytest.mark.parametrize('card,shielded', [('knight',False),('dark_prince',True)])
def test_clone_has_only_original_shield_presence_and_one_hp_ceiling(card,shielded):
    cards,_=load_public_gamedata(level=11)
    battle=Battle()
    original=make_unit(0,cards[card].unit,1,Point(9000,21000),0)
    battle.add(original)
    original_hp,original_shield=original.hitpoints,original.shield_hitpoints
    assert (original_shield>0)==shielded
    clone_spell=load_public_scenario_spells(engine_capabilities={'clone_preserves_shield_presence'}).spells['clone']
    cast_spell(battle,clone_spell,original.pos,1)
    clone=next(e for e in battle.entities.values() if e.is_clone)
    assert clone.hitpoints==clone.max_hitpoints==1
    assert clone.shield_hitpoints==clone.shield_max_hitpoints==int(shielded)
    assert clone.heal(1000)==0 and clone.hitpoints==1
    clone.take_damage(1)
    assert clone.alive==shielded
    if shielded:
        clone.take_damage(1)
        assert not clone.alive
    assert (original.hitpoints,original.shield_hitpoints)==(original_hp,original_shield)
