"""Unused pickups are distinct from combat troops; no fabricated expiry."""
from sim.arena import Point
from sim.entities import make_unit
from sim.public_gamedata import load_public_gamedata


def test_santa_present_role_is_derived_from_zero_hp_boost_reference_not_ttl():
    cards,units=load_public_gamedata()
    present=units['SantaPresent']
    assert present.raw['hitpoints']==0
    assert present.raw['boost_items']=='SantaPresentBoostClone'
    assert present.object_role=='pickup_effect_carrier'
    assert present.lifetime_ms==0
    assert present.raw['_public_pickup_effect_reference']=='SantaPresentBoostClone'
    assert 'pickup_activation_and_clone_bottle_action_graph_not_modeled' in present.raw['_public_profile_limitations']
    body=make_unit(0,present,-1,Point(9000,1500),0)
    assert body.object_role=='pickup_effect_carrier' and body.untargetable
    normal=make_unit(0,cards['knight'].unit,1,Point(9000,2000),0)
    assert normal.object_role=='combat'
    assert not normal.is_valid_target(body)
