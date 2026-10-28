from types import SimpleNamespace as E
import pytest
from sim.tower_support import TowerSupport


def fixture():
    tower=E(uid=1,side=1,is_tower=True)
    enemy=E(uid=2,side=-1,is_tower=False,target_uid=3,elixir_value=5000,max_hitpoints=1000)
    defender=E(uid=3,side=1,is_tower=False,alive=True)
    return E(entities={1:tower,2:enemy,3:defender},damage_log=[])


def test_only_own_defender_aggro_is_credited_and_value_capped():
    b=fixture();t=TowerSupport(b).attach()
    b.damage_log.append((50,1,2,250))
    assert t.assisted_value==1.25
    b.entities[2].target_uid=1
    b.damage_log.append((100,1,2,250))
    assert t.assisted_value==1.25
    b.entities[2].target_uid=3
    b.damage_log.append((150,1,2,2000))
    assert t.assisted_value==5
    assert t.total_damage==2500
    assert len(b.damage_log)==3
    with pytest.raises(ValueError):TowerSupport(b).attach()


def test_dead_defender_or_non_tower_source_never_gets_bonus():
    b=fixture();t=TowerSupport(b).attach();b.entities[3].alive=False
    b.damage_log.append((50,1,2,100))
    b.entities[3].alive=True;b.damage_log.append((100,3,2,100))
    assert t.assisted_value==0
    assert t.total_damage==100


def test_snapshot_does_not_recredit_damage_and_rebinds_callback():
    from copy import deepcopy
    b=fixture();t=TowerSupport(b).attach()
    b.damage_log.append((50,1,2,250))
    copied=deepcopy(t)
    assert copied.assisted_value==t.assisted_value==1.25
    copied.battle.damage_log.append((100,1,2,250))
    assert copied.assisted_value==2.5
    assert t.assisted_value==1.25


def test_all_damage_includes_tower_aggro_but_caps_healing_and_ignores_enemy_towers():
    b=fixture();t=TowerSupport(b).attach();b.entities[2].target_uid=1
    b.damage_log.append((50,1,2,250))
    assert t.total_paid_value==1.25 and t.assisted_value==0
    b.damage_log.append((100,1,2,2000))
    b.damage_log.append((150,1,2,1000))
    assert t.total_paid_value==5
    b.entities[1].side=-1
    b.damage_log.append((200,1,2,1000))
    assert t.total_paid_value==5
