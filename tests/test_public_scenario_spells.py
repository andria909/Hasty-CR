from sim.public_scenario_spells import load_public_scenario_spells


def test_playable_spell_subset_has_no_npc_rows_or_phantom_heal():
    catalog = load_public_scenario_spells()
    assert set(catalog.spells) == {'rage', 'rocket', 'arrows', 'goblin_barrel'}
    assert not catalog.audit['heal']['playable_card']
    assert catalog.audit['heal_spirit']['playable_card']
    assert not catalog.audit['heal_spirit']['supported']


def test_missing_buff_integration_fails_closed():
    catalog = load_public_scenario_spells()
    assert catalog.audit['freeze']['missing'] == ['engine_load_buffs:Freeze']
    assert catalog.audit['lightning']['missing'] == ['engine_load_buffs:ZapFreeze']
    assert catalog.buff_fallback['Freeze'] == (-100, -100, 0)
    assert catalog.buff_fallback['ZapFreeze'] == (-100, -100, 0)
    enabled = load_public_scenario_spells(available_buffs=catalog.buff_fallback)
    assert enabled.spells['freeze'].damage == 115
    assert enabled.spells['lightning'].damage == 1056
    assert enabled.spells['lightning'].sequential_targets


def test_current_rage_overrides_stale_linked_buff():
    catalog = load_public_scenario_spells()
    rage = catalog.spells['rage']
    assert 'Rage' not in catalog.buff_fallback  # stale +35% must never reach engine
    assert (rage.damage, rage.tower_damage_override, rage.life_duration_ms) == (179, 45, 4500)
    assert rage.area_speed_pct == rage.area_hit_speed_pct == 30
    assert rage.area_only_own_troops


def test_barrel_uses_playable_projectile_not_npc_single_goblin():
    barrel = load_public_scenario_spells().spells['goblin_barrel']
    assert barrel.spawn_count == 3
    assert barrel.spawn_character == 'Goblin'
    assert barrel.spawn_deploy_time_ms == 1100
    assert barrel.travels_from_king
    assert not barrel.can_place_on_water


def test_unsupported_levels_and_incomplete_scaling_are_explicit():
    assert not load_public_scenario_spells(level=12).spells
    catalog = load_public_scenario_spells()
    assert 'portable_current_spawn_action_graph' in catalog.audit['graveyard']['missing']
    assert 'level_11_damage_per_second' in catalog.audit['poison']['missing']


def test_accepted_specs_actually_apply_damage_and_aura():
    from sim.arena import Point
    from sim.engine import Battle
    from sim.entities import make_unit
    from sim.gamedata import load_gamedata
    from sim.spells import cast_spell
    cards = load_gamedata()
    # Unit lookup source names are deliberately explicit in this standalone check.
    knight = cards['knight'].unit
    battle = Battle()
    target = make_unit(0, knight, -1, Point(9000, 12000), 0)
    ally = make_unit(0, knight, 1, Point(9100, 12000), 0)
    battle.add(target)
    battle.add(ally)
    catalog = load_public_scenario_spells()
    before = target.hitpoints
    cast_spell(battle, catalog.spells['rage'], target.pos, 1)
    assert target.hitpoints == before - 179
    battle._tick_areas()
    assert ally.buff_speed_pct == 30
    assert target.buff_speed_pct == 0
    before = target.hitpoints
    cast_spell(battle, catalog.spells['arrows'], target.pos, 1)
    for now in (0, 200, 400):
        battle.now_ms = now
        battle._tick_areas()
    assert target.hitpoints == before - 366


def test_freeze_requires_real_buff_consumer_and_interrupts_ramp(monkeypatch):
    from sim.arena import Point
    from sim.engine import Battle
    from sim.entities import make_unit
    from sim.gamedata import load_gamedata
    from sim.spells import cast_spell
    fallback = load_public_scenario_spells().buff_fallback
    catalog = load_public_scenario_spells(available_buffs=fallback)
    monkeypatch.setattr('sim.gamedata.load_buffs', lambda: fallback)
    battle = Battle()
    target = make_unit(0, load_gamedata()['knight'].unit, -1, Point(9000, 12000), 0)
    battle.add(target)
    target.ramp_target_uid = 123
    target.ramp_started_ms = 100
    cast_spell(battle, catalog.spells['freeze'], target.pos, 1)
    assert target.buff_speed_pct == target.buff_hit_speed_pct == -100
    assert target.buff_until_ms == 4000
    assert target.ramp_target_uid is None
