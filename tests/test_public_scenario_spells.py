from sim.public_scenario_spells import load_public_scenario_spells


def test_playable_spell_subset_has_no_npc_rows_or_phantom_heal():
    catalog = load_public_scenario_spells()
    assert set(catalog.spells) == {'rage', 'rocket', 'arrows', 'goblin_barrel', 'barbarian_barrel', 'royal_delivery'}
    assert not catalog.audit['heal']['playable_card']
    assert catalog.audit['heal_spirit']['playable_card']
    assert catalog.audit['heal_spirit']['supported']
    assert 'heal_spirit' not in catalog.spells


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


def test_remaining_card_gaps_are_explicit_not_aliased_to_other_spells():
    catalog=load_public_scenario_spells()
    remaining={'barbarian_barrel','clone','earthquake','elixir_collector','giant_snowball',
               'graveyard','heal_spirit','mirror','party_rocket','poison','tornado'}
    assert remaining <= catalog.audit.keys()
    assert catalog.audit['elixir_collector']['missing'] == ['engine_public_elixir_income_table','starting_hand_exclusion']
    assert 'projectile_or_action_definition' in catalog.audit['party_rocket']['missing']
    assert 'level_11_damage_per_second' in catalog.audit['tornado']['missing']
    assert 'engine_clone_invents_shield_for_unshielded_troops' in catalog.audit['clone']['missing']
    assert 'mirror' not in catalog.spells
    assert 'party_rocket' not in catalog.spells


def test_clone_requires_explicit_verified_engine_capability():
    catalog=load_public_scenario_spells(engine_capabilities={'clone_preserves_shield_presence'})
    clone=catalog.spells['clone']
    assert clone.clone and clone.only_own_troops and clone.ignore_buildings
    assert clone.radius_mt==3000


def test_snowball_requires_real_slow_buff_and_preserves_current_override(monkeypatch):
    from sim.arena import Point
    from sim.engine import Battle
    from sim.entities import make_unit
    from sim.gamedata import load_gamedata
    from sim.spells import cast_spell
    raw=load_public_scenario_spells()
    assert 'giant_snowball' not in raw.spells
    assert raw.buff_fallback['IceWizardSlowDown']==(-35,-35,0)
    catalog=load_public_scenario_spells(available_buffs=raw.buff_fallback)
    monkeypatch.setattr('sim.gamedata.load_buffs',lambda:raw.buff_fallback)
    spec=catalog.spells['giant_snowball']
    assert spec.damage==179 and spec.tower_damage_override==45
    battle=Battle()
    unit=make_unit(0,load_gamedata()['knight'].unit,-1,Point(9000,20000),0)
    battle.add(unit)
    before=unit.hitpoints
    cast_spell(battle,spec,unit.pos,1)
    assert unit.hitpoints==before-179
    assert unit.buff_speed_pct==unit.buff_hit_speed_pct==-35
    assert unit.buff_until_ms==2500


def test_barbarian_barrel_rolls_ground_only_once_and_spawns_one_barbarian():
    from sim.arena import Point
    from sim.engine import Battle
    from sim.entities import make_unit
    from sim.gamedata import load_gamedata
    from sim.spells import cast_spell
    cards=load_gamedata()
    battle=Battle()
    battle.unit_lookup=lambda name: cards['barbarians'].unit if name=='Barbarian' else None
    knight=make_unit(0,cards['knight'].unit,-1,Point(9000,20000),0)
    air=make_unit(0,cards['baby_dragon'].unit,-1,Point(9000,20000),0)
    battle.add(knight);battle.add(air)
    spec=load_public_scenario_spells().spells['barbarian_barrel']
    assert spec.damage==241 and spec.rolling_range_mt==4500 and not spec.hits_air
    before,air_before=knight.hitpoints,air.hitpoints
    cast_spell(battle,spec,Point(9000,22000),1)
    for _ in range(30):
        battle.now_ms+=50
        battle._tick_rolling_spells(50)
    assert knight.hitpoints==before-241
    assert air.hitpoints==air_before
    spawned=[e for e in battle.entities.values() if e.side==1]
    assert len(spawned)==1 and spawned[0].name=='barbarian'
    assert spawned[0].deploy_remaining_ms==500


def test_projectile_rates_use_client_sixtieths_not_walking_speed_multiplier():
    from sim.arena import Point
    from sim.engine import Battle
    from sim.spells import cast_spell
    catalog=load_public_scenario_spells()
    assert catalog.spells['rocket'].projectile_speed_mt_per_sec==350*1000//60
    assert catalog.spells['goblin_barrel'].projectile_speed_mt_per_sec==400*1000//60
    assert catalog.spells['barbarian_barrel'].rolling_speed_mt_per_sec==200*1000//60
    battle=Battle()
    cast_spell(battle,catalog.spells['rocket'],Point(9000,22000),1,Point(9000,28000))
    assert battle.spell_impacts[0][0]==6000*1000//(350*1000//60)
