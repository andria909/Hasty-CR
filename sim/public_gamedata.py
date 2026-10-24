"""Portable fallback built from the repository's versioned RoyaleAPI data.

The full simulator prefers extracted client TOMLs.  A clean clone does not
ship those copyrighted client assets, but it does ship audited RoyaleAPI JSON
snapshots with level arrays and combat fields.  This loader keeps local smoke
training possible without inventing stats or downloading an APK.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, fields
from pathlib import Path
from types import UnionType
from typing import get_args, get_origin

from .entities import speed_to_mt_per_sec
from .gamedata import CardSpec, UnitSpec, to_snake_case

ROOT = Path(__file__).resolve().parents[1] / "data" / "royaleapi"


def _rows(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def _empty_value(field):
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:
        return field.default_factory()
    annotation = field.type
    origin = get_origin(annotation)
    if annotation is bool:
        return False
    if annotation is int:
        return 0
    if annotation is str:
        return ""
    if annotation is dict or origin is dict:
        return {}
    if origin is tuple or str(annotation).startswith("tuple"):
        return ()
    if origin in (UnionType,) or type(None) in get_args(annotation):
        return None
    return 0


def _at_level(row: dict, field: str, level: int, default=0):
    values = row.get(field)
    if isinstance(values, list) and values:
        rarity = str(row.get("rarity", "Common"))
        starts = {"Common": 1, "Rare": 3, "Epic": 6,
                  "Legendary": 9, "Champion": 11, "Experimental": 9}
        return values[min(len(values) - 1, max(0, level - starts.get(rarity, 1)))]
    return row.get(field.removesuffix("_per_level"), default) or default


def _unit(row: dict, projectiles: dict, level: int, *, building: bool,
          areas=None, buffs=None, rarities=None, rules=None) -> UnitSpec:
    kwargs = {field.name: _empty_value(field) for field in fields(UnitSpec)}
    projectile = projectiles.get(str(row.get("projectile") or ""), {})
    damage = _at_level(row, "damage_per_level", level)
    damage_source = None
    inspected = set()
    if not damage:
        # The default projectile may be decorative (Princess). A projectile
        # may also spawn the damaging one (Firecracker). Follow explicit links,
        # matching the client loader rather than assigning a guessed total hit.
        queue = [row.get("projectile"), row.get("custom_first_projectile")]
        while queue and not damage:
            name = queue.pop(0)
            if not name or name in inspected:
                continue
            inspected.add(name)
            candidate = projectiles.get(name, {})
            damage = _at_level(candidate, "damage_per_level", level)
            if damage:
                damage_source = name
            else:
                queue.append(candidate.get("spawn_projectile"))
    kwargs.update({
        "name": to_snake_case(str(row["name"])),
        "hitpoints": int(_at_level(row, "hitpoints_per_level", level, 1)),
        "damage": int(damage),
        "hit_speed_ms": int(row.get("hit_speed") or 1000),
        "load_time_ms": int(row.get("load_time") or 0),
        "range_mt": int(row.get("range") or 0),
        "sight_range_mt": int(row.get("sight_range") or 5500),
        # Historical UnitSpec field name: this stores raw client Speed, just
        # like the TOML loader. make_unit converts it to millitiles/second.
        # Converting here as well accelerated fallback units by another x20.
        "speed_mt_per_sec": int(row.get("speed") or 0),
        "collision_radius_mt": int(row.get("collision_radius") or 500),
        "mass": int(row.get("mass") or 1),
        "deploy_time_ms": int(row.get("deploy_time") or 1000),
        "attacks_ground": bool(row.get("attacks_ground", True)),
        "attacks_air": bool(row.get("attacks_air", False)),
        "flying": bool(row.get("flying_height", 0)),
        "target_only_buildings": bool(row.get("target_only_buildings", False)),
        "target_only_troops": bool(row.get("target_only_troops", False)),
        "splash_radius_mt": int(row.get("area_damage_radius") or
                                 projectile.get("projectile_radius") or 0),
        "jump_enabled": bool(row.get("jump_enabled", False)),
        "jump_speed_mt_per_sec": speed_to_mt_per_sec(int(row.get("jump_speed") or 0)),
        "retarget_after_attack": bool(row.get("retarget_after_attack", False)),
        "projectile_speed_mt_per_sec": int(projectile.get("speed") or 0) * 1000 // 60,
        "projectile_homing": bool(projectile.get("homing", False)),
        "charge_range_mt": int(row.get("charge_range") or 0),
        "charge_speed_multiplier": int(row.get("charge_speed_multiplier") or 0),
        "damage_special": int(row.get("damage_special") or 0),
        "attack_self_pushback_mt": int(row.get("attack_push_back") or 0),
        "shield_hitpoints": int(row.get("shield_hitpoints") or 0),
        "initial_shield_pct": 100,
        "death_damage": int(row.get("death_damage") or 0),
        "death_damage_radius_mt": int(row.get("death_damage_radius") or 0),
        "death_damage_pushback_mt": int(row.get("death_push_back") or 0),
        "death_spawn_character": str(row.get("death_spawn_character") or ""),
        "death_spawn_count": int(row.get("death_spawn_count") or 0),
        "death_spawn_radius_mt": int(row.get("death_spawn_radius") or 0),
        "death_spawn_deploy_ms": int(row.get("death_spawn_deploy_time") or 0),
        "spawn_character": str(row.get("spawn_character") or ""),
        "spawn_count": int(row.get("spawn_number") or 0),
        "spawn_pause_ms": int(row.get("spawn_pause_time") or 0),
        "spawn_start_ms": int(row.get("spawn_start_time") or 0),
        "target_buff": str(projectile.get("target_buff") or ""),
        "buff_time_ms": int(projectile.get("buff_time") or 0),
        "multiple_targets": int(row.get("multiple_targets") or 0),
        "all_targets_hit": bool(row.get("all_targets_hit", False)),
        "kamikaze": bool(row.get("kamikaze", False)),
        "lifetime_ms": int(row.get("life_time") or 0),
        # Miner is deployed at the target point but remains underground for
        # its client SpawnPathfindSpeed travel time. This is already a speed
        # in mt/s (unlike UnitSpec's historical walking-speed field).
        "burrow_speed_mt_per_sec": int(row.get("spawn_pathfind_speed") or 0) * 1000 // 60,
        "tower_damage_pct": max(0, 100 + int(row.get("crown_tower_damage_percent") or 0)),
        "ignore_pushback": bool(row.get("ignore_pushback", False)),
        "raw": dict(row),
        "from_building_card": building,
    })
    if damage_source:
        kwargs['raw']['_public_damage_projectile'] = damage_source
        carried = projectiles[damage_source]
        if int(carried.get('spawn_count') or 0) > 1:
            kwargs['raw']['_public_profile_limitations'] = [
                'projectile_fragment_count_and_scatter_geometry_not_modeled']
            kwargs['raw']['_public_fragment_count'] = int(carried['spawn_count'])
    area = (areas or {}).get(projectile.get("spawn_area_effect_object"), {})
    buff = (buffs or {}).get(area.get("buff"), {})
    if buff.get("heal_per_second"):
        from .public_stat_scaling import scale_public_stat
        kwargs.update(
            projectile_heal_per_second=scale_public_stat(
                buff["heal_per_second"], buff["rarity"], level, rarities),
            projectile_heal_radius_mt=int(area["radius"]),
            projectile_heal_duration_ms=int(area["life_duration"]),
            projectile_heal_tick_ms=int(buff["hit_frequency"]))
    if area.get('follow_behaviour') == 'FollowParent' and buff.get('damage_per_second'):
        from .public_stat_scaling import scale_public_stat
        kwargs['projectile_follow_area'] = dict(
            damage_per_second=scale_public_stat(buff['damage_per_second'], buff['rarity'], level, rarities),
            radius_mt=int(area['radius']), duration_ms=int(area['life_duration']),
            hit_frequency_ms=int(buff['hit_frequency']),
            attract_percentage=int(buff.get('attract_percentage') or 0),
            crown_tower_percent=100+int(buff.get('crown_tower_damage_percent') or 0),
            projectile_speed_mt_per_sec=int(projectile['speed'])*1000//60,
            projectile_range_mt=int(projectile['projectile_range']),
            declared_homing_time_ms=int(projectile.get('homing_time') or 0),
            declared_homing_min_distance_mt=int(projectile.get('homing_min_distance') or 0),
            declared_attract_max_angle=int(buff.get('attract_max_angle') or 0),
            hits_air=bool(area['hits_air']), hits_ground=bool(area['hits_ground']))
        kwargs['raw']['_public_profile_limitations'] = [
            'follow_parent_projectile_initial_homing_semantics_not_modeled',
            'follow_parent_area_angular_pull_mask_not_modeled',
            'follow_parent_area_first_damage_tick_offset_not_source_verified']
    if row.get('name') == 'SantaPresent':
        # A zero-HP container points to a BOOST area, not a combat action.
        # The linked area's duration is NOT evidence of a pickup lifetime.
        linked = (areas or {}).get(row.get('boost_items'), {})
        if not linked.get('boost') or int(row.get('hitpoints') or 0) != 0:
            raise ValueError('SantaPresent no longer matches the public pickup shape')
        kwargs.update(object_role='pickup_effect_carrier', always_untargetable=True)
        kwargs['raw']['_public_object_role'] = 'pickup_effect_carrier'
        kwargs['raw']['_public_pickup_effect_reference'] = str(row['boost_items'])
        kwargs['raw']['_public_profile_limitations'] = [
            'pickup_activation_and_clone_bottle_action_graph_not_modeled',
            'pickup_expiry_not_declared_do_not_infer_from_buff_duration']
    if row.get('name') == 'BombTowerBomb':
        from .public_stat_scaling import scale_public_stat
        # Zero-HP death-effect carrier, not a persistent attackable building.
        # The only declared countdown is DeployTime; retain that interpretation
        # explicitly until the original action graph is available.
        kwargs.update(lifetime_ms=int(row['deploy_time']),
                      always_untargetable=True,
                      death_damage=scale_public_stat(row['death_damage'], row['rarity'], level, rarities))
        kwargs['raw']['_public_profile_limitations'] = [
            'death_bomb_deploy_time_as_fuse_not_verified_against_original_action_graph']
    rule = (rules or {}).get('heal_spirit', {}) if kwargs['name'] == 'heal_spirit' else {}
    if rule and rule.get('level') == level:
        if 'hitpoints_override' in rule:
            kwargs['hitpoints'] = int(rule['hitpoints_override'])
        if 'cannot_target_towers' in rule:
            kwargs['cannot_target_towers'] = bool(rule['cannot_target_towers'])
    return UnitSpec(**kwargs)


def load_public_gamedata(level: int = 11) -> tuple[dict[str, CardSpec], dict]:
    projectiles = {row["name"]: row for row in _rows("cards_stats_projectile.json")}
    areas = {row["name"]: row for row in _rows("cards_stats_spell.json")}
    buffs = {row["name"]: row for row in _rows("cards_stats_character_buff.json")}
    rarities = {row["name"]: row for row in _rows("rarities.json")}
    rules = _rows("combat_rules.json")["rules"]
    characters = _rows("cards_stats_characters.json")
    buildings = _rows("cards_stats_building.json")
    units = {}
    by_source = {}
    for row, building in [(row, False) for row in characters] + [
            (row, True) for row in buildings]:
        if not row.get("name"):
            continue
        spec = _unit(row, projectiles, level, building=building,
                     areas=areas, buffs=buffs, rarities=rarities, rules=rules)
        units[row["name"]] = spec
        units[spec.name] = spec
        by_source[row["name"]] = spec

    cards: dict[str, CardSpec] = {}
    catalogue = _rows("cards.json")["cards"]
    types = {row['key']: row['type'] for row in catalogue}
    card_rows = _rows("cards_stats_troop.json") + _rows("cards_stats_building.json")
    # Heal is a card-definition row in the spell table but the playable card
    # summons HealSpirit: catalogue type, not the source table, decides this.
    card_rows += [row for row in areas.values()
                  if types.get(row.get('key')) == 'Troop' and row.get('summon_character')]
    for row in card_rows:
        key = str(row.get("key") or "").replace("-", "_")
        source = str(row.get("summon_character") or row.get("name") or "")
        unit = by_source.get(source)
        if not key or unit is None:
            continue
        cards[key] = CardSpec(
            name=key, cost=int(row.get("mana_cost") or row.get("elixir") or 0),
            rarity=str(row.get("rarity") or "Common"), unit=unit,
            summon_number=max(1, int(row.get("summon_number") or 0)),
            summon_radius_mt=int(row.get("summon_radius") or 0),
            summon_deploy_delay_ms=int(row.get("summon_deploy_delay") or 0),
            deploy_time_ms=int(row.get("custom_deploy_time") or unit.deploy_time_ms))

    for raw_key in ("fireball", "the-log", "zap", "royal-delivery"):
        row = next(item for item in catalogue if item["key"] == raw_key)
        key = raw_key.replace("-", "_")
        cards[key] = CardSpec(
            name=key, cost=int(row["elixir"]), rarity=str(row["rarity"]),
            unit=None, summon_number=0, summon_radius_mt=0,
            summon_deploy_delay_ms=0, deploy_time_ms=0)
    aliases = {
        "ice_spirit": "ice_spirits", "ice_golem": "ice_golemite",
        "the_log": "log", "archers": "archer", "mini_pekka": "minipekka",
        "royal_ghost": "ghost", "goblins": "goblin",
        "spear_goblins": "spear_goblin", "guards": "guard",
        "barbarians": "barbarian", "bats": "bat", "minions": "minion",
    }
    # The catalogue names this "Elixir Collector" while its exact public
    # building identifier is ElixirCollector; its stats row has no card key.
    collector = next(row for row in catalogue if row['key'] == 'elixir-collector')
    cards['elixir_collector'] = CardSpec(
        name='elixir_collector', cost=int(collector['elixir']), rarity=collector['rarity'],
        unit=by_source['ElixirCollector'], summon_number=1, summon_radius_mt=0,
        summon_deploy_delay_ms=0, deploy_time_ms=by_source['ElixirCollector'].deploy_time_ms)
    mirror = next(row for row in catalogue if row['key'] == 'mirror')
    cards['mirror'] = CardSpec(
        name='mirror', cost=int(mirror['elixir']), rarity=mirror['rarity'],
        unit=None, summon_number=0, summon_radius_mt=0,
        summon_deploy_delay_ms=0, deploy_time_ms=0, mirror_units_only=True)
    for source, alias in aliases.items():
        if source in cards and alias not in cards:
            spec = cards[source]
            cards[alias] = CardSpec(**{**spec.__dict__, "name": alias})
    return cards, units


def load_public_spells(level: int = 11):
    from .spells import SpellSpec

    rows = {row["name"]: row for row in _rows("cards_stats_projectile.json")}

    def damage(row, rarity):
        return int(_at_level({**row, "rarity": rarity}, "damage_per_level", level))

    fireball = rows["FireballSpell"]
    log = rows["LogProjectileRolling"]
    fire = SpellSpec(
        name="fireball", damage=damage(fireball, "Rare"),
        radius_mt=int(fireball["radius"]),
        radius_y_mt=int(fireball.get("radius_y") or fireball["radius"]),
        crown_tower_percent=100 + int(fireball["crown_tower_damage_percent"]),
        pushback_mt=int(fireball.get("pushback") or 0),
        # 77 native flight fits: median599.48mt/tick; client clock20Hz.
        projectile_speed_mt_per_sec=int(fireball["speed"]) * 20,
        travels_from_king=True,
        deflect_behaviour=str(fireball.get("deflect_behaviour") or ""))
    rolling = SpellSpec(
        name="the_log", damage=damage(log, "Legendary"),
        radius_mt=int(log["projectile_radius"]),
        radius_y_mt=int(log["projectile_radius_y"]),
        crown_tower_percent=100 + int(log["crown_tower_damage_percent"]),
        pushback_mt=int(log.get("pushback") or 0), hits_air=False,
        deflect_behaviour=str(log.get("deflect_behaviour") or ""),
        rolling_range_mt=int(log["projectile_range"]),
        rolling_speed_mt_per_sec=int(log["speed"]) * 1000 // 60)
    spell_rows = {row.get("key"): row for row in _rows("cards_stats_spell.json")}
    zap_row = spell_rows["zap"]
    zap = SpellSpec(
        name="zap",
        damage=int(_at_level(zap_row, "damage_per_level", level)),
        radius_mt=int(zap_row["radius"]),
        radius_y_mt=int(zap_row["radius"]),
        crown_tower_percent=100 + int(zap_row["crown_tower_damage_percent"]),
        pushback_mt=0,
        target_buff=str(zap_row.get("buff") or ""),
        buff_time_ms=int(zap_row.get("buff_time") or 0),
    )
    return {"fireball": fire, "zap": zap, "the_log": rolling,
            "log": SpellSpec(**{**rolling.__dict__, "name": "log"})}


def load_public_elixir_table():
    """Portable Collector economy in milli-elixir; client entries take precedence."""
    row = next(row for row in _rows('cards_stats_building.json') if row['name']=='ElixirCollector')
    return {'elixir_collector': dict(
        generate_ms=int(row['mana_generate_time_ms']),
        amount_milli=int(row['mana_collect_amount'])*1000,
        death_milli=int(row['mana_on_death'])*1000,
        death_opp_milli=int(row.get('mana_on_death_for_opponent') or 0)*1000)}
