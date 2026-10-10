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


def _unit(row: dict, projectiles: dict, level: int, *, building: bool) -> UnitSpec:
    kwargs = {field.name: _empty_value(field) for field in fields(UnitSpec)}
    projectile = projectiles.get(str(row.get("projectile") or ""), {})
    damage = _at_level(row, "damage_per_level", level)
    if not damage:
        damage = _at_level(projectile, "damage_per_level", level)
    kwargs.update({
        "name": to_snake_case(str(row["name"])),
        "hitpoints": int(_at_level(row, "hitpoints_per_level", level, 1)),
        "damage": int(damage),
        "hit_speed_ms": int(row.get("hit_speed") or 1000),
        "load_time_ms": int(row.get("load_time") or 0),
        "range_mt": int(row.get("range") or 0),
        "sight_range_mt": int(row.get("sight_range") or 5500),
        "speed_mt_per_sec": speed_to_mt_per_sec(int(row.get("speed") or 0)),
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
        "projectile_speed_mt_per_sec": speed_to_mt_per_sec(
            int(projectile.get("speed") or 0)),
        "projectile_homing": bool(projectile.get("homing", False)),
        "charge_range_mt": int(row.get("charge_range") or 0),
        "charge_speed_multiplier": int(row.get("charge_speed_multiplier") or 0),
        "damage_special": int(row.get("damage_special") or 0),
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
        "ignore_pushback": bool(row.get("ignore_pushback", False)),
        "raw": dict(row),
        "from_building_card": building,
    })
    return UnitSpec(**kwargs)


def load_public_gamedata(level: int = 11) -> tuple[dict[str, CardSpec], dict]:
    projectiles = {row["name"]: row for row in _rows("cards_stats_projectile.json")}
    characters = _rows("cards_stats_characters.json")
    buildings = _rows("cards_stats_building.json")
    units = {}
    by_source = {}
    for row, building in [(row, False) for row in characters] + [
            (row, True) for row in buildings]:
        if not row.get("name"):
            continue
        spec = _unit(row, projectiles, level, building=building)
        units[row["name"]] = spec
        units[spec.name] = spec
        by_source[row["name"]] = spec

    cards: dict[str, CardSpec] = {}
    card_rows = _rows("cards_stats_troop.json") + _rows("cards_stats_building.json")
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

    catalogue = _rows("cards.json")["cards"]
    for raw_key in ("fireball", "the-log"):
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
        projectile_speed_mt_per_sec=speed_to_mt_per_sec(int(fireball["speed"])),
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
        rolling_speed_mt_per_sec=speed_to_mt_per_sec(int(log["speed"])))
    return {"fireball": fire, "the_log": rolling,
            "log": SpellSpec(**{**rolling.__dict__, "name": "log"})}
