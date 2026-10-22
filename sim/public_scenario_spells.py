"""Audited portable spell subset for defense scenarios.

This module does not alter global spell/buff caches. Callers must explicitly
merge the accepted specs, then route them through ``cast_spell`` with a King
Tower origin. ``audit`` records missing mechanics rather than inventing them.
The snapshot is historical data, not a claim about a newer game build.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .entities import speed_to_mt_per_sec
from .spells import SpellSpec

DATA = Path(__file__).resolve().parents[1] / 'data' / 'royaleapi'
REQUESTED = ('graveyard', 'goblin_barrel', 'rage', 'freeze', 'heal', 'heal_spirit',
             'arrows', 'poison', 'rocket', 'lightning')


@dataclass
class PublicScenarioSpells:
    spells: dict[str, SpellSpec]
    # Candidate buff deltas can be merged into load_buffs by the caller after
    # that integration is implemented. Merely returning these enables nothing.
    buff_fallback: dict[str, tuple[int, int, int]]
    audit: dict[str, dict]


def _read(root, filename):
    return json.loads((root / filename).read_text(encoding='utf-8'))


def _level_damage(row, level):
    rarity_start = {'Common': 1, 'Rare': 3, 'Epic': 6, 'Legendary': 9,
                    'Champion': 11}
    start = rarity_start[row['rarity']]
    index = level - start
    values = row.get('damage_per_level')
    if not isinstance(values, list) or not 0 <= index < len(values):
        raise ValueError('missing exact damage_per_level entry')
    return int(values[index])


def _buff_delta(value):
    value = int(value)
    return value - 100 if value > 0 else value


def load_public_scenario_spells(level=11, *, root=DATA,
                                available_buffs: Mapping | None = None):
    """Return supported specs and explicit exclusions, with provenance.

    ``available_buffs`` must be the mapping actually used by engine.load_buffs,
    not just candidate data. Without Freeze/ZapFreeze integration the affected
    spells are excluded, preventing damage-only freeze/lightning behavior.
    Current combat overrides are level 11: other levels fail closed.
    """
    root = Path(root)
    available_buffs = available_buffs or {}
    spells = {r['name']: r for r in _read(root, 'cards_stats_spell.json')}
    projectiles = {r['name']: r for r in _read(root, 'cards_stats_projectile.json')}
    buff_rows = {r['name']: r for r in _read(root, 'cards_stats_character_buff.json')}
    playable = {r['key'].replace('-', '_'): r for r in _read(root, 'cards.json')['cards']}
    rules = _read(root, 'combat_rules.json')['rules']
    fallback = {}
    # Zero-heal movement/attack buffs are representable without inventing
    # level scaling. HealSpiritBuff and Poison need level-aware healing/DPS.
    for name in ('Freeze', 'ZapFreeze'):
        row = buff_rows[name]
        fallback[name] = (_buff_delta(row['speed_multiplier']),
                          _buff_delta(row['hit_speed_multiplier']), 0)
    report = {}
    accepted = {}

    def status(name, *, missing=(), notes=(), sources=(), spec=None):
        report[name] = {'playable_card': name in playable,
                        'supported': spec is not None,
                        'missing': list(missing), 'limitations': list(notes),
                        'sources': list(sources), 'level': level}
        if spec is not None:
            accepted[name] = spec

    for name in REQUESTED:
        if name not in playable:
            status(name, missing=('not_in_playable_card_catalogue',),
                   notes=('Heal row belongs to heal_spirit; not a separate playable Heal spell.',))
        elif level != 11:
            status(name, missing=('current_combat_overrides_require_level_11',))
    if level != 11:
        return PublicScenarioSpells(accepted, fallback, report)

    # Rage's linked raw buff says +35%, contradicted by the newer audited rule
    # (+30%). Use the explicit current override for both damage and aura.
    r = rules['rage']
    accepted_rage = SpellSpec(
        name='rage', damage=int(r['damage_override']), radius_mt=int(r['radius_mt']),
        radius_y_mt=int(r['radius_mt']), crown_tower_percent=0, pushback_mt=0,
        tower_damage_override=int(r['tower_damage_override']),
        life_duration_ms=int(r['life_duration_ms']),
        hit_frequency_ms=int(r['hit_frequency_ms']),
        area_speed_pct=int(r['area_speed_pct']),
        area_hit_speed_pct=int(r['area_hit_speed_pct']),
        area_only_own_troops=bool(r['area_only_own_troops']),
        area_buff_linger_ms=int(r['area_buff_linger_ms']),
        resolves_card_as_spell=True)
    status('rage', spec=accepted_rage, sources=('combat_rules.json:rage',),
           notes=('Engine aura does not alter spawner production cadence.',))

    p, r = projectiles['RocketSpell'], rules['rocket']
    status('rocket', spec=SpellSpec(
        name='rocket', damage=_level_damage(p, level), radius_mt=int(p['radius']),
        radius_y_mt=int(p['radius']), crown_tower_percent=100+int(p['crown_tower_damage_percent']),
        pushback_mt=int(p['pushback']), tower_damage_override=int(r['tower_damage_override']),
        projectile_speed_mt_per_sec=speed_to_mt_per_sec(int(p['speed'])),
        travels_from_king=bool(r['travels_from_king']),
        deflect_behaviour=str(p['deflect_behaviour'])),
        sources=('cards_stats_projectile.json:RocketSpell', 'combat_rules.json:rocket'),
        notes=('Common cast_spell uses constant-speed travel; projectile gravity trajectory is not modeled.',))

    p, r = projectiles['GoblinBarrelSpell'], rules['goblin_barrel']
    status('goblin_barrel', spec=SpellSpec(
        name='goblin_barrel', damage=0, radius_mt=int(p['radius']),
        radius_y_mt=int(p['radius']), crown_tower_percent=0, pushback_mt=0,
        spawn_character=str(p['spawn_character']), spawn_count=int(p['spawn_character_count']),
        spawn_deploy_time_ms=int(p['spawn_character_deploy_time']),
        projectile_speed_mt_per_sec=speed_to_mt_per_sec(int(p['speed'])),
        travels_from_king=bool(r['travels_from_king']),
        can_place_on_water=bool(r['can_place_on_water'])),
        sources=('cards_stats_projectile.json:GoblinBarrelSpell', 'combat_rules.json:goblin_barrel'),
        notes=('Requires battle.unit_lookup("Goblin").',
               'Common engine uses a fixed 0.5-tile circular spawn layout, not an audited barrel placement graph.',
               'Common cast_spell uses constant-speed travel; projectile gravity trajectory is not modeled.'))

    p, r = projectiles['ArrowsSpell'], rules['arrows']
    status('arrows', spec=SpellSpec(
        name='arrows', damage=int(r['damage_override']), radius_mt=int(r['radius_mt']),
        radius_y_mt=int(r['radius_mt']), crown_tower_percent=0, pushback_mt=0,
        tower_damage_override=int(r['tower_damage_override']),
        volley_waves=int(r['volley_waves']), volley_interval_ms=int(r['volley_interval_ms']),
        deflect_behaviour=str(p['deflect_behaviour'])),
        sources=('combat_rules.json:arrows', 'cards_stats_projectile.json:ArrowsSpell'),
        notes=('Cast-to-first-wave projectile flight timing is absent from the portable rule; waves start at cast.',))

    row, r = spells['Freeze'], rules['freeze']
    if available_buffs.get('Freeze') != fallback['Freeze']:
        status('freeze', missing=('engine_load_buffs:Freeze',),
               sources=('cards_stats_spell.json:Freeze', 'cards_stats_character_buff.json:Freeze'))
    else:
        status('freeze', spec=SpellSpec(
            name='freeze', damage=_level_damage(row, level), radius_mt=int(row['radius']),
            radius_y_mt=int(row['radius']), crown_tower_percent=100+int(row['crown_tower_damage_percent']),
            pushback_mt=0, target_buff=str(row['buff']), buff_time_ms=int(row['buff_time']),
            tower_damage_override=int(r['tower_damage_override'])),
            sources=('cards_stats_spell.json:Freeze', 'combat_rules.json:freeze'))

    row, p, r = spells['Lightning'], projectiles['LighningSpell'], rules['lightning']
    if available_buffs.get('ZapFreeze') != fallback['ZapFreeze']:
        status('lightning', missing=('engine_load_buffs:ZapFreeze',))
    else:
        status('lightning', spec=SpellSpec(
            name='lightning', damage=_level_damage(p, level), radius_mt=int(row['radius']),
            radius_y_mt=int(row['radius']), crown_tower_percent=100+int(p['crown_tower_damage_percent']),
            pushback_mt=0, target_buff=str(p['target_buff']), buff_time_ms=int(p['buff_time']),
            tower_damage_override=int(r['tower_damage_override']),
            target_limit=3, target_highest_hitpoints=bool(row['hit_biggest_targets']),
            life_duration_ms=int(row['life_duration']), hit_frequency_ms=int(row['hit_speed']),
            falls_from_sky=bool(row['projectile_start_height']), sequential_targets=bool(r['sequential_targets']),
            deflect_behaviour=str(p['deflect_behaviour'])),
            sources=('cards_stats_spell.json:Lightning', 'cards_stats_projectile.json:LighningSpell',
                     'combat_rules.json:lightning'))

    status('graveyard', missing=('portable_current_spawn_action_graph',),
           sources=('cards_stats_spell.json:Graveyard',),
           notes=('Legacy JSON interval=500ms is not the client loader current 12-spawn action list.',
                  'Do not synthesize periodic skeletons from this stale interval.'))
    status('poison', missing=('level_11_damage_per_second',),
           sources=('cards_stats_spell.json:Poison', 'cards_stats_character_buff.json:Poison'),
           notes=('Buff has raw DPS=57 without a level array; tower override168 alone is insufficient.',))
    status('heal_spirit', missing=('troop_heal_area_in_public_loader_and_engine',),
           sources=('cards.json:heal-spirit', 'cards_stats_spell.json:HealSpirit'),
           notes=('Playable Heal Spirit is a troop, not an independently cast heal area.',
                  'Raw HealSpiritBuff heal_per_second lacks exact level scaling.'))
    return PublicScenarioSpells(accepted, fallback, report)
