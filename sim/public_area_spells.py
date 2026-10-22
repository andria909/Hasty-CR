"""Explicit public-source area specs, with unresolved simulator semantics exposed.

This opt-in loader changes no global caches. Damage is represented by temporal
area ticks, never a fabricated instantaneous hit. Exact data provenance is
separate from engine fidelity: public buff clocks and attraction procedures are
not proof of the real client's tick phase or displacement.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from sim.public_stat_scaling import scale_public_stat
from sim.spells import SpellSpec

DATA=Path(__file__).resolve().parents[1]/'data/royaleapi'
REQUESTED=('poison','earthquake','tornado')


@dataclass
class PublicAreaSpells:
    spells: dict[str,SpellSpec]
    audit: dict[str,dict]


def load_public_area_spells(level=11,*,root=DATA):
    root=Path(root)
    files=('cards_stats_spell.json','cards_stats_character_buff.json','rarities.json','combat_rules.json')
    raw={name:(root/name).read_bytes() for name in files}
    tables={name:json.loads(contents)for name,contents in raw.items()}
    rows={row.get('key',row['name'].lower()):row for row in tables['cards_stats_spell.json']}
    buffs={row['name']:row for row in tables['cards_stats_character_buff.json']}
    rarities={row['name']:row for row in tables['rarities.json']}
    rules=tables['combat_rules.json']['rules']
    hashes={name:hashlib.sha256(contents).hexdigest()for name,contents in raw.items()}
    accepted={};audit={}
    for key in REQUESTED:
        report=dict(supported=False,source_stats_exact=False,mechanics_verified=False,level=level,
            sources=list(files),source_sha256=hashes,missing=[],limitations=[
                'public_snapshot_is_not_verified_current_client_mechanics',
                'per_target_buff_tick_phase_approximated_by_shared_area_clock',
                'area_boundary_and_lingering_buff_behavior_require_live_validation',
                'temporal_spell_reward_hits_not_logged_by_existing_area_engine'])
        audit[key]=report
        try:
            row=rows[key];buff=buffs[row['buff']]
            duration=int(row['life_duration']);period=int(buff['hit_frequency'])
            if duration<=0 or period<=0 or int(buff['damage_per_second'])<=0:
                raise ValueError('Missing positive exact area duration/frequency/DPS')
            if not row['only_enemies'] or row['only_own_troops']:
                raise ValueError('Unsupported area ownership semantics')
            rule=rules.get(key,{})
            tower_override=int(rule.get('tower_damage_override',0))
            if tower_override and int(rule['level'])!=int(level):
                raise ValueError('Current crown damage override applies only to its explicit level')
            dps=scale_public_stat(buff['damage_per_second'],buff['rarity'],level,rarities)
            # Existing source convention: 0 denotes ordinary building damage;
            # a nonzero field is the explicit multiplier passed to the engine.
            building_percent=int(buff['building_damage_percent'])or 100
            crown_percent=0 if buff['no_effect_to_crown_towers'] else 100+int(buff['crown_tower_damage_percent'])
            if not 0<=crown_percent<=100 or building_percent<0:
                raise ValueError('Unrepresentable damage percentage')
            ticks=(duration+period-1)//period
            if tower_override and tower_override%ticks:
                raise ValueError('Crown total cannot be represented by equal exact ticks')
            speed=int(buff['speed_multiplier']);hit=int(buff['hit_speed_multiplier'])
            speed=speed-100 if speed>0 else speed
            hit=hit-100 if hit>0 else hit
            spec=SpellSpec(name=key,damage=0,radius_mt=int(row['radius']),radius_y_mt=int(row['radius']),
                crown_tower_percent=crown_percent,pushback_mt=int(row['pushback']),
                life_duration_ms=duration,damage_per_second=dps,hit_frequency_ms=period,
                tower_damage_override=tower_override,
                tower_damage_per_hit=tower_override//ticks if tower_override else 0,
                building_damage_percent=building_percent,area_speed_pct=speed,area_hit_speed_pct=hit,
                area_buff_linger_ms=int(row['buff_time']),attract_percentage=int(buff['attract_percentage']),
                hits_ground=bool(row['hits_ground']),hits_air=bool(row['hits_air']),
                ignore_buildings=bool(row['ignore_buildings']or buff['ignore_buildings']),
                initial_damage=False,can_place_on_water=bool(row['can_place_on_water']),
                deploy_own_side_only=not bool(row['can_deploy_on_enemy_side']))
            accepted[key]=spec
            report.update(supported=True,source_stats_exact=True,raw_buff_damage_per_second=buff['damage_per_second'],
                scaled_damage_per_second=dps,area_damage_tick_ms=period,area_duration_ms=duration,
                area_scan_ms=row['hit_speed'],area_buff_time_ms=row['buff_time'],
                initial_area_damage=False,engine_tick_count_stationary=ticks,
                tower_total_override=tower_override or None,building_damage_percent_source=buff['building_damage_percent'],
                combat_rule_sources=rule.get('sources',[rule['source']]if rule.get('source')else []))
            if int(row['hit_speed'])!=period:
                report['limitations'].append('area_scan_frequency_differs_from_buff_damage_frequency')
            if spec.attract_percentage:
                report['limitations'].extend(['attraction_displacement_uses_existing_uniform_pull_approximation',
                    'controlled_parent_buff_and_pull_mass_interaction_not_verified'])
            if building_percent!=100:
                report['limitations'].append('building_multiplier_interpretation_follows_existing_engine_not_live_verified')
        except (KeyError,ValueError,TypeError)as error:
            report['missing'].append(str(error))
    return PublicAreaSpells(accepted,audit)
