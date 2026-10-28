"""Native hit/miss fixtures exercise real spell damage and its predictor."""
from dataclasses import replace
from copy import deepcopy
import json
import math
import pickle
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from scripts.gamestate.audit_fireball_crown_overlap import inspect_clip, replay_flight_timing
from sim.arena import Point
from sim.attack_env import AttackEnv
from sim.engine import Battle
from sim.entities import make_tower, make_unit
from sim.gamedata import load_gamedata
from sim.spells import apply_spell, cast_spell, load_spells, spell_overlaps

ROOT = Path(__file__).resolve().parents[1]
REPORTS = [ROOT/'reports/defense_plan'/name for name in (
    'native_fireball_crown_overlap_verified_20261028.json',
    'camp_fireball_crown_overlap_20261028.json')]
CASES = [c for report in REPORTS for c in json.loads(report.read_text())['cases']
         if c['accepted']]


@pytest.mark.parametrize('case', CASES, ids=[f"{c['projectile']['entity_id']}-{c['last_frame']}-{c['target_before']['name']}" for c in CASES])
def test_recorded_native_hit_or_miss(case):
    native = case['target_before']
    b = Battle()
    target = b.add(make_tower(1, 1, Point(native['x'], native['y']), 10000,
                             0, 1000, 0, king=native['name'] == 'KingTower'))
    aim = Point(case['aim']['x'], case['aim']['y'])
    spec = load_spells(11)['fireball']
    predicted = AttackEnv._spell_covers(spec, aim, target)
    apply_spell(b, spec, aim, -1)
    assert (target.hitpoints < 10000) == (case['observed_hp_drop'] > 0)
    assert predicted == (target.hitpoints < 10000)
    # Native level/damage scaling is intentionally not asserted here.


@pytest.mark.parametrize('dx,dy,hit', [(2000,3000,True),(3000,2000,True),
    (0,4000,False),(1000,4000,False),(4000,3000,False),(3000,3000,False)])
@pytest.mark.parametrize('sx,sy', [(1,1),(1,-1),(-1,1),(-1,-1)])
def test_corner_correction_does_not_expand_every_direction(dx, dy, hit, sx, sy):
    target = make_tower(1, 1, Point(9000,16000), 10000, 0, 1000, 0)
    spec = load_spells(11)['fireball']
    aim = Point(9000+sx*dx,16000+sy*dy)
    assert spell_overlaps(spec, aim, target) == hit
    assert target.collision_radius_mt == 1000


def test_legal_cell_centers_add_only_eight_previously_missed_corner_positions():
    target=make_tower(1,1,Point(9500,16500),10000,0,1000,0)
    new=load_spells(11)['fireball']
    old=replace(new,princess_hitbox_mode='collision_circle_v1')
    changed=set()
    for dx in range(-9,9):
        for dy in range(-16,16):
            aim=Point(target.pos.x+dx*1000,target.pos.y+dy*1000)
            previous,current=spell_overlaps(old,aim,target),spell_overlaps(new,aim,target)
            assert not(previous and not current)
            if current!=previous:changed.add((dx,dy))
    assert changed=={(sx*x,sy*y) for x,y in [(2,3),(3,2)]
                     for sx in [-1,1] for sy in [-1,1]}


def test_spell_travel_resolves_corner_damage_at_arrival_and_records_hit():
    battle = Battle()
    target = battle.add(make_tower(1,1,Point(3500,25500),10000,0,1000,0))
    spec = load_spells(11)['fireball']
    cast_spell(battle,spec,Point(5500,22500),-1,origin=Point(9000,3000))
    due = battle.spell_cast_log[0]['due_ms']
    while battle.now_ms+50 < due:
        battle.step(50)
        assert target.hitpoints == 10000
    battle.step(50)
    assert target.hitpoints == 10000-spec.damage_to(target)
    event = battle.spell_cast_log[0]
    assert event['resolved'] and event['hits'][0]['target_uid'] == target.uid


def test_only_calibrated_princess_spell_geometry_changes():
    cards = load_gamedata(11)
    fire = load_spells(11)['fireball']
    old = replace(fire,princess_hitbox_mode='collision_circle_v1')
    point = Point(11000,19000)
    princess = make_tower(1,1,Point(9000,16000),10000,0,1000,0)
    king = make_tower(2,1,Point(9000,16000),10000,0,1000,0,king=True)
    assert spell_overlaps(fire,point,princess) and not spell_overlaps(old,point,princess)
    for entity in [king,make_unit(3,cards['cannon'].unit,1,Point(9000,16000)),
                   make_unit(4,cards['knight'].unit,1,Point(9000,16000))]:
        assert spell_overlaps(fire,point,entity) == spell_overlaps(old,point,entity)
    assert load_spells(11)['zap'].princess_hitbox_mode == 'collision_circle_v1'
    # Old boundary snapshots don't silently acquire the new calibrated mode.
    old_serialized = replace(fire)
    old_serialized.__dict__.pop('princess_hitbox_mode')
    restored = pickle.loads(pickle.dumps(old_serialized))
    assert not spell_overlaps(restored,point,princess)


def test_public_and_client_spell_loaders_select_the_same_mode(tmp_path,monkeypatch):
    assert load_spells(11,root=tmp_path/'missing')['fireball'].princess_hitbox_mode == 'rectangle_v1'
    import sim.gamedata
    monkeypatch.setattr(sim.gamedata,'load_gamedata',lambda **kw:
        {'fireball':NS(unit=None,form='Normal',rarity='Rare')})
    characters=tmp_path/'characters';characters.mkdir()
    (characters/'fireballspell.toml').write_text(
        '[PROJECTILE.FireballSpell]\nDamage = 100\nRadius = 2500\nSpeed = 600\n')
    assert load_spells(11,root=tmp_path,scale=lambda damage,*_:damage)['fireball'].princess_hitbox_mode == 'rectangle_v1'


def synthetic_native_flight():
    origin, aim = dict(x=9000,y=3000),dict(x=5500,y=22500)
    length=math.hypot(aim['x']-origin['x'],aim['y']-origin['y'])
    king=dict(entity_id=1,address='king',name='KingTower',side=0,hp=5000,**origin)
    tower=dict(entity_id=2,address='tower',name='PrincessTower',side=1,hp=1000,x=3500,y=25500)
    shot=dict(entity_id=10,address='shot',name='FireballSpell',side=0,hp=0,
              target_position_candidate=aim,source_position_fields=origin)
    frames,points=[],[]
    for tick in range(7):
        bodies={1:deepcopy(king),2:dict(tower,hp=891 if tick>=5 else 1000)}
        if tick<6:
            remaining=(5-tick)*600
            x=round(aim['x']-(aim['x']-origin['x'])*remaining/length)
            y=round(aim['y']-(aim['y']-origin['y'])*remaining/length)
            bodies[10]=dict(deepcopy(shot),x=x,y=y)
            points.append([tick,x,y])
        frames.append(dict(tick=tick,world=('world',),bodies=bodies))
    clip=dict(source='synthetic',first_frame=0,last_frame=5,
              identity=[10,'shot','FireballSpell',0],points=points)
    return clip,frames


def test_native_audit_anchors_before_explosion_without_using_hp():
    clip,frames=synthetic_native_flight()
    cases,rejection=inspect_clip(clip,frames)
    assert rejection is None and len(cases)==1
    case=cases[0]
    # Last visible point is already at the blast with damage applied. It must
    # not erase the real hit by using that sample as the HP-before baseline.
    assert case['preimpact_frame']==4 and case['last_frame']==5
    assert case['accepted'] and case['observed_hp_drop']==109
    frames[5]['bodies'][2]['hp']=1000
    delayed,_=inspect_clip(clip,frames)
    assert delayed[0]['preimpact_frame']==4
    assert delayed[0]['observed_hp_drop']==109


@pytest.mark.parametrize('change,reason',[
    ('aim','unstable_aim_or_identity'),
    ('address','unstable_aim_or_identity'),
    ('world','flight_discontinuity'),
    ('clock','flight_discontinuity'),
    ('missing','flight_discontinuity'),
    ('long_disappearance','no_contiguous_disappearance'),
    ('changed_points','manifest_points_disagree'),
])
def test_native_audit_rejects_ambiguous_or_mismatched_flights(change,reason):
    clip,frames=synthetic_native_flight()
    if change=='aim':frames[2]['bodies'][10]['target_position_candidate']['x']+=1000
    elif change=='address':frames[2]['bodies'][10]['address']='different-shot'
    elif change=='world':frames[2]['world']=('other-world',)
    elif change=='clock':frames[2]['tick']=20
    elif change=='missing':frames[2]=None
    elif change=='long_disappearance':frames[6]['tick']=11
    elif change=='changed_points':clip['points'][2][1]+=1000
    assert inspect_clip(clip,frames)==(None,reason)


def test_native_audit_does_not_attribute_damage_with_another_hostile_nearby():
    clip,frames=synthetic_native_flight()
    frames[5]['bodies'][20]=dict(entity_id=20,address='hog',name='HogRider',
                               side=0,hp=100,x=3500,y=24500)
    cases,rejection=inspect_clip(clip,frames)
    assert rejection is None and len(cases)==1
    assert cases[0]['observed_hp_drop']==109
    assert not cases[0]['accepted'] and cases[0]['threats']==[(20,'HogRider')]


def test_native_birth_phase_and_actual_delayed_resolution_agree():
    case=CASES[0]
    timing=replay_flight_timing(case['projectile'],case['flight_points'],
        case['origin'],case['aim'],*case['interval_ticks'])
    assert timing['accepted'] and timing['engine_in_observed_impact_window']
    changed=dict(case['projectile'])
    changed['deployment_tick_candidate']-=3
    rejected=replay_flight_timing(changed,case['flight_points'],
        case['origin'],case['aim'],*case['interval_ticks'])
    assert rejected['reason']=='birth_candidate_disagrees_with_flight'
