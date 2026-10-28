"""Recorded post-river paths, real acquisition, and frozen-physics contracts."""
from copy import deepcopy
import json
import pickle
from pathlib import Path

import pytest

from scripts.gamestate.replay_recorded_episode import replay
from scripts.attack.evaluation_execution import (HOG_CLIPPED,CORRECTED,CURRENT,LEGACY,
    configure_battle,environment_kwargs,training_execution,execution_for,
    validate_execution,check_row_execution)
from sim.arena import Point
from sim.engine import Battle
from sim.entities import make_unit,make_tower
from sim.gamedata import load_gamedata,build_unit_spec
from sim.continuous_push import ContinuousPushEnv

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=json.loads((ROOT/'data/recorded_mechanics_20261028/hog_postriver_sight_clips.json').read_text())


@pytest.mark.parametrize('item',FIXTURE['episodes'],ids=lambda i:i['clip']['name'])
def test_real_postriver_paths_do_not_regress(item):
    clip=item['clip'];before=deepcopy(clip)
    old=replay(clip,battle_options={'hog_sight_mode':'legacy_radial_v1'})
    new=replay(clip,battle_options={'hog_sight_mode':'side_clip_v1'})
    assert clip==before
    assert new['mean_position_error_tiles']<=old['mean_position_error_tiles']+.02
    # Require the measured large false-pull examples to improve, not merely
    # all cases to stay unchanged after an accidentally disconnected flag.
    if old['mean_position_error_tiles']>3:
        assert new['mean_position_error_tiles']<.25


def setup(cannon_x=8500,*,mirror_x=False,mirror_y=False,mode='side_clip_v1',card='hog_rider'):
    cards=load_gamedata(11)
    def p(x,y):return Point(18000-x if mirror_x else x,32000-y if mirror_y else y)
    own=1 if mirror_y else -1
    battle=Battle(hog_sight_mode=mode)
    hog=battle.add(make_unit(1,cards[card].unit,own,p(15356,16753)))
    cannon=battle.add(make_unit(2,cards['cannon'].unit,-own,p(cannon_x,18500)))
    tower=battle.add(make_tower(3,-own,p(14500,25500),10000,0,1000,7500))
    for e in battle.entities.values():e.deploy_remaining_ms=0
    return battle,hog,cannon,tower


@pytest.mark.parametrize('mirror_x,mirror_y',[(False,False),(True,False),(False,True),(True,True)])
@pytest.mark.parametrize('cannon_x,pulls',[(8500,False),(9500,True)])
def test_measured_adjacent_cannon_cells_and_reflections(cannon_x,pulls,mirror_x,mirror_y):
    battle,hog,cannon,tower=setup(cannon_x,mirror_x=mirror_x,mirror_y=mirror_y)
    battle._acquire_target(hog)
    assert hog.target_uid==(cannon.uid if pulls else tower.uid)
    legacy,old,old_cannon,_=setup(cannon_x,mode='legacy_radial_v1',mirror_x=mirror_x,mirror_y=mirror_y)
    legacy._acquire_target(old)
    assert old.target_uid==old_cannon.uid


def test_forward_sight_and_existing_target_are_not_reduced_to_5500():
    battle,hog,cannon,tower=setup()
    hog.pos=Point(14500,15500);cannon.pos=Point(14500,24500)
    battle._acquire_target(hog)
    assert hog.target_uid==cannon.uid
    battle,hog,cannon,tower=setup()
    hog.target_uid=cannon.uid
    battle.now_ms=450 # Force the normal scheduled scan for UID1.
    battle._acquire_target(hog)
    assert hog.target_uid==cannon.uid


@pytest.mark.parametrize('card',['giant','musketeer','knight'])
def test_other_troops_keep_previous_acquisition(card):
    battle,unit,_,_=setup(card=card);battle._acquire_target(unit)
    old,previous,_,_=setup(card=card,mode='legacy_radial_v1');old._acquire_target(previous)
    assert unit.target_uid==previous.target_uid


def test_loader_fields_and_legacy_serialized_state():
    spec=load_gamedata(11)['hog_rider'].unit
    assert (spec.sight_clip_mt,spec.sight_clip_side_mt)==(4000,4000)
    unit=make_unit(1,spec,1,Point(9000,16000))
    assert (unit.sight_clip_mt,unit.sight_clip_side_mt)==(4000,4000)
    local=build_unit_spec('HogRider',dict(SightRange=9500,SightClip=4000,SightClipSide=4000),
                          11,'Rare',{}, {})
    assert (local.sight_clip_mt,local.sight_clip_side_mt)==(4000,4000)
    # Serialized simulator snapshots predate this field: preserve their old
    # radial behavior, even though new standalone Battle defaults are fixed.
    battle=Battle();battle.__dict__.pop('hog_sight_mode')
    assert pickle.loads(pickle.dumps(battle)).hog_sight_mode=='legacy_radial_v1'
    unit.__dict__.pop('sight_clip_side_mt')
    assert pickle.loads(pickle.dumps(unit)).sight_clip_side_mt==0


@pytest.mark.parametrize('execution',[LEGACY,CURRENT,CORRECTED,HOG_CLIPPED])
def test_explicit_execution_selects_actual_training_and_duel_physics(execution):
    expected=execution.get('hog_sight_mode','legacy_radial_v1')
    battle=Battle();configure_battle(battle,execution)
    assert battle.hog_sight_mode==expected
    kwargs=environment_kwargs(execution)
    env=ContinuousPushEnv(**kwargs);_,info=env.reset(seed=1)
    assert env.match.battle.hog_sight_mode==expected
    row=dict(info['metrics'],execution=execution)
    check_row_execution(row,execution)
    if execution is HOG_CLIPPED:
        config=dict(kwargs,continuous_profile=env.profile)
        assert training_execution(config)==HOG_CLIPPED
        row['hog_sight_mode']='legacy_radial_v1'
        with pytest.raises(ValueError,match='Actual battle geometry'):
            check_row_execution(row,execution)


def test_execution_version_and_incomplete_new_flags_fail_closed():
    assert execution_for(dict(schema='bound-full-candidate-evaluation-v4',execution=HOG_CLIPPED))==HOG_CLIPPED
    with pytest.raises(ValueError):execution_for(dict(schema='bound-full-candidate-evaluation-v4',execution=CORRECTED))
    bad=dict(HOG_CLIPPED);bad.pop('hog_sight_mode')
    with pytest.raises(ValueError):validate_execution(bad)
    with pytest.raises(ValueError):validate_execution(dict(HOG_CLIPPED,hog_sight_mode='guess'))
    with pytest.raises(ValueError):Battle(hog_sight_mode='guess')
