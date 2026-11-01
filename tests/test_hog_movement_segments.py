import pytest

from scripts.gamestate.analyze_hog_jump_segments import decode,segment
from scripts.gamestate.experimental_hog_edge_jump import EdgeJumpBattle
from sim.arena import Point
from sim.entities import make_tower,make_unit
from sim.gamedata import load_gamedata


def test_state_fit_preserves_stops_and_switches_without_speed_prefilter():
    points=[dict(tick=0,x=0,y=0)];y=0
    for step in [0]*8+[120]*8+[160]*12+[120]*8:
        y+=step;points.append(dict(tick=len(points),x=0,y=y))
    states,_=decode(points)
    assert states==[0]*8+[1]*8+[2]*12+[1]*8
    fit=segment(points,16,28)
    assert fit['fitted_speed_mt_per_sec']==pytest.approx(3200)
    assert fit['straight_line_rmse_tiles']<1e-10
    assert fit['fixed_speed_mean_error_tiles']['3200']<1e-10


@pytest.mark.parametrize('ticks',[[1,1],[2,1],[1,7]])
def test_clock_gap_or_duplicate_cannot_be_fit_as_a_jump(ticks):
    with pytest.raises(ValueError):decode([dict(tick=t,x=0,y=100*i)for i,t in enumerate(ticks)])


def run(side,x,y,target_y):
    battle=EdgeJumpBattle();spec=load_gamedata(11)['hog_rider'].unit
    hog=battle.add(make_unit(1,spec,side,Point(x,y)))
    tower=battle.add(make_tower(2,-side,Point(14500 if x>9000 else 3500,target_y),10000,0,1000,0))
    hog.target_uid=tower.uid;hog.deploy_remaining_ms=0
    positions=[]
    for _ in range(20):battle._move(hog,50);positions.append(hog.pos)
    return battle,hog,positions


def test_experimental_edge_route_is_mirrored_and_lands():
    a,ha,pa=run(-1,16500,14500,25500)
    b,hb,pb=run(1,16500,17500,6500)
    assert ha.uid in a.experimental_completed and hb.uid in b.experimental_completed
    # First airborne steps are exactly symmetric; normal post-landing routing
    # remains the existing engine rather than being overridden by this test.
    assert all(x.x==y.x and x.y+y.y==32000 for x,y in zip(pa[:12],pb[:12]))


def test_same_half_target_does_not_trigger_a_river_jump():
    battle,hog,_=run(-1,16500,14500,12000)
    assert not battle.experimental_jumps and hog.uid not in battle.experimental_completed


def test_restricted_prototype_waits_until_takeoff_zone_and_requires_crown_target():
    battle=EdgeJumpBattle(takeoff_offset_mt=1500,crown_target_only=True)
    cards=load_gamedata(11)
    hog=battle.add(make_unit(1,cards['hog_rider'].unit,1,Point(1500,17900)))
    tower=battle.add(make_tower(2,-1,Point(3500,6500),10000,0,1000,0))
    hog.target_uid=tower.uid;hog.deploy_remaining_ms=0
    battle._move(hog,50)
    assert not battle.experimental_jumps
    cannon=battle.add(make_unit(3,cards['cannon'].unit,-1,Point(8500,13500)))
    hog.pos=Point(1500,17499);hog.target_uid=cannon.uid
    battle._move(hog,50)
    assert not battle.experimental_jumps
