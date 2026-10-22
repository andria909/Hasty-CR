"""Do not issue unreachable waypoints inside an allied tower's footprint."""
import pytest

from sim import arena, pathfind
from sim.arena import Point
from sim.env import DECK_26
from sim.match import Match
from sim.public_gamedata import load_public_gamedata


def giant_match(side=-1,x=3500):
    cards,_=load_public_gamedata()
    deck=['giant']+list(DECK_26)[:7]
    decks=(deck,DECK_26) if side==1 else (DECK_26,deck)
    match=Match(cards=cards,decks=decks,seed=42,spells={})
    player=match.players[side]
    player.hand=deck[:4];player.queue=deck[4:];player.elixir=10000
    assert match.play_card(side,'giant',Point(x,1500 if side==-1 else 30500))
    giant=next(e for e in match.battle.entities.values()if e.name=='giant')
    return match,giant


def test_flow_waypoint_excludes_actual_tower_plus_body_radius():
    obstacle=(3500,7500,1770)
    goal=Point(3500,24500)
    waypoint=pathfind.next_step(Point(3807,5747),goal,(obstacle,))
    assert waypoint is not None
    assert arena.distance(waypoint,Point(*obstacle[:2]))>=obstacle[2]
    # This exact old waypoint lies inside the solid disk and was unreachable.
    assert waypoint!=Point(3500,6500)


@pytest.mark.parametrize('side,x',[(-1,3500),(-1,14500),(1,3500),(1,14500)])
def test_giant_from_behind_own_princess_crosses_bridge_without_clipping(side,x):
    match,giant=giant_match(side,x)
    own=[e for e in match.battle.entities.values()if e.is_tower and e.side==side]
    for _ in range(400):
        match.step()
        for tower in own:
            assert arena.distance(giant.pos,tower.pos)>=giant.collision_radius_mt+tower.collision_radius_mt-3
    assert giant.alive
    assert giant.pos.y>arena.RIVER_BOTTOM if side==-1 else giant.pos.y<arena.RIVER_TOP


def test_lane_and_player_reflection_preserve_detour_progress():
    final={}
    for side in (-1,1):
        for x in (3500,14500):
            match,giant=giant_match(side,x)
            for _ in range(400):match.step()
            final[side,x]=giant.pos
    for side in (-1,1):
        left,right=final[side,3500],final[side,14500]
        assert abs(left.x+right.x-18000)<100
        assert abs(left.y-right.y)<100
    for x in (3500,14500):
        top,bottom=final[-1,x],final[1,x]
        # Integer movement and cell-centre routing introduce sub-tile drift.
        assert abs(top.y+bottom.y-32000)<350


def test_full_hold_only_solo_giant_back_finishes_without_timeout():
    from sim.defense_env import DefenseEnv
    from sim.defense_scenarios import Scenario,ScheduledPlay
    scenario=Scenario('giant-back-regression','solo:giant','simple','train','left',
                      (ScheduledPlay('giant',position='back'),))
    env=DefenseEnv(scenario=scenario,split='train',max_ms=180000)
    env.reset(seed=42)
    for _ in range(360):
        _,_,terminal,truncated,info=env.step(0)
        if terminal or truncated:break
    assert terminal and not truncated
    assert info['terminal_reason']=='cleared'
    assert info['metrics']['elapsed_ms']<60000
