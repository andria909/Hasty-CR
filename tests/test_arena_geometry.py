"""The board's published dimensions, checked against the ones we simulate.

Arena geometry is not in the shipped data - `arenas.csv` is trophy and
progression metadata, and the playfield is identical in every arena, so it
lives in the engine binary. That binary is packed: `libg.so` is 28MB, stripped,
with its strings encrypted into 8-character noise, and `base.apk` ships an
encrypted `.text.ecc` section decrypted at runtime. Reading the constants out
of it is a reverse-engineering project, not an afternoon.

The dimensions are published, though, and that is enough to check them.

  https://clashroyale.fandom.com/wiki/Arenas

This caught a real error. Bridges are two tiles wide in Arenas 1-6 and 8-9, and
*three* tiles wide in Arena 7 and Arenas 10-23 - which is every arena this
project cares about, Path of Legends included. The simulator modelled the
narrow bridge, funnelling a push harder than the real board does, and it
matters most for the bridge-spam decks the opponent pool is full of.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim import arena                                          # noqa: E402


def test_the_board_is_the_published_size():
    """32 tiles long, 18 wide."""
    assert (arena.TILES_W, arena.TILES_H) == (18, 32)


def test_the_bridges_are_the_wide_kind():
    """Three tiles, which is Arena 7 and 10-23 - everything that matters."""
    assert arena.BRIDGE_TILES == 3
    assert arena.BRIDGE_HALF_WIDTH == 3 * arena.MT // 2


def test_each_bridge_spans_three_whole_tiles():
    for centre in arena.BRIDGE_X:
        low = centre - arena.BRIDGE_HALF_WIDTH
        high = centre + arena.BRIDGE_HALF_WIDTH
        assert (high - low) == arena.BRIDGE_TILES * arena.MT
        # Whole tiles, not a strip straddling two halves.
        assert low % arena.MT == 0 and high % arena.MT == 0


def test_there_are_two_bridges_and_they_are_symmetric():
    assert len(arena.BRIDGE_X) == 2
    left, right = sorted(arena.BRIDGE_X)
    assert left + right == arena.WIDTH, (left, right, arena.WIDTH)


def test_the_river_band_is_centred_on_the_halfway_line():
    """An offset band once cost one seat a whole tile per push."""
    assert arena.RIVER_TOP < arena.RIVER_Y < arena.RIVER_BOTTOM
    assert arena.RIVER_Y - arena.RIVER_TOP == arena.RIVER_BOTTOM - arena.RIVER_Y
    assert arena.RIVER_Y * 2 == arena.HEIGHT


def test_the_towers_mirror_across_the_river():
    for lane in ("left", "right"):
        ours = arena.ALLY_PRINCESS[lane]
        theirs = arena.ENEMY_PRINCESS[lane]
        assert ours.x == theirs.x, lane
        assert ours.y + theirs.y == arena.HEIGHT, lane
    assert arena.ALLY_KING.x == arena.ENEMY_KING.x
    assert arena.ALLY_KING.y + arena.ENEMY_KING.y == arena.HEIGHT


def test_a_unit_on_the_bridge_can_cross_and_one_beside_it_cannot():
    from sim.arena import Point
    for centre in arena.BRIDGE_X:
        assert arena.on_bridge(Point(centre, arena.RIVER_Y))
        # A tile and a half out is the edge; two tiles out is the river.
        assert not arena.on_bridge(Point(centre + 2 * arena.MT, arena.RIVER_Y))


# Opt-in per-match anchor profile regressions.
from copy import deepcopy
from dataclasses import FrozenInstanceError,replace
import pickle
from types import SimpleNamespace

import pytest

from sim import arena
from sim.engine import Battle
from sim.env import DECK_26
from sim.match import Match
from sim.defense_assets import load_defense_assets
from sim.runner import resolve_deck


def make_match(profile=None):
    cards,spells,_,_=load_defense_assets()
    cards.update(resolve_deck(cards,DECK_26))
    options={}if profile is None else {'geometry':profile}
    return Match(cards=cards,decks=(list(DECK_26),list(DECK_26)),seed=42,spells=spells,**options)


def anchors(match):
    return {side:{lane:t.pos for lane,t in towers.items()}for side,towers in match.towers.items()}


def test_default_geometry_keeps_exact_legacy_anchors_and_initial_state():
    default=make_match();explicit=make_match('legacy_v1')
    assert default.geometry is arena.LEGACY_GEOMETRY
    assert anchors(default)==anchors(explicit)=={
        1:dict(left=arena.Point(3500,24500),right=arena.Point(14500,24500),king=arena.Point(9500,28500)),
        -1:dict(left=arena.Point(3500,7500),right=arena.Point(14500,7500),king=arena.Point(9500,3500))}
    for side in (1,-1):
        assert default.players[side]==explicit.players[side]
        for lane in ('left','right','king'):
            assert default.towers[side][lane]==explicit.towers[side][lane]
    assert Battle().geometry is arena.LEGACY_GEOMETRY


def test_native_geometry_is_per_match_and_has_measured_symmetric_anchors():
    legacy=make_match();native=make_match('native_anchors_candidate_v1')
    assert native.geometry is native.battle.geometry
    assert legacy.geometry is legacy.battle.geometry is arena.LEGACY_GEOMETRY
    assert anchors(native)=={
        1:dict(left=arena.Point(3500,25500),right=arena.Point(14500,25500),king=arena.Point(9000,29000)),
        -1:dict(left=arena.Point(3500,6500),right=arena.Point(14500,6500),king=arena.Point(9000,3000))}
    for lane,other in [('left','right'),('right','left'),('king','king')]:
        p=native.towers[1][lane].pos
        assert arena.Point(arena.WIDTH-p.x,arena.HEIGHT-p.y)==native.towers[-1][other].pos
    assert legacy.towers[1]['king'].pos==arena.Point(9500,28500)
    native.geometry.princess(1)['left']=arena.Point(0,0)
    assert native.geometry.princess(1)['left']==arena.Point(3500,25500)
    assert 'native_entity_anchor_hitbox_offset_unverified'in native.geometry.limitations


@pytest.mark.parametrize('side,lane,point',[
    (1,'left',arena.Point(3500,6000)),(-1,'right',arena.Point(14500,26000))])
def test_profile_deployment_and_match_play_share_the_same_pocket(side,lane,point):
    legacy=make_match();native=make_match('native_anchors_candidate_v1')
    for match in (legacy,native):
        match.towers[-side][lane].hitpoints=0
        match.players[side].hand[0]='knight'
        match.players[side].elixir=10000
    assert arena.deploy_area_ok(point,side,[lane],geometry=native.geometry)
    assert not arena.deploy_area_ok(point,side,[lane],geometry=legacy.geometry)
    assert not arena.deploy_area_ok(point,side,[],geometry=native.geometry)
    assert native.play_card(side,'knight',point)
    assert not legacy.play_card(side,'knight',point)


def test_geometry_is_immutable_pickle_deepcopy_safe_and_rejects_unknown_terrain():
    profile=arena.NATIVE_ANCHORS_CANDIDATE_GEOMETRY
    with pytest.raises(FrozenInstanceError):profile.profile_id='changed'
    assert pickle.loads(pickle.dumps(profile))==profile
    clone=deepcopy(make_match(profile))
    assert clone.geometry==profile and clone.battle.geometry is clone.geometry
    assert anchors(clone)[1]['king']==arena.Point(9000,29000)
    assert profile.fingerprint==pickle.loads(pickle.dumps(profile)).fingerprint
    assert profile.fingerprint!=arena.LEGACY_GEOMETRY.fingerprint
    with pytest.raises(ValueError,match='terrain'):
        replace(profile,terrain_id='different_river')
    with pytest.raises(ValueError,match='Unknown'):
        Battle(geometry='not-a-profile')
    with pytest.raises(ValueError,match='Unknown'):
        make_match('not-a-profile')


def test_drill_anchor_lookup_uses_battle_profile_without_global_mutation():
    native=Battle(geometry='native_anchors_candidate_v1')
    legacy=Battle()
    entity=SimpleNamespace(side=1,pos=arena.Point(4500,6500))
    assert native._drill_resurface_at(entity)==arena.Point(3500,7500)
    assert legacy._drill_resurface_at(entity)==arena.Point(4500,8500)
    assert arena.ENEMY_PRINCESS['left']==arena.Point(3500,7500)


def test_older_snapshot_state_without_geometry_restores_legacy_profile():
    battle=Battle()
    del battle.geometry
    restored=pickle.loads(pickle.dumps(battle))
    assert restored.geometry is arena.LEGACY_GEOMETRY
    match=make_match()
    state=dict(match.__dict__)
    state.pop('geometry')
    match.__setstate__(state)
    assert match.geometry is match.battle.geometry is arena.LEGACY_GEOMETRY


def test_uppercut_fallback_reads_own_instance_king_anchor():
    source=SimpleNamespace(side=1,uppercut_push_mt=1000,uppercut_flight_ms=100,uppercut_root_ms=0)
    endpoints=[]
    for profile in (arena.LEGACY_GEOMETRY,arena.NATIVE_ANCHORS_CANDIDATE_GEOMETRY):
        battle=Battle(geometry=profile)
        target=SimpleNamespace(pos=arena.Point(9000,28000))
        battle._start_uppercut(source,target)
        endpoints.append(battle.forced_moves[0][2])
    assert endpoints==[arena.Point(9707,28707),arena.Point(9000,29000)]


def test_burrow_travel_uses_profile_home_without_mutating_card_cache():
    for profile in (arena.LEGACY_GEOMETRY,arena.NATIVE_ANCHORS_CANDIDATE_GEOMETRY):
        match=make_match(profile)
        card=match.cards['knight']
        match.cards=dict(match.cards)
        match.cards['knight']=replace(card,unit=replace(card.unit,burrow_speed_mt_per_sec=1000,deploy_time_ms=0))
        match.players[1].hand[0]='knight'
        match.players[1].elixir=10000
        at=arena.Point(9000,27000)
        first=match.battle.next_uid
        assert match.play_card(1,'knight',at)
        unit=match.battle.entities[first]
        assert unit.deploy_remaining_ms==arena.distance(profile.king(1),unit.pos)
