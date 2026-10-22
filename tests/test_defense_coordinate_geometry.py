import numpy as np
import pytest

from sim import arena
from sim.adapter import grid_to_point
from sim.defense_env import DefenseEnv
from sim.env import _placeable


def test_native_pocket_cache_and_mask_agree_with_actual_placement():
    candidate = DefenseEnv(geometry='native_anchors_candidate_v1',
                           coordinate_mode='physical_live_cells_v2')
    legacy = DefenseEnv()
    for env in (candidate, legacy):
        env.reset(seed=42)
        env.match.towers[-1]['left'].hitpoints = 0
        env.match.players[1].hand[0] = 'knight'
        env.match.players[1].elixir = 10000
    point = grid_to_point(3, 5, 1)
    for env, expected in ((candidate, True), (legacy, False)):
        assert bool(env._action_mask()[1+5*18+3]) is expected
        assert env.match.play_card(1, 'knight', point) is expected
    # Interleaving profiles cannot reuse the other's deployment cache entry.
    native = _placeable(1, ('left',), False, geometry=candidate.geometry)
    old = _placeable(1, ('left',), False, geometry=legacy.geometry)
    assert native[5, 3] and not old[5, 3]
    assert not np.array_equal(native, old)
    for side in (-1, 1):
        for profile in (candidate.geometry, legacy.geometry):
            mask = _placeable(side, ('left',), False, geometry=profile)
            for y, x in np.ndindex(mask.shape):
                assert mask[y, x] == arena.deploy_area_ok(
                    grid_to_point(x, y, side), side, ('left',), geometry=profile)


def test_snapshot_cannot_silently_change_geometry_or_coordinate_contract():
    env = DefenseEnv(geometry='native_anchors_candidate_v1',
                     coordinate_mode='physical_live_cells_v2')
    observation, info = env.reset(seed=42)
    snapshot = env.snapshot()
    same = DefenseEnv(geometry='native_anchors_candidate_v1',
                      coordinate_mode='physical_live_cells_v2')
    restored, restored_info = same.restore(snapshot)
    assert np.array_equal(restored['planes'], observation['planes'])
    assert np.array_equal(restored_info['action_mask'], info['action_mask'])
    assert same.match.geometry == same.match.battle.geometry == env.geometry
    with pytest.raises(ValueError, match='contract differs'):
        DefenseEnv().restore(snapshot)
    with pytest.raises(ValueError, match='contract differs'):
        DefenseEnv(geometry='native_anchors_candidate_v1').restore(snapshot)
