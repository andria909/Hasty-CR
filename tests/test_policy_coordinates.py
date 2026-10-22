"""Observation-only contract checks: no battles, training, or native reader."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from sim.adapter import grid_to_point
from sim.arena import Point, to_tiles
from sim import env
from sim.policy_coordinates import (
    LEGACY_TRAINING, PHYSICAL_LIVE_CELLS_V2, policy_coordinates,
)


def fake_match(entities):
    player = SimpleNamespace(elixir=7000, hand=env.DECK_26[:4],
                             next_card=env.DECK_26[4])
    return SimpleNamespace(
        battle=SimpleNamespace(entities=dict(enumerate(entities))),
        players={1: player, -1: player}, elapsed_ms=3500, play_log=[],
        regen_ms=lambda: 2800,
        tower_fractions=lambda side: {'left': .7, 'right': .8, 'king': 1.0})


def body(pos, side=1, **kwargs):
    return SimpleNamespace(pos=pos, side=side, hitpoints=1234,
                           damage=123, hit_speed_ms=1100, range_mt=4000,
                           speed_mt_per_sec=900, **kwargs)


@pytest.mark.parametrize('side', [1, -1])
def test_all_action_centers_roundtrip_in_v2(side):
    cells = set()
    for y in range(32):
        for x in range(18):
            point = grid_to_point(x, y, side)
            cell = PHYSICAL_LIVE_CELLS_V2.cell(point, side)
            assert cell == (x, y)
            cells.add(cell)
            observed = env.observe(fake_match([body(point, side)]), side,
                                   coordinate_mode='physical_live_cells_v2')
            assert observed['planes'][0, y, x] == 1
            assert observed['planes'][0].sum() == 1
    assert len(cells) == 576


@pytest.mark.parametrize('side', [1, -1])
def test_legacy_aliasing_is_preserved(side):
    cells = {LEGACY_TRAINING.cell(grid_to_point(x, y, side), side)
             for y in range(32) for x in range(18)}
    assert len(cells) == 160


@pytest.mark.parametrize('side', [1, -1])
def test_default_and_explicit_legacy_match_original_whole_observation(side, monkeypatch):
    entities = [body(grid_to_point(x, y, 1), -1 if x % 2 else 1,
                     flying=bool(y % 2), is_building=bool(x % 3 == 0),
                     attacks_air=True, splash_radius_mt=1500)
                for y in range(32) for x in range(18)]
    entities += [body(Point(3500, 6500), is_tower=True),
                 body(Point(1000, 2000), alive=False)]
    match = fake_match(entities)
    default = env.observe(match, side)
    explicit = env.observe(match, side, coordinate_mode='legacy_training')

    class OriginalEncoder:
        def cell(self, point, side):
            x, y = to_tiles(point)
            gx = int(np.clip(round(x), 0, 17))
            gy = int(np.clip(round(31 - y), 0, 31))
            return (17 - gx, 31 - gy) if side < 0 else (gx, gy)

    monkeypatch.setattr(env, 'policy_coordinates', lambda name: OriginalEncoder())
    original = env.observe(match, side)
    for key in ('planes', 'scalars'):
        assert np.array_equal(default[key], explicit[key])
        assert default[key].tobytes() == original[key].tobytes()


@pytest.mark.parametrize('side', [1, -1])
def test_clipping_only_affects_output_not_native_positions(side):
    points = [Point(-1050, 33000), Point(19000, -5),
              Point(3500, 6500), Point(9000, 29000)]
    entities = [body(p) for p in points]
    match = fake_match(entities)
    env.observe(match, side, coordinate_mode='physical_live_cells_v2')
    assert all(entity.pos is point for entity, point in zip(entities, points))
    assert [(p.x, p.y) for p in points] == [(-1050, 33000), (19000, -5),
                                         (3500, 6500), (9000, 29000)]
    expected = [(0, 31), (17, 0)]
    if side < 0:
        expected = [(17-x, 31-y) for x, y in expected]
    assert [PHYSICAL_LIVE_CELLS_V2.cell(p, side) for p in points[:2]] == expected


def test_contract_immutable_explicit_and_partial_observation():
    with pytest.raises(FrozenInstanceError):
        PHYSICAL_LIVE_CELLS_V2.name = 'legacy_training'
    with pytest.raises(ValueError):
        policy_coordinates('physical_live')
    with pytest.raises(ValueError):
        env.observe(None, coordinate_mode='physical_live')
    for name in ('legacy_training', 'physical_live_cells_v2'):
        obs = env.observe(None, coordinate_mode=name)
        assert obs['planes'].shape == (env.NUM_PLANES, 32, 18)
        assert obs['scalars'].shape == (env.NUM_SCALARS,)
        assert not obs['planes'].any() and not obs['scalars'].any()
