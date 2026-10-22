"""Offline ARM64-shaped fixtures: no device, inference, or stepping."""
from copy import deepcopy

import numpy as np
import pytest

from tests.test_memory_model_bridge import fixture
from scripts.gamestate.model_bridge import NativeModelAdapter, MemoryPolicy, MissingModelInput
from sim import arena
from sim.env import legal_mask, observe


@pytest.mark.parametrize('rotated', [False, True])
@pytest.mark.parametrize('own_side', [0, 1])
@pytest.mark.parametrize('geometry', ['legacy_v1', 'native_anchors_candidate_v1'])
def test_v2_physical_observation_and_mechanical_mask_parity(rotated, own_side, geometry):
    match, snapshot, own = fixture(rotated=rotated)
    match.geometry = arena.resolve_geometry(geometry)
    # Native measured crown-anchor fixture; preserve positions independently of
    # selected profile. Anchors do not prove collision/deployment parity.
    for row in snapshot['units']:
        if own_side:
            row['side'] = 1 - row['side']
        if row['name'] == 'KingTower':
            row['x'] = 9000
            row['y'] = 3000 if row['y'] < 16000 else 29000
        elif row['name'] == 'PrincessTower':
            row['y'] = 6500 if row['y'] < 16000 else 25500
        uid = row['entity_id'] - 5_000_000
        match.battle.entities[uid].pos = arena.Point(
            arena.WIDTH-row['x'] if rotated else row['x'],
            arena.HEIGHT-row['y'] if rotated else row['y'])
    original = deepcopy(snapshot)
    view = NativeModelAdapter(own_side=own_side, geometry=geometry,
        coordinate_mode='physical_live_cells_v2').encode(snapshot, own)
    expected = observe(match, coordinate_mode='physical_live_cells_v2')
    for key in ('planes', 'scalars'):
        np.testing.assert_array_equal(view.observation[key], expected[key])
    np.testing.assert_array_equal(view.action_mask,
                                 legal_mask(match, match.cards, strategic=False))
    assert snapshot == original
    assert view.metadata['action_mask_mode'] == 'mechanical'
    assert view.metadata['geometry'] == geometry
    assert view.metadata['geometry_sha256'] == match.geometry.fingerprint
    assert view.rotated == rotated


@pytest.mark.parametrize('mode', ['legacy_training', 'physical_live'])
def test_existing_modes_default_mask_unchanged(mode):
    _, snapshot, own = fixture()
    default = NativeModelAdapter(own_side=0, coordinate_mode=mode).encode(snapshot, own)
    explicit = NativeModelAdapter(own_side=0, coordinate_mode=mode,
        action_mask_mode='legacy_strategic').encode(snapshot, own)
    for key in ('planes', 'scalars'):
        assert default.observation[key].tobytes() == explicit.observation[key].tobytes()
    assert default.action_mask.tobytes() == explicit.action_mask.tobytes()


def test_v2_strategic_rejected_and_mask_option_validated():
    with pytest.raises(ValueError, match='not implemented'):
        NativeModelAdapter(own_side=0, coordinate_mode='physical_live_cells_v2',
                           action_mask_mode='legacy_strategic')
    with pytest.raises(ValueError, match='mask mode'):
        NativeModelAdapter(own_side=0, action_mask_mode='typo')


def test_checkpoint_metadata_binding_without_loading_weights():
    _, snapshot, own = fixture()
    view = NativeModelAdapter(own_side=0, coordinate_mode='physical_live_cells_v2').encode(snapshot, own)
    policy = object.__new__(MemoryPolicy)
    policy.input_contract = {}  # old v25 does not invent new metadata restrictions
    policy.validate_view_contract(view)
    policy.input_contract = {k: view.metadata[k] for k in
        ('coordinate_mode', 'geometry', 'geometry_sha256', 'action_mask_mode')}
    policy.validate_view_contract(view)
    for key in policy.input_contract:
        original = policy.input_contract[key]
        policy.input_contract[key] = 'different'
        with pytest.raises(MissingModelInput, match=key):
            policy.validate_view_contract(view)
        policy.input_contract[key] = original
