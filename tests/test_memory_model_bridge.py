from pathlib import Path

import numpy as np
import pytest

from scripts.gamestate.elixir import ElixirTracker
from scripts.gamestate.model_bridge import (
    MemoryPolicy, MissingModelInput, NativeModelAdapter, OwnState)
from sim import arena
from sim.env import ClashEnv, legal_mask, observe
from sim.entities import make_unit


def fixture(*, rotated=False):
    env = ClashEnv(seed=17, opponent='brain')
    env.reset()
    m = env.match
    m.battle.add(make_unit(0, m.cards['knight'].unit, 1, arena.tile(5, 21)))
    m.battle.add(make_unit(0, m.cards['musketeer'].unit, -1, arena.tile(8, 10)))
    m.battle.add(make_unit(0, m.cards['cannon'].unit, -1, arena.tile(9, 12)))
    # Damaged bodies must retain actual HP rather than nominal maximum HP.
    for entity in m.battle.entities.values():
        entity.hitpoints = max(1, entity.hitpoints - 27)
    player = m.players[1]
    player.hand = ['hog_rider', 'fireball', 'knight', 'skeletons']
    player.queue = ['zap', 'cannon', 'electro_spirit', 'musketeer']
    own = OwnState(0, tuple(player.hand), player.next_card,
                   ElixirTracker.from_match_start().estimate(), (), True, battle_active=True)
    rows = []
    for entity in m.battle.entities.values():
        name = ('KingTower' if entity.name == 'king_tower' else 'PrincessTower') if entity.is_tower else entity.name
        x, y = entity.pos.x, entity.pos.y
        if rotated:
            x, y = arena.WIDTH - x, arena.HEIGHT - y
        rows.append(dict(entity_id=5_000_000 + entity.uid, name=name,
                         side=0 if entity.side == 1 else 1, x=x, y=y,
                         hp=entity.hitpoints, max_hp=entity.max_hitpoints))
    return m, {'status': 'ok', 'units': rows}, own


@pytest.mark.parametrize('enemy_present',[False,True])
def test_joint_live_appends_only_public_task_and_hidden_deadline_zeros(enemy_present):
    import torch
    _,row,own=fixture()
    if not enemy_present:
        row['units']=[u for u in row['units']if u['side']==0 or u['name']in ('KingTower','PrincessTower')]
    view=NativeModelAdapter(own_side=0,coordinate_mode='physical_live_cells_v2').encode(row,own)
    class Recorder:
        def __call__(self,planes,scalars):
            self.scalars=scalars.clone()
            return torch.zeros((1,len(view.action_mask))),torch.zeros(1)
    policy=MemoryPolicy.__new__(MemoryPolicy)
    policy.torch=torch;policy.joint=True;policy.input_contract={};policy.network=Recorder()
    policy.decide(view)
    values=policy.network.scalars.numpy()[0]
    assert values.shape==(309,)
    np.testing.assert_array_equal(values[:306],view.observation['scalars'])
    np.testing.assert_array_equal(values[-3:],[0.,0.,float(not enemy_present)])


@pytest.mark.parametrize('rotated', [False, True])
def test_memory_encoding_matches_training_exactly(rotated):
    match, row, own = fixture(rotated=rotated)
    view = NativeModelAdapter(own_side=0).encode(row, own)
    expected = observe(match, 1)
    np.testing.assert_array_equal(view.observation['planes'], expected['planes'])
    np.testing.assert_array_equal(view.observation['scalars'], expected['scalars'])
    np.testing.assert_array_equal(view.action_mask, legal_mask(match, match.cards, 1))
    assert view.rotated == rotated


def test_world_placement_roundtrip_retains_training_convention():
    from sim.adapter import grid_to_point
    for rotated in (False, True):
        _, row, own = fixture(rotated=rotated)
        view = NativeModelAdapter(own_side=0).encode(row, own)
        action = next(int(i) for i in np.flatnonzero(view.action_mask) if i)
        placement = view.placement_world(action)
        point = grid_to_point(placement['grid_x'], placement['grid_y'], 1)
        assert placement['x'] == (arena.WIDTH - point.x if rotated else point.x)
        assert placement['y'] == (arena.HEIGHT - point.y if rotated else point.y)


def test_unknown_hp_and_missing_tower_fail_closed():
    _, row, own = fixture()
    row['units'][-1]['hp'] = -1
    with pytest.raises(MissingModelInput, match='HP'):
        NativeModelAdapter(own_side=0).encode(row, own)
    _, row, own = fixture()
    row['units'] = [u for u in row['units'] if not (u['side'] == 1 and u['name'] == 'PrincessTower' and u['x'] < 9000)]
    with pytest.raises(MissingModelInput, match='Missing tower'):
        NativeModelAdapter(own_side=0).encode(row, own)


def test_incomplete_history_holds_instead_of_inventing_complete_inputs():
    from dataclasses import replace
    _, row, own = fixture()
    view = NativeModelAdapter(own_side=0).encode(row, replace(own, public_history_complete=False))
    assert np.flatnonzero(view.action_mask).tolist() == [0]
    assert 'incomplete_opponent_play_history' in view.metadata['blockers']


def test_unknown_unit_never_substitutes_card_source():
    _, row, own = fixture()
    row['units'][-1]['name'] = 'UnmappedNewChampion'
    row['units'][-1]['card_id'] = 26000000
    with pytest.raises(MissingModelInput, match='No verified unit profile'):
        NativeModelAdapter(own_side=0).encode(row, own)


def test_v25_runs_on_memory_encoded_observation_without_taps():
    checkpoint = Path('checkpoints/portable_v25/portable_v25_cannon_economy_best.pt')
    if not checkpoint.exists():
        pytest.skip('Local v25 checkpoint is not distributed')
    import torch
    torch.set_num_threads(1)
    _, row, own = fixture()
    view = NativeModelAdapter(own_side=0).encode(row, own)
    decision = MemoryPolicy(checkpoint).decide(view)
    assert view.action_mask[decision['action']]
    assert np.isfinite(decision['value'])


def test_real_native_body_fixtures_with_explicit_synthetic_own_context():
    """Tests recorded coordinates/HP/names; never claims to recover a real hand."""
    import json
    from dataclasses import replace
    payload = json.loads(Path('data/validation/parser_memory_alignment_20261022.json').read_text())
    _, _, own = fixture()
    for snapshot in payload['cases']:
        view = NativeModelAdapter(own_side=1).encode(snapshot, replace(own, battle_active=False))
        assert view.observation['planes'].shape == (20, 32, 18)
        assert view.observation['scalars'].shape == (306,)
        assert np.isfinite(view.observation['planes']).all()
        assert view.metadata['hp_source'] == 'native_current_hp'
        assert np.flatnonzero(view.action_mask).tolist() == [0]


def test_legacy_observation_action_y_mismatch_is_visible_not_silently_fixed():
    """Pin v25 compatibility while reporting the physical coordinate defect."""
    from sim.adapter import grid_to_point
    _, row, own = fixture()
    view = NativeModelAdapter(own_side=0).encode(row, own)
    # The friendly Knight is physically at y=21.5 tiles.
    assert view.observation['planes'][0, round(31 - 21.5), round(5.5)] > 0
    assert grid_to_point(5, round(31 - 21.5), 1).y != 21500


def test_unconfirmed_battle_and_inconsistent_own_state_fail_closed():
    from dataclasses import replace
    _, row, own = fixture()
    adapter = NativeModelAdapter(own_side=0)
    assert adapter.encode(row, replace(own, battle_active=False)).metadata['blockers'] == ['active_battle_not_confirmed']
    with pytest.raises(MissingModelInput, match='different times'):
        adapter.encode(row, replace(own, game_seconds=1))
    with pytest.raises(MissingModelInput, match='elixir estimate'):
        adapter.encode(row, replace(own, elixir=replace(own.elixir, minimum=float('nan'))))
    with pytest.raises(MissingModelInput, match='opponent'):
        adapter.encode(row, replace(own, opponent_events=({'at_ms': -1, 'cost': 1, 'card': 'skeletons'},)))


@pytest.mark.parametrize('native_name,character,card_id',[('GoblinBrawler','GoblinBrawler',27000012),('Goblin_Stab','Goblin',26000002)])
def test_native_child_and_guarded_melee_alias_match_simulator_tactical_planes(native_name,character,card_id):
    from sim.gamedata import load_characters
    from sim.eval_defense_duel import policy_observation
    match,row,own=fixture()
    character_spec=load_characters(11)[character]
    entity=match.battle.add(make_unit(0,character_spec,-1,arena.tile(4,14)))
    entity.hitpoints-=31
    row['units'].append(dict(entity_id=5000000+entity.uid,name=native_name,card_id=card_id,side=1,
        x=entity.pos.x,y=entity.pos.y,hp=entity.hitpoints,max_hp=entity.max_hitpoints))
    view=NativeModelAdapter(own_side=0,coordinate_mode='physical_live_cells_v2').encode(row,own)
    expected=policy_observation(match,1,'physical_live_cells_v2')
    np.testing.assert_array_equal(view.observation['planes'],expected['planes'])
    assert not any('unknown_tactical_profile:'+native_name==s for s in view.metadata['warnings'])
    assert bool(view.metadata['resolved_profile_aliases'])==(native_name=='Goblin_Stab')


def test_guarded_alias_does_not_apply_to_an_unverified_card_or_hero():
    adapter=NativeModelAdapter(own_side=0)
    assert adapter.unit_profile(dict(name='Goblin_Stab',card_id=123))[0]is None
    assert adapter.unit_profile(dict(name='KnightHero',card_id=203000001))[0]is None


def test_exact_capture_sees_augmented_input_and_final_mask_without_changing_decision():
    import torch
    _,row,own=fixture()
    view=NativeModelAdapter(own_side=0,coordinate_mode='physical_live_cells_v2').encode(row,own)
    view.action_mask[1:]=False  # pending/cooldown masks force WAIT
    class Network:
        def __call__(self,planes,scalars):
            self.inputs=(planes.clone(),scalars.clone())
            return torch.arange(len(view.action_mask)).float().unsqueeze(0),torch.tensor([.25])
    policy=MemoryPolicy.__new__(MemoryPolicy)
    policy.torch=torch;policy.joint=True;policy.input_contract={};policy.network=Network()
    ordinary=policy.decide(view);captured=[]
    observed=policy.decide(view,capture=lambda *args:captured.append(args))
    assert observed==ordinary and observed['action']==0
    planes,scalars,mask,result=captured[0]
    np.testing.assert_array_equal(planes,policy.network.inputs[0][0].numpy())
    np.testing.assert_array_equal(scalars,policy.network.inputs[1][0].numpy())
    assert scalars.shape==(309,) and mask.sum()==1 and result==observed
