from copy import deepcopy
import numpy as np
import pytest
import torch

from sim.entity_policy import ARCHITECTURES, EntityTransformer, encode_entities
from sim.env import ACTIONS
from sim.train_attack_ppo import make_training_env
from tests.test_memory_model_bridge import fixture
from scripts.gamestate.model_bridge import NativeModelAdapter


@pytest.mark.parametrize('rotated', [False, True])
def test_native_and_simulator_share_exact_public_entities(rotated):
    match, snapshot, own = fixture(rotated=rotated)
    view = NativeModelAdapter(own_side=0, coordinate_mode='physical_live_cells_v2',
                             entity_inputs=True).encode(snapshot, own)
    np.testing.assert_array_equal(view.observation['entities'], encode_entities(match))


def test_hidden_state_and_dictionary_order_do_not_change_input():
    m, _, _ = fixture()
    original = encode_entities(m)
    m.players[-1].elixir = 1234
    for e in m.battle.entities.values():
        e.target_uid = 789
        e.next_attack_ms = 99999
    m.battle.entities = dict(reversed(list(m.battle.entities.items())))
    np.testing.assert_array_equal(original, encode_entities(m))


@pytest.mark.parametrize('arch', list(ARCHITECTURES))
def test_permutation_padding_empty_and_backward(arch):
    torch.set_num_threads(1); torch.manual_seed(7)
    net = EntityTransformer(309, ACTIONS, arch).eval()
    m, _, _ = fixture()
    entities = torch.from_numpy(encode_entities(m))[None]
    scalars = torch.zeros(1, 309)
    logits, value = net(entities, scalars)
    changed = entities[:, torch.randperm(128)].clone()
    changed[:, :, 2:] += (changed[:, :, :1] == 0) * 999
    other, other_value = net(changed, scalars)
    torch.testing.assert_close(logits, other, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(value, other_value, atol=1e-6, rtol=1e-5)
    net.train()
    full, full_value = net(entities, scalars)
    torch.testing.assert_close(logits, full, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(value, full_value, atol=1e-6, rtol=1e-5)
    net.eval()
    empty, v = net(torch.zeros_like(entities), scalars)
    assert empty.shape == (1, ACTIONS) and torch.isfinite(empty).all()
    (empty.square().mean() + v.square().mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())


def test_overflow_preserves_count_and_hp():
    m, _, _ = fixture()
    entity = next(e for e in m.battle.entities.values() if not e.is_tower)
    m.battle.entities = {i: deepcopy(entity) for i in range(200)}
    rows = encode_entities(m)
    assert rows[:, 18].sum() == 200
    assert rows[:, 17].sum() == 1
    assert rows[:, 5].sum() == pytest.approx(entity.hitpoints * 200 / 10000)


def test_wrapper_keeps_actions_rewards_masks_and_resume_identical():
    opts = dict(task_mode='joint', curriculum='continuous_v1', seed=8,
                action_delay_ms=1350, continuous_profile={'lane_reflection_fraction': 1.})
    old = make_training_env(**opts)
    new = make_training_env(**opts, entity_inputs=True)
    a, ia = old.reset(seed=8); b, ib = new.reset(seed=8)
    for _ in range(12):
        np.testing.assert_array_equal(a['scalars'], b['scalars'])
        np.testing.assert_array_equal(ia['action_mask'], ib['action_mask'])
        np.testing.assert_array_equal(b['planes'], encode_entities(new.match, reflected=True))
        action = int(np.flatnonzero(ia['action_mask'])[-1])
        a, ra, ta, xa, ia = old.step(action)
        b, rb, tb, xb, ib = new.step(action)
        assert (ra, ta, xa) == (rb, tb, xb)
    state = new.snapshot()
    expected = new.step(0)
    new.restore(state)
    actual = new.step(0)
    np.testing.assert_array_equal(expected[0]['planes'], actual[0]['planes'])
    assert expected[1:4] == actual[1:4]


def test_live_loader_requires_entity_vocabulary_and_uses_entity_input(tmp_path):
    from scripts.gamestate.model_bridge import MemoryPolicy, MissingModelInput
    from sim.entity_policy import CONTRACT, vocabulary_sha
    from sim.train_ppo import checkpoint_architecture
    arch=next(iter(ARCHITECTURES))
    net=EntityTransformer(309,ACTIONS,arch)
    blob=dict(architecture=arch,state_dict=net.state_dict(),config=dict(task_mode='joint',
        coordinate_mode='physical_live_cells_v2',observation_contract=CONTRACT,
        entity_vocabulary_sha256=vocabulary_sha()))
    path=tmp_path/'entity.pt';torch.save(blob,path)
    policy=MemoryPolicy(path)
    _,snapshot,own=fixture()
    view=NativeModelAdapter(own_side=0,coordinate_mode='physical_live_cells_v2',entity_inputs=True).encode(snapshot,own)
    assert checkpoint_architecture(blob)==arch
    assert view.action_mask[policy.decide(view)['action']]
    view.observation['planes'][:]=np.nan
    assert view.action_mask[policy.decide(view)['action']]  # Raster is not consumed.
    blob['config']['entity_vocabulary_sha256']='different'
    bad=tmp_path/'wrong_vocab.pt';torch.save(blob,bad)
    with pytest.raises(MissingModelInput,match='vocabulary'):
        MemoryPolicy(bad)


def test_relative_geometry_has_same_parameters_and_changes_spatial_response():
    torch.manual_seed(13)
    old=EntityTransformer(309,ACTIONS,'entity_transformer_d64_l2_v1').eval()
    torch.manual_seed(13)
    new=EntityTransformer(309,ACTIONS,'entity_transformer_d64_l2_relative_v1').eval()
    assert all(torch.equal(p,new.state_dict()[k])for k,p in old.state_dict().items())
    m,_,_=fixture()
    entities=torch.from_numpy(encode_entities(m))[None]
    scalars=torch.zeros(1,309)
    a,va=old(entities,scalars);b,vb=new(entities,scalars)
    torch.testing.assert_close(va,vb,atol=0,rtol=0)
    assert not torch.allclose(a[:,1:2305],b[:,1:2305])
    (b.square().mean()+vb.square().mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all()for p in new.parameters())
