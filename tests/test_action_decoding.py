import pytest
import torch

from sim.action_decoding import greedy_actions


def test_grouped_rule_resolves_probability_dilution_and_honors_masks():
    logits = torch.full((3, 2307), -20.)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[:, 0] = True
    logits[:, 0] = 0.
    mask[0, 1:577] = True
    logits[0, 1:577] = -2.
    logits[0, 19] = -1.
    logits[0, 578] = 1000.  # Illegal card cannot win either decision.
    mask[2, 2306] = True
    logits[2, 2306] = 1.
    assert greedy_actions(logits, mask).tolist() == [0, 0, 2306]
    assert greedy_actions(logits, mask, 'grouped_argmax').tolist() == [19, 0, 2306]


def test_grouped_rule_matches_explicit_probability_marginalization():
    torch.manual_seed(94)
    logits = torch.randn(23, 2308)
    mask = torch.rand_like(logits) > .4
    mask[:, 0] = True
    p = logits.masked_fill(~mask, -torch.inf).softmax(-1)
    mass = torch.cat((p[:, :1], p[:, 1:2305].reshape(-1, 4, 576).sum(-1), p[:, 2305:]), -1)
    actions = greedy_actions(logits, mask, 'grouped_argmax')
    for i, action in enumerate(actions.tolist()):
        group = 0 if action == 0 else 1 + (action - 1) // 576 if action < 2305 else action - 2300
        assert group == int(mass[i].argmax())
        assert mask[i, action]
        if 1 <= group <= 4:
            start = 1 + (group - 1) * 576
            assert action == start + int(p[i, start:start + 576].argmax())
    assert torch.equal(greedy_actions(logits, mask), logits.masked_fill(~mask, -torch.inf).argmax(-1))


def test_invalid_modes_and_empty_masks_rejected():
    logits = torch.zeros(1, 2305)
    with pytest.raises(ValueError):
        greedy_actions(logits, torch.zeros_like(logits, dtype=torch.bool))
    with pytest.raises(ValueError):
        greedy_actions(logits, torch.ones_like(logits, dtype=torch.bool), 'unknown')


def test_native_policy_uses_explicit_decoding_mode():
    from scripts.gamestate.model_bridge import MemoryPolicy, NativeModelAdapter
    from tests.test_memory_model_bridge import fixture
    _,snapshot,own=fixture()
    view=NativeModelAdapter(own_side=0,coordinate_mode='physical_live_cells_v2').encode(snapshot,own)
    logits=torch.full((1,len(view.action_mask)),-2.)
    logits[0,0]=0.
    class Network:
        def __call__(self,*args):return logits.clone(),torch.zeros(1)
    policy=MemoryPolicy.__new__(MemoryPolicy)
    policy.torch=torch;policy.joint=True;policy.input_contract={};policy.network=Network()
    assert policy.decide(view)['action']==0
    policy.action_mode='grouped_argmax'
    expected=int(greedy_actions(logits,torch.from_numpy(view.action_mask)[None],policy.action_mode)[0])
    assert expected!=0 and policy.decide(view)['action']==expected
