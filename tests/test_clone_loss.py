import pytest

torch = pytest.importorskip("torch")

from sim.clone import action_gate_loss, near_play_correct, spatial_cross_entropy
from sim.env import ACTIONS, GRID_W, TILES


def _placement(slot, x, y):
    return 1 + slot * TILES + y * GRID_W + x


def test_neighboring_legal_tile_gets_soft_credit():
    target = torch.tensor([_placement(2, 9, 20)])
    mask = torch.ones((1, ACTIONS), dtype=torch.bool)
    neighbor = torch.zeros((1, ACTIONS))
    neighbor[0, _placement(2, 10, 20)] = 5.0
    distant = torch.zeros((1, ACTIONS))
    distant[0, _placement(2, 2, 28)] = 5.0
    assert spatial_cross_entropy(neighbor, mask, target, radius=1).item() < \
        spatial_cross_entropy(distant, mask, target, radius=1).item()


def test_smoothing_never_crosses_to_another_card_slot():
    target = torch.tensor([_placement(1, 9, 20)])
    mask = torch.ones((1, ACTIONS), dtype=torch.bool)
    same_cell_wrong_card = torch.zeros((1, ACTIONS))
    same_cell_wrong_card[0, _placement(2, 9, 20)] = 5.0
    nearby_right_card = torch.zeros((1, ACTIONS))
    nearby_right_card[0, _placement(1, 10, 20)] = 5.0
    assert spatial_cross_entropy(nearby_right_card, mask, target).item() < \
        spatial_cross_entropy(same_cell_wrong_card, mask, target).item()


def test_hold_target_remains_exact():
    target = torch.tensor([0])
    mask = torch.ones((1, ACTIONS), dtype=torch.bool)
    logits = torch.zeros((1, ACTIONS))
    logits[0, 0] = 3.0
    smoothed = spatial_cross_entropy(logits, mask, target, radius=2)
    exact = spatial_cross_entropy(logits, mask, target, radius=0)
    assert torch.equal(smoothed, exact)


def test_action_gate_prefers_a_play_logit_for_a_play_target():
    target = torch.tensor([_placement(0, 9, 20)])
    mask = torch.ones((1, ACTIONS), dtype=torch.bool)
    holding = torch.zeros((1, ACTIONS))
    holding[0, 0] = 5.0
    acting = torch.zeros((1, ACTIONS))
    acting[0, target.item()] = 5.0
    assert action_gate_loss(acting, mask, target).item() < \
        action_gate_loss(holding, mask, target).item()


def test_action_gate_prefers_hold_for_a_hold_target():
    target = torch.tensor([0])
    mask = torch.ones((1, ACTIONS), dtype=torch.bool)
    holding = torch.zeros((1, ACTIONS))
    holding[0, 0] = 5.0
    acting = torch.zeros((1, ACTIONS))
    acting[0, _placement(0, 9, 20)] = 5.0
    assert action_gate_loss(holding, mask, target).item() < \
        action_gate_loss(acting, mask, target).item()


def test_near_accuracy_requires_same_card_and_nearby_cell():
    truth = torch.tensor([_placement(1, 9, 20)] * 3)
    predicted = torch.tensor([
        _placement(1, 10, 19),
        _placement(2, 9, 20),
        _placement(1, 12, 20),
    ])
    assert near_play_correct(predicted, truth, radius=1) == [True, False, False]
