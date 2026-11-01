import torch
import pytest

from scripts.attack.evaluate_support_closed_loop import episode
from sim.env import ACTIONS, DECK_26, LANES


class WaitPolicy:
    def __call__(self, planes, scalars):
        return torch.zeros((len(planes), ACTIONS)), torch.zeros(len(planes))


class MusketeerOnlyPolicy:
    def __call__(self, planes, scalars):
        logits = torch.full((len(planes), ACTIONS), -1.)
        logits[:, 0] = 0
        start = 3 + 2 * len(LANES)
        for i, row in enumerate(scalars):
            hand = row[start:start + 4 * len(DECK_26)].reshape(4, len(DECK_26))
            for slot, card in enumerate(hand):
                if card[DECK_26.index('musketeer')] == 1:
                    logits[i, 1 + slot * 576 + 20 * 18 + 3] = 10
        return logits, torch.zeros(len(planes))


def test_closed_loop_wait_controller_does_not_receive_teacher_commands():
    result = episode(WaitPolicy(), dict(seed=352, lane='left', bank=6000, hand='musketeer'), 1350)
    assert result['commands'] == result['rejected'] == []
    assert result['total_damage'] > 0
    assert result['wait_decisions'] > 0


@pytest.mark.parametrize('delay,delivery', [(1350, 1350), (2025, 2050)])
def test_closed_loop_blocks_pending_slot_and_delivers_one_paid_musketeer(delay, delivery):
    result = episode(MusketeerOnlyPolicy(), dict(seed=352, lane='left', bank=6000, hand='musketeer'), delay)
    assert result['rejected'] == []
    assert result['cards'] == {'musketeer': 1}
    assert result['commands'][0]['requested_ms'] == result['start_ms']
    assert result['commands'][0]['at_ms'] - result['start_ms'] == delivery


def test_opposite_wave_uses_existing_hand_and_pays_from_regenerated_enemy_bank():
    result=episode(WaitPolicy(),dict(seed=352,lane='left',bank=6000,hand='musketeer'),1350,'opposite')
    assert len(result['enemy_wave'])==1
    wave=result['enemy_wave'][0]
    assert wave['lane']=='right'
    assert wave['at_ms']>=result['start_ms']+12000
    assert wave['elixir_after']>=0
    assert wave['elixir_before']-wave['elixir_after']==wave['cost']
