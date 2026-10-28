import json
from pathlib import Path

import pytest

from scripts.gamestate.hand_readiness import HandReadiness


def test_new_card_needs_a_later_tick_but_other_slots_stay_available():
    ready = HandReadiness()
    assert ready.observe(100, [0, 1, 2, 3]) == ()
    assert ready.observe(128, [0, 4, 2, 3]) == (1,)
    # Re-reading the same tick, even after host time passes, is not fresh.
    assert ready.observe(128, [0, 4, 2, 3]) == (1,)
    assert ready.observe(129, [0, 4, 2, 3]) == ()


def test_independent_rotations_and_new_match_reset():
    ready = HandReadiness()
    ready.observe(100, [0, 1, 2, 3])
    assert ready.observe(101, [4, 1, 5, 3]) == (0, 2)
    assert ready.observe(102, [4, 6, 5, 3]) == (1,)
    assert ready.observe(103, [4, 6, 5, 3]) == ()
    assert HandReadiness().observe(0, [4, 6, 5, 3]) == ()


def test_same_tick_rotation_is_not_treated_as_a_settled_hand():
    ready = HandReadiness()
    ready.observe(100, [0, 1, 2, 3])
    assert ready.observe(100, [0, 1, 2, 4]) == (3,)
    assert ready.observe(101, [0, 1, 2, 4]) == ()


@pytest.mark.parametrize('tick,hand', [
    (99, [0, 1, 2, 3]), (101, [0, 1, 2]), (101, [0, 1, 1, 3]),
    (101, [0, 1, 2, 8]), (101.0, [0, 1, 2, 3]),
])
def test_invalid_observation_cannot_release_a_slot(tick, hand):
    ready = HandReadiness()
    ready.observe(100, [0, 1, 2, 3])
    with pytest.raises(ValueError):
        ready.observe(tick, hand)
    assert ready.hand == (0, 1, 2, 3)


RECORDED = json.loads((Path(__file__).resolve().parents[1]/
    'data/recorded_mechanics_20261028/hand_rotation_20261029.json').read_text())


@pytest.mark.parametrize('case', RECORDED['cases'],
                         ids=lambda c: f"{c['command']['outcome']}_{c['command']['tick']}")
def test_recorded_rotations_defer_only_until_a_fresh_observation(case):
    ready = HandReadiness()
    for row in case['observations']:
        assert list(ready.observe(row['tick'], row['hand'])) == row['blocked_slots']
    slot = case['command']['slot']
    assert slot in case['observations'][1]['blocked_slots']
    assert slot not in ready.blocked_slots()
