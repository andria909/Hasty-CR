import json

import numpy as np
import pytest

from scripts.gamestate.analyze_live_decisions import command_audit, decision_audit, native_audit
from sim.env import ACTIONS, NUM_PLANES, NUM_SCALARS


def play(tick, card='zap', slot=0):
    return dict(event='PLAY', match=1, tick=tick, card=card, slot=slot,
                grid=[8, 18], host_ns=tick*1000+500)


def ack(source, tick, card='zap'):
    return dict(event='PLAY_ACK', match=1, source_tick=source, tick=tick, card=card)


def test_attempts_ack_timeout_and_pending_are_not_conflated():
    report = command_audit([play(100), ack(100, 128), play(128, 'skeletons'),
        dict(event='PLAY_NOT_CONFIRMED', match=1, card='skeletons', tick=210),
        play(250, 'hog_rider')])
    assert report['counts'] == dict(confirmed=1, not_confirmed=1, unresolved=1)
    assert report['same_tick_as_previous_slot_ack'] == dict(not_confirmed=1)
    assert report['by_card']['skeletons']['confirmed'] == 0
    assert report['by_card']['zap']['confirmed_left'] == 1


@pytest.mark.parametrize('events', [
    [play(100), ack(100, 128), ack(100, 128)],
    [play(100), ack(101, 128)],
    [play(100), play(101, 'skeletons')],
    [play(100), ack(100, 99)],
    [play(100), dict(event='PLAY_NOT_CONFIRMED', match=1, card='zap',
                    tick=200, source_tick=100, slot=1)],
])
def test_ambiguous_or_conflicting_command_link_is_rejected(events):
    with pytest.raises(ValueError):
        command_audit(events)


def capture(path, *, tick=100, available=2., reserved=3., action=0, legal_slot=None):
    scalars = np.zeros(NUM_SCALARS, dtype=np.float32)
    scalars[0] = available/10
    mask = np.zeros(ACTIONS, dtype=bool)
    mask[0] = True
    if legal_slot is not None:
        mask[1+legal_slot*576] = True
    metadata = dict(match=1, tick=tick, checkpoint_sha256='test',
        coordinate_mode='physical_live_cells_v2', geometry='native_anchors_candidate_v1',
        hand=['hog_rider', 'fireball', 'musketeer', 'zap'], reserved_elixir=reserved,
        decision=dict(action=action))
    np.savez(path, planes=np.zeros((NUM_PLANES, 32, 18), dtype=np.float32),
             scalars=scalars, action_mask=mask, metadata=np.asarray(json.dumps(metadata)))


def test_forced_wait_is_distinct_from_choice_and_reserved_mana_is_not_raw(tmp_path):
    a, b, c = (tmp_path/name for name in ('a.npz', 'b.npz', 'c.npz'))
    capture(a)
    capture(b, tick=102, available=2, reserved=0, legal_slot=3)
    capture(c, tick=104, available=4, reserved=0, legal_slot=1, action=577)
    result = decision_audit([a, b, c], expected_sha='test')
    assert result['counts'] == dict(decisions=3, available_below4=2, native_below4=1,
        reservation_makes_below4=1, forced_wait=1, voluntary_wait=1, play=1)
    assert result['by_card']['fireball'] == dict(in_hand=3, legal=1, selected=1)
    assert result['by_card']['musketeer']['legal'] == 0
    with pytest.raises(ValueError, match='Mixed checkpoints'):
        decision_audit([a], expected_sha='different')
    with pytest.raises(ValueError, match='Duplicate decision'):
        decision_audit([a, a], expected_sha='test')


def frame(tick, hand, mana=3):
    return dict(status='ok', received_host_ns=tick*1000,
        own_state=dict(status='ok', world='world1', tick=tick, hand_indices=hand, elixir_raw=int(mana*10000)),
        frame=dict(battle_tick=tick, entities=[dict(name='KingTower', side=1, entity_id=1, address='king')]))


def test_native_coverage_does_not_fill_invalid_frames_or_large_gaps():
    frames = [frame(100, [0, 1, 2, 3]), frame(102, [0, 1, 2, 3]),
              dict(status='no_battle'), frame(104, [0, 1, 2, 3]),
              frame(110, [4, 1, 2, 3]), frame(112, [4, 1, 2, 3], mana=5)]
    command = dict(play(110), outcome='not_confirmed')
    result = native_audit(frames, [dict(world='world1', number=1)], [command])
    assert result['covered_game_seconds'] == pytest.approx(.2)
    assert result['left_sample_below4_seconds'] == pytest.approx(.2)
    assert result['gaps_over_5_ticks'] == 1
    assert result['fresh_tick_guard_would_defer'] == dict(not_confirmed=1)
    assert result['commands_with_native_source'] == 1


def test_future_snapshot_cannot_change_original_command_readiness():
    a = frame(100, [0, 1, 2, 3])
    b = frame(100, [4, 1, 2, 3])
    b['received_host_ns'] += 600
    result = native_audit([a, b], [dict(world='world1', number=1)],
                          [dict(play(100), outcome='confirmed')])
    assert result['fresh_tick_guard_would_defer'] == {}
