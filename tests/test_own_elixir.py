import math

import pytest

from scripts.gamestate.elixir import ElixirTracker


def test_start_regen_and_cap_discards_overflow():
    tracker = ElixirTracker.from_match_start()
    assert tracker.advance(14).value == 10
    tracker.confirm_play(event_id='hog', cost=4, game_seconds=28)
    assert tracker.estimate().value == 6
    assert tracker.advance(30.8).value == pytest.approx(7)


def test_phase_boundaries_integrated_piecewise():
    tracker = ElixirTracker(game_seconds=119, initial=0)
    assert tracker.advance(121).value == pytest.approx(3 / 2.8)
    tracker = ElixirTracker(game_seconds=239, initial=0)
    assert tracker.advance(241).value == pytest.approx(5 / 2.8)


def test_late_attachment_has_no_invented_point_value():
    tracker = ElixirTracker(game_seconds=35)
    assert tracker.estimate().value is None
    assert not tracker.can_afford(1)
    assert tracker.advance(37.8).minimum == pytest.approx(1)
    tracker.confirm_play(event_id='confirmed', cost=4, game_seconds=37.8)
    assert tracker.estimate().minimum == 0
    assert tracker.estimate().maximum == 6


def test_deduplicate_grouped_units_and_reject_conflicts():
    tracker = ElixirTracker.from_match_start()
    tracker.confirm_play(event_id='skeletons', cost=1, game_seconds=0)
    tracker.advance(1)
    tracker.confirm_play(event_id='skeletons', cost=1, game_seconds=0)
    assert tracker.estimate().value == pytest.approx(4 + 1 / 2.8)
    with pytest.raises(ValueError):
        tracker.confirm_play(event_id='skeletons', cost=2, game_seconds=0)


def test_impossible_spend_is_atomic_and_time_cannot_reverse():
    tracker = ElixirTracker(initial=0)
    with pytest.raises(ValueError):
        tracker.confirm_play(event_id='bad', cost=4, game_seconds=1)
    assert tracker.estimate().game_seconds == 0
    tracker.advance(5)
    with pytest.raises(ValueError):
        tracker.advance(4)


def test_reconnect_invalidates_and_explicit_measurement_restores():
    tracker = ElixirTracker.from_match_start()
    assert tracker.invalidate(10).value is None
    assert tracker.reconcile(game_seconds=11, value=3).value == 3


@pytest.mark.parametrize('bad', [math.nan, math.inf, -1])
def test_invalid_time(bad):
    with pytest.raises(ValueError):
        ElixirTracker(game_seconds=bad)


def test_custom_mode_and_pause():
    tracker = ElixirTracker(initial=0, schedule=((0, 7),))
    assert tracker.advance(2).value == pytest.approx(5)
    assert tracker.advance(2).value == pytest.approx(5)


def test_gain_is_capped_deduplicated_and_shared_id_conflicts():
    tracker = ElixirTracker(initial=9)
    assert tracker.confirm_gain(event_id='pump-1', amount=2, game_seconds=0).value == 10
    tracker.confirm_gain(event_id='pump-1', amount=2, game_seconds=0)
    assert tracker.estimate().value == 10
    with pytest.raises(ValueError, match='Conflicting duplicate'):
        tracker.confirm_play(event_id='pump-1', cost=2, game_seconds=0)
    tracker.confirm_play(event_id='hog', cost=4, game_seconds=0)
    assert tracker.advance(2.8).value == pytest.approx(7)


def test_delayed_play_replays_regeneration_lost_at_cap():
    tracker = ElixirTracker(initial=10)
    tracker.advance(5.6)
    # Confirmation arrives later, but the actual spend happened at game time 0.
    estimate = tracker.confirm_play(event_id='hog', cost=4, game_seconds=0)
    assert estimate.game_seconds == 5.6
    assert estimate.value == pytest.approx(8)


def test_out_of_order_events_match_chronological_events():
    ordered = ElixirTracker.from_match_start()
    delayed = ElixirTracker.from_match_start()
    ordered.confirm_play(event_id='first', cost=1, game_seconds=1)
    ordered.confirm_play(event_id='second', cost=4, game_seconds=3)
    ordered.confirm_gain(event_id='credit', amount=2, game_seconds=4)
    ordered.advance(10)
    delayed.confirm_gain(event_id='credit', amount=2, game_seconds=4)
    delayed.confirm_play(event_id='second', cost=4, game_seconds=3)
    delayed.advance(10)
    delayed.confirm_play(event_id='first', cost=1, game_seconds=1)
    assert delayed.estimate().value == pytest.approx(ordered.estimate().value)


def test_bad_delayed_event_does_not_corrupt_later_spending():
    tracker = ElixirTracker.from_match_start()
    tracker.confirm_play(event_id='later', cost=5, game_seconds=1)
    before = tracker.estimate()
    with pytest.raises(ValueError, match='Spend exceeds'):
        tracker.confirm_play(event_id='early', cost=4, game_seconds=0)
    assert tracker.estimate() == before
    # The failed event ID is not poisoned and can be retried with valid data.
    tracker.confirm_gain(event_id='early', amount=1, game_seconds=0)


def test_pending_command_prevents_double_spending_and_confirms_once():
    tracker = ElixirTracker.from_match_start()
    estimate = tracker.reserve_play(action_id='tap-1', cost=4, game_seconds=0)
    assert estimate.value == 5  # A tap has not yet been confirmed.
    assert estimate.available_minimum == 1
    assert estimate.pending_actions == 1
    assert not tracker.can_afford(2)
    with pytest.raises(ValueError, match='Insufficient unreserved'):
        tracker.reserve_play(action_id='tap-2', cost=4, game_seconds=0)
    assert tracker.estimate() == estimate
    assert tracker.reserve_play(action_id='tap-1', cost=4, game_seconds=0) == estimate
    tracker.advance(2.8)
    estimate = tracker.confirm_play(event_id='accepted-1', action_id='tap-1',
                                    cost=4, game_seconds=1)
    assert estimate.value == pytest.approx(2)
    assert estimate.reserved == 0
    assert tracker.confirm_play(event_id='accepted-1', action_id='tap-1',
                                cost=4, game_seconds=1) == estimate
    with pytest.raises(ValueError, match='Cannot cancel'):
        tracker.cancel_play('tap-1')
    with pytest.raises(ValueError, match='already completed'):
        tracker.reserve_play(action_id='tap-1', cost=4, game_seconds=2.8)


def test_pending_reservation_never_expires_without_proof_of_failure():
    tracker = ElixirTracker(initial=10)
    tracker.reserve_play(action_id='tap', cost=4, game_seconds=0)
    assert tracker.advance(100).reserved == 4
    assert tracker.cancel_play('tap').value == 10
    assert tracker.estimate().reserved == 0
    assert tracker.cancel_play('tap').value == 10  # Idempotent rejection handling.
    with pytest.raises(ValueError, match='different result'):
        tracker.confirm_play(event_id='oops', action_id='tap', cost=4, game_seconds=1)


def test_cannot_confirm_wrong_pending_action_or_reuse_event():
    tracker = ElixirTracker(initial=10)
    tracker.reserve_play(action_id='a', cost=4, game_seconds=1)
    before = tracker.estimate()
    with pytest.raises(ValueError, match='does not match'):
        tracker.confirm_play(event_id='p', action_id='a', cost=3, game_seconds=1)
    with pytest.raises(ValueError, match='does not match'):
        tracker.confirm_play(event_id='p', action_id='a', cost=4, game_seconds=0)
    assert tracker.estimate() == before
    tracker.confirm_play(event_id='p', action_id='a', cost=4, game_seconds=1)
    tracker.reserve_play(action_id='b', cost=4, game_seconds=1)
    with pytest.raises(ValueError, match='already accounted for'):
        tracker.confirm_play(event_id='p', action_id='b', cost=4, game_seconds=1)
    assert tracker.estimate().reserved == 4


def test_forecast_and_wait_cross_phase_boundaries_without_mutation():
    tracker = ElixirTracker(game_seconds=119, initial=0)
    before = tracker.estimate()
    # 1s at x1, then 0.9s at x2 -> 1 elixir.
    assert tracker.seconds_until_affordable(1) == pytest.approx(1.9)
    assert tracker.forecast(120.9).value == pytest.approx(1)
    assert tracker.estimate() == before
    tracker = ElixirTracker(game_seconds=239, initial=0)
    assert tracker.seconds_until_affordable(1) == pytest.approx(1 + 0.8 / 3)
    assert tracker.forecast(240 + 0.8 / 3).value == pytest.approx(1)


def test_wait_uses_reservations_and_handles_impossible_budget():
    tracker = ElixirTracker(initial=5)
    tracker.reserve_play(action_id='a', cost=4, game_seconds=0)
    assert tracker.seconds_until_affordable(1) == 0
    assert tracker.seconds_until_affordable(2) == pytest.approx(2.8)
    assert math.isinf(tracker.seconds_until_affordable(7))
    tracker.invalidate(0)
    assert tracker.seconds_until_affordable(0) == 0
    assert tracker.estimate().reserved == 4


def test_reconciliation_keeps_dedup_but_rejects_unseen_older_events():
    tracker = ElixirTracker.from_match_start()
    tracker.confirm_play(event_id='known', cost=1, game_seconds=1)
    tracker.reconcile(game_seconds=10, value=5)
    assert tracker.confirm_play(event_id='known', cost=1, game_seconds=1).value == 5
    with pytest.raises(ValueError, match='predates'):
        tracker.confirm_play(event_id='unknown', cost=1, game_seconds=9)
    assert tracker.estimate().value == 5


def test_custom_mode_has_honest_source_and_usable_budget_in_json():
    tracker = ElixirTracker(initial=5, schedule=((0, 2),))
    tracker.reserve_play(action_id='a', cost=4, game_seconds=0)
    row = tracker.estimate().to_dict()
    assert row['source'] == 'mathematical_custom_mode'
    assert row['available_minimum'] == row['available_maximum'] == 1
    assert row['pending_actions'] == 1


@pytest.mark.parametrize('method,kwargs', [
    ('confirm_gain', dict(event_id='bad', amount=math.nan, game_seconds=0)),
    ('confirm_play', dict(event_id='', cost=1, game_seconds=0)),
    ('reserve_play', dict(action_id='bad', cost=-1, game_seconds=0)),
    ('reserve_play', dict(action_id='bad', cost=1, game_seconds=-1)),
    ('forecast', dict(game_seconds=math.inf)),
])
def test_rejected_inputs_leave_state_intact(method, kwargs):
    tracker = ElixirTracker.from_match_start()
    before = tracker.estimate()
    with pytest.raises(ValueError):
        getattr(tracker, method)(**kwargs)
    assert tracker.estimate() == before
