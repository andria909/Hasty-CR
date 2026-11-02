import pytest

from scripts.gamestate.observation_freshness import (
    ObservationInterrupted, check_input_age, check_poll_gap,
)


def test_host_sleep_does_not_become_a_fresh_observation_on_resume():
    # Observed fourth-game event gap, while the engine barely advanced.
    with pytest.raises(ObservationInterrupted):
        check_poll_gap(1793637336630073000, 1793637506661588000)
    check_poll_gap(1_000_000_000, 1_150_000_000)


def test_delay_inside_card_selection_prevents_the_target_tap():
    snapshot={'request_host_ns':1_000_000_000,'received_host_ns':1_035_000_000}
    check_input_age(snapshot,1_045_000_000)
    with pytest.raises(ObservationInterrupted):
        check_input_age(snapshot,1_600_000_000)


@pytest.mark.parametrize('requested,received,now', [(3,2,4),(1,4,3)])
def test_clock_discontinuities_fail_closed(requested,received,now):
    with pytest.raises(ObservationInterrupted):
        check_input_age({'request_host_ns':requested,'received_host_ns':received},now)
