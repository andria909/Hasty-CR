"""Reject stale control after host sleep or a stalled memory/ADB request."""


class ObservationInterrupted(RuntimeError):
    pass


def check_poll_gap(previous_ns, now_ns, *, max_gap_ns=5_000_000_000):
    if previous_ns is not None and not 0 <= now_ns - previous_ns <= max_gap_ns:
        raise ObservationInterrupted('Observation interrupted; active battle requires reconciliation')


def check_input_age(snapshot, now_ns, *, max_age_ns=500_000_000):
    request = snapshot['request_host_ns']
    received = snapshot['received_host_ns']
    if not request <= received <= now_ns or now_ns - request > max_age_ns:
        raise ObservationInterrupted('Stale observation; refusing Android input')
