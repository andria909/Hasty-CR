"""Timings at the live loop's boundaries, before logging or sending taps."""

from datetime import datetime, timezone


def decision_sample(boundaries, *, match, tick, elapsed_s, decision,
                    status="decision", screen="in_game"):
    start, captured, detected, checked, decided = boundaries
    if any(right < left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("latency boundaries must be monotonic")
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "match": match,
        "tick": tick,
        "elapsed_s": elapsed_s,
        "status": status,
        "screen": screen,
        "kind": ("hold" if decision is None else "play") if status == "decision" else None,
        "card": None if decision is None else decision.card,
        "tag": None if decision is None else decision.tag,
        "capture_ms": (captured - start) * 1000,
        "perception_ms": (detected - captured) * 1000,
        "checks_ms": (checked - detected) * 1000,
        "policy_ms": (decided - checked) * 1000,
        "after_capture_ms": (decided - captured) * 1000,
        "total_ms": (decided - start) * 1000,
    }


def placement_sample(boundaries, tap_boundaries, **metadata):
    """ADB input completion is an upper bound on injection, not troop spawn."""
    selected_start, selected_end, placement_start, placement_end = tap_boundaries
    if (selected_start < boundaries[-1]
            or any(right < left for left, right in
                   zip(tap_boundaries, tap_boundaries[1:]))):
        raise ValueError("tap boundaries must follow the decision monotonically")
    sample = decision_sample(boundaries, **metadata)
    sample.update({
        "status": "action_sent",
        "select_card_ms": (selected_end - selected_start) * 1000,
        "between_taps_ms": (placement_start - selected_end) * 1000,
        "placement_command_ms": (placement_end - placement_start) * 1000,
        "decision_to_input_ms": (placement_end - boundaries[-1]) * 1000,
        "capture_start_to_input_ms": (placement_end - boundaries[0]) * 1000,
        "capture_ready_to_input_ms": (placement_end - boundaries[1]) * 1000,
    })
    return sample
