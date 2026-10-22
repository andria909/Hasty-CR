"""Own elixir from explicit GAME time and confirmed economy events; no CV.

The default schedule matches our standard-mode simulator. Host/ADB time and
entity first appearances are not game-clock/play events. ``exact`` means exact
within this mathematical model, conditional on complete, trusted inputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

STANDARD_SCHEDULE = ((0.0, 1.0), (120.0, 2.0), (240.0, 3.0))


@dataclass(frozen=True)
class ElixirEstimate:
    game_seconds: float
    minimum: float
    maximum: float
    exact: bool
    source: str = 'mathematical_standard_mode'
    reserved: float = 0.0
    pending_actions: int = 0

    @property
    def value(self) -> float | None:
        return self.minimum if self.exact else None

    @property
    def available_minimum(self) -> float:
        return max(0.0, self.minimum - self.reserved)

    @property
    def available_maximum(self) -> float:
        return max(0.0, self.maximum - self.reserved)

    def to_dict(self) -> dict:
        return dict(asdict(self), value=self.value,
                    available_minimum=self.available_minimum,
                    available_maximum=self.available_maximum)


@dataclass(frozen=True)
class _Event:
    kind: str
    game_seconds: float
    amount: float


class ElixirTracker:
    """Model own economy while ALL own spending/gains are accounted for.

    Start at 5 only with a known match start. A mid-match attachment defaults
    to [0,10]. Invalidate on missed events/reconnect. A tap is NOT confirmation.
    Explicit game time must stop during pauses; never feed wall-clock time.
    Use one instance per match. IDs belong to that match and must be unique.
    """

    def __init__(self, *, game_seconds: float = 0.0,
                 initial: float | None = None,
                 schedule: tuple[tuple[float, float], ...] = STANDARD_SCHEDULE):
        self._time = self._number(game_seconds)
        if self._time < 0:
            raise ValueError('Negative game time')
        self.schedule = tuple((self._number(t), self._number(rate))
                              for t, rate in schedule)
        if (not self.schedule or self.schedule[0][0] != 0 or
                any(rate <= 0 for _, rate in self.schedule) or
                any(b[0] <= a[0] for a, b in zip(self.schedule, self.schedule[1:]))):
            raise ValueError('Schedule must start at 0 with increasing times and positive rates')
        if initial is not None and not 0 <= self._number(initial) <= 10:
            raise ValueError('Initial elixir must be in [0,10]')
        self._low = 0.0 if initial is None else float(initial)
        self._high = 10.0 if initial is None else float(initial)
        self._anchor = (self._time, self._low, self._high)
        self._events: dict[str, _Event] = {}
        self._seen: dict[str, _Event] = {}
        self._pending: dict[str, tuple[float, float]] = {}
        # Retain completed action IDs so retries cannot reserve/spend twice.
        self._completed: dict[str, tuple[str | None, float]] = {}

    @staticmethod
    def _number(value: float) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('Expected finite number')
        return result

    @staticmethod
    def _id(value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError('Expected a nonempty ID')
        return value

    @classmethod
    def _amount(cls, value: float) -> float:
        value = cls._number(value)
        if not 0 <= value <= 10:
            raise ValueError('Elixir amount must be in [0,10]')
        return value

    @classmethod
    def from_match_start(cls) -> ElixirTracker:
        return cls(initial=5.0)

    def _regen(self, low: float, high: float, start: float,
               end: float) -> tuple[float, float]:
        for index, (phase, multiplier) in enumerate(self.schedule):
            boundary = self.schedule[index + 1][0] if index + 1 < len(self.schedule) else end
            left, right = max(start, phase), min(end, boundary)
            if right > left:
                gain = (right - left) * multiplier / 2.8
                low, high = min(10.0, low + gain), min(10.0, high + gain)
        return low, high

    def _replay(self, end: float, events: dict[str, _Event]) -> tuple[float, float]:
        cursor, low, high = self._anchor
        # Stable sort: equal timestamps retain the caller's event ordering.
        for event in sorted(events.values(), key=lambda e: e.game_seconds):
            low, high = self._regen(low, high, cursor, event.game_seconds)
            if event.kind == 'spend':
                if high + 1e-9 < event.amount:
                    raise ValueError('Spend exceeds estimate; invalidate or reconcile state')
                low, high = max(0.0, low - event.amount), max(0.0, high - event.amount)
            else:
                low, high = min(10.0, low + event.amount), min(10.0, high + event.amount)
            cursor = event.game_seconds
        return self._regen(low, high, cursor, end)

    def advance(self, game_seconds: float) -> ElixirEstimate:
        end = self._number(game_seconds)
        if end < self._time:
            raise ValueError('Game time moved backwards; reset for a new match')
        self._low, self._high = self._regen(self._low, self._high, self._time, end)
        self._time = end
        return self.estimate()

    def _confirm(self, event_id: str, event: _Event,
                 action_id: str | None = None) -> ElixirEstimate:
        self._id(event_id)
        if action_id is not None:
            self._id(action_id)
            if event.kind != 'spend':
                raise ValueError('Only spending can confirm a pending play')
            if action_id in self._completed:
                if self._completed[action_id] != (event_id, event.amount):
                    raise ValueError('Action already completed with a different result')
            elif action_id not in self._pending:
                raise ValueError('Unknown pending action')
            else:
                cost, submitted = self._pending[action_id]
                if cost != event.amount or event.game_seconds < submitted:
                    raise ValueError('Confirmation does not match pending action')
        if event_id in self._seen:
            if self._seen[event_id] != event:
                raise ValueError('Conflicting duplicate event ID')
            if action_id is not None and action_id in self._pending:
                raise ValueError('Event already accounted for; cannot confirm another action')
            return self.estimate()
        if event.game_seconds < self._anchor[0]:
            raise ValueError('Event predates last reconciliation; cannot replay across anchor')
        candidate = {**self._events, event_id: event}
        end = max(self._time, event.game_seconds)
        low, high = self._replay(end, candidate)
        # Commit only after validating the complete replay, including later plays.
        self._events = candidate
        self._seen[event_id] = event
        self._time, self._low, self._high = end, low, high
        if action_id is not None:
            self._pending.pop(action_id, None)
            self._completed[action_id] = (event_id, event.amount)
        return self.estimate()

    def confirm_play(self, *, event_id: str, cost: float,
                     game_seconds: float, action_id: str | None = None) -> ElixirEstimate:
        """Record one confirmed play/ability at its actual GAME timestamp.

        Delayed confirmations replay history; future events advance the clock.
        Do not use arrival time, spawned-body IDs, or the time a tap was sent.
        """
        return self._confirm(event_id, _Event('spend', self._number(game_seconds),
                                             self._amount(cost)), action_id)

    def confirm_gain(self, *, event_id: str, amount: float,
                     game_seconds: float) -> ElixirEstimate:
        """Account for a confirmed collector/golem/other credit, with cap 10."""
        return self._confirm(event_id, _Event('gain', self._number(game_seconds),
                                             self._amount(amount)))

    def reserve_play(self, *, action_id: str, cost: float,
                     game_seconds: float) -> ElixirEstimate:
        """Reserve BEFORE sending input to prevent spending the same budget twice.

        No automatic timeout: missing confirmation does not prove rejection.
        Confirm with the same action_id or cancel only after a known failure.
        """
        self._id(action_id)
        cost, at = self._amount(cost), self._number(game_seconds)
        if action_id in self._completed:
            raise ValueError('Action already completed; use a new action ID')
        if action_id in self._pending:
            if self._pending[action_id] != (cost, at):
                raise ValueError('Conflicting duplicate action ID')
            return self.estimate()
        if self.forecast(at).available_minimum + 1e-9 < cost:
            raise ValueError('Insufficient unreserved elixir')
        self.advance(at)
        self._pending[action_id] = (cost, at)
        return self.estimate()

    def cancel_play(self, action_id: str) -> ElixirEstimate:
        """Release a command known to have failed. Never refunds confirmed spend."""
        self._id(action_id)
        if action_id in self._completed:
            if self._completed[action_id][0] is not None:
                raise ValueError('Cannot cancel a confirmed play')
            return self.estimate()
        if action_id not in self._pending:
            raise ValueError('Unknown pending action')
        cost, _ = self._pending.pop(action_id)
        self._completed[action_id] = (None, cost)
        return self.estimate()

    def invalidate(self, game_seconds: float) -> ElixirEstimate:
        self.advance(game_seconds)
        self._low, self._high = 0.0, 10.0
        self._anchor = (self._time, self._low, self._high)
        self._events.clear()
        return self.estimate()

    def reconcile(self, *, game_seconds: float, value: float) -> ElixirEstimate:
        """Anchor to a trusted current measurement. Pending commands stay reserved.

        Older unseen events cannot be replayed across this new measurement.
        """
        value = self._amount(value)
        self.advance(game_seconds)
        self._low = self._high = value
        self._anchor = (self._time, value, value)
        self._events.clear()
        return self.estimate()

    def _estimate(self, at: float, low: float, high: float) -> ElixirEstimate:
        return ElixirEstimate(at, low, high, abs(high - low) < 1e-9,
                              source=('mathematical_standard_mode' if self.schedule ==
                                      STANDARD_SCHEDULE else 'mathematical_custom_mode'),
                              reserved=sum(cost for cost, _ in self._pending.values()),
                              pending_actions=len(self._pending))

    def estimate(self) -> ElixirEstimate:
        return self._estimate(self._time, self._low, self._high)

    def forecast(self, game_seconds: float) -> ElixirEstimate:
        """Non-mutating projection assuming no additional economy events."""
        at = self._number(game_seconds)
        if at < self._time:
            raise ValueError('Cannot forecast into the past')
        low, high = self._regen(self._low, self._high, self._time, at)
        return self._estimate(at, low, high)

    def can_afford(self, cost: float) -> bool:
        return self.estimate().available_minimum + 1e-9 >= self._amount(cost)

    def seconds_until_affordable(self, cost: float) -> float:
        """Conservative game-seconds to afford, assuming no additional events.

        Returns infinity if outstanding reservations make the target exceed
        the cap. This forecasts the model, not unknown future player actions.
        """
        cost = self._amount(cost)
        estimate = self.estimate()
        if estimate.available_minimum + 1e-9 >= cost:
            return 0.0
        target = cost + estimate.reserved
        if target > 10:
            return math.inf
        missing = max(0.0, target - self._low)
        cursor = self._time
        for index, (start, multiplier) in enumerate(self.schedule):
            boundary = (self.schedule[index + 1][0] if index + 1 < len(self.schedule)
                        else math.inf)
            left = max(cursor, start)
            if boundary <= left:
                continue
            duration = missing * 2.8 / multiplier
            if left + duration <= boundary:
                return left - self._time + duration
            missing -= (boundary - left) * multiplier / 2.8
            cursor = boundary
        raise AssertionError('Validated schedule must have an unbounded final phase')
