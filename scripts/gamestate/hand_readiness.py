"""Wait for a fresh native observation of a newly rotated card, per slot.

The hand vector can rotate before the corresponding UI card accepts a tap.
This is a conservative observation guard, not a measured game cooldown: it
does not delay other slots and does not assert that the next tap will succeed.
"""


class HandReadiness:
    def __init__(self):
        self.hand = None
        self.tick = None
        self.changed_at = {}

    def observe(self, tick, hand):
        hand = tuple(hand)
        if (type(tick) is not int or tick < 0 or len(hand) != 4
                or any(type(card) is not int or not 0 <= card < 8 for card in hand)
                or len(set(hand)) != 4):
            raise ValueError('A valid native tick and four distinct deck indices are required')
        if self.tick is not None and tick < self.tick:
            raise ValueError('Hand readiness clock moved backwards')
        if self.hand is not None:
            for slot, (before, after) in enumerate(zip(self.hand, hand)):
                if before != after:
                    self.changed_at[slot] = tick
        self.hand, self.tick = hand, tick
        return self.blocked_slots()

    def blocked_slots(self):
        return tuple(slot for slot, tick in sorted(self.changed_at.items())
                     if self.tick <= tick)
