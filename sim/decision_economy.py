"""Decision-time economy counters; diagnostics only, never an extra reward."""
from copy import deepcopy


def empty():
    return dict(decisions=0, forced_wait=0, voluntary_wait=0, plays=0, invalid=0,
                native_below4=0, available_below4=0, native_elixir_sum=0.,
                available_elixir_sum=0., cards={})


def record(stats, *, hand, mask, action, native_elixir, available_elixir):
    legal = [bool(mask[1+i*576:1+(i+1)*576].any()) for i in range(4)]
    valid = isinstance(action, int) and 0 <= action < len(mask) and bool(mask[action])
    stats['decisions'] += 1
    stats['native_below4'] += int(native_elixir < 4)
    stats['available_below4'] += int(available_elixir < 4)
    stats['native_elixir_sum'] += native_elixir
    stats['available_elixir_sum'] += available_elixir
    stats['invalid' if not valid else 'plays' if action else
          'voluntary_wait' if any(legal) else 'forced_wait'] += 1
    for slot, card in enumerate(hand):
        row = stats['cards'].setdefault(card, dict(in_hand=0, legal=0, selected=0))
        row['in_hand'] += 1
        row['legal'] += int(legal[slot])
        row['selected'] += int(valid and 0 < action < 2305 and (action-1)//576 == slot)


def snapshot(stats):
    return deepcopy(stats)
