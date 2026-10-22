"""Exact public rarity multipliers; no guessed growth or extrapolation."""
from __future__ import annotations


def scale_public_stat(raw, rarity, level, rarities):
    row = rarities[rarity]
    index = int(level) - (int(row['relative_level']) + 1)
    if index < 0:
        raise ValueError('Requested level predates rarity base level')
    if index == 0:
        return int(raw)
    multipliers = row['power_level_multiplier']
    if index > len(multipliers):
        raise ValueError('Exact public rarity multiplier is unavailable')
    return int(raw) * int(multipliers[index-1]) // 100
