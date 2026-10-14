"""Public opponent memory shared by simulator training and live inference."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DECK_SIZE = 8
CYCLE_GAP = 4
MAX_ELIXIR = 10.0
START_ELIXIR = 5.0
SUMMARY_FEATURES = 9

ALIASES = {
    "the_log": "log",
    "ice_spirits": "ice-spirit",
    "ice_golemite": "ice-golem",
    "ice_spirit": "ice-spirit",
    "ice_golem": "ice-golem",
    # Live vision reports troop entities, while the snapshot contains card
    # keys. A singular child is evidence of the parent card only when the
    # observer has accepted it as a new deployment.
    "archer": "archers",
    "barbarian": "barbarians",
    "bat": "bats",
    "elite_barbarian": "elite-barbarians",
    "goblin": "goblins",
    "guard": "guards",
    "minion": "minions",
    "royal_hog": "royal-hogs",
    "royal_recruit": "royal-recruits",
    "skeleton": "skeletons",
    "skeleton_dragon": "skeleton-dragons",
    "spear_goblin": "spear-goblins",
    "wall_breaker": "wall-breakers",
    "zappy": "zappies",
    "minipekka": "mini-pekka",
}

HOG_ANSWERS = frozenset({
    "cannon", "tesla", "inferno_tower", "bomb_tower", "goblin_cage",
    "tombstone", "furnace", "goblin_hut", "barbarian_hut",
    "elixir_collector", "skeletons", "skeleton_army", "goblins",
    "guards", "barbarians", "minipekka", "valkyrie", "knight", "bomber",
    "tornado", "electro_spirit", "ice_spirits", "bats", "minions",
    "mega_minion",
})

# Buildings are strategically different from the other Hog answers above.
# A troop can often be logged, tanked or out-damaged; a building changes the
# Hog's path and is best attacked by outcycling it or switching lane. Keep the
# subset explicit so the policy does not have to infer this category from a
# sparse one-hot during a short fine-tune.
BUILDING_ANSWERS = frozenset({
    "cannon", "tesla", "inferno_tower", "bomb_tower", "goblin_cage",
    "tombstone", "furnace", "goblin_hut", "barbarian_hut",
    "elixir_collector",
})


@lru_cache(maxsize=1)
def card_vocab() -> tuple[str, ...]:
    """Stable card vocabulary from the repository's versioned snapshot."""
    path = Path(__file__).resolve().parents[1] / "data" / "royaleapi" / "cards.json"
    try:
        cards = json.loads(path.read_text(encoding="utf-8"))["cards"]
        names = {str(card["key"]) for card in cards}
    except (OSError, KeyError, TypeError, ValueError):
        names = set()
    names.update(ALIASES.values())
    # Keep the two old synthetic slots so checkpoint input dimensions and
    # indices do not shift. New observations use the real snapshot keys.
    names.update(("ice_golemite", "ice_spirits"))
    return tuple(sorted(names))


def public_feature_count() -> int:
    # seen one-hot, definitely-unavailable one-hot, unknown bucket for each,
    # then compact elixir/deck/counter/building summaries.
    return 2 * (len(card_vocab()) + 1) + SUMMARY_FEATURES


@lru_cache(maxsize=256)
def _canon(card: str) -> str:
    name = ALIASES.get(card, card)
    slug = name.replace("_", "-")
    return slug if slug in card_vocab() else name


# The summary counters must use the same keys as the seen-card one-hot.
# Otherwise Inferno Tower, Skeleton Army, etc. appear as known cards while
# silently disappearing from the Hog-answer and building counters.
HOG_ANSWERS = frozenset(_canon(card) for card in HOG_ANSWERS)
BUILDING_ANSWERS = frozenset(_canon(card) for card in BUILDING_ANSWERS)


def _regen_period_ms(at_ms: int) -> int:
    if at_ms >= 240_000:
        return 933
    if at_ms >= 120_000:
        return 1400
    return 2800


def _regenerate(low: float, high: float, start_ms: int,
                end_ms: int) -> tuple[float, float]:
    """Integrate elixir exactly across the 2x and 3x phase boundaries."""
    cursor = start_ms
    while cursor < end_ms:
        boundary = min(end_ms, 120_000 if cursor < 120_000 else
                       240_000 if cursor < 240_000 else end_ms)
        gain = (boundary - cursor) / _regen_period_ms(cursor)
        low, high = min(MAX_ELIXIR, low + gain), min(MAX_ELIXIR, high + gain)
        cursor = boundary
    return low, high


def _elixir_interval(events: list[dict], now_ms: int) -> tuple[float, float, float]:
    """Conservative min/mean/max from observed spending only."""
    low = high = START_ELIXIR
    last = 0
    recent = 0.0
    for event in events:
        at = min(now_ms, max(last, int(event.get("at_ms", last))))
        # Phase changes are rare, so integrate in short exact segments.
        low, high = _regenerate(low, high, last, at)
        cost = max(0.0, float(event.get("cost", 0)))
        low, high = max(0.0, low - cost), max(0.0, high - cost)
        if now_ms - at <= 5000:
            recent += cost
        last = at
    if last < now_ms:
        low, high = _regenerate(low, high, last, now_ms)
    return low, high, min(MAX_ELIXIR, recent)


def vector_from_events(events: list[dict], now_ms: int) -> np.ndarray:
    """Encode only opponent plays visible to a real player."""
    vocab = card_vocab()
    index = {card: i for i, card in enumerate(vocab)}
    width = len(vocab) + 1
    out = np.zeros(public_feature_count(), dtype=np.float32)
    seen_order: list[str] = []
    last_play: dict[str, int] = {}
    clean = []
    for play_index, raw in enumerate(events, 1):
        card = _canon(str(raw.get("card", "")))
        if not card:
            continue
        clean.append(raw)
        if card not in seen_order and len(seen_order) < DECK_SIZE:
            seen_order.append(card)
        last_play[card] = play_index
    unknown = len(vocab)
    for card in seen_order:
        out[index.get(card, unknown)] = 1.0
    plays = len(clean)
    for card, last in last_play.items():
        if plays - last < CYCLE_GAP:
            out[width + index.get(card, unknown)] = 1.0

    low, high, recent = _elixir_interval(clean, now_ms)
    answers = [card for card in seen_order if card in HOG_ANSWERS]
    answer_ready = (not answers or any(plays - last_play[card] >= CYCLE_GAP
                                       for card in answers))
    buildings = [card for card in seen_order if card in BUILDING_ANSWERS]
    building_ready = bool(buildings) and any(
        plays - last_play[card] >= CYCLE_GAP for card in buildings)
    base = 2 * width
    out[base:] = (
        (low + high) / 2 / MAX_ELIXIR,
        low / MAX_ELIXIR,
        high / MAX_ELIXIR,
        len(seen_order) / DECK_SIZE,
        recent / MAX_ELIXIR,
        len(answers) / DECK_SIZE,
        float(answer_ready),
        len(buildings) / DECK_SIZE,
        float(building_ready),
    )
    return out


def vector_from_live(opponent, economy, now: float) -> np.ndarray:
    """Same layout from the live observer's already-deduplicated models."""
    vocab = card_vocab()
    index = {card: i for i, card in enumerate(vocab)}
    width = len(vocab) + 1
    unknown = len(vocab)
    out = np.zeros(public_feature_count(), dtype=np.float32)
    for raw in opponent.deck:
        out[index.get(_canon(raw), unknown)] = 1.0
    for raw in opponent.last_play_index:
        if opponent.definitely_unavailable(raw):
            out[width + index.get(_canon(raw), unknown)] = 1.0
    base = 2 * width
    estimate = max(0.0, min(MAX_ELIXIR, float(economy.elixir)))
    confidence = min(1.0, len(opponent.deck) / DECK_SIZE)
    uncertainty = 2.0 * (1.0 - confidence)
    out[base:base + 5] = (
        estimate / MAX_ELIXIR,
        max(0.0, estimate - uncertainty) / MAX_ELIXIR,
        min(MAX_ELIXIR, estimate + uncertainty) / MAX_ELIXIR,
        confidence,
        min(MAX_ELIXIR, economy.recent_spend(now)) / MAX_ELIXIR,
    )
    answers = [raw for raw in opponent.deck
               if _canon(raw) in HOG_ANSWERS]
    out[base + 5] = len(answers) / DECK_SIZE
    out[base + 6] = float(not answers or any(
        not opponent.definitely_unavailable(raw) for raw in answers))
    buildings = [raw for raw in opponent.deck
                 if _canon(raw) in BUILDING_ANSWERS]
    out[base + 7] = len(buildings) / DECK_SIZE
    out[base + 8] = float(bool(buildings) and any(
        not opponent.definitely_unavailable(raw) for raw in buildings))
    return out
