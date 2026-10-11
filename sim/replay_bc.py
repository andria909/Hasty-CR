"""Adapt trusted classic Hog 2.6 replay rows to the live deck's PPO tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .env import (ACTIONS, BASE_SCALARS, DECK_26, GRID_H, GRID_W, NUM_PLANES,
                  NUM_SCALARS, SPELLS, TILES, _placeable,
                  building_lane_risks, tactical_plane_values)
from .public_state import (BUILDING_ANSWERS, HOG_ANSWERS, card_vocab as public_vocab,
                           public_feature_count)
from .replay_rows import PAD, card_vocab as replay_card_vocab, unit_vocab

# Semantic substitutions for the account's actual deck.  The unit-plane code
# below also substitutes the allied on-board body, because a Knight is not an
# Ice Golem with another label: it has different HP, speed and DPS.
ADAPT_CARD = {
    "ice_golemite": "knight",
    "ice_spirits": "electro_spirit",
    "log": "zap",
}
ADAPT_UNIT = {
    "ice_golemite": "knight",
    "ice_spirit": "electro_spirit",
}


def _inverse(mapping: dict[str, int]) -> dict[int, str]:
    return {value: key for key, value in mapping.items()}


def _card(name: str) -> str:
    return ADAPT_CARD.get(name, name)


def _unit_specs() -> dict:
    from .entities import speed_to_mt_per_sec
    from .gamedata import load_gamedata

    specs = {}
    for card in load_gamedata(11).values():
        unit = getattr(card, "unit", None)
        if unit is not None:
            specs.setdefault(unit.name, unit)
    # Keep the conversion function with the table so call sites cannot forget
    # that game data stores speed per tick while entities store speed/second.
    return specs, speed_to_mt_per_sec


def _public(row, reverse_cards: dict[int, str]) -> np.ndarray:
    vocab = public_vocab()
    index = {name: i for i, name in enumerate(vocab)}
    width = len(vocab) + 1
    unknown = len(vocab)
    out = np.zeros(public_feature_count(), dtype=np.float32)

    seen = [_card(reverse_cards.get(int(value), ""))
            for value in row["opp_seen"] if int(value) != PAD]
    recent = [_card(reverse_cards.get(int(value), ""))
              for value in row["opp_last"] if int(value) != PAD]
    for name in seen:
        out[index.get(name, unknown)] = 1.0
    for name in recent:
        out[width + index.get(name, unknown)] = 1.0

    answers = [name for name in seen if name in HOG_ANSWERS]
    buildings = [name for name in seen if name in BUILDING_ANSWERS]
    base = 2 * width
    # The compact replay row intentionally does not store oracle opponent
    # elixir.  Preserve that uncertainty instead of fabricating a precise bar.
    out[base:base + 5] = (0.5, 0.0, 1.0, len(seen) / 8.0, 0.0)
    out[base + 5:] = (
        len(answers) / 8.0,
        float(not answers or any(name not in recent for name in answers)),
        len(buildings) / 8.0,
        float(bool(buildings) and any(name not in recent for name in buildings)),
    )
    return out


def _observation(row, reverse_cards: dict[int, str],
                 reverse_units: dict[int, str], specs: dict,
                 speed_to_mt_per_sec) -> tuple[dict, list[str]]:
    planes = np.zeros((NUM_PLANES, GRID_H, GRID_W), dtype=np.float32)
    enemy_buildings = []
    for uid, features in zip(row["unit_ids"], row["unit_feats"]):
        if int(uid) == PAD:
            continue
        ours = bool(float(features[0]) >= 0.5)
        name = reverse_units.get(int(uid), "")
        if ours:
            name = ADAPT_UNIT.get(name, name)
        spec = specs.get(name)
        x = int(np.clip(float(features[1]) * GRID_W, 0, GRID_W - 1))
        y = int(np.clip(float(features[2]) * GRID_H, 0, GRID_H - 1))
        base = 0 if ours else 1
        planes[base, y, x] += 1.0
        max_hp = float(getattr(spec, "hitpoints", 0) or 0)
        planes[2 + base, y, x] += float(features[3]) * max_hp / 1000.0
        if bool(features[4]):
            planes[4 + base, y, x] += 1.0
        if bool(features[5]):
            planes[6 + base, y, x] += 1.0
            if not ours:
                enemy_buildings.append((x, y))
        if spec is not None:
            values = tactical_plane_values(
                damage=getattr(spec, "damage", 0),
                hit_speed_ms=getattr(spec, "hit_speed_ms", 1000),
                range_mt=getattr(spec, "range_mt", 0),
                speed_mt_per_sec=speed_to_mt_per_sec(
                    int(getattr(spec, "speed_mt_per_sec", 0) or 0)),
                attacks_air=getattr(spec, "attacks_air", False),
                target_only_buildings=getattr(
                    spec, "target_only_buildings", False),
                splash_radius_mt=getattr(spec, "splash_radius_mt", 0))
            for pair, value in enumerate(values):
                planes[8 + 2 * pair + base, y, x] += value

    hand = [_card(reverse_cards.get(int(value), ""))
            for value in row["hand"]]
    scalars = np.zeros(NUM_SCALARS, dtype=np.float32)
    tick = int(row["tick"])
    scalars[0] = float(row["scalars"][0])
    scalars[1] = min(1.0, tick * 50 / 180_000.0)
    scalars[2] = (3.0 if tick >= 4800 else 2.0 if tick >= 2400 else 1.0) / 3.0
    towers = row["towers"]
    scalars[3:7] = (towers[0], towers[1], towers[3], towers[4])
    base = 7
    for slot, name in enumerate(hand):
        if name in DECK_26:
            scalars[base + slot * len(DECK_26) + DECK_26.index(name)] = 1.0
    next_name = _card(reverse_cards.get(int(row["next"]), ""))
    if next_name in DECK_26:
        scalars[base + 4 * len(DECK_26) + DECK_26.index(next_name)] = 1.0
    public_end = BASE_SCALARS + public_feature_count()
    scalars[BASE_SCALARS:public_end] = _public(row, reverse_cards)
    scalars[public_end:] = building_lane_risks(enemy_buildings)
    return {"planes": planes, "scalars": scalars}, hand


def load_replay_rows(path: Path, *, winners_only: bool = True,
                     hold_ratio: float = 2.0) -> tuple[dict, np.ndarray, np.ndarray]:
    """Return PPO-format observations, actions and fixed corpus splits.

    Only exact-hand, pre-divergence rows from the Hog side are eligible.  Holds
    are deterministically capped relative to actual plays so a once-per-second
    sampling cadence cannot teach the trivial always-wait policy.
    """
    files = sorted(path.glob("shard-*.npz")) if path.is_dir() else [path]
    reverse_cards = _inverse(replay_card_vocab())
    reverse_units = _inverse(unit_vocab())
    specs, speed_converter = _unit_specs()
    candidates = []
    for filename in files:
        shard = np.load(filename)
        for index in range(len(shard["tick"])):
            if not (bool(shard["hog"][index])
                    and bool(shard["trusted"][index])
                    and bool(shard["hand_exact"][index])):
                continue
            if winners_only and float(shard["won"][index]) <= 0.5:
                continue
            row = {key: shard[key][index] for key in shard.files}
            candidates.append(row)

    plays = [row for row in candidates if int(row["act_card"]) != PAD]
    holds = [row for row in candidates if int(row["act_card"]) == PAD]
    keep_holds = min(len(holds), int(round(max(0.0, hold_ratio) * len(plays))))
    candidates = plays + holds[:keep_holds]

    observations, actions, splits = [], [], []
    for row in candidates:
        observation, hand = _observation(
            row, reverse_cards, reverse_units, specs, speed_converter)
        mask = np.zeros(ACTIONS, dtype=bool)
        mask[0] = True
        for slot, name in enumerate(hand):
            if name not in DECK_26:
                continue
            grid = _placeable(1, (), name in SPELLS)
            mask[1 + slot * TILES:1 + (slot + 1) * TILES] = grid.reshape(-1)

        played = _card(reverse_cards.get(int(row["act_card"]), ""))
        if int(row["act_card"]) == PAD:
            action = 0
        elif played in hand:
            slot = hand.index(played)
            x = int(np.clip(row["act_x"], 0, GRID_W - 1))
            y = int(np.clip(row["act_y"], 0, GRID_H - 1))
            action = 1 + slot * TILES + y * GRID_W + x
            mask[action] = True
        else:
            continue
        observation["mask"] = mask
        observations.append(observation)
        actions.append(action)
        splits.append(str(row["split"]))

    data = {
        "planes": np.stack([row["planes"] for row in observations]),
        "scalars": np.stack([row["scalars"] for row in observations]),
        "masks": np.stack([row["mask"] for row in observations]),
    }
    return data, np.asarray(actions, dtype=np.int64), np.asarray(splits)
