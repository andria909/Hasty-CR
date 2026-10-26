"""A headless Gym-style environment, for training a policy rather than tuning one.

Everything else in `sim/` answers "is setting X better than setting Y" by
playing whole matches with the hand-written policy. This exposes the same engine
one decision at a time, so a learned policy can act in it.

    env = ClashEnv(seed=0)
    obs, info = env.reset()
    while not done:
        action = pick(obs, info["action_mask"])
        obs, reward, terminated, truncated, info = env.step(action)

Design notes worth knowing before training anything on it.

**The opponent is the hand-written brain, not a random agent.** A random
opponent is not a test: this policy beat one 25-0, which says nothing about the
policy. The vendored ClashAI simulator fell into exactly that trap - random
actions won 12 of its 20 matches.

**Actions are masked, not penalised.** The legal set changes every step (elixir,
what is in hand, which half a card may be placed in), and 2,305 actions with
maybe 40 legal is far too sparse to learn from rejection alone. `info` carries a
boolean mask; illegal actions are treated as no-ops and flagged in `info`.

**The reward is shaped by tower damage, not only by the result.** A match is
~900 decisions and one terminal signal, which is a hard credit assignment
problem. Tower fractions are dense, monotone, and exactly what the game scores,
so the shaping is the objective rather than a proxy for it.

**Elixir is symmetric and honest.** The opponent's elixir is *not* in the
observation, because it is not observable in the real game either - the live
bot has to infer it (see `brain/economy.py`). Training on an oracle the deployed
agent will not have is how a simulator policy stops transferring.
"""

from __future__ import annotations

import random

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from sim import arena as sim_arena          # noqa: E402
from sim.adapter import grid_to_point       # noqa: E402
from sim.match import Match                 # noqa: E402
from sim.policy_coordinates import policy_coordinates  # noqa: E402
from brain import spellinfo                 # noqa: E402

GRID_W, GRID_H = 18, 32
TILES = GRID_W * GRID_H
NUM_SLOTS = 4
CARD_ACTIONS = 1 + NUM_SLOTS * TILES       # index 0 is "hold elixir"
# An ability targets the currently living friendly champion/hero by stable UID
# rank, so the policy gets a real button press without exposing an arbitrary
# internal UID as part of its fixed action tensor.
ABILITY_SLOTS = 16
ACTIONS = CARD_ACTIONS + ABILITY_SLOTS

DECIDE_EVERY_MS = 500
ACTION_DELAY_MS = 500
MAX_MS = 310_000

# The deck actually used by the live account.  Keep the order stable: the
# policy's hand/next-card one-hots use these indices, so the three replacements
# deliberately inherit the closest 2.6 semantics from the old checkpoint.
DECK_26 = ["cannon", "fireball", "hog_rider", "knight",
           "electro_spirit", "musketeer", "skeletons", "zap"]
SPELLS = {"fireball", "zap"}
KITE_CARDS = {"knight", "electro_spirit", "skeletons"}

# Spatial planes. One plane per card type would be 100+ mostly-empty channels,
# but the original eight-plane summary was too lossy: a Hog, Knight and Giant
# could become nearly indistinguishable once their name was discarded.  The
# tactical planes keep the representation compact while exposing the properties
# that determine a defence: damage rate, range, speed, air targeting, building
# targeting and splash.  They are paired ally/enemy exactly like the original
# count/HP/air/building planes so the live adapter can reproduce them.
PLANES = ("ally_units", "enemy_units", "ally_hp", "enemy_hp",
          "ally_air", "enemy_air", "ally_buildings", "enemy_buildings",
          "ally_dps", "enemy_dps", "ally_range", "enemy_range",
          "ally_speed", "enemy_speed", "ally_attacks_air", "enemy_attacks_air",
          "ally_building_targeter", "enemy_building_targeter",
          "ally_splash", "enemy_splash")
NUM_PLANES = len(PLANES)


def tactical_plane_values(*, damage: float, hit_speed_ms: float,
                          range_mt: float, speed_mt_per_sec: float,
                          attacks_air: bool, target_only_buildings: bool,
                          splash_radius_mt: float) -> tuple[float, ...]:
    """Normalised public unit traits for the six tactical plane pairs.

    Values are clipped so a single exotic card cannot dominate a whole input
    tensor.  These are descriptors, not hidden simulator state, and the live
    detector can reconstruct every value from the detected unit name.
    """
    dps = float(damage) * 1000.0 / max(100.0, float(hit_speed_ms))
    return (
        min(3.0, dps / 500.0),
        min(2.0, max(0.0, float(range_mt)) / 6000.0),
        min(2.0, max(0.0, float(speed_mt_per_sec)) / 1000.0),
        float(bool(attacks_air)),
        float(bool(target_only_buildings)),
        min(2.0, max(0.0, float(splash_radius_mt)) / 2000.0),
    )

# Scalars: elixir, elapsed, elixir multiplier, our two tower fractions and
# their two, hand one-hot (4 slots x 8 cards), next card one-hot (8).
LANES = ("left", "right")
RIVER_ROW = 16                 # our half is y >= 16 in the policy's grid
from sim.public_state import public_feature_count, vector_from_events  # noqa: E402

BASE_SCALARS = 3 + 2 * len(LANES) + NUM_SLOTS * len(DECK_26) + len(DECK_26)
ACTIVE_BUILDING_FEATURES = 2
NUM_SCALARS = BASE_SCALARS + public_feature_count() + ACTIVE_BUILDING_FEATURES


# --------------------------------------------------------------- both seats
#
# A policy is written to play the bottom of the board, so the top player's view
# is the board mirrored: `grid_to_point` already flips a top placement through
# (17-x, 31-y), and these flip the observation the same way. One frame of
# reference, two seats - which is what lets the same network drive both sides
# in self-play without learning each orientation separately.


def ability_entities(match, side: int):
    """Living friendly units with an ability, in stable UID order."""
    if match is None:
        return []
    return sorted((entity for entity in match.battle.entities.values()
                   if entity.side == side and entity.alive and entity.ability_buff),
                  key=lambda entity: entity.uid)


def building_lane_risks(building_cells) -> tuple[float, float]:
    """Public pull risk at the left/right bridge from active buildings.

    A central building can threaten both lanes; an off-centre one should make
    the opposite lane preferable. Distances use the same bottom-oriented grid
    the policy acts on and a conservative ten-tile Hog acquisition radius.
    """
    anchors = ((4.0, float(RIVER_ROW)), (13.0, float(RIVER_ROW)))
    risks = []
    for anchor_x, anchor_y in anchors:
        risk = max((max(0.0, 1.0 - (((x - anchor_x) ** 2
                                     + (y - anchor_y) ** 2) ** 0.5) / 10.0)
                    for x, y in building_cells), default=0.0)
        risks.append(risk)
    return risks[0], risks[1]


def observe(match, side: int = 1, *,
            coordinate_mode: str = "legacy_training") -> dict:
    """Encode the board with an explicit, opt-in coordinate contract.

    Defaults retain legacy rounding/Y flip for existing checkpoints. V2 bins
    physical positions with floor before mirroring the cell for the top seat.
    This option does not change action geometry or strategic masks.
    """
    coordinates = policy_coordinates(coordinate_mode)
    planes = np.zeros((NUM_PLANES, GRID_H, GRID_W), dtype=np.float32)
    if match is None:
        return {"planes": planes, "scalars": np.zeros(NUM_SCALARS, dtype=np.float32)}
    enemy_building_cells = []
    for entity in match.battle.entities.values():
        if not getattr(entity, "alive", True):
            continue
        if getattr(entity, "is_tower", False):
            continue
        grid_x, grid_y = coordinates.cell(entity.pos, side)
        ours = (getattr(entity, "side", 1) > 0) == (side > 0)
        base = 0 if ours else 1
        planes[base, grid_y, grid_x] += 1.0
        planes[2 + base, grid_y, grid_x] += float(
            getattr(entity, "hitpoints", 0)) / 1000.0
        if getattr(entity, "flying", False):
            planes[4 + base, grid_y, grid_x] += 1.0
        if getattr(entity, "is_building", False):
            planes[6 + base, grid_y, grid_x] += 1.0
            if not ours:
                enemy_building_cells.append((grid_x, grid_y))
        for pair, value in enumerate(tactical_plane_values(
                damage=getattr(entity, "damage", 0),
                hit_speed_ms=getattr(entity, "hit_speed_ms", 1000),
                range_mt=getattr(entity, "range_mt", 0),
                speed_mt_per_sec=getattr(entity, "speed_mt_per_sec", 0),
                attacks_air=getattr(entity, "attacks_air", False),
                target_only_buildings=getattr(
                    entity, "target_only_buildings", False),
                splash_radius_mt=getattr(entity, "splash_radius_mt", 0))):
            planes[8 + 2 * pair + base, grid_y, grid_x] += value

    scalars = np.zeros(NUM_SCALARS, dtype=np.float32)
    player = match.players[side]
    scalars[0] = player.elixir / 10_000.0
    scalars[1] = min(1.0, match.elapsed_ms / 180_000.0)
    # Elixir regen expressed as a multiplier (1x, 2x, 3x) rather than a
    # period, so the scale matches how the game talks about it.
    scalars[2] = 2800.0 / max(1, match.regen_ms()) / 3.0
    ours = match.tower_fractions(side)
    theirs = match.tower_fractions(-side)
    base = 3
    for index, key in enumerate(LANES):
        scalars[base + index] = ours.get(key, 0.0)
        scalars[base + len(LANES) + index] = theirs.get(key, 0.0)
    base += 2 * len(LANES)
    for slot, card in enumerate(player.hand[:NUM_SLOTS]):
        if card in DECK_26:
            scalars[base + slot * len(DECK_26) + DECK_26.index(card)] = 1.0
    nxt = player.next_card
    if nxt in DECK_26:
        scalars[base + NUM_SLOTS * len(DECK_26) + DECK_26.index(nxt)] = 1.0
    public = [event for event in match.play_log if event["side"] == -side]
    public_end = BASE_SCALARS + public_feature_count()
    scalars[BASE_SCALARS:public_end] = vector_from_events(
        public, match.elapsed_ms)
    scalars[public_end:] = building_lane_risks(enemy_building_cells)
    return {"planes": planes, "scalars": scalars}


# Which cells a card may go in, by (side, which enemy towers are down, spell).
# The answer depends only on geometry, and it was being recomputed from
# scratch every step: 4 slots x 32 rows x 18 columns of `grid_to_point` plus
# `deploy_area_ok`, in Python, twice per step once the opponent also learned.
# There are eight distinct answers in a whole run.
_PLACEMENT_CACHE: Dict[tuple, np.ndarray] = {}


def _placeable(side: int, enemy_down: tuple, spell: bool, *, geometry=None) -> np.ndarray:
    profile = sim_arena.resolve_geometry(geometry)
    key = (side, enemy_down, spell, profile)
    cached = _PLACEMENT_CACHE.get(key)
    if cached is not None:
        return cached
    grid = np.zeros((GRID_H, GRID_W), dtype=bool)
    for y in range(GRID_H):
        # Deliberately *not* short-circuiting the enemy half for troops here.
        # It reads like a free optimisation - troops go on your own side - but
        # it is only true while both princess towers stand. Taking one opens a
        # strip in front of it, which is how a crown snowballs into the next
        # one, and `deploy_area_ok` has always implemented that. Skipping those
        # rows meant the rule was unreachable from the action space: the agent
        # could take a tower and gain none of the ground it should have won.
        for x in range(GRID_W):
            point = grid_to_point(x, y, side)
            grid[y, x] = (sim_arena.in_arena(point) if spell
                          else sim_arena.deploy_area_ok(point, side,
                                                        list(enemy_down), geometry=profile))
    _PLACEMENT_CACHE[key] = grid
    return grid


def legal_mask(match, cards, side: int = 1, *, strategic: bool = True) -> np.ndarray:
    """Which actions `side` may take right now. Index 0 (hold) always is."""
    mask = np.zeros(ACTIONS, dtype=bool)
    mask[0] = True
    if match is None or match.finished:
        return mask
    player = match.players[side]
    enemy_down = [key for key, value in match.tower_fractions(-side).items()
                  if value <= 0.0 and key in ("left", "right")]
    for slot, card in enumerate(player.hand[:NUM_SLOTS]):
        spec = cards.get(card)
        cost = getattr(spec, "cost", 0) * 1000 if spec else 0
        if player.elixir < cost:
            continue
        grid = _placeable(side, tuple(enemy_down), card in SPELLS,
                          geometry=getattr(match, "geometry", None))
        base = 1 + slot * TILES
        mask[base:base + TILES] = grid.reshape(-1)
    if strategic:
        mask = strategic_action_mask(match, player.hand[:NUM_SLOTS], mask, side)
    for rank, entity in enumerate(ability_entities(match, side)[:ABILITY_SLOTS]):
        if match.can_activate_ability(side, entity.uid):
            mask[CARD_ACTIONS + rank] = True
    return mask


def _policy_cell(entity, side: int) -> tuple[int, int]:
    """Entity position in the same bottom-oriented grid the policy acts on."""
    x, y = sim_arena.to_tiles(entity.pos)
    gx = int(np.clip(round(x), 0, GRID_W - 1))
    gy = int(np.clip(round(31 - y), 0, GRID_H - 1))
    if side < 0:
        gx, gy = GRID_W - 1 - gx, GRID_H - 1 - gy
    return gx, gy


def strategic_action_mask(match, hand, mask: np.ndarray,
                          side: int = 1) -> np.ndarray:
    """Training-time counterpart of the live ladder safety mask.

    PPO must never be rewarded for actions deployment will remove.  Hog is
    withheld while enemies are on our half and otherwise constrained to the
    two bridge pockets.  Area spells require a current cluster (or a reset
    target for Zap), with Fireball tower finishing retained.
    """
    safe = mask.copy()
    enemies = [entity for entity in match.battle.entities.values()
               if getattr(entity, "alive", True)
               and not getattr(entity, "is_tower", False)
               and (getattr(entity, "side", 1) > 0) != (side > 0)]
    cells = [(entity, *_policy_cell(entity, side)) for entity in enemies]
    pressure = [(entity, x, y) for entity, x, y in cells
                if not getattr(entity, "is_building", False)
                and (y >= RIVER_ROW or
                     (getattr(entity, "target_only_buildings", False)
                      and y >= RIVER_ROW - 2))]
    ground_pressure = [(entity, x, y) for entity, x, y in pressure
                       if not getattr(entity, "flying", False)]
    air_pressure = [(entity, x, y) for entity, x, y in pressure
                    if getattr(entity, "flying", False)]
    pure_air = bool(air_pressure) and not ground_pressure
    player = match.players[side]
    elixir = player.elixir / 1000.0
    enemy_events = [event for event in match.play_log if event["side"] == -side]
    # Same public estimate that is fed to the policy. Do not leak the
    # simulator's exact opponent bar through the action mask.
    enemy_public = vector_from_events(enemy_events, match.elapsed_ms)
    enemy_elixir = float(enemy_public[-9]) * 10.0
    allies = [entity for entity in match.battle.entities.values()
              if getattr(entity, "alive", True)
              and not getattr(entity, "is_tower", False)
              and (getattr(entity, "side", 1) > 0) == (side > 0)]
    allied_names = {str(getattr(entity, "name", "")) for entity in allies}
    allied_hogs = [entity for entity in match.battle.entities.values()
                   if getattr(entity, "alive", True)
                   and (getattr(entity, "side", 1) > 0) == (side > 0)
                   and str(getattr(entity, "name", "")) == "hog_rider"
                   and _policy_cell(entity, side)[1] <= 21]
    hog_inbound = bool(allied_hogs)
    hog_lane_left = (_policy_cell(allied_hogs[-1], side)[0] < GRID_W / 2
                     if allied_hogs else None)
    ours = match.tower_fractions(side)
    theirs = match.tower_fractions(-side)
    crown_lead = (sum(float(ours.get(lane, 0.0)) <= 0.0
                      for lane in ("left", "right"))
                  < sum(float(theirs.get(lane, 0.0)) <= 0.0
                        for lane in ("left", "right")))

    def entity_card(entity):
        direct = match.cards.get(str(getattr(entity, "name", "")))
        if direct is not None:
            return direct
        for spec in match.cards.values():
            unit = getattr(spec, "unit", None)
            if unit is not None and unit.name == getattr(entity, "name", ""):
                return spec
        return None

    def slot_grid(card):
        try:
            slot = list(hand).index(card)
        except ValueError:
            return None
        return safe[1 + slot * TILES:1 + (slot + 1) * TILES].reshape(
            GRID_H, GRID_W)

    def focus_of(units):
        lane_risk = {"left": 0.0, "right": 0.0}
        for entity, x, y in units:
            lane = "left" if x < GRID_W / 2 else "right"
            depth = max(0.0, y - 16.0)
            lane_risk[lane] += (1.0 + depth / 5.0) * (
                1.7 if getattr(entity, "target_only_buildings", False) else 1.0)
        lane = max(lane_risk, key=lane_risk.get)
        return max((item for item in units
                    if (item[1] < GRID_W / 2) == (lane == "left")),
                   key=lambda item: item[2])

    if not pressure:
        for card in ("cannon", "musketeer", "knight"):
            grid = slot_grid(card)
            if grid is not None:
                grid[:] = False
        skeletons = slot_grid("skeletons")
        if skeletons is not None:
            combo_ready = "hog_rider" in hand and "electro_spirit" in hand
            if elixir < 10.0 or combo_ready:
                skeletons[:] = False
            else:
                back = np.zeros_like(skeletons)
                back[28:32, 6:12] = True
                skeletons &= back
        spirit = slot_grid("electro_spirit")
        if spirit is not None:
            if not hog_inbound:
                spirit[:] = False
            else:
                allowed = np.zeros_like(spirit)
                xs = slice(2, 7) if hog_lane_left else slice(11, 16)
                allowed[15:21, xs] = True
                spirit &= allowed

    if pure_air:
        for card in ("cannon", "knight", "skeletons", "hog_rider"):
            grid = slot_grid(card)
            if grid is not None:
                grid[:] = False
        musketeer = slot_grid("musketeer")
        if musketeer is not None:
            lane_left = sum(x for _entity, x, _y in air_pressure) / len(air_pressure) < 8.5
            x = 9 if lane_left else 8
            y = int(np.clip(max(y for _entity, _x, y in air_pressure) + 4, 21, 26))
            allowed = np.zeros_like(musketeer)
            allowed[max(16, y - 1):min(GRID_H, y + 2), max(0, x - 1):x + 2] = True
            musketeer &= allowed

    if pressure:
        focus_entity, focus_x, focus_y = focus_of(pressure)
        lane_left = focus_x < GRID_W / 2
        late_contact = focus_y >= 22.5
        musketeer = slot_grid("musketeer")
        if musketeer is not None:
            allowed = np.zeros_like(musketeer)
            centre_x = 9 if lane_left else 8
            allowed[23:28, centre_x - 2:centre_x + 3] = True
            musketeer &= allowed

        knight = slot_grid("knight")
        if knight is not None:
            allowed = np.zeros_like(knight)
            from sim.kite_geometry import close_kite_cell
            close_x,close_y=close_kite_cell(focus_x,focus_y)
            kite_x = focus_x if late_contact else close_x
            y = (int(np.clip(focus_y, 21, 24)) if late_contact
                 else close_y)
            allowed[max(16, y - 1):min(GRID_H, y + 2),
                    max(0, kite_x - 1):min(GRID_W, kite_x + 2)] = True
            knight &= allowed

        skeletons = slot_grid("skeletons")
        if skeletons is not None:
            if all(bool(getattr(entity, "splash_radius_mt", 0))
                   for entity, _x, _y in ground_pressure):
                skeletons[:] = False
            else:
                allowed = np.zeros_like(skeletons)
                targeter = (bool(getattr(focus_entity, "target_only_buildings", False))
                            and not bool(getattr(focus_entity, "flying", False)))
                y = int(np.clip(focus_y + 1, 19, 24 if targeter else 23))
                x = focus_x if targeter else 9
                allowed[y - 1:y + 2,
                        max(0, x - 1):min(GRID_W, x + 2)] = True
                skeletons &= allowed

        spirit = slot_grid("electro_spirit")
        if spirit is not None:
            allowed = np.zeros_like(spirit)
            y = int(np.clip(focus_y, 18, 22))
            allowed[y - 1:y + 2, 7:11] = True
            spirit &= allowed

    cannon = slot_grid("cannon")
    if cannon is not None and ground_pressure:
        cannon_spec = match.cards.get("cannon")
        cannon_range = float(getattr(getattr(cannon_spec, "unit", None),
                                     "range_mt", 5500) or 5500)
        pullable = [(entity, x, y) for entity, x, y in ground_pressure
                    if not getattr(entity, "range_mt", 0)
                    or getattr(entity, "range_mt", 0) <= cannon_range]
        pullable = [(entity, x, y) for entity, x, y in pullable
                    if not (getattr(entity, "target_only_buildings", False)
                            and y >= 22.5)]
        cheap_ranged = (len(pullable) == 1
                        and not getattr(pullable[0][0], "target_only_buildings", False)
                        and getattr(pullable[0][0], "range_mt", 0) > 1000
                        and float(getattr(entity_card(pullable[0][0]), "cost", 99)) <= 3)
        cheaper_answer = any(
            grid is not None and grid.any()
            for grid in (slot_grid("knight"), slot_grid("skeletons"),
                         slot_grid("electro_spirit")))
        if not pullable or (cheap_ranged and cheaper_answer):
            cannon[:] = False
        else:
            focus_entity, focus_x, _focus_y = focus_of(pullable)
            x = 8 if focus_x < 8.5 else 9
            allowed = np.zeros_like(cannon)
            fast_pull = any(str(getattr(entity, "name", "")) in {
                "hog", "hog_rider", "battle_ram", "ram_rider",
                "royal_hog", "wall_breaker",
            } for entity, _x, _y in pullable)
            y = 20 if fast_pull else 21
            allowed[y:y + 1, x - 1:x + 2] = True
            cannon &= allowed
            needs_pull = (any(bool(getattr(entity, "target_only_buildings", False))
                              for entity, _x, _y in pullable)
                          and "cannon" not in allied_names)
            if needs_pull and cannon.any():
                for card in ("knight", "skeletons", "electro_spirit"):
                    candidate = slot_grid(card)
                    if candidate is not None:
                        candidate[:] = False

    recent_spells = []
    for event in reversed(match.play_log):
        age = match.elapsed_ms - int(event.get("at_ms", 0))
        if age > 4500:
            break
        if event.get("side") != side or event.get("card") not in SPELLS:
            continue
        tx = float(event.get("x", 0)) / sim_arena.MT
        ty = 31.0 - float(event.get("y", 0)) / sim_arena.MT
        if side < 0:
            tx, ty = GRID_W - 1 - tx, GRID_H - 1 - ty
        recent_spells.append((tx, ty))

    for slot, card in enumerate(hand):
        grid = safe[1 + slot * TILES:1 + (slot + 1) * TILES].reshape(
            GRID_H, GRID_W)
        if card == "hog_rider":
            counter_units = [entity for entity in allies
                             if str(getattr(entity, "name", "")) in {
                                 "knight", "musketeer", "electro_spirit"}
                             and _policy_cell(entity, side)[1] <= 24]
            counterpush = bool(counter_units)
            advantage = elixir - enemy_elixir >= (2.5 if crown_lead else 2.0)
            minimum = 7.0 if crown_lead else 6.0
            if (pressure or elixir < minimum
                    or (elixir < 9.5 and not (advantage or counterpush))):
                grid[:] = False
            else:
                allowed = np.zeros_like(grid)
                allowed[RIVER_ROW:RIVER_ROW + 3, 2:6] = True
                allowed[RIVER_ROW:RIVER_ROW + 3, 12:16] = True
                alive = {lane: float(theirs.get(lane, 1.0))
                         for lane in ("left", "right")
                         if float(theirs.get(lane, 1.0)) > 0.0}
                if len(alive) == 2 and abs(alive["left"] - alive["right"]) > 0.03:
                    target = min(alive, key=alive.get)
                    if target == "left":
                        allowed[:, 9:] = False
                    else:
                        allowed[:, :9] = False
                elif counter_units:
                    lead = min(counter_units,
                               key=lambda entity: _policy_cell(entity, side)[1])
                    if _policy_cell(lead, side)[0] < GRID_W / 2:
                        allowed[:, 9:] = False
                    else:
                        allowed[:, :9] = False
                grid &= allowed
        elif card in ("fireball", "zap"):
            radius, minimum = ((3.0, 2) if card == "fireball" else (2.0, 2))
            valuable = np.zeros_like(grid)
            for y in range(GRID_H):
                for x in range(GRID_W):
                    hit = [(entity, ex, ey) for entity, ex, ey in cells
                           if (ex - x) ** 2 + (ey - y) ** 2 <= radius ** 2]
                    if any((cast_x - x) ** 2 + (cast_y - y) ** 2 <= 4.0 ** 2
                           for cast_x, cast_y in recent_spells):
                        continue
                    reset = (card == "zap" and any(
                        any(key in str(getattr(entity, "name", ""))
                            for key in ("inferno", "sparky"))
                        for entity, _ex, _ey in hit))
                    troops = [(entity, ex, ey) for entity, ex, ey in hit
                              if not getattr(entity, "is_building", False)]
                    fireball_targets = [
                        (entity, ex, ey) for entity, ex, ey in troops
                        if (0 < getattr(entity, "max_hitpoints", 0) <= 1500
                            and (getattr(entity, "range_mt", 0) > 1000
                                 or getattr(entity, "flying", False)))]
                    fireball_value = sum(
                        float(getattr(entity_card(entity), "cost", 0) or 0)
                        for entity, _ex, _ey in fireball_targets)
                    push_support = [entity for entity, _ex, _ey in troops
                                    if getattr(entity, "max_hitpoints", 0) <= 1000
                                    and float(getattr(entity_card(entity),
                                                      "cost", 0) or 0) >= 3
                                    and (getattr(entity, "range_mt", 0) > 1000
                                         or getattr(entity, "flying", False))]
                    incoming_push = (
                        len(troops) >= 2
                        and sum(float(getattr(entity_card(entity), "cost", 0) or 0)
                                for entity, _ex, _ey in troops) >= 6
                        and push_support
                        and any(ey >= RIVER_ROW for _entity, _ex, ey in troops))
                    medium_ground_pack = sum(
                        300 <= getattr(entity, "max_hitpoints", 0) <= 1000
                        and not getattr(entity, "flying", False)
                        and ey >= RIVER_ROW
                        for entity, _ex, ey in troops) >= 3
                    zap_targets = [
                        (entity, ex, ey) for entity, ex, ey in troops
                        if any(key in str(getattr(entity, "name", ""))
                               for key in ("skeleton", "bat", "spear_goblin",
                                           "goblin", "fire_spirit", "ice_spirit",
                                           "electro_spirit"))]
                    building = any(getattr(entity, "is_building", False)
                                   for entity, _ex, _ey in hit)
                    building_support = building and hog_inbound and any(
                        (ex < GRID_W / 2) == hog_lane_left
                        for entity, ex, _ey in hit
                        if getattr(entity, "is_building", False))
                    defensive_air = (len(fireball_targets) >= 2 and any(
                        getattr(entity, "flying", False) and ey >= RIVER_ROW
                        for entity, _ex, ey in fireball_targets))
                    valuable_cast = ((card == "fireball" and
                                      ((len(fireball_targets) >= 2
                                        and fireball_value >= 5)
                                       or incoming_push
                                       or medium_ground_pack
                                       or building_support
                                       or defensive_air))
                                     or (card == "zap" and
                                         (len(zap_targets) >= minimum or reset)))
                    if valuable_cast:
                        valuable[y, x] = True
            if card in ("fireball", "zap"):
                for lane, (x, y) in (("left", (4, 7)), ("right", (14, 7))):
                    hp = float(match.tower_fractions(-side).get(lane, 1.0))
                    if spellinfo.can_finish(card, hp):
                        valuable[max(0, y - 1):y + 2,
                                 max(0, x - 1):x + 2] = True
            grid &= valuable
    if not safe[1:].any() and ground_pressure and "cannon" in hand:
        slot = list(hand).index("cannon")
        base = mask[1 + slot * TILES:1 + (slot + 1) * TILES].reshape(
            GRID_H, GRID_W)
        cannon = slot_grid("cannon")
        _entity, focus_x, focus_y = focus_of(ground_pressure)
        x = 8 if focus_x < GRID_W / 2 else 9
        y = 22 if focus_y >= 22.5 else 21
        cannon[y, x - 1:x + 2] = base[y, x - 1:x + 2]
    return safe


@dataclass
class RewardWeights:
    """What the agent is actually optimising.

    Tower damage dominates because it is what the game scores, and it is
    symmetric: dealing and taking a tower fraction are worth the same, so the
    shaping cannot make the policy structurally attack-happy or turtle-happy.

    `elixir_traded` is the term that docstring said to add if it were ever
    added: net milli-elixir of enemy bodies destroyed minus our own lost, per
    step, in whole elixir. It defaults to zero, so nothing changes unless a run
    asks for it.

    It exists because tower damage cannot see most of the game. Kiting a
    Musketeer with an Ice Golem, pulling a tank to the centre so both Princess
    Towers work on it, answering a five-elixir push with two - none of those
    move a tower's hitpoints at all, and all of them are why one player beats
    another. Scored on elixir they are exactly what they are: +4, +6, +3.

    It is still a prior about how to play, which is why it is off by default
    and named rather than folded into the tower weight.
    """
    tower_damage_dealt: float = 10.0
    tower_damage_taken: float = -10.0
    crown_for: float = 3.0
    crown_against: float = -3.0
    win: float = 10.0
    loss: float = -10.0
    draw: float = 0.0
    elixir_traded: float = 0.0
    spell_quality: float = 0.0
    spell_prediction: float = 0.0
    spell_miss: float = 0.0
    kite_quality: float = 0.0
    hog_spirit_combo: float = 0.0
    hold_at_cap: float = 0.0
    illegal_action: float = -0.01
    step_cost: float = 0.0


@dataclass
class EpisodeStats:
    steps: int = 0
    plays: int = 0
    illegal: int = 0
    cards: Dict[str, int] = field(default_factory=dict)
    # Raw, unweighted reward events. Keeping these separate from the scalar
    # return lets training runs answer whether a shaping term was useful,
    # merely rare, or completely absent instead of guessing from win rate.
    tower_fraction_dealt: float = 0.0
    tower_fraction_taken: float = 0.0
    elixir_trade: float = 0.0
    spell_quality: float = 0.0
    spell_prediction: float = 0.0
    spell_casts: int = 0
    spell_misses: int = 0
    kite_quality: float = 0.0
    hog_spirit_combos: int = 0
    reward_total: float = 0.0

    @property
    def hog_share(self) -> float:
        return self.cards.get("hog_rider", 0) / self.plays if self.plays else 0.0


class ClashEnv:
    """One agent on the bottom side; the hand-written brain on the top."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0, level: int = 11, opponent: str = "brain",
                 decide_every_ms: int = DECIDE_EVERY_MS, max_ms: int = MAX_MS,
                 action_delay_ms: int = ACTION_DELAY_MS,
                 rewards: Optional[RewardWeights] = None,
                 opponent_config: Optional[Path] = None):
        from sim.gamedata import load_gamedata
        from sim.runner import BrainPolicy, SimpleOpponent, resolve_deck
        from sim.spells import load_spells

        self._spells = load_spells(level=level)
        self._brain_policy = BrainPolicy
        self._simple_opponent = SimpleOpponent
        self._opponent_kind = opponent
        self._opponent_config = opponent_config

        # `meta` draws a different real ladder deck every episode. The card
        # table has to carry every card in the pool, not just our eight, or the
        # opponent's deck resolves to nothing and it stands there.
        all_cards = load_gamedata(level=level)
        self._deck_pool = []
        if opponent == "meta":
            from sim.meta_decks import deck_pool
            self._deck_pool = deck_pool(all_cards)
            if not self._deck_pool:
                raise RuntimeError(
                    "opponent='meta' needs a deck pool; run "
                    "scripts/sync_meta_decks.py")
        wanted = set(DECK_26)
        for _name, _style, deck in self._deck_pool:
            wanted |= set(deck)
        self._cards = resolve_deck(all_cards, sorted(wanted))

        self.seed = seed
        self.decide_every_ms = decide_every_ms
        self.action_delay_ms = max(0, int(action_delay_ms))
        self.max_ms = max_ms
        # `or` here calls bool() on whatever is passed, which raises on a
        # numpy array - and a caller with a local named `rewards` holding
        # the per-step reward buffer is an easy mistake to make.
        self.rewards = RewardWeights() if rewards is None else rewards

        self.match: Optional[Match] = None
        self.opponent = None
        self._policy_opponent = None
        self.opponent_deck_name = ""
        self.opponent_deck_style = ""
        self.stats = EpisodeStats()
        self._turn = 0
        self._last_towers: Tuple[float, float] = (1.0, 1.0)
        self._last_elixir: Tuple[int, int] = (0, 0)
        self._spell_reward_cursor = 0
        self._kite_events: List[dict] = []
        self._last_hog_play: Optional[Tuple[int, bool]] = None
        self._episodes = 0

    # ------------------------------------------------------------- spaces

    def set_policy_opponent(self, opponent) -> None:
        """Swap in a learned opponent for the next episode, or None to revert.

        Set between episodes rather than mid-match: changing who is playing the
        top seat half way through a game makes the episode unattributable.
        """
        self._policy_opponent = opponent

    def set_opponent_kind(self, kind: str) -> None:
        """Choose which scripted opponent the next episode faces.

        Training against a single opponent produces a policy that beats that
        opponent and nothing else. Measured here on 60 held-out games: a run
        trained 50% against the rule engine reached 93.3% against it while
        falling to 76.7% against meta decks, from a starting point of 41.7%
        and 86.7%. The reverse arrangement had already failed the same way in
        the other direction - 82.5% against meta decks and 16.7% against the
        rule engine.

        So the diet has to contain both, and which one an episode uses is a
        per-episode choice rather than a property of the environment.

        `meta` requires the deck pool, which is only loaded when the
        environment was constructed with it; asking for it otherwise is a
        silent no-op rather than an error, because a worker that raises
        mid-run takes the whole rollout down.
        """
        if kind == "meta" and not self._deck_pool:
            return
        if kind in ("brain", "meta", "simple", "mirror"):
            self._opponent_kind = kind

    @property
    def observation_shape(self) -> Dict[str, tuple]:
        return {"planes": (NUM_PLANES, GRID_H, GRID_W), "scalars": (NUM_SCALARS,)}

    @property
    def action_count(self) -> int:
        return ACTIONS

    @staticmethod
    def decode(action: int) -> Optional[Tuple[int, int, int] | Tuple[str, int]]:
        """Card action -> ``(slot, x, y)``, ability -> ``('ability', rank)``."""
        if action <= 0:
            return None
        if action >= CARD_ACTIONS:
            return ("ability", action - CARD_ACTIONS)
        index = action - 1
        slot, cell = divmod(index, TILES)
        y, x = divmod(cell, GRID_W)
        return slot, x, y

    @staticmethod
    def encode(slot: int, x: int, y: int) -> int:
        return 1 + slot * TILES + y * GRID_W + x

    @staticmethod
    def encode_ability(rank: int) -> int:
        return CARD_ACTIONS + rank

    def _ability_entities(self):
        return ability_entities(self.match, 1)

    # -------------------------------------------------------------- reset

    def reset(self, seed: Optional[int] = None) -> Tuple[dict, dict]:
        if seed is not None:
            self.seed = seed
        else:
            self.seed += 1
        self._episodes += 1

        opponent_deck = DECK_26
        if self._opponent_kind == "meta" and self._policy_opponent is None:
            # Seeded by the episode seed, so a run is reproducible and two
            # workers on different seeds see different decks.
            picker = random.Random(self.seed)
            name, style, opponent_deck = picker.choice(self._deck_pool)
            self.opponent_deck_name = name
            self.opponent_deck_style = style
        self.match = Match(cards=self._cards, decks=(DECK_26, opponent_deck),
                           seed=self.seed, spells=self._spells)
        if self._policy_opponent is not None:
            # A learned opponent keeps its own deck fixed to ours: it is a past
            # version of this policy, and it reads a 2.6 hand one-hot.
            self.opponent = self._policy_opponent
            self.opponent_deck_name = getattr(self.opponent, "name", "self")
            self.opponent_deck_style = "self"
        elif self._opponent_kind == "brain":
            self.opponent = self._brain_policy(self._cards, side=-1,
                                               config_path=self._opponent_config)
        elif self._opponent_kind == "meta":
            from sim.opponents import ScriptedOpponent
            self.opponent = ScriptedOpponent(
                self._cards, side=-1, deck=list(opponent_deck),
                style=self.opponent_deck_style, seed=self.seed + 1)
        elif self._opponent_kind == "mirror":
            # Our own deck, piloted by the deck-agnostic heuristics rather than
            # by the rule engine or a past self. It exists so a mirror-trained
            # policy meets more than two ways of playing 2.6: the style sets
            # reaction time (0.6s cycling, 1.1s beatdown) and how much elixir
            # is banked before committing, and `reset` jitters the reaction by
            # 0.6-1.6x on top. A human playing hog cycle is not the rule
            # engine and is not a past self, and a policy that has only ever
            # seen those two will have learned their timings specifically.
            from sim.opponents import ScriptedOpponent
            style = random.Random(self.seed).choice(
                ["cycle", "cycle", "control", "beatdown"])
            self.opponent_deck_name = f"mirror_{style}"
            self.opponent_deck_style = style
            self.opponent = ScriptedOpponent(
                self._cards, side=-1, deck=list(DECK_26),
                style=style, seed=self.seed + 1)
        else:
            self.opponent = self._simple_opponent(self._cards, side=-1,
                                                  seed=self.seed + 1)
        self.opponent.reset()
        self.stats = EpisodeStats()
        self._turn = 0
        self._last_towers = self._tower_totals()
        self._last_elixir = (0, 0)
        self._spell_reward_cursor = 0
        self._kite_events = []
        self._last_hog_play = None
        return self._observe(), self._info()

    # --------------------------------------------------------------- step

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        assert self.match is not None, "call reset() first"
        match = self.match
        reward = self.rewards.step_cost
        illegal = False

        decoded = self.decode(int(action))
        if decoded is None and match.players[1].elixir >= 9_950:
            # Banking is good until the bar is full; after that, continued
            # holding throws regeneration away. Kept separately weighted so
            # it cannot silently turn every training run hyper-aggressive.
            reward += self.rewards.hold_at_cap
        # A live decision is based on a captured frame that is already about
        # half a second old by the time the Android tap lands. Advance the
        # world before applying the chosen action so PPO has to aim at where a
        # moving enemy will be, not at the stale position in its observation.
        # The total decision cadence stays unchanged; with the production
        # defaults the action lands at the end of the 500 ms interval.
        self._turn += 1
        if self._turn % 2 == 0 and self.opponent is not None:
            self.opponent.act(match)
        step_started = match.elapsed_ms
        execute_at = min(step_started + self.action_delay_ms,
                         step_started + self.decide_every_ms)
        while not match.finished and match.elapsed_ms < execute_at:
            match.step()

        if decoded is not None and not match.finished:
            if decoded[0] == "ability":
                rank = decoded[1]
                entities = self._ability_entities()
                illegal = (rank >= len(entities)
                           or not match.activate_ability(1, entities[rank].uid))
            else:
                slot, x, y = decoded
                hand = match.players[1].hand
                if slot >= len(hand):
                    illegal = True
                else:
                    card = hand[slot]
                    point = grid_to_point(x, y, 1)
                    first_uid = match.battle.next_uid
                    if not match.play_card(1, card, point):
                        illegal = True
                    else:
                        self.stats.plays += 1
                        self.stats.cards[card] = self.stats.cards.get(card, 0) + 1
                        lane_left = x < GRID_W / 2
                        if card == "hog_rider":
                            self._last_hog_play = (match.elapsed_ms, lane_left)
                        elif (card == "electro_spirit" and self._last_hog_play
                              and match.elapsed_ms - self._last_hog_play[0] <= 4500
                              and lane_left == self._last_hog_play[1]):
                            # Small shaping term: it teaches ordering and lane
                            # coordination, while tower damage still decides
                            # whether the push was actually good.
                            reward += self.rewards.hog_spirit_combo
                            self.stats.hog_spirit_combos += 1
                            self._last_hog_play = None
                        if card in KITE_CARDS:
                            self._start_kite(first_uid)
        if illegal:
            reward += self.rewards.illegal_action
            self.stats.illegal += 1

        # Alternate which side acts at the execution boundary. This retains
        # the contested-tile fairness of the old immediate-action loop while
        # keeping our configured input-to-tap delay explicit.
        if self._turn % 2 == 1 and self.opponent is not None and not match.finished:
            self.opponent.act(match)

        target = step_started + self.decide_every_ms
        while not match.finished and match.elapsed_ms < target:
            match.step()

        reward += self._shaping_reward()
        self.stats.steps += 1

        terminated = bool(match.finished)
        truncated = (not terminated) and match.elapsed_ms >= self.max_ms
        if terminated or truncated:
            reward += self._terminal_reward()
        self.stats.reward_total += reward
        return self._observe(), reward, terminated, truncated, self._info()

    # ------------------------------------------------------------ rewards

    def _tower_totals(self) -> Tuple[float, float]:
        ours = sum(self.match.tower_fractions(1).values())
        theirs = sum(self.match.tower_fractions(-1).values())
        return ours, theirs

    def _shaping_reward(self) -> float:
        ours, theirs = self._tower_totals()
        was_ours, was_theirs = self._last_towers
        self._last_towers = (ours, theirs)
        dealt = max(0.0, was_theirs - theirs)
        taken = max(0.0, was_ours - ours)
        self.stats.tower_fraction_dealt += dealt
        self.stats.tower_fraction_taken += taken
        reward = (dealt * self.rewards.tower_damage_dealt
                  + taken * self.rewards.tower_damage_taken)
        if self.rewards.elixir_traded:
            trade = self._elixir_trade()
            self.stats.elixir_trade += trade
            reward += trade * self.rewards.elixir_traded
        if (self.rewards.spell_quality or self.rewards.spell_prediction
                or self.rewards.spell_miss):
            reward += self._spell_reward()
        if self.rewards.kite_quality:
            reward += self._kite_reward() * self.rewards.kite_quality
        return reward

    def _spell_reward(self) -> float:
        from sim.reward_events import score_spell_cast

        events = self.match.battle.spell_cast_log
        reward = 0.0
        # Casts are ordered. Keep the cursor on an airborne spell until it
        # resolves, then score it exactly once.
        while self._spell_reward_cursor < len(events):
            event = events[self._spell_reward_cursor]
            if not event.get("resolved"):
                break
            score = score_spell_cast(event)
            self.stats.spell_casts += int(event.get("caster_side") == 1)
            self.stats.spell_quality += score.quality
            self.stats.spell_prediction += score.prediction
            self.stats.spell_misses += int(score.missed)
            reward += score.quality * self.rewards.spell_quality
            reward += score.prediction * self.rewards.spell_prediction
            if score.missed:
                reward += self.rewards.spell_miss
            self._spell_reward_cursor += 1
        return reward

    def _start_kite(self, first_uid: int) -> None:
        """Start measuring a cheap defence only when a real threat exists."""
        match = self.match
        cheap = [uid for uid in range(first_uid, match.battle.next_uid)
                 if match.battle.get(uid) is not None]
        if not cheap:
            return
        towers = tuple(match.towers[1][lane] for lane in ("left", "right", "king"))
        threats = {}
        for entity in match.battle.entities.values():
            if not entity.alive or entity.side != -1 or entity.is_tower:
                continue
            nearest = min(sim_arena.distance(entity.pos, tower.pos) for tower in towers)
            if entity.pos.y >= sim_arena.RIVER_Y or nearest <= 8000:
                threats[entity.uid] = nearest
        if not threats:
            return
        self._kite_events.append({
            "started_ms": match.elapsed_ms,
            "expires_ms": match.elapsed_ms + 4000,
            "cheap": set(cheap),
            "cheap_max_hp": sum(match.battle.get(uid).max_hitpoints for uid in cheap),
            "threats": threats,
            "retargeted": set(),
            "defenders": {entity.uid for entity in match.battle.entities.values()
                          if entity.alive and entity.side == 1
                          and not entity.is_tower},
            "center_time_ms": 0,
            "sampled_ms": match.elapsed_ms,
            "damage_cursor": len(match.battle.damage_log),
            "tower_total": self._tower_totals()[0],
        })

    def _kite_reward(self) -> float:
        from sim.reward_events import score_kite

        match = self.match
        towers = tuple(match.towers[1][lane] for lane in ("left", "right", "king"))
        pending = []
        reward = 0.0
        for event in self._kite_events:
            for uid in event["threats"]:
                entity = match.battle.get(uid)
                if entity is not None and entity.target_uid in event["cheap"]:
                    event["retargeted"].add(uid)
            sample_ms = max(0, match.elapsed_ms - event["sampled_ms"])
            event["sampled_ms"] = match.elapsed_ms
            centred = 0
            for uid in event["retargeted"]:
                entity = match.battle.get(uid)
                if entity is None or not entity.alive:
                    continue
                x, y = sim_arena.to_tiles(entity.pos)
                centred += int(6 <= x <= 12 and 16 <= y <= 26)
            if event["threats"]:
                event["center_time_ms"] += round(
                    sample_ms * centred / len(event["threats"]))
            done = (match.elapsed_ms >= event["expires_ms"] or match.finished
                    or all(not (match.battle.get(uid) and match.battle.get(uid).alive)
                           for uid in event["threats"]))
            if not done:
                pending.append(event)
                continue
            displaced = 0
            for uid in event["retargeted"]:
                entity = match.battle.get(uid)
                if entity is None:
                    continue
                current = min(sim_arena.distance(entity.pos, tower.pos) for tower in towers)
                displaced += max(0, current - event["threats"][uid])
            absorbed = sum(
                max(0, int(amount))
                for _at, source, target, amount
                in match.battle.damage_log[event["damage_cursor"]:]
                if source in event["threats"] and target in event["cheap"])
            tower_lost = max(0.0, event["tower_total"] - self._tower_totals()[0])
            defenders = event["defenders"]
            defender_survival = (sum(
                bool(match.battle.get(uid) and match.battle.get(uid).alive)
                for uid in defenders) / max(1, len(defenders)))
            quality = score_kite(
                retargeted=len(event["retargeted"]),
                threats=len(event["threats"]), displaced_mt=displaced,
                absorbed_hp=absorbed, cheap_max_hp=event["cheap_max_hp"],
                tower_fraction_lost=tower_lost,
                center_time_ms=event["center_time_ms"],
                defender_survival=defender_survival)
            self.stats.kite_quality += quality
            reward += quality
        self._kite_events = pending
        return reward

    def _elixir_trade(self) -> float:
        """Net elixir destroyed since the last step, in whole elixir.

        Symmetric on purpose: killing four elixir of theirs and losing four of
        ours nets nothing, so this pays for *trades* rather than for kills. A
        term that only counted their losses would reward throwing bodies at
        anything.
        """
        destroyed = self.match.battle.elixir_destroyed
        ours = destroyed.get(1, 0)
        theirs = destroyed.get(-1, 0)
        was_ours, was_theirs = self._last_elixir
        self._last_elixir = (ours, theirs)
        return ((ours - was_ours) - (theirs - was_theirs)) / 1000.0

    def _terminal_reward(self) -> float:
        match = self.match
        mine = match.crowns_for(1)
        theirs = match.crowns_for(-1)
        reward = (mine * self.rewards.crown_for
                  + theirs * self.rewards.crown_against)
        if match.result == "bottom":
            reward += self.rewards.win
        elif match.result == "top":
            reward += self.rewards.loss
        else:
            reward += self.rewards.draw
        return reward

    # -------------------------------------------------------- observation

    def _observe(self) -> dict:
        return observe(self.match, 1)

    def action_mask(self) -> np.ndarray:
        """Which actions are legal right now. Index 0 (hold) always is."""
        return legal_mask(self.match, self._cards, 1)

    def _info(self) -> dict:
        return {
            "action_mask": self.action_mask(),
            "elapsed_s": (self.match.elapsed_ms / 1000.0) if self.match else 0.0,
            "action_delay_ms": self.action_delay_ms,
            "crowns": ((self.match.crowns_for(1), self.match.crowns_for(-1))
                       if self.match else (0, 0)),
            "result": self.match.result if self.match else None,
            "stats": self.stats,
        }

    def close(self) -> None:
        self.match = None
        self.opponent = None
