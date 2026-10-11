"""Drive the live bot with a checkpoint trained in the simulator.

`scripts/cr_bot.py` says, at the top of the file, "No learned checkpoint in the
hot path" - the runner has always been the hand-written `Brain`. That was the
right call when the learned thing was a StARformer that cost too much per
frame, but it left the RL work with nowhere to go: a policy could get as good
as it liked in the simulator and never touch a real match.

This is the bridge. It exposes the one method the runner calls -
`decide(state, elapsed, now, frame=None) -> Candidate | None` - so swapping it
in is a one-line change and everything downstream (tap timing, hand tracking,
match filtering) is untouched.

**Why the coordinates line up.** The simulator's action grid and the live grid
are the same convention, checked rather than assumed: both are 18x32 with the
river at row 16, our half at y >= 16, enemy princess towers at y=7 and ours at
y=24. BuildABot's bottom-up tile coordinate is already flipped once by
`brain.arena.to_grid`; the resulting live cell is therefore the plane row and
must not be flipped a second time here.
`tests/test_rl_policy.py` asserts both, because a silent mirror is the single
most expensive bug this project has hit.

**What is approximated, honestly.** The simulator observes each unit's current
hit points; live vision does not measure HP, so a detected unit contributes its
card's full HP. A half-dead Musketeer therefore looks healthy to the policy.
That is a real gap, not a rounding detail, and it is the first thing to suspect
if live behaviour diverges from simulated behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.env import (ACTIONS, BASE_SCALARS, CARD_ACTIONS, DECK_26, GRID_H, GRID_W,
                     NUM_PLANES, NUM_SCALARS, NUM_SLOTS, SPELLS, TILES,
                     _placeable, building_lane_risks, tactical_plane_values)
from sim.public_state import (public_feature_count, vector_from_events,
                              vector_from_live)

_CARDS = None
# Detected unit names are not always the card key. A Skeletons card puts three
# entities called `skeleton` on the board, Archers puts two `archer`, and the
# card table is keyed by the card. Looking a unit up by its own name silently
# returns nothing, and nothing means zero hit points and "not a building" -
# which is how a Cannon came out looking like a troop.
_UNIT_TO_CARD = {
    "skeleton": "skeletons", "archer": "archers", "goblin": "goblins",
    "spear_goblin": "spear_goblins", "minion": "minions", "barbarian": "barbarians",
    "royal_recruit": "royal_recruits", "royal_hog": "royal_hogs",
    "bat": "bats", "elite_barbarian": "elite_barbarians",
    "three_musketeer": "three_musketeers", "guard": "guards",
    "lava_pup": "lava_hound",
}

# Some detector entities are not present in the resolved training deck pool,
# but their targeting rule is stable and essential to live Cannon priority.
_BUILDING_TARGETERS = frozenset({
    "balloon", "battle_ram", "electro_giant", "giant", "goblin_giant",
    "golem", "hog", "hog_rider", "lava_hound", "ram_rider",
    "royal_giant", "royal_hog", "wall_breaker",
})
_KNOWN_SPLASH = frozenset({
    "baby_dragon", "bomber", "bowler", "dark_prince", "executioner",
    "firecracker", "mega_knight", "valkyrie", "wizard",
})


def _card_key(name: str) -> str:
    """Prefer an exact resolved key before applying plural/unit aliases."""
    return name if name in _cards() else _UNIT_TO_CARD.get(name, name)


def _cards():
    """The card table, resolved exactly the way `ClashEnv` resolves it.

    Not `load_gamedata` directly: the loader does not key everything under the
    name this project plays it as - `ice_golem` is simply absent from it, and
    `resolve_deck` is what maps our names onto the client's. Using the raw
    table here meant Ice Golem had no cost, fell back to a default of four
    elixir, and was masked out as unaffordable at two.
    """
    global _CARDS
    if _CARDS is None:
        from sim.gamedata import load_gamedata
        from sim.meta_decks import deck_pool
        from sim.runner import resolve_deck
        all_cards = load_gamedata(11)
        wanted = set(DECK_26)
        try:
            for _name, _style, deck in deck_pool(all_cards):
                wanted |= set(deck)
        except (RuntimeError, OSError):
            pass                      # no pool synced; our own deck is enough
        _CARDS = resolve_deck(all_cards, sorted(wanted))
    return _CARDS


def card_cost(card: str) -> float:
    """Elixir cost, or 99 for something we cannot price and must not play."""
    spec = _cards().get(card)
    cost = getattr(spec, "cost", None) if spec else None
    return float(cost) if cost else 99.0


def _unit_traits(name: str):
    """(max_hp, flying, is_building) for a detected unit name."""
    spec = _cards().get(_card_key(name))
    unit = getattr(spec, "unit", None) if spec else None
    if unit is None:
        return 0.0, False, False
    # `is_building` is derived, not declared - `sim.entities.make_unit` reads
    # it as "cannot move, or the client declares it in spells_buildings.csv".
    # Reading a field of that name off the spec gets None for every building.
    building = (getattr(unit, "speed_mt_per_sec", 1) == 0
                or bool(getattr(unit, "from_building_card", False)))
    return (float(getattr(unit, "hitpoints", 0) or 0),
            bool(getattr(unit, "flying", False)), building)


def _unit_profile(name: str) -> dict:
    """Public tactical facts used by the live action guard."""
    from sim.entities import speed_to_mt_per_sec

    key = _card_key(name)
    spec = _cards().get(key)
    unit = getattr(spec, "unit", None) if spec else None
    hp, flying, building = _unit_traits(name)
    return {
        "hp": hp,
        "flying": flying,
        "building": building,
        "cost": float(getattr(spec, "cost", 0) or 0),
        "range": float(getattr(unit, "range_mt", 0) or 0) if unit else 0.0,
        "building_targeter": (bool(getattr(unit, "target_only_buildings", False))
                                if unit else name in _BUILDING_TARGETERS),
        "splash": (bool(getattr(unit, "splash_radius_mt", 0))
                   if unit else name in _KNOWN_SPLASH),
        "damage": float(getattr(unit, "damage", 0) or 0) if unit else 0.0,
        "hit_speed_ms": float(getattr(unit, "hit_speed_ms", 1000) or 1000)
        if unit else 1000.0,
        "speed": float(speed_to_mt_per_sec(
            int(getattr(unit, "speed_mt_per_sec", 0) or 0)))
        if unit else 0.0,
        "attacks_air": bool(getattr(unit, "attacks_air", False))
        if unit else False,
        "splash_radius": float(getattr(unit, "splash_radius_mt", 0) or 0)
        if unit else (2000.0 if name in _KNOWN_SPLASH else 0.0),
    }


def ally_cells_with_names(state):
    """Our units as (cell, name), which `Observation.allies` throws away.

    `obs.allies` is a list of bare cells - the hand-written policy only ever
    needed to know where our units are. The network was trained with four
    planes per side (count, hit points, air, building), so feeding it nameless
    allies would zero three of its four ally planes: a train/serve mismatch
    that does not crash and does quietly change every decision. `state` still
    has the names, so they are recovered here rather than lost.
    """
    from brain import arena as live_arena
    out = []
    for ally in getattr(state, "allies", ()) or ():
        cell = live_arena.to_grid(ally.position.tile_x, ally.position.tile_y)
        out.append((cell, str(ally.unit.name)))
    return out


def observation_from_live(obs, allies=None, opponent=None, economy=None) -> dict:
    """A live `brain.policy.Observation` in the simulator's own frame.

    `obs.tracks` are the enemies. `allies` is an optional list of
    `(cell, name)` from `ally_cells_with_names`; without it our own units fall
    back to `obs.allies`, which carries positions but no identity.
    """
    planes = np.zeros((NUM_PLANES, GRID_H, GRID_W), dtype=np.float32)
    enemy_building_cells = []

    def place(cell_x: float, cell_y: float, base: int, name: Optional[str]):
        # `cell_y` is already in the top-down policy grid. The perception path
        # applied tile-y -> 31-y in brain.arena.to_grid.
        col = int(np.clip(round(cell_x), 0, GRID_W - 1))
        row = int(np.clip(round(cell_y), 0, GRID_H - 1))
        planes[base, row, col] += 1.0
        hp, flying, building = _unit_traits(name) if name else (0.0, False, False)
        planes[2 + base, row, col] += hp / 1000.0
        if flying:
            planes[4 + base, row, col] += 1.0
        if building:
            planes[6 + base, row, col] += 1.0
            if base == 1:
                enemy_building_cells.append((col, row))
        if name:
            profile = _unit_profile(name)
            values = tactical_plane_values(
                damage=profile["damage"],
                hit_speed_ms=profile["hit_speed_ms"],
                range_mt=profile["range"],
                speed_mt_per_sec=profile["speed"],
                attacks_air=profile["attacks_air"],
                target_only_buildings=profile["building_targeter"],
                splash_radius_mt=profile["splash_radius"])
            for pair, value in enumerate(values):
                planes[8 + 2 * pair + base, row, col] += value

    if allies is not None:
        for cell, name in allies:
            place(cell.x, cell.y, 0, name)
    else:
        for cell in obs.allies:
            place(cell.x, cell.y, 0, None)
    for track in obs.tracks:
        place(track.cell.x, track.cell.y, 1, str(track.name))

    scalars = np.zeros(NUM_SCALARS, dtype=np.float32)
    scalars[0] = float(obs.elixir) / 10.0
    scalars[1] = min(1.0, float(obs.elapsed) / 180.0)
    # The simulator stores 2800 / regen_ms / 3; live carries the multiplier
    # directly, and those are the same number.
    scalars[2] = float(obs.multiplier) / 3.0
    base = 3
    for index, lane in enumerate(("left", "right")):
        scalars[base + index] = float(obs.ally_hp.get(lane, 1.0))
        scalars[base + 2 + index] = float(obs.enemy_hp.get(lane, 1.0))
    base += 4
    for card, slot in obs.hand.items():
        if card in DECK_26 and 0 <= slot < NUM_SLOTS:
            scalars[base + slot * len(DECK_26) + DECK_26.index(card)] = 1.0
    # The next card. These eight scalars were left at zero for the whole of
    # every live match while the simulator always set one of them, so the
    # policy was reading a state that never occurred once in training.
    #
    # It matters most for exactly this deck. 2.6 wins by cycling back to the
    # Hog faster than the opponent cycles their answer, and "how far away is
    # the Hog" lives entirely in this one-hot. Measured over five live
    # matches against a person, hog share was 3-5% against 11-13% for the
    # same checkpoint in the simulator, with 266-360 holds a match.
    nxt = getattr(obs, "next_card", "")
    if nxt in DECK_26:
        scalars[base + NUM_SLOTS * len(DECK_26) + DECK_26.index(nxt)] = 1.0
    if opponent is not None and economy is not None:
        public_end = BASE_SCALARS + public_feature_count()
        scalars[BASE_SCALARS:public_end] = vector_from_live(
            opponent, economy, float(getattr(obs, "now", 0.0)))
    elif hasattr(obs, "opponent_events"):
        public_end = BASE_SCALARS + public_feature_count()
        scalars[BASE_SCALARS:public_end] = vector_from_events(
            list(obs.opponent_events), int(float(obs.elapsed) * 1000))
    else:
        public_end = BASE_SCALARS + public_feature_count()
    scalars[public_end:] = building_lane_risks(enemy_building_cells)
    return {"planes": planes, "scalars": scalars}


def mask_from_live(obs) -> np.ndarray:
    """Legal actions live: what is in hand, affordable, and placeable."""
    mask = np.zeros(ACTIONS, dtype=bool)
    mask[0] = True
    enemy_down = tuple(sorted(lane for lane in ("left", "right")
                              if float(obs.enemy_hp.get(lane, 1.0)) <= 0.0))
    for card, slot in obs.hand.items():
        if not (0 <= slot < NUM_SLOTS):
            continue
        # Millielixir, matching the simulator: comparing 3.999 against 4.0 in
        # floats and 3999 against 4000 in integers must give the same answer.
        if round(float(obs.elixir) * 1000) < round(card_cost(card) * 1000):
            continue
        grid = _placeable(1, enemy_down, card in SPELLS)
        start = 1 + slot * TILES
        mask[start:start + TILES] = grid.reshape(-1)
    return mask


def apply_live_safety_mask(mask: np.ndarray, obs, allies_named=None,
                           recent_spells=()) -> np.ndarray:
    """Remove strategically unsafe live actions before PPO sees the logits.

    Simulation exploration is allowed to make a bad choice and learn from it;
    ladder control is not.  In particular, the transferred policy repeatedly
    sent Hog into a push and aimed spells at cells whose targets had already
    gone.  This mask is deliberately conservative and leaves the network to
    choose among the remaining legal actions.
    """
    safe = mask.copy()
    hand = getattr(obs, "hand", {})

    tracks = list(getattr(obs, "tracks", ()) or ())
    enemy_cells = []
    for track in tracks:
        cell = getattr(track, "cell", None)
        if cell is not None:
            name = str(getattr(track, "name", ""))
            enemy_cells.append({"x": float(np.clip(round(cell.x), 0, GRID_W - 1)),
                                "y": float(np.clip(round(cell.y), 0, GRID_H - 1)),
                                "name": name, **_unit_profile(name)})

    pressure = [unit for unit in enemy_cells
                if unit["y"] >= GRID_H // 2 and not unit["building"]]
    ground_pressure = [unit for unit in pressure if not unit["flying"]]
    air_pressure = [unit for unit in pressure if unit["flying"]]
    pure_air = bool(air_pressure) and not ground_pressure
    elixir = float(getattr(obs, "elixir", 0.0) or 0.0)
    enemy_elixir = float(getattr(obs, "enemy_elixir", 10.0) or 0.0)
    allies_named = list(allies_named or ())
    allied_names = {name for _cell, name in allies_named}
    allied_hogs = [(cell, name) for cell, name in allies_named
                   if name == "hog_rider" and cell.y <= 21]
    hog_inbound = bool(allied_hogs)
    hog_lane_left = (allied_hogs[-1][0].x < GRID_W / 2) if allied_hogs else None
    ours = getattr(obs, "ally_hp", {})
    theirs = getattr(obs, "enemy_hp", {})
    crown_lead = (sum(float(ours.get(lane, 0.0)) <= 0.0
                      for lane in ("left", "right"))
                  < sum(float(theirs.get(lane, 0.0)) <= 0.0
                        for lane in ("left", "right")))

    def card_grid(card: str):
        slot = hand.get(card)
        if slot is None:
            return None
        return safe[1 + slot * TILES:1 + (slot + 1) * TILES].reshape(
            GRID_H, GRID_W)

    # Empty board: preserve every defender.  The only attacking package is
    # Hog + Electro Spirit; Skeletons may cycle only at full elixir when that
    # package is not in hand.  A one-crown lead raises the commitment bar.
    if not pressure:
        for card in ("cannon", "musketeer", "knight"):
            grid = card_grid(card)
            if grid is not None:
                grid[:] = False
        skeletons = card_grid("skeletons")
        if skeletons is not None:
            combo_ready = "hog_rider" in hand and "electro_spirit" in hand
            if elixir < 10.0 or combo_ready:
                skeletons[:] = False
            else:
                back = np.zeros_like(skeletons)
                back[28:32, 6:12] = True
                skeletons &= back
        spirit = card_grid("electro_spirit")
        if spirit is not None:
            if not hog_inbound:
                spirit[:] = False
            else:
                allowed = np.zeros_like(spirit)
                xs = slice(2, 7) if hog_lane_left else slice(11, 16)
                allowed[15:21, xs] = True
                spirit &= allowed

    # Against a pure air push, ground-only defenders only burn elixir. The
    # Musketeer is the real answer; Spirit/Zap/Fireball may support it below.
    if pure_air:
        for card in ("cannon", "knight", "skeletons", "hog_rider"):
            grid = card_grid(card)
            if grid is not None:
                grid[:] = False
        musketeer = card_grid("musketeer")
        if musketeer is not None:
            lane_left = sum(unit["x"] for unit in air_pressure) / len(air_pressure) < 8.5
            x = 9 if lane_left else 8
            y = int(np.clip(max(unit["y"] for unit in air_pressure) + 4, 21, 26))
            allowed = np.zeros_like(musketeer)
            allowed[max(16, y - 1):min(GRID_H, y + 2), max(0, x - 1):x + 2] = True
            musketeer &= allowed

    # A 2.6 defence is a formation, not eight unrelated card choices. Keep
    # the ranged pieces protected in the middle/back and use cheap troops in
    # front to pull attackers through their fire.
    if pressure:
        focus = max(pressure, key=lambda unit: unit["y"])
        lane_left = focus["x"] < GRID_W / 2
        musketeer = card_grid("musketeer")
        if musketeer is not None:
            allowed = np.zeros_like(musketeer)
            centre_x = 9 if lane_left else 8
            allowed[23:28, centre_x - 2:centre_x + 3] = True
            musketeer &= allowed

        knight = card_grid("knight")
        if knight is not None:
            allowed = np.zeros_like(knight)
            kite_x = 10 if lane_left else 7
            allowed[19:23, kite_x - 1:kite_x + 2] = True
            knight &= allowed

        skeletons = card_grid("skeletons")
        if skeletons is not None:
            if all(unit["splash"] for unit in ground_pressure):
                skeletons[:] = False
            else:
                allowed = np.zeros_like(skeletons)
                y = int(np.clip(focus["y"] + 1, 19, 23))
                allowed[y - 1:y + 2, 7:11] = True
                skeletons &= allowed

        spirit = card_grid("electro_spirit")
        if spirit is not None:
            allowed = np.zeros_like(spirit)
            y = int(np.clip(focus["y"], 18, 22))
            allowed[y - 1:y + 2, 7:11] = True
            spirit &= allowed

    # Cannon is a pull building, not a card to drop on the edge or into a
    # ranged unit. Keep it deep in the central pocket, shifted toward the lane,
    # so both towers and a Musketeer behind it can keep shooting.
    cannon = card_grid("cannon")
    if cannon is not None and ground_pressure:
        cannon_range = _unit_profile("cannon")["range"] or 5500.0
        pullable = [unit for unit in ground_pressure
                    if not unit["range"] or unit["range"] <= cannon_range]
        if not pullable:
            cannon[:] = False
        else:
            focus = max(pullable, key=lambda unit: unit["y"])
            # Expert replays put Cannon almost exactly between the towers
            # (x median 9.5, IQR 8.5-9.5), shifted only one cell toward the
            # incoming lane.  The older x=6/11 mask was lane defence, not a
            # central pull, and denied the crossfire the deck is built around.
            x = 8 if focus["x"] < 8.5 else 9
            allowed = np.zeros_like(cannon)
            # Fast win conditions need the exact 4-3 row to guarantee the
            # pull. Slow tanks can be met one tile deeper, as requested after
            # the live review, which buys more crossfire time.
            fast_pull = any(unit["name"] in {
                "hog", "hog_rider", "battle_ram", "ram_rider",
                "royal_hog", "wall_breaker",
            } for unit in pullable)
            y = 20 if fast_pull else 21
            allowed[y:y + 1, x - 1:x + 2] = True
            cannon &= allowed

            # Establish the pull before spending the same hand on a
            # discretionary ground screen. Musketeer stays available for air.
            needs_pull = (any(unit["building_targeter"] for unit in pullable)
                          and "cannon" not in allied_names)
            if needs_pull and cannon.any():
                for card in ("knight", "skeletons", "electro_spirit"):
                    grid = card_grid(card)
                    if grid is not None:
                        grid[:] = False

    hog_slot = hand.get("hog_rider")
    threats = list(getattr(obs, "threats", ()) or ())
    threat_score = float(getattr(obs, "threat_score", 0.0) or 0.0)
    if hog_slot is not None:
        grid = safe[1 + hog_slot * TILES:1 + (hog_slot + 1) * TILES].reshape(
            GRID_H, GRID_W)
        combo_ready = "electro_spirit" in hand
        counterpush = any(name in {"knight", "musketeer", "electro_spirit"}
                          and cell.y <= 24 for cell, name in allies_named)
        advantage = elixir - enemy_elixir >= (2.5 if crown_lead else 2.0)
        minimum = 8.0 if combo_ready else 10.0
        if (threats or threat_score >= 5.0 or pressure
                or elixir < minimum or not (advantage or counterpush)):
            grid[:] = False
        else:
            allowed = np.zeros_like(grid)
            # Hog belongs at either bridge, not behind our towers or directly
            # on top of defenders in the middle of our half.
            allowed[GRID_H // 2:GRID_H // 2 + 3, 2:6] = True
            allowed[GRID_H // 2:GRID_H // 2 + 3, 12:16] = True
            # Focus the more damaged surviving princess tower instead of
            # splitting damage across both lanes.
            alive = {lane: float(theirs.get(lane, 1.0))
                     for lane in ("left", "right")
                     if float(theirs.get(lane, 1.0)) > 0.0}
            if len(alive) == 2 and abs(alive["left"] - alive["right"]) > 0.03:
                target = min(alive, key=alive.get)
                if target == "left":
                    allowed[:, 9:] = False
                else:
                    allowed[:, :9] = False
            grid &= allowed

    now = float(getattr(obs, "now", 0.0) or 0.0)
    for spell, radius, minimum in (("fireball", 3.0, 2), ("zap", 2.0, 2)):
        slot = hand.get(spell)
        if slot is None:
            continue
        grid = safe[1 + slot * TILES:1 + (slot + 1) * TILES].reshape(
            GRID_H, GRID_W)
        valuable = np.zeros_like(grid)
        for y in range(GRID_H):
            for x in range(GRID_W):
                hit = [unit for unit in enemy_cells
                       if (unit["x"] - x) ** 2 + (unit["y"] - y) ** 2 <= radius ** 2]
                # Vision cannot see an in-flight spell. Remember confirmed
                # casts so Fireball and Zap cannot hit the same resolved area.
                overlaps_cast = any(
                    now - cast_at <= 4.5
                    and (cast_x - x) ** 2 + (cast_y - y) ** 2 <= 4.0 ** 2
                    for cast_at, cast_x, cast_y in recent_spells)
                if overlaps_cast:
                    continue
                reset = (spell == "zap" and any(
                    any(key in unit["name"] for key in ("inferno", "sparky"))
                    for unit in hit))
                troops = [unit for unit in hit if not unit["building"]]
                fireball_targets = [
                    unit for unit in troops
                    if (0 < unit["hp"] <= 1500
                        and (unit["range"] > 1000 or unit["flying"]))
                ]
                fireball_value = sum(unit["cost"] for unit in fireball_targets)
                zap_targets = [
                    unit for unit in troops
                    if any(key in unit["name"] for key in (
                        "skeleton", "bat", "spear_goblin", "goblin",
                        "fire_spirit", "ice_spirit", "electro_spirit",
                    ))
                ]
                building = any(unit["building"] for unit in hit)
                building_support = building and hog_inbound and any(
                    (unit["x"] < GRID_W / 2) == hog_lane_left
                    for unit in hit if unit["building"])
                defensive_air = (len(fireball_targets) >= 2
                                  and any(unit["flying"] and unit["y"] >= 16
                                          for unit in fireball_targets))
                valuable_cast = ((spell == "fireball" and
                                  ((len(fireball_targets) >= 2
                                    and fireball_value >= 5)
                                   or building_support
                                   or defensive_air))
                                 or (spell == "zap" and
                                     (len(zap_targets) >= minimum or reset)))
                if valuable_cast:
                    valuable[y, x] = True
        # Fireball may always finish a princess tower, even without troops.
        if spell == "fireball":
            for side, point in (("left", (4, 7)), ("right", (14, 7))):
                if 0.0 < float(getattr(obs, "enemy_hp", {}).get(side, 1.0)) <= 0.18:
                    x, y = point
                    valuable[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = True
        grid &= valuable
    return safe


class RLBrain:
    """A trained checkpoint behind the `Brain.decide` interface.

    `temperature` of 0 is greedy, and greedy is the right default live:
    sampling is exploration, and there is nothing to explore on ladder.
    """

    def __init__(self, checkpoint: Path, temperature: float = 0.0,
                 device: str = "cpu", min_confidence: float = 0.0):
        import torch
        from sim.train_ppo import (build_network, checkpoint_architecture,
                                   load_policy_state)
        self.torch = torch
        self.device = torch.device(device)
        blob = torch.load(Path(checkpoint), map_location=self.device,
                          weights_only=False)
        self.network = build_network(
            NUM_PLANES, NUM_SCALARS, ACTIONS,
            checkpoint_architecture(blob))
        load_policy_state(self.network, blob["state_dict"])
        self.network.to(self.device).eval()
        self.step_trained = int(blob.get("step", 0))
        self.temperature = temperature
        self.checkpoint_name = Path(checkpoint).name
        self.min_confidence = min_confidence
        self.last_obs = None
        self.holds = 0
        # Why the mask came back empty. `holds` alone says the policy
        # did nothing; it does not say whether that was its choice.
        # It never is: decide() returns before the network runs when
        # nothing is playable, so a hold is the mask's verdict, and
        # 266-360 of them a match against 52-68 plays in the simulator
        # is the gap worth explaining.
        self.hold_reasons = {"no_hand": 0, "no_elixir": 0,
                             "no_placement": 0}
        self.recent_spells = []
        # Built here rather than on the first decision. `cr_bot` calls
        # `brain.reset()` the moment it sees a battle, before it ever asks for
        # a decision, and a lazily-built observer made that throw
        # `AttributeError` on every frame - the bot sat in a live match doing
        # nothing at all while the log filled with the same line.
        self._observer = self._build_observer()

    @staticmethod
    def _build_observer():
        from brain.policy import Brain
        # Perception only, and explicitly no advisor: this reuses the
        # hand-written brain's tracking, hand smoothing and elixir model, and
        # none of its judgement.
        return Brain(learn=False, use_advisor=False)

    # ------------------------------------------------------------- interface
    #
    # `cr_bot` drives whichever brain it is given through one interface, so a
    # learned policy has to answer the whole of it and not only `decide`.
    # Everything that is bookkeeping rather than judgement is delegated to the
    # observer, which already does it correctly: hand tracking has to be told
    # what was played or the next observation is wrong, and the elixir model
    # has to be told or it thinks we are richer than we are.

    def reset(self) -> None:
        """New match: clear perception state and the play counters."""
        self._observer.reset()
        self.last_obs = None
        self.holds = 0
        for key in self.hold_reasons:
            self.hold_reasons[key] = 0
        self.recent_spells = []

    def confirm(self, candidate, obs_now=None) -> None:
        """A card was actually tapped. The observer needs to know."""
        self._observer.confirm(candidate, obs_now)
        if candidate.card in SPELLS:
            cast_at = float(obs_now if obs_now is not None else 0.0)
            self.recent_spells.append((cast_at, candidate.x, candidate.y))
            self.recent_spells = self.recent_spells[-4:]

    @property
    def committed_elixir(self) -> float:
        return self._observer.committed_elixir

    @property
    def plays(self) -> int:
        return self._observer.plays

    @property
    def play_counts(self):
        return self._observer.play_counts

    @property
    def hand_tracker(self):
        return self._observer.hand_tracker

    @property
    def classifier_overrides(self) -> int:
        return self._observer.classifier_overrides

    @property
    def advice_used(self) -> int:
        return self._observer.advice_used

    @property
    def book(self):
        return self._observer.book        # None: learn=False

    # There is no advisor in this path and there is not going to be one. The
    # attribute exists so the match record can say so rather than crash.
    advisor = None

    def _note_hold(self, obs) -> None:
        """Which constraint emptied the mask, in order of severity."""
        hand = {c: sl for c, sl in getattr(obs, "hand", {}).items()
                if c in DECK_26 and 0 <= sl < NUM_SLOTS}
        if not hand:
            self.hold_reasons["no_hand"] += 1
            return
        cheapest = min(card_cost(card) for card in hand)
        if float(getattr(obs, "elixir", 0.0)) < cheapest:
            self.hold_reasons["no_elixir"] += 1
            return
        self.hold_reasons["no_placement"] += 1

    def summary(self) -> str:
        # A string, because that is what the match record stores and what
        # `Brain.summary` returns; returning a dict here wrote a different
        # shape into every match json depending on which brain played.
        return (f"{self._observer.summary()} brain=rl "
                f"ckpt={self.checkpoint_name} step={self.step_trained} "
                f"holds={self.holds} "
                f"hold_why=" + ",".join(
                    f"{k}:{v}" for k, v in self.hold_reasons.items()))

    # The runner reloads config between matches; nothing to reload here.
    def reload_config(self) -> None:
        pass

    def decide(self, state, elapsed: float, now: Optional[float] = None,
               frame=None):
        from brain.policy import Candidate
        obs = self._observer.observe(state, elapsed, now, frame)
        self.last_obs = obs

        allies_named = ally_cells_with_names(state)
        view = observation_from_live(
            obs, allies_named,
            opponent=self._observer.opponent, economy=self._observer.enemy)
        mask = apply_live_safety_mask(mask_from_live(obs), obs, allies_named,
                                      self.recent_spells)
        if not mask[1:].any():
            self.holds += 1
            self._note_hold(obs)
            return None

        torch = self.torch
        with torch.no_grad():
            p = torch.from_numpy(view["planes"]).unsqueeze(0).to(self.device)
            s = torch.from_numpy(view["scalars"]).unsqueeze(0).to(self.device)
            logits, _ = self.network(p, s)
            logits = logits[0].float()
            logits[~torch.from_numpy(mask).to(logits.device)] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            if self.temperature <= 0:
                action = int(torch.argmax(logits).item())
            else:
                action = int(torch.multinomial(
                    torch.softmax(logits / self.temperature, dim=-1), 1).item())
            confidence = float(probs[action].item())

        if action == 0 or confidence < self.min_confidence:
            self.holds += 1
            return None
        if action >= CARD_ACTIONS:
            self.holds += 1          # abilities: no live activation path yet
            return None

        index = action - 1
        slot, cell = divmod(index, TILES)
        y, x = divmod(cell, GRID_W)
        card = next((name for name, held in obs.hand.items() if held == slot), None)
        if card is None:
            self.holds += 1
            return None
        return Candidate(card=card, slot=slot, x=int(x), y=int(y),
                         tag=f"rl_p{confidence:.2f}", weight_key="rl",
                         features={"confidence": confidence}, score=confidence)
