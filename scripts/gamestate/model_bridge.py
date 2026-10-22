"""ARM64 snapshots -> policy inputs through explicit shared coordinate contracts.

No screenshot processing, tracking guesses, taps or automatic battle start.
Own state and confirmed public play history must come from a trusted caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.gamestate.elixir import ElixirEstimate
from sim import arena
from sim.adapter import grid_to_point
from sim.entities import make_tower, make_unit
from sim.env import ACTIONS, DECK_26, NUM_PLANES, NUM_SCALARS, legal_mask, observe
from sim.gamedata import load_gamedata, to_snake_case
from sim.match import PlayerState
from sim.runner import resolve_deck


class MissingModelInput(ValueError):
    """A native snapshot alone cannot supply all of the checkpoint's inputs."""


@dataclass(frozen=True)
class OwnState:
    game_seconds: float
    hand: tuple[str, str, str, str]
    next_card: str
    elixir: ElixirEstimate
    opponent_events: tuple[dict, ...]
    public_history_complete: bool = False
    battle_active: bool = False  # accepted native world may be a frozen result/replay
    destroyed_towers: tuple[tuple[int, str], ...] = ()  # engine side, left/right after orientation


@dataclass
class ModelView:
    observation: dict
    action_mask: np.ndarray
    metadata: dict
    rotated: bool

    def placement_world(self, action: int) -> dict | None:
        """Decode placement to native world units. This sends NO input.

        The model's existing grid_to_point convention is retained, while the observation encoder has its legacy Y flip.
        Screen pixels require a separate calibrated mapping.
        """
        if not isinstance(action, (int, np.integer)) or not 0 <= action < ACTIONS:
            raise ValueError('Invalid action index')
        if not self.action_mask[action]:
            raise ValueError('Action is masked')
        if action == 0:
            return None
        if action >= 1 + 4 * 18 * 32:
            raise ValueError('Native ability activation is not implemented')
        slot, cell = divmod(int(action) - 1, 18 * 32)
        y, x = divmod(cell, 18)
        point = grid_to_point(x, y, 1)
        wx, wy = point.x, point.y
        if self.rotated:
            wx, wy = arena.WIDTH - wx, arena.HEIGHT - wy
        return {'slot': slot, 'grid_x': x, 'grid_y': y, 'x': wx, 'y': wy}


class NativeModelAdapter:
    def __init__(self, *, own_side: int, stats_level: int = 11,
                 allow_incomplete_history: bool = False,
                 allow_unknown_profiles: bool = False,
                 coordinate_mode: str = 'legacy_training',
                 geometry=None, action_mask_mode: str | None = None):
        if own_side not in (0, 1):
            raise ValueError('Own engine side must be explicitly supplied as 0 or 1')
        self.own_side, self.stats_level = own_side, stats_level
        self.allow_incomplete_history = allow_incomplete_history
        self.allow_unknown_profiles = allow_unknown_profiles
        if coordinate_mode not in ('legacy_training', 'physical_live', 'physical_live_cells_v2'):
            raise ValueError('Unknown coordinate mode')
        self.coordinate_mode = coordinate_mode
        self.geometry = arena.resolve_geometry(geometry)
        self.action_mask_mode = action_mask_mode or (
            'mechanical' if coordinate_mode == 'physical_live_cells_v2' else 'legacy_strategic')
        if self.action_mask_mode not in ('mechanical', 'legacy_strategic'):
            raise ValueError('Unknown action mask mode')
        if coordinate_mode == 'physical_live_cells_v2' and self.action_mask_mode != 'mechanical':
            raise ValueError('V2 coordinate-consistent strategic mask is not implemented')
        self.cards = load_gamedata(level=stats_level)
        self.cards.update(resolve_deck(self.cards, DECK_26))
        self.units = {}
        for spec in self.cards.values():
            if spec.unit is not None:
                key = to_snake_case(spec.unit.name)
                if key in self.units and self.units[key] != spec.unit:
                    raise ValueError(f'Ambiguous unit profile: {key}')
                self.units[key] = spec.unit

    def encode(self, snapshot: dict, own: OwnState) -> ModelView:
        if snapshot.get('status') != 'ok':
            raise MissingModelInput('A fresh accepted native snapshot is required')
        elapsed = float(own.game_seconds)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise MissingModelInput('Need finite nonnegative elapsed GAME time')
        if abs(own.elixir.game_seconds - elapsed) > 1e-6:
            raise MissingModelInput('Elixir and game clock refer to different times')
        balance = own.elixir
        if (balance.source not in ('mathematical_standard_mode', 'arm64_memory') or
                not all(math.isfinite(v) for v in (balance.minimum, balance.maximum,
                                                   balance.reserved, balance.game_seconds)) or
                not 0 <= balance.minimum <= balance.maximum <= 10 or balance.reserved < 0 or
                balance.exact != (abs(balance.maximum - balance.minimum) < 1e-9)):
            raise MissingModelInput('Need a valid standard-mode elixir estimate')
        if (len(own.hand) != 4 or len(set(own.hand)) != 4 or
                any(name not in DECK_26 for name in own.hand) or
                own.next_card not in DECK_26 or own.next_card in own.hand):
            raise MissingModelInput('Need four ordered hand slots and a distinct known next card')
        rows = snapshot.get('units')
        if not isinstance(rows, list):
            raise MissingModelInput('Missing native units')
        kings = [u for u in rows if u['name'] == 'KingTower']
        ours = [u for u in kings if u['side'] == self.own_side]
        theirs = [u for u in kings if u['side'] != self.own_side]
        if len(ours) != 1 or len(theirs) != 1:
            raise MissingModelInput('Need both king anchors to establish orientation')
        rotated = ours[0]['y'] < arena.HEIGHT / 2
        if (theirs[0]['y'] < arena.HEIGHT / 2) == rotated:
            raise MissingModelInput('King anchors do not define opposite ends')
        entities, towers, seen = {}, {1: {}, -1: {}}, set()
        unknown_profiles = set()
        for raw in rows:
            uid, side = raw['entity_id'], raw['side']
            if side not in (0, 1) or uid in seen or uid // 1_000_000 != 5:
                raise MissingModelInput('Invalid side, namespace or duplicate body ID')
            seen.add(uid)
            x, y = float(raw['x']), float(raw['y'])
            if not all(math.isfinite(v) for v in (x, y)) or not (-4000 <= x <= 22000 and -4000 <= y <= 36000):
                raise MissingModelInput('Invalid native coordinates')
            if rotated:
                x, y = arena.WIDTH - x, arena.HEIGHT - y
            side = 1 if side == self.own_side else -1
            pos = arena.Point(round(x), round(y))
            hp, maximum = raw.get('hp', -1), raw.get('max_hp', -1)
            if not (isinstance(hp, int) and isinstance(maximum, int) and
                    0 <= hp <= maximum and maximum > 0):
                raise MissingModelInput(f'Unknown/invalid HP for body {uid}')
            if raw['name'] in ('KingTower', 'PrincessTower'):
                entity = make_tower(uid, side, pos, maximum, 0, 1000, 0,
                                    king=raw['name'] == 'KingTower')
                lane = 'king' if raw['name'] == 'KingTower' else ('left' if x < arena.WIDTH / 2 else 'right')
                if lane in towers[side]:
                    raise MissingModelInput('Ambiguous crown tower lane')
                towers[side][lane] = entity
            else:
                key = to_snake_case(raw['name'])
                if key not in self.units:
                    if not self.allow_unknown_profiles:
                        raise MissingModelInput(f'No verified unit profile for native body {raw["name"]}')
                    # Explicit trial mode: retain measured bodies and HP, but
                    # do not substitute source-card or invented tactical traits.
                    unknown_profiles.add(raw['name'])
                    entity = SimpleNamespace(uid=uid, name=key, side=side, pos=pos,
                                             alive=hp > 0, is_tower=False, ability_buff=None)
                else:
                    entity = make_unit(uid, self.units[key], side, pos)
            entity.hitpoints = hp
            entities[uid] = entity
        for engine_side in (0, 1):
            side = 1 if engine_side == self.own_side else -1
            for lane in ('left', 'right'):
                if lane not in towers[side] and (engine_side, lane) not in own.destroyed_towers:
                    raise MissingModelInput('Missing tower is not proof of destruction')
        if any(side not in (0, 1) or lane not in ('left', 'right')
               for side, lane in own.destroyed_towers):
            raise MissingModelInput('Invalid destroyed-tower declaration')
        for engine_side, lane in own.destroyed_towers:
            side = 1 if engine_side == self.own_side else -1
            if lane in towers[side] and towers[side][lane].hitpoints > 0:
                raise MissingModelInput('Destroyed-tower declaration conflicts with live HP')

        def fractions(side):
            return {lane: (towers[side][lane].hitpoints / towers[side][lane].max_hitpoints
                           if lane in towers[side] else 0.0) for lane in ('left', 'right')}

        events = []
        last = -1
        for event in own.opponent_events:
            at = event.get('at_ms')
            cost = event.get('cost')
            if (not isinstance(at, int) or not 0 <= at <= elapsed * 1000 or at < last or
                    not isinstance(cost, (int, float)) or not math.isfinite(cost) or
                    not 0 <= cost <= 10 or not isinstance(event.get('card'), str) or not event['card']):
                raise MissingModelInput('Need ordered, confirmed opponent card-play events')
            last = at
            events.append(dict(event, side=-1))
        player = PlayerState(1, list(DECK_26), hand=list(own.hand),
                             queue=[own.next_card],
                             elixir=own.elixir.available_minimum * 1000)
        match = SimpleNamespace(
            cards=self.cards, geometry=self.geometry, battle=SimpleNamespace(entities=entities),
            players={1: player}, elapsed_ms=round(elapsed * 1000),
            finished=False, play_log=events, towers=towers,
            tower_fractions=fractions,
            regen_ms=lambda: 2800 if elapsed < 120 else 1400 if elapsed < 240 else 933,
            can_activate_ability=lambda side, uid: False)
        encoder_match = match
        if self.coordinate_mode == 'physical_live':
            # v25's working CV path feeds top-down screen coordinates. The
            # current training encoder instead applies a legacy 31-y flip to
            # already top-down positions. Cancel it for BOTH observation and
            # action constraints; leave the physical snapshot untouched.
            encoder_entities = {}
            for uid, entity in entities.items():
                view_entity = copy(entity)
                view_entity.pos = arena.Point(entity.pos.x, 31000 - entity.pos.y)
                encoder_entities[uid] = view_entity
            encoder_match = copy(match)
            encoder_match.battle = SimpleNamespace(entities=encoder_entities)
        encoding = ('physical_live_cells_v2' if self.coordinate_mode == 'physical_live_cells_v2'
                    else 'legacy_training')
        observation = observe(encoder_match, side=1, coordinate_mode=encoding)
        mask = legal_mask(encoder_match, self.cards, side=1,
                          strategic=self.action_mask_mode == 'legacy_strategic')
        blockers = []
        if not own.battle_active:
            blockers.append('active_battle_not_confirmed')
        if not own.public_history_complete and not self.allow_incomplete_history:
            blockers.append('incomplete_opponent_play_history')
        if not own.elixir.exact:
            blockers.append('uncertain_own_elixir')
        if blockers:
            mask[:] = False
            mask[0] = True
        return ModelView(observation, mask, {
            'encoder': 'sim.env.observe', 'mask': 'sim.env.legal_mask',
            'own_engine_side': self.own_side, 'rotated_180': rotated,
            'coordinate_mode': self.coordinate_mode,
            'geometry': self.geometry.profile_id,
            'geometry_sha256': self.geometry.fingerprint,
            'action_mask_mode': self.action_mask_mode,
            'stats_level': self.stats_level,
            'hp_source': 'native_current_hp',
            'traits_source': 'sim_unit_profile_at_explicit_stats_level',
            'blockers': blockers, 'ready_for_inference': not blockers,
            'warnings': (['incomplete_opponent_play_history']
                         if not own.public_history_complete and self.allow_incomplete_history else []) +
                        [f'unknown_tactical_profile:{name}' for name in sorted(unknown_profiles)] +
                        ([f'geometry:{flag}' for flag in self.geometry.limitations]
                         if self.geometry != arena.LEGACY_GEOMETRY else []),
            'units': len(entities),
            'note': 'Ready means complete model inputs, not validated live mechanics or permission to tap.'
        }, rotated)


class MemoryPolicy:
    """Read-only v25 inference on encoded memory; independent of the CV runner."""
    def __init__(self, checkpoint: Path):
        import torch
        from sim.train_ppo import build_network, checkpoint_architecture
        self.torch = torch
        # Checkpoints are trusted local project files. No architecture migration.
        blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
        config = blob.get('config', {})
        self.input_contract = {key: config[key] for key in
            ('geometry', 'geometry_sha256', 'coordinate_mode', 'action_mask_mode') if key in config}
        self.network = build_network(NUM_PLANES, NUM_SCALARS, ACTIONS,
                                     checkpoint_architecture(blob))
        self.network.load_state_dict(blob['state_dict'], strict=True)
        self.network.eval()

    def validate_view_contract(self, view: ModelView):
        """Historical weights remain compatible; explicit new metadata is binding."""
        for key, expected in self.input_contract.items():
            if view.metadata.get(key) != expected:
                raise MissingModelInput(f'Checkpoint/view contract mismatch: {key}')

    def decide(self, view: ModelView) -> dict:
        self.validate_view_contract(view)
        if not view.metadata['ready_for_inference']:
            return {'action': 0, 'placement': None, 'blockers': view.metadata['blockers']}
        torch = self.torch
        with torch.inference_mode():
            planes = torch.from_numpy(view.observation['planes']).unsqueeze(0)
            scalars = torch.from_numpy(view.observation['scalars']).unsqueeze(0)
            logits, value = self.network(planes, scalars)
            logits[0, ~torch.from_numpy(view.action_mask)] = -torch.inf
            action = int(logits.argmax(dim=-1)[0])
        return {'action': action, 'placement': view.placement_world(action),
                'value': float(value[0]), 'blockers': []}
