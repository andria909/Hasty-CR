"""Opt-in TRAIN-only synthetic defensive starts, followed by normal full games.

These are sampled midgame exercises, not claimed reconstructions of reachable
match histories. Setup troops are paid; policy observations remain unchanged.
"""
import math
import random

from sim.adapter import grid_to_point
from sim.continuous_spell_waste import SpellWasteContinuousPushEnv
from sim.eval_defense_duel import mechanical_mask
from sim.tower_support import TowerSupport

PRIMARY = dict(air='balloon', hog='hog_rider', giant='giant', golem='golem',
               bridge='battle_ram', swarm='knight')


class TacticalStartContinuousPushEnv(SpellWasteContinuousPushEnv):
    def __init__(self, *args, tactical_start_fraction=0., **kwargs):
        if (isinstance(tactical_start_fraction, bool) or
                not math.isfinite(tactical_start_fraction) or not 0 <= tactical_start_fraction <= 1):
            raise ValueError('Tactical start fraction must be finite and within [0,1]')
        self.tactical_start_fraction = float(tactical_start_fraction)
        self.tactical_start = None
        super().__init__(*args, **kwargs)
        if (self.profile['cannon_center_only'] != 1 or self.profile['action_cooldown_ms'] != 0 or
                self.profile['tower_damage_all'] != 1 or self.action_delay_ms != 1350):
            raise ValueError('Tactical starts require central Cannon, zero slot lock, all tower damage and 1350ms delivery')

    def reset(self, *args, **kwargs):
        self.tactical_start = None
        observation, info = super().reset(*args, **kwargs)
        # Independent setup RNG: a zero fraction and held-out splits preserve
        # the original opponent RNG, observations and physics exactly.
        rng = random.Random(f'tactical-start-v1:{self.seed}')
        if self.split != 'train' or rng.random() >= self.tactical_start_fraction:
            return observation, info
        match = self.match
        clock = rng.choice((30000, 60000, 90000, 120000))
        match.battle.now_ms = clock
        masks = {side:rng.randrange(4) for side in (1, -1)}
        for side in (1, -1):
            for bit, lane in enumerate(('left', 'right')):
                if not masks[side] & (1 << bit):
                    match.towers[side][lane].hitpoints = 0
        match._wake_kings()
        own, enemy = match.players[1], match.players[-1]
        own.elixir = rng.randrange(3, 11)*1000
        enemy.elixir = 10000
        primary = PRIMARY[self.scenario.family]
        rest = [c for c in enemy.deck if c != primary]
        rng.shuffle(rest)
        order = [primary, *rest]
        enemy.hand, enemy.queue = order[:4], order[4:]
        lane = rng.choice(('left', 'right'))
        x = rng.choice((0, 1, 2, 3)) if primary == 'balloon' else rng.choice((2, 3, 4))
        if lane == 'right':
            x = 17-x
        setup = []

        def pay(card, cell):
            slot = enemy.hand.index(card)
            # Mechanical legality is expressed in enemy-relative cells.
            ex, ey = 17-cell[0], 31-cell[1]
            mask = mechanical_mask(match, self.cards, self.spells, -1)
            if not mask[1+slot*576+ey*18+ex]:
                raise ValueError('Illegal tactical setup placement')
            before = enemy.elixir
            if not match.play_card(-1, card, grid_to_point(*cell, 1)):
                raise ValueError('Paid tactical setup was rejected')
            record = dict(at_ms=match.elapsed_ms, card=card, cell=list(cell),
                elixir_before=before, elixir_after=enemy.elixir)
            setup.append(record)
            self.enemy_actions.append(record)

        pay(primary, (x, 15))
        warmup = rng.choice((500, 1000, 1500, 2000))
        support_requested = rng.random() < .5
        while match.elapsed_ms < clock+warmup:
            if match.elapsed_ms == clock+250 and support_requested:
                affordable = [c for c in enemy.hand if c not in self.spells and match.can_play(-1, c)]
                if affordable:
                    pay(rng.choice(affordable), (x, 12))
            match.step(50)
        # No setup credit: begin reward accounting at the returned observation.
        # Keep the historical damage events, detach only the old observer.
        match.battle.damage_log = list(match.battle.damage_log)
        self.tower_support = TowerSupport(match.battle).attach()
        self.next_enemy = match.elapsed_ms+500
        self.scripted.last_play_ms = match.elapsed_ms
        self.tactical_start = dict(schema='paid-tactical-start-v1', family=self.scenario.family,
            lane=lane, own_princess_mask=masks[1], enemy_princess_mask=masks[-1],
            clock_ms=clock, decision_start_ms=match.elapsed_ms, warmup_ms=warmup,
            own_hand=list(own.hand), own_queue=list(own.queue), own_elixir=own.elixir,
            enemy_setup=setup, synthetic_history=True,
            initial_own_fraction_loss=sum(1-t.hp_fraction for t in match.towers[1].values()),
            initial_enemy_fraction_damage=sum(1-t.hp_fraction for t in match.towers[-1].values()),
            initial_elixir_overflow=own.natural_elixir_overflow_milli/1000)
        self._initial_board_potential = self._metrics()['board_potential']
        self._previous = self._metrics()
        return self._observe(), self._info()

    def _metrics(self):
        metrics = super()._metrics()
        metrics.update(tactical_start_fraction=self.tactical_start_fraction,
                       tactical_start=self.tactical_start)
        return metrics

    def restore(self, state):
        if state.get('tactical_start_fraction') != self.tactical_start_fraction:
            raise ValueError('Tactical start fraction differs')
        super().restore(state)
