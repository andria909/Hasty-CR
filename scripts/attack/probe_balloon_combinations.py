"""TRAIN-only paid Balloon defense sweep; no physics changes or live override."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from sim.adapter import grid_to_point
from sim.arena import MT
from sim.continuous_push import DECKS
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26
from sim.eval_defense_duel import mechanical_mask
from sim.match import Match
from sim.runner import resolve_deck


def probe(output, seed_count=3):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve earlier sweep')
    if not 1 <= seed_count <= 3:
        raise ValueError('One to three grouped TRAIN seeds supported')
    cards, spells, _, _ = load_defense_assets()
    cards.update(resolve_deck(cards, list(DECK_26)))
    def make(seed):
        return Match(cards=cards, spells=spells, decks=(list(DECK_26), list(DECKS['air'])),
                     seed=seed, geometry='native_anchors_candidate_v1')
    seeds = []
    for seed in range(6200000, 6210000):
        match = make(seed)
        if {'cannon', 'electro_spirit', 'fireball', 'zap'} <= set(match.players[1].hand) and {'balloon', 'baby_dragon'} <= set(match.players[-1].hand):
            seeds.append(seed)
        if len(seeds) == seed_count:
            break
    if len(seeds) != seed_count:
        raise ValueError('Insufficient actual legal opening hands')
    def advance(match, target):
        while not match.finished and match.elapsed_ms < target:
            match.step(50)
    def play(match, card, x, y, ready):
        if card not in match.players[1].hand:
            return None
        slot = match.players[1].hand.index(card)
        if match.elapsed_ms < ready.get(slot, 0):
            return None
        action = 1 + slot * 576 + y * 18 + x
        if not mechanical_mask(match, cards, spells, 1)[action]:
            return None
        decision_ms = match.elapsed_ms
        advance(match, decision_ms + 225)
        if not match.play_card(1, card, grid_to_point(x, y, 1)):
            return None
        ready[slot] = match.elapsed_ms + 2100
        return dict(card=card, x=x, y=y, action=action, decision_ms=decision_ms,
                    placed_ms=match.elapsed_ms, cost=cards[card].cost, slot=slot)
    records = []
    for seed in seeds:
        for lane in ('left', 'right'):
            for support in (False, True):
                initial = make(seed)
                initial.players[1].elixir = 7000
                initial.players[-1].elixir = 10000
                x = 3 if lane == 'left' else 14
                assert initial.play_card(-1, 'balloon', grid_to_point(17 - x, 16, -1))
                advance(initial, 500)
                if support:
                    assert initial.play_card(-1, 'baby_dragon', grid_to_point(17 - x, 18, -1))
                hp = {k: tower.hitpoints for k, tower in initial.towers[1].items()}
                def finish(trial, commands, name):
                    advance(trial, 65000)
                    damage = {k: before - trial.towers[1][k].hitpoints for k, before in hp.items()}
                    remaining = sum(e.alive and e.side == -1 and not e.is_tower for e in trial.battle.entities.values())
                    records.append(dict(seed=seed, lane=lane, support=support, sequence=name,
                        commands=commands, elixir_cost=sum(c['cost'] for c in commands),
                        own_tower_damage=damage, total_damage=sum(damage.values()),
                        remaining_enemies=remaining, clear_without_tower_damage=not any(damage.values()) and remaining == 0))
                finish(deepcopy(initial), [], 'no_cards')
                for cx, cy in ((7, 19), (8, 20), (7, 22), (8, 23)):
                    px = cx if lane == 'left' else 17 - cx
                    first = deepcopy(initial)
                    ready = {}
                    cannon = play(first, 'cannon', px, cy, ready)
                    if cannon is None:
                        continue
                    finish(deepcopy(first), [cannon], 'cannon_only')
                    for helper in ('electro_spirit', 'zap', 'fireball'):
                        for delay in (500, 1500, 2500, 4000):
                            positions = ((x, 21), (6 if lane == 'left' else 11, 22), (x, 25)) if helper == 'electro_spirit' else (None,)
                            for position in positions:
                                trial = deepcopy(first)
                                advance(trial, first.elapsed_ms + delay)
                                if trial.finished:
                                    continue
                                if position is None:
                                    balloons = [e for e in trial.battle.entities.values() if e.alive and e.side == -1 and e.name == 'balloon']
                                    if not balloons:
                                        continue
                                    target = balloons[0]
                                    tx, ty = int(target.pos.x / MT), int(target.pos.y / MT)
                                    position = (min(17, max(0, tx)), min(31, max(0, ty)))
                                command = play(trial, helper, *position, dict(ready))
                                if command is not None:
                                    finish(trial, [cannon, command], 'cannon+' + helper)
        print(json.dumps(dict(seed=seed, completed_trials=len(records))), flush=True)
    groups = {}
    for name in sorted({r['sequence'] for r in records}):
        rows = [r for r in records if r['sequence'] == name]
        groups[name] = dict(trials=len(rows), successful=sum(r['clear_without_tower_damage'] for r in rows),
                           minimum_damage=min(r['total_damage'] for r in rows))
    result = dict(schema='train-balloon-combination-sweep-v1', split='train_diagnostic', seeds=seeds,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), groups=groups, records=records,
        command_delay_ms=225, slot_cooldown_ms=2100, initial_own_elixir=7,
        enemy_opening_paid=True, live_games_started=False, physics_changed=False, quality_accepted=False,
        limitations=['Local fixed paid opening, no enemy reinforcement',
                     'Grid/timing search is not proof of optimality or real-game success',
                     'Seven-elixir opening is shared; actual card costs are reported separately',
                     'Spell target is visible Balloon position at decision, not a future oracle'])
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(groups), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed-count', type=int, default=3)
    args = parser.parse_args()
    probe(args.output, args.seed_count)
