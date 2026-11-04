"""Exhaustive one-command TRAIN continuations for hidden own queue positions.

The search is optimal only within one legal command now, followed by WAIT.
It does not measure a learned policy, full-game optimality, or live win rate.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.attack.evaluation_execution import HOG_CLIPPED, configure_battle
from scripts.attack.search_late_air_defense import rollout, rank
from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.adapter import grid_to_point
from sim.continuous_push import ContinuousPushEnv, DECKS
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26, ClashEnv
from sim.match import Match
from sim.pending_inputs import append_pending
from sim.runner import resolve_deck


def opening(cards, spells, lane, support):
    match = Match(cards=cards, spells=spells,
                  decks=(list(DECK_26), list(DECKS['air'])), seed=3800057,
                  geometry=HOG_CLIPPED['geometry'])
    configure_battle(match.battle, HOG_CLIPPED)
    match.players[1].elixir = match.players[-1].elixir = 10000
    hp = {k: t.hitpoints for k, t in match.towers[1].items()}
    x = 3 if lane == 'left' else 14
    if not match.play_card(-1, 'balloon', grid_to_point(17-x, 16, -1)):
        raise ValueError('Paid Balloon unavailable')
    while match.elapsed_ms < 500:
        match.step(50)
    if support and not match.play_card(-1, 'baby_dragon', grid_to_point(17-x, 18, -1)):
        raise ValueError('Paid Baby Dragon unavailable')
    while match.elapsed_ms < 2000:
        match.step(50)
    return match, hp


def queued_view(initial, cards, spells, cell):
    env = object.__new__(ContinuousPushEnv)
    env.match = deepcopy(initial)
    env.cards, env.spells = cards, spells
    env.profile = {'cannon_center_only': 1}
    env.slot_ready, env.reflected = {}, False
    slot = env.match.players[1].hand.index('musketeer')
    env.pending = [(initial.elapsed_ms+1350, 'musketeer', slot, *cell)]
    # The first request is legal and pays from the same native bank in both worlds.
    base = env.pending
    env.pending = []
    action = 1 + slot*576 + cell[1]*18 + cell[0]
    if not env._mask()[action]:
        raise ValueError('First request is illegal')
    env.pending = base
    for _ in range(5):
        env.match.step(50)
    return env


def compare(rows):
    """Compare action-indexed outcomes, retaining ties and separate HP/cost claims."""
    common = set(rows[0]) & set(rows[1])
    if not common or set(rows[0]) != set(rows[1]):
        raise ValueError('Both worlds require identical exhaustive legal action sets')
    best = [min(rank(r) for r in world.values()) for world in rows]
    opt = [{a for a, r in world.items() if rank(r) == score}
           for world, score in zip(rows, best)]
    hp = [min(r['total_damage'] for r in world.values()) for world in rows]
    hp_opt = [{a for a, r in world.items() if r['total_damage'] == damage}
              for world, damage in zip(rows, hp)]
    hp_regret = {a: max(rows[i][a]['total_damage']-hp[i] for i in range(2))
                 for a in common}
    lowest = min(hp_regret.values())
    return dict(legal_actions=len(common), best_rank=best,
                optimal_actions=[sorted(x) for x in opt],
                common_optimal_actions=sorted(opt[0] & opt[1]),
                minimum_damage=hp, common_minimum_damage_actions=sorted(hp_opt[0] & hp_opt[1]),
                best_shared_worst_case_hp_regret=lowest,
                shared_minimax_hp_actions=sorted(a for a in common if hp_regret[a] == lowest))


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pins = physics_pins()
    sources = {str(Path(p)): file_sha(p) for p in (
        __file__, 'scripts/attack/search_late_air_defense.py',
        'scripts/attack/sequence_execution.py', 'sim/pending_inputs.py')}
    protocol = dict(schema='pending-action-aliasing-train-v1', execution=HOG_CLIPPED,
                    physics_pins=pins, sources=sources, seed=3800057,
                    split='constructed_train_diagnostic', delay_ms=1350,
                    second_request_offset_ms=250, horizon_after_first_request_ms=60000,
                    first_card='musketeer', first_cells_left=[[1, 18], [8, 28]],
                    second_commands='Every legal action, including WAIT, then WAIT only',
                    rank='King survival, enemy clearance, own tower HP loss, spent elixir',
                    enemy_reinforcements=False, initial_elixir_each=10,
                    policy_inference=False, training_modified=False, live_games=False)
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (output/'driver.py').write_bytes(Path(__file__).read_bytes())
    cards, spells, *_ = load_defense_assets()
    cards.update(resolve_deck(cards, list(DECK_26)))
    results = []
    for lane in ('left', 'right'):
        for support in (False, True):
            context = f'{lane}_support{int(support)}'
            initial, hp = opening(cards, spells, lane, support)
            cells = [[x if lane == 'left' else 17-x, y] for x, y in protocol['first_cells_left']]
            envs = [queued_view(initial, cards, spells, c) for c in cells]
            obs = [e._observe() for e in envs]
            equal = {k: np.array_equal(obs[0][k], obs[1][k]) for k in ('planes', 'scalars')}
            equal['mask'] = np.array_equal(envs[0]._mask(), envs[1]._mask())
            if not all(equal.values()):
                raise ValueError('Base inputs differ; not an aliasing counterexample')
            extended = [append_pending(o, e.pending, e.match.elapsed_ms) for o, e in zip(obs, envs)]
            if np.array_equal(extended[0]['planes'], extended[1]['planes']):
                raise ValueError('Own queue encoder did not distinguish positions')
            actions = list(map(int, np.flatnonzero(envs[0]._mask())))
            if any(a >= 2305 for a in actions):
                raise ValueError('Ability actions unsupported by this local diagnostic')
            worlds = [{}, {}]
            path = output/f'{context}.jsonl'
            with path.open('x') as stream:
                for action in actions:
                    helper = None
                    if action:
                        slot, x, y = ClashEnv.decode(action)
                        helper = dict(card=envs[0].match.players[1].hand[slot],
                                      cell=[x, y], offset_ms=250)
                    for i, cell in enumerate(cells):
                        row = rollout(initial, hp, cards, spells, lane,
                                      dict(card='musketeer', cell=cell), helper)
                        if not row['valid']:
                            raise ValueError('Masked legal request failed paid execution')
                        worlds[i][action] = row
                        stream.write(json.dumps(dict(world=i, action=action, helper=helper, outcome=row))+'\n')
                    if len(worlds[0]) % 128 == 0:
                        stream.flush()
                        print(json.dumps(dict(context=context, actions=len(worlds[0]), total=len(actions))), flush=True)
            result = dict(context=context, lane=lane, support=support, pending_cells=cells,
                          identical_base_inputs=equal, pending_inputs_distinguish=True,
                          observation_sha256=hashlib.sha256(b''.join(obs[0][k].tobytes() for k in ('planes', 'scalars'))).hexdigest(),
                          raw_file=path.name, raw_sha256=file_sha(path), **compare(worlds))
            results.append(result)
            (output/f'{context}_summary.json').write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(dict(context=context, complete=True,
                                  common_optima=len(result['common_optimal_actions']),
                                  hp_regret=result['best_shared_worst_case_hp_regret'])), flush=True)
    if physics_pins() != pins or any(file_sha(p) != sha for p, sha in sources.items()):
        raise ValueError('Diagnostic source changed during execution')
    report = dict(protocol=protocol, complete=True, contexts=results,
                  quality_accepted=False, limitations=[
                      'Four synthetic contexts share one seed; not independent matches.',
                      'Optimality only over one immediate command followed by WAIT, not all future policies.',
                      'No prevalence estimate, trained-model comparison or live attribution.',
                      'Queue inputs were tried previously; this diagnostic alone cannot justify a new GPU study.'])
    (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
