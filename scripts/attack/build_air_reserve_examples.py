"""Successful wait-then-Musketeer sequences from paid TRAIN fixtures.

Local teacher lookahead labels actions; no future schedule enters model inputs.
Only zero-tower-damage clears are accepted. This is not a full-game expert.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from sim.adapter import grid_to_point
from sim.continuous_push import DECKS
from sim.defense_assets import load_defense_assets
from sim.env import ACTIONS, DECK_26
from sim.eval_defense_duel import policy_observation, mechanical_mask, JOINT_MODE
from sim.match import Match
from sim.runner import resolve_deck


def build(output, base='data/joint_skill_examples_20261027'):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    base = Path(base)
    source = json.loads((base / 'manifest.json').read_text())
    if source['split'] != 'train' or hashlib.sha256((base / 'examples.npz').read_bytes()).hexdigest() != source['data_sha256']:
        raise ValueError('Pinned TRAIN base required')
    with np.load(base / 'examples.npz', allow_pickle=False) as arrays:
        data = {key: list(arrays[key]) for key in ('planes', 'scalars', 'masks', 'success_masks')}
    records = deepcopy(source['records'])
    cards, spells, _, _ = load_defense_assets()
    cards.update(resolve_deck(cards, list(DECK_26)))
    trials = []
    for seed in source['seeds']:
        for lane in ('left', 'right'):
            for support in (False, True):
                for bank in (2000, 3000):
                    match = Match(cards=cards, spells=spells, decks=(list(DECK_26), list(DECKS['air'])),
                                  seed=seed, geometry='native_anchors_candidate_v1')
                    match.players[1].elixir = bank
                    match.players[-1].elixir = 10000
                    x = 3 if lane == 'left' else 14
                    assert match.play_card(-1, 'balloon', grid_to_point(17 - x, 22, -1))
                    while match.elapsed_ms < 500:
                        match.step(50)
                    if support:
                        assert match.play_card(-1, 'baby_dragon', grid_to_point(17 - x, 24, -1))
                    initial_hp = {k: t.hitpoints for k, t in match.towers[1].items()}
                    waiting = []
                    while match.players[1].elixir < 4000 and not match.finished:
                        waiting.append(deepcopy(match))
                        for _ in range(10):
                            match.step(50)
                    obs = policy_observation(match, 1, JOINT_MODE)
                    mask = mechanical_mask(match, cards, spells, 1)
                    slot = match.players[1].hand.index('musketeer')
                    safe = np.zeros(ACTIONS, dtype=np.bool_)
                    outcomes = []
                    for cx in (1, 3, 6):
                        for y in (17, 20, 23, 26):
                            px = cx if lane == 'left' else 17 - cx
                            action = 1 + slot * 576 + y * 18 + px
                            if not mask[action]:
                                continue
                            trial = deepcopy(match)
                            for _ in range(5):
                                trial.step(50)
                            if not trial.play_card(1, 'musketeer', grid_to_point(px, y, 1)):
                                continue
                            deadline = trial.elapsed_ms + 60000
                            while not trial.finished and trial.elapsed_ms < deadline:
                                trial.step(50)
                            remaining = sum(e.alive and e.side == -1 and not e.is_tower
                                            for e in trial.battle.entities.values())
                            damage = sum(hp - trial.towers[1][k].hitpoints for k, hp in initial_hp.items())
                            safe[action] = remaining == 0 and damage == 0
                            outcomes.append(dict(action=action, damage=damage, remaining=remaining))
                    accepted = bool(safe.any())
                    meta = dict(seed=seed, lane=lane, support=support, bank=bank / 1000,
                                wait_frames=len(waiting), accepted=accepted, candidates=outcomes)
                    trials.append(meta)
                    if not accepted:
                        continue
                    def append(state, label, kind):
                        observation = policy_observation(state, 1, JOINT_MODE)
                        legal = mechanical_mask(state, cards, spells, 1)
                        if (label & ~legal).any() or not label.any():
                            raise ValueError('Successful labels must be legal')
                        index = len(data['planes'])
                        for key, value in [('planes', observation['planes']), ('scalars', observation['scalars']),
                                           ('masks', legal), ('success_masks', label)]:
                            data[key].append(value)
                        records.append(dict(seed=seed, accepted=True, array_index=index, skill=kind,
                            lane=lane, support=support, decision_ms=state.elapsed_ms,
                            own_elixir=state.players[1].elixir / 1000))
                    for state in waiting:
                        label = np.zeros(ACTIONS, dtype=np.bool_)
                        label[0] = True
                        append(state, label, 'air_reserve_wait')
                    append(match, safe, 'air_reserve_deploy')
    added = len(data['planes']) - source['examples']
    if not added:
        raise ValueError('No successful reserve sequences; do not train on unsupported labels')
    np.savez_compressed(root / 'examples.npz', **{key: np.stack(value) for key, value in data.items()})
    manifest = dict(schema='joint-air-reserve-sequence-labels-v1', split='train', seeds=source['seeds'],
        examples=len(data['planes']), base_examples=source['examples'], added_examples=added, records=records,
        fixture_trials=trials, data_sha256=hashlib.sha256((root / 'examples.npz').read_bytes()).hexdigest(),
        base_data_sha256=source['data_sha256'], source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Successful local wait-then-deploy sequences, not optimal or full-game expert trajectories',
        own_inputs='Unchanged public20-plane309-scalar; no future schedule/hidden enemy bank',
        grouping='Entire maximum seed held out across base and added sequences',
        command_delay_ms=250, physics_changed=False, live_games_started=False,
        limitations=['Local fixtures stop enemy reinforcement after paid opening',
                     'Successful waiting is not proof every other legal action is inferior',
                     'Waiting frames are correlated; local accuracy is not full-game quality'])
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(examples=manifest['examples'], added=added,
        accepted_sequences=sum(r['accepted'] for r in trials), sequences=len(trials))), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--base', default='data/joint_skill_examples_20261027')
    args = parser.parse_args()
    build(args.output, args.base)
