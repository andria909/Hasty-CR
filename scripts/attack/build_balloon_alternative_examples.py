"""Add efficient Cannon defenses and useful recovery spells/spirit TRAIN labels.

Helper labels require strictly better tower damage than the identical Cannon
placement alone. Thus successful but unnecessary spell spending is excluded.
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
from sim.eval_defense_duel import mechanical_mask, policy_observation, JOINT_MODE
from sim.match import Match
from sim.runner import resolve_deck


def build(output, sweep='reports/defense_plan/balloon_combinations_20261027.json',
          base='data/air_reserve_examples_20261027'):
    output, base, sweep = Path(output), Path(base), Path(sweep)
    output.mkdir(parents=True, exist_ok=False)
    report = json.loads(sweep.read_text())
    manifest = json.loads((base / 'manifest.json').read_text())
    if report['split'] != 'train_diagnostic' or manifest['split'] != 'train':
        raise ValueError('TRAIN-only sources required')
    if hashlib.sha256((base / 'examples.npz').read_bytes()).hexdigest() != manifest['data_sha256']:
        raise ValueError('Base dataset changed')
    with np.load(base / 'examples.npz', allow_pickle=False) as arrays:
        data = {key:list(arrays[key]) for key in arrays.files}
    records = deepcopy(manifest['records'])
    rows = report['records']
    def context(row):
        return row['seed'], row['lane'], row['support']
    def cannon_key(row):
        first = row['commands'][0]
        return (*context(row), first['x'], first['y'])
    alone = {cannon_key(row):row for row in rows if row['sequence'] == 'cannon_only'}
    chosen = {}
    for row in rows:
        if not row['clear_without_tower_damage'] or not row['commands']:
            continue
        if len(row['commands']) == 1:
            continue
        comparison = alone[cannon_key(row)]
        if comparison['total_damage'] <= row['total_damage']:
            continue
        key = (*context(row), row['sequence'])
        if key not in chosen or row['commands'][1]['decision_ms'] < chosen[key]['commands'][1]['decision_ms']:
            chosen[key] = row
    cards, spells, _, _ = load_defense_assets()
    cards.update(resolve_deck(cards, list(DECK_26)))
    def advance(match, target):
        while match.elapsed_ms < target and not match.finished:
            match.step(50)
    def opening(row):
        match = Match(cards=cards, spells=spells, decks=(list(DECK_26), list(DECKS['air'])),
                      seed=row['seed'], geometry='native_anchors_candidate_v1')
        match.players[1].elixir = 7000
        match.players[-1].elixir = 10000
        x = 3 if row['lane'] == 'left' else 14
        assert match.play_card(-1, 'balloon', grid_to_point(17-x, 16, -1))
        advance(match, 500)
        if row['support']:
            assert match.play_card(-1, 'baby_dragon', grid_to_point(17-x, 18, -1))
        return match
    def append(match, row, commands, skill, ready=None):
        obs = policy_observation(match, 1, JOINT_MODE)
        mask = mechanical_mask(match, cards, spells, 1)
        for slot, due in (ready or {}).items():
            if match.elapsed_ms < due:
                mask[1+slot*576:1+(slot+1)*576] = False
        label = np.zeros(ACTIONS, dtype=np.bool_)
        for command in commands:
            label[command['action']] = True
        if not label.any() or (label & ~mask).any():
            raise ValueError('Illegal replayed successful action')
        index = len(data['planes'])
        for key, value in [('planes', obs['planes']), ('scalars', obs['scalars']),
                           ('masks', mask), ('success_masks', label)]:
            data[key].append(value)
        records.append(dict(array_index=index, accepted=True, seed=row['seed'], lane=row['lane'],
            support=row['support'], skill=skill, decision_ms=match.elapsed_ms,
            source_sequence=row['sequence'], commands=deepcopy(row['commands']),
            cannon_alone_damage=(alone[cannon_key(row)]['total_damage'] if len(row['commands']) > 1 else 0)))
    efficient = {}
    for row in alone.values():
        if row['clear_without_tower_damage']:
            efficient.setdefault(context(row), []).append(row)
    for group in efficient.values():
        append(opening(group[0]), group[0], [r['commands'][0] for r in group], 'air_cannon_efficient')
    for row in chosen.values():
        match = opening(row)
        first, helper = row['commands']
        advance(match, first['placed_ms'])
        assert match.play_card(1, first['card'], grid_to_point(first['x'], first['y'], 1))
        advance(match, helper['decision_ms'])
        append(match, row, [helper], 'air_' + helper['card'] + '_recovery',
               ready={first['slot']:first['placed_ms'] + 2100})
    added = len(data['planes']) - manifest['examples']
    if not added:
        raise ValueError('No verified alternative labels')
    np.savez_compressed(output / 'examples.npz', **{key:np.stack(value) for key, value in data.items()})
    result = dict(schema='joint-air-alternative-labels-v1', split='train',
        seeds=sorted(set(manifest['seeds']) | set(report['seeds'])), records=records,
        examples=len(data['planes']), base_examples=manifest['examples'], added_examples=added,
        data_sha256=hashlib.sha256((output/'examples.npz').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_data_sha256=manifest['data_sha256'], sweep_sha256=hashlib.sha256(sweep.read_bytes()).hexdigest(),
        scope='Efficient Cannon openings plus recovery actions that strictly reduce damage over identical Cannon-only control',
        limitations=['Only single Balloon had zero-damage clears; supported Balloon not solved',
                     'Maximum seed held out now tests only new alternative fixtures; local accuracy is not cross-skill quality',
                     'Former local base holdout becomes TRAIN; full-game validation/test contexts are untouched',
                     'Recovery action labels do not by themselves teach complete timing sequences'],
        command_delay_ms=225, slot_cooldown_ms=2100, physics_changed=False, live_games_started=False)
    (output/'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(examples=result['examples'], added=added, recovery_sequences=len(chosen))), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build(args.output)
