"""Add explicit Fireball/Zap restraint coverage to each whole split fold.

This labels only avoiding empty spells. It does not label every accepted troop
placement as an economical or tactically good decision.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.attack.sequence_physics import physics_pins, require_current_physics
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.cannon_placement import CENTRAL_MODE, restrict_cannon
from sim.continuous_push import DECKS
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26, LANES
from sim.eval_defense_duel import JOINT_MODE, mechanical_mask, policy_observation
from sim.match import Match
from sim.runner import resolve_deck


def coverage(manifest, arrays):
    result = {}
    for r in manifest['records']:
        if not r['accepted'] or r['skill'] != 'spell_restraint':
            continue
        i = r['array_index']
        start = 3 + 2 * len(LANES)
        hand = arrays['scalars'][i, start:start + 4 * len(DECK_26)].reshape(4, len(DECK_26))
        fold = result.setdefault(str(r['seed']), {c: dict(examples=0, slots=[]) for c in ('fireball', 'zap')})
        for slot, onehot in enumerate(hand):
            if onehot.sum() != 1:
                raise ValueError('Public hand must be one-hot')
            card = DECK_26[int(onehot.argmax())]
            if card not in fold:
                continue
            segment = slice(1 + slot * 576, 1 + (slot + 1) * 576)
            if arrays['masks'][i, segment].any() and not arrays['success_masks'][i, segment].any():
                fold[card]['examples'] += 1
                fold[card]['slots'] = sorted(set(fold[card]['slots']) | {slot})
    return result


def build(base, output):
    base, output = Path(base), Path(output)
    m = json.loads((base / 'manifest.json').read_text())
    require_current_physics(m)
    if m['schema'] != 'mixed-bank-public-sequences-v1' or m['seeds'] != [352, 353, 354]:
        raise ValueError('Existing three-fold mixed curriculum required')
    if file_sha(base / 'examples.npz') != m['data_sha256']:
        raise ValueError('Source data changed')
    with np.load(base / 'examples.npz', allow_pickle=False) as a:
        arrays = {k: list(a[k]) for k in a.files}
        before = coverage(m, a)
    cards, spells, _, _ = load_defense_assets()
    cards.update(resolve_deck(cards, list(DECK_26)))
    records = deepcopy(m['records'])
    used = {r.get('simulation_seed', r['seed']) for r in records}
    chosen = []
    next_seed = 3810200
    for fold in (352, 353, 354):
        missing = {('fireball', i) for i in range(4)} | {('zap', i) for i in range(4)}
        while missing:
            seed = next_seed
            next_seed += 1
            if seed >= 3811000:
                raise ValueError('Development seed range exhausted')
            if seed in used:
                continue
            initial = Match(cards=cards, spells=spells, decks=(list(DECK_26), list(DECKS['hog'])),
                seed=seed, geometry='native_anchors_candidate_v1')
            hand = initial.players[1].hand
            if not {'fireball', 'zap'} <= set(hand):
                continue
            traits = {(card, hand.index(card)) for card in ('fireball', 'zap')}
            if not traits & missing:
                continue
            missing -= traits
            used.add(seed)
            chosen.append(dict(fold=fold, simulation_seed=seed, hand=list(hand)))
            for bank in (5000, 10000):
                for age in (1500, 4500):
                    match = deepcopy(initial)
                    match.players[1].elixir = bank
                    while match.elapsed_ms < age:
                        match.step(50)
                    mask = restrict_cannon(mechanical_mask(match, cards, spells, 1), hand, CENTRAL_MODE)
                    success = mask.copy()
                    for card in ('fireball', 'zap'):
                        slot = hand.index(card)
                        success[1 + slot * 576:1 + (slot + 1) * 576] = False
                    obs = policy_observation(match, 1, JOINT_MODE)
                    state = dict(obs, masks=mask, success_masks=success)
                    index = len(records)
                    for key in arrays:
                        arrays[key].append(state[key])
                    records.append(dict(accepted=True, array_index=index, seed=fold,
                        simulation_seed=seed, skill='spell_restraint', action=None,
                        context=f'spell-coverage:{seed}:{bank}:{age}', pending_count=0,
                        reserved_elixir=0, hand=list(hand), initial_elixir=bank / 1000, decision_ms=age))
    identities = {}
    for r in records:
        i = r['array_index']
        digest = hashlib.sha256(b''.join(arrays[k][i].tobytes() for k in ('planes', 'scalars', 'masks'))).hexdigest()
        if digest in identities and identities[digest] != r['seed']:
            raise ValueError('Identical input across folds')
        identities[digest] = r['seed']
        r['input_sha256'] = digest
    packed = {k: np.stack(v) for k, v in arrays.items()}
    result = dict(m, records=records, examples=len(records),
        builder_sha256=file_sha(__file__), source_data=str(base), source_data_sha256=m['data_sha256'],
        source_manifest_sha256=file_sha(base / 'manifest.json'),
        restraint_seed_selection=chosen, spell_restraint_coverage_before=before,
        full_game_training_ready=False, quality_accepted=False,
        limitations=m['limitations'] + ['Empty-spell labels do not certify accepted unit placement quality or positive Zap timing'])
    after = coverage(result, packed)
    for fold in ('352', '353', '354'):
        if any(after[fold][card]['slots'] != [0, 1, 2, 3] for card in ('fireball', 'zap')):
            raise ValueError('Spell/slot trait absent from a fold')
    result['spell_restraint_coverage'] = after
    require_current_physics(result)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'examples.npz', **packed)
    result['data_sha256'] = file_sha(output / 'examples.npz')
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    m = build(a.base, a.output)
    print(json.dumps(dict(examples=m['examples'], coverage=m['spell_restraint_coverage'])))
