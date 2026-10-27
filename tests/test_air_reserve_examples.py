import hashlib
import json
from pathlib import Path

import numpy as np
from sim.supervised_replay import SuccessfulActionReplay


def test_reserve_labels_keep_legal_alternatives_and_end_waiting_at_affordability():
    root = Path('data/air_reserve_examples_20261027')
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['data_sha256'] == hashlib.sha256((root / 'examples.npz').read_bytes()).hexdigest()
    with np.load(root / 'examples.npz', allow_pickle=False) as arrays:
        records = [r for r in manifest['records'] if r.get('skill', '').startswith('air_reserve_')]
        assert records
        for row in records:
            i = row['array_index']
            mask, label = arrays['masks'][i], arrays['success_masks'][i]
            assert label.any() and not (label & ~mask).any()
            if row['skill'] == 'air_reserve_wait':
                assert row['own_elixir'] < 4
                assert arrays['scalars'][i, 0] < .4
                assert np.flatnonzero(label).tolist() == [0]
                assert mask[1:].any()  # Waiting is learned, not imposed by the mask.
            else:
                assert row['own_elixir'] >= 4
                assert not label[0]
                assert arrays['scalars'][i, 0] >= .4
    replay = SuccessfulActionReplay(root, 'cpu')
    assert replay.metadata['heldout_seed'] == max(manifest['seeds'])
    assert replay.metadata['heldout_seed'] not in replay.metadata['train_seeds']
    assert replay.metadata['train_examples'] + replay.metadata['heldout_examples'] == manifest['examples']


def test_existing_joint_examples_preserved_and_all_accepted_sequences_clear():
    root = Path('data/air_reserve_examples_20261027')
    manifest = json.loads((root / 'manifest.json').read_text())
    with np.load(root / 'examples.npz', allow_pickle=False) as arrays, np.load(
            'data/joint_skill_examples_20261027/examples.npz', allow_pickle=False) as base:
        for key in base.files:
            assert np.array_equal(arrays[key][:manifest['base_examples']], base[key])
    accepted = [r for r in manifest['fixture_trials'] if r['accepted']]
    assert len(accepted) == 12
    assert all(any(c['damage'] == c['remaining'] == 0 for c in r['candidates']) for r in accepted)
