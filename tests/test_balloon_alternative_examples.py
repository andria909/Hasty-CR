import hashlib
import json
from pathlib import Path

import numpy as np
from sim.supervised_replay import SuccessfulActionReplay


def test_alternatives_keep_base_and_only_teach_useful_recovery_actions():
    root=Path('data/air_alternative_examples_20261027')
    report=json.loads((root/'manifest.json').read_text())
    assert hashlib.sha256((root/'examples.npz').read_bytes()).hexdigest()==report['data_sha256']
    with np.load(root/'examples.npz',allow_pickle=False)as arrays, np.load(
            'data/air_reserve_examples_20261027/examples.npz',allow_pickle=False)as base:
        for key in base.files:
            assert np.array_equal(arrays[key][:report['base_examples']],base[key])
        added=[r for r in report['records']if r.get('array_index',-1)>=report['base_examples']]
        skills={r['skill']for r in added}
        assert skills=={'air_cannon_efficient','air_electro_spirit_recovery','air_zap_recovery','air_fireball_recovery'}
        for row in added:
            i=row['array_index'];label=arrays['success_masks'][i];mask=arrays['masks'][i]
            assert label.any()and not(label&~mask).any()
            if row['skill'].endswith('_recovery'):
                assert row['cannon_alone_damage']>0
                assert len(row['commands'])==2
                assert row['commands'][0]['card']=='cannon'
    replay=SuccessfulActionReplay(root,'cpu')
    assert replay.metadata['heldout_seed']==max(report['seeds'])
    assert replay.metadata['heldout_seed']not in replay.metadata['train_seeds']


def test_search_successes_are_paid_and_respect_observed_command_delay():
    report=json.loads(Path('reports/defense_plan/balloon_combinations_20261027.json').read_text())
    for row in report['records']:
        assert row['elixir_cost']==sum(c['cost']for c in row['commands'])
        if row['clear_without_tower_damage']:
            assert row['total_damage']==row['remaining_enemies']==0
        for command in row['commands']:
            assert command['placed_ms']-command['decision_ms']==250
    for name in ('cannon+electro_spirit','cannon+zap','cannon+fireball'):
        assert any(r['sequence']==name and r['clear_without_tower_damage']for r in report['records'])
