import json
from pathlib import Path
import numpy as np
from scripts.attack.sequence_physics import require_current_physics
from scripts.attack.sequence_sampling import sequence_probabilities


def test_stable_expansion_keeps_holdout_and_matches_current_physics():
    source=Path('data/late_air_stable_sequences_20261027')
    target=Path('data/late_air_stable_diverse_sequences_20261027')
    a=json.loads((source/'manifest.json').read_text());b=json.loads((target/'manifest.json').read_text())
    require_current_physics(a);require_current_physics(b)
    assert b['records'][:a['examples']]==a['records']
    assert max(a['seeds'])==max(b['seeds'])==3800085
    assert all(r['seed']!=3800085 for r in b['records'][a['examples']:])
    with np.load(source/'examples.npz')as x,np.load(target/'examples.npz')as y:
        for key in x.files:assert np.array_equal(x[key],y[key][:a['examples']])
        assert not np.any(y['success_masks']&~y['masks'])
    ids=[r['array_index']for r in b['records']if r['seed']!=3800085]
    p=sequence_probabilities(b['records'],ids,.25)
    assert np.isclose(p.sum(),1)
    assert sum(v for v,i in zip(p,ids)if b['records'][i]['action']==0)<=.25
