import json
from pathlib import Path
import numpy as np
from scripts.attack.sequence_sampling import sequence_probabilities


def test_diverse_sequences_preserve_all_original_frames_and_holdout():
    base=Path('data/late_air_sequences_20261027')
    new=Path('data/late_air_diverse_sequences_20261027_v2')
    a=json.loads((base/'manifest.json').read_text()); b=json.loads((new/'manifest.json').read_text())
    assert b['records'][:a['examples']]==a['records']
    assert max(a['seeds'])==max(b['seeds'])==b['unchanged_holdout_seed']
    assert all(r['seed']!=max(a['seeds']) for r in b['records'][a['examples']:])
    with np.load(base/'examples.npz')as x,np.load(new/'examples.npz')as y:
        for k in x.files:assert np.array_equal(x[k],y[k][:a['examples']])
        assert not (y['success_masks'] & ~y['masks']).any()
    train=[r['array_index']for r in b['records']if r['seed']!=max(b['seeds'])]
    p=sequence_probabilities(b['records'],train)
    assert np.isclose(p.sum(),1)
    assert sum(v for v,i in zip(p,train) if b['records'][i]['action']==0)<=.25


def test_transfer_audit_retains_failed_contexts():
    path=Path('data/late_air_diverse_sequences_20261027_v2')
    m=json.loads((path/'manifest.json').read_text());audit=json.loads((path/'transfer_audit.json').read_text())
    assert len(audit['trials'])==24*8==m['new_contexts_attempted']
    assert sum(r['accepted']for r in audit['trials'])==m['new_contexts_accepted']
    assert any(not r['accepted']for r in audit['trials'])
    assert all(r['best']['valid'] and r['best']['clear_without_damage']for r in audit['accepted'])
