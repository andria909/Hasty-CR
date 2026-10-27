import json
from pathlib import Path
import numpy as np
from sim.supervised_replay import SuccessfulActionReplay
from sim.env import DECK_26

DATA=Path('data/joint_skill_examples_20261027')


def test_shared_whole_seed_holdout_covers_every_skill_and_never_reuses_validation():
    m=json.loads((DATA/'manifest.json').read_text());heldout=max(m['seeds'])
    replay=SuccessfulActionReplay(DATA,'cpu')
    assert replay.metadata['train_examples']==68 and replay.metadata['heldout_examples']==34
    assert all(3800000<=s<3801000 for s in m['seeds'])
    for skill in m['by_skill']:
        records=[r for r in m['records']if r['accepted']and r['skill']==skill]
        assert any(r['seed']==heldout for r in records)
        assert any(r['seed']!=heldout for r in records)
    assert m['validation_data_reused']is False


def test_fireball_success_masks_match_resolved_hits_and_restraint_preserves_other_cards():
    m=json.loads((DATA/'manifest.json').read_text())
    with np.load(DATA/'examples.npz')as arrays:
        for row in m['records']:
            if not row['accepted']:continue
            index=row['array_index'];mask=arrays['masks'][index];success=arrays['success_masks'][index]
            assert success.any()and not np.any(success&~mask)
            if row['skill']=='fireball_group':
                expected={r['action']for r in row['candidate_outcomes']if r['damaged_units']>=3 and r['fireball_exchange_value']>0}
                assert set(np.flatnonzero(success))==expected
                assert all(r['impact_ms']>=r['cast_ms']>=r['request_ms']+225 for r in row['candidate_outcomes'])
            if row['skill']=='fireball_restraint':
                hand=arrays['scalars'][index,7:39].reshape(4,8);slot=int(np.flatnonzero(hand[:,DECK_26.index('fireball')])[0])
                expected=mask.copy();expected[1+slot*576:1+(slot+1)*576]=False
                np.testing.assert_array_equal(success,expected)
                assert success[0]
