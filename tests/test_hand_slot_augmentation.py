import itertools
import json
from pathlib import Path

import numpy as np
from sim.continuous_push import ContinuousPushEnv
from sim.env import ACTIONS,CARD_ACTIONS
from sim.hand_slot_augmentation import permute
from scripts.attack.build_slot_augmented_examples import identity


def test_slot_permutation_matches_actual_match_hand_reordering():
    env=ContinuousPushEnv(split='train')
    obs,info=env.reset(seed=6200647,scenario_id='defense::air/push')
    hand=list(env.match.players[1].hand)
    for order in itertools.permutations(range(4)):
        scalars,mask=permute(obs['scalars'],info['action_mask'],order)
        env.match.players[1].hand=[hand[i]for i in order]
        actual=env._observe()
        assert np.array_equal(actual['planes'],obs['planes'])
        assert np.array_equal(actual['scalars'],scalars)
        assert np.array_equal(env._mask(),mask)


def test_wait_and_living_unit_ability_actions_do_not_move_with_hand():
    scalars=np.arange(309,dtype=np.float32);actions=np.arange(ACTIONS)
    order=(3,0,2,1);inverse=np.argsort(order)
    changed,mapped=permute(scalars,actions,order)
    assert mapped[0]==actions[0]
    assert np.array_equal(mapped[CARD_ACTIONS:],actions[CARD_ACTIONS:])
    restored,restored_actions=permute(changed,mapped,inverse)
    assert np.array_equal(restored,scalars)and np.array_equal(restored_actions,actions)


def test_augmentation_preserves_heldout_and_excludes_exact_input_overlap():
    root=Path('data/hand_slot_examples_20261027')
    report=json.loads((root/'manifest.json').read_text())
    with np.load(root/'examples.npz',allow_pickle=False)as arrays,np.load(
            'data/air_lowbank_examples_20261027/examples.npz',allow_pickle=False)as base:
        arrays={key:arrays[key]for key in arrays.files}
        base={key:base[key]for key in base.files}
        groups={r['array_index']:r['seed']for r in report['records']if r['accepted']}
        heldout=[i for i,g in groups.items()if g==report['heldout_seed']]
        train=[i for i,g in groups.items()if g!=report['heldout_seed']]
        heldout_hashes={identity(arrays['planes'][i],arrays['scalars'][i],arrays['masks'][i])for i in heldout}
        assert not heldout_hashes.intersection(identity(arrays['planes'][i],arrays['scalars'][i],arrays['masks'][i])for i in train)
        assert all(i<report['base_examples']for i in heldout)
        for key in arrays:
            assert np.array_equal(arrays[key][heldout],base[key][heldout])
        assert not(base['success_masks']&~arrays['success_masks'][:len(base['planes'])]).any()
        assert arrays['success_masks'].any(-1).all()
        assert not(arrays['success_masks']&~arrays['masks']).any()
        assert report['exact_heldout_overlap_candidates_excluded']>0
