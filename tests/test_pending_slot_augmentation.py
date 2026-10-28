import itertools
import numpy as np
from sim.continuous_pending import PendingContinuousPushEnv
from sim.pending_slot_augmentation import permute_pending


def test_all_permutations_match_actual_pending_environment_and_masks():
    env=PendingContinuousPushEnv(split='train')
    env.reset(seed=6200647,scenario_id='defense::air/push')
    hand=list(env.match.players[1].hand)
    pending=[(1350,hand[0],0,3,20),(2025,hand[3],3,12,24)]
    ready={1:2100}
    env.pending=list(pending);env.slot_ready=dict(ready)
    obs,mask=env._observe(),env._mask()
    for order in itertools.permutations(range(4)):
        changed,mapped=permute_pending(obs,mask,order);inverse=np.argsort(order)
        env.match.players[1].hand=[hand[i]for i in order]
        env.pending=[(due,card,int(inverse[slot]),x,y)for due,card,slot,x,y in pending]
        env.slot_ready={int(inverse[slot]):due for slot,due in ready.items()}
        actual=env._observe()
        assert np.array_equal(actual['planes'],changed['planes'])
        assert np.array_equal(actual['scalars'],changed['scalars'])
        assert np.array_equal(env._mask(),mapped)
        restored,original_mask=permute_pending(changed,mapped,inverse)
        assert np.array_equal(restored['planes'],obs['planes'])
        assert np.array_equal(restored['scalars'],obs['scalars'])
        assert np.array_equal(original_mask,mask)
