"""TRAIN-only sampling: equal mass per teacher sequence, bounded WAIT share."""
import numpy as np


def sequence_probabilities(records, train_ids, wait_fraction=0.25):
    if not 0 < wait_fraction < 1:
        raise ValueError('WAIT fraction must be between zero and one')
    ids = list(map(int, train_ids))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Unique nonempty training indices required')
    by_index = {r['array_index']: r for r in records if r['accepted']}
    groups = {}
    for position, index in enumerate(ids):
        r = by_index[index]
        if 'context' not in r or 'action' not in r:
            raise ValueError('Explicit sequential context/action required')
        groups.setdefault((r['seed'], r['context']), []).append((position, r['action']==0))
    p = np.zeros(len(ids), dtype=np.float64)
    for group in groups.values():
        wait = [i for i,w in group if w]
        play = [i for i,w in group if not w]
        if not play:
            raise ValueError('Sequence without a deployment is unsupported')
        mass = 1 / len(groups)
        if wait:
            p[wait] = mass*wait_fraction/len(wait)
        p[play] = mass*(1-wait_fraction if wait else 1)/len(play)
    if not np.isclose(p.sum(), 1):
        raise ValueError('Invalid probability mass')
    return p
