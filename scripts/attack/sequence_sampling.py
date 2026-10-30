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


def mixed_skill_probabilities(records, train_ids, wait_fraction=.5):
    """Equal skill/context mass; flexible restraint labels are not WAIT labels."""
    if not 0<wait_fraction<1:raise ValueError('WAIT fraction must be between zero and one')
    ids=list(map(int,train_ids))
    if not ids or len(ids)!=len(set(ids)):raise ValueError('Unique nonempty training indices required')
    by_index={r['array_index']:r for r in records if r['accepted']}
    skills={}
    for position,index in enumerate(ids):
        r=by_index[index]
        skills.setdefault(r['skill'],{}).setdefault((r['seed'],r['context']),[]).append((position,r['action']))
    p=np.zeros(len(ids),dtype=np.float64)
    for groups in skills.values():
        mass=1/len(skills)/len(groups)
        for group in groups.values():
            wait=[i for i,a in group if a==0]
            deploy=[i for i,a in group if a is not None and a!=0]
            choice=[i for i,a in group if a is None]
            if choice:
                if wait or deploy:raise ValueError('Flexible-choice and trajectory labels must have separate contexts')
                p[choice]=mass/len(choice)
            elif wait and deploy:
                p[wait]=mass*wait_fraction/len(wait)
                p[deploy]=mass*(1-wait_fraction)/len(deploy)
            elif deploy:p[deploy]=mass/len(deploy)
            else:raise ValueError('An always-WAIT sequence is not a complete teacher plan')
    if not np.isclose(p.sum(),1):raise ValueError('Invalid mixed skill probability mass')
    return p
