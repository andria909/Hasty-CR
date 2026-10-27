"""Augment TRAIN groups only; preserve heldout and exclude exact input overlap."""
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from sim.hand_slot_augmentation import permute


def identity(planes, scalars, mask):
    return hashlib.sha256(planes.tobytes()+scalars.tobytes()+mask.tobytes()).hexdigest()


def build(base, output):
    base, output = Path(base), Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous augmented dataset')
    manifest = json.loads((base/'manifest.json').read_text())
    if manifest['split'] != 'train' or hashlib.sha256((base/'examples.npz').read_bytes()).hexdigest() != manifest['data_sha256']:
        raise ValueError('Pinned TRAIN source required')
    with np.load(base/'examples.npz',allow_pickle=False) as arrays:
        data = {key:list(arrays[key]) for key in ('planes','scalars','masks','success_masks')}
    groups = {r['array_index']:r['seed'] for r in manifest['records'] if r['accepted']}
    if set(groups) != set(range(len(data['planes']))):
        raise ValueError('Every example needs an explicit seed group')
    heldout = max(groups.values())
    heldout_inputs = {identity(data['planes'][i],data['scalars'][i],data['masks'][i])
                      for i,seed in groups.items() if seed == heldout}
    seen = {identity(data['planes'][i],data['scalars'][i],data['masks'][i]):i for i in groups}
    if any(identity(data['planes'][i],data['scalars'][i],data['masks'][i]) in heldout_inputs
           for i,seed in groups.items() if seed != heldout):
        raise ValueError('Original TRAIN/holdout inputs already overlap')
    records = deepcopy(manifest['records']); skipped = merged = 0
    for i,seed in groups.items():
        if seed == heldout:
            continue
        for permutation in itertools.permutations(range(4)):
            scalars,mask = permute(data['scalars'][i],data['masks'][i],permutation)
            _,label = permute(data['scalars'][i],data['success_masks'][i],permutation)
            key = identity(data['planes'][i],scalars,mask)
            if key in heldout_inputs:
                skipped += 1
                continue
            if key in seen:
                # Different successful continuations can share the same public
                # state (e.g. Spirit versus Zap). Keep their union, TRAIN only.
                existing=seen[key]
                union=np.logical_or(data['success_masks'][existing],label)
                if not np.array_equal(union,data['success_masks'][existing]):
                    merged+=1;data['success_masks'][existing]=union
                continue
            seen[key] = len(data['planes'])
            records.append(dict(array_index=len(data['planes']),seed=seed,accepted=True,
                skill='hand_slot_permutation',source_array_index=i,permutation=list(permutation)))
            data['planes'].append(data['planes'][i])
            data['scalars'].append(scalars);data['masks'].append(mask);data['success_masks'].append(label)
    output.mkdir(parents=True)
    np.savez_compressed(output/'examples.npz',**{key:np.stack(value) for key,value in data.items()})
    report = dict(schema='public-hand-slot-augmented-examples-v1',split='train',seeds=manifest['seeds'],
        records=records,examples=len(data['planes']),base_examples=manifest['examples'],
        added_examples=len(data['planes'])-manifest['examples'],heldout_seed=heldout,
        exact_heldout_overlap_candidates_excluded=skipped,merged_successful_alternative_sets=merged,
        data_sha256=hashlib.sha256((output/'examples.npz').read_bytes()).hexdigest(),
        base_data_sha256=manifest['data_sha256'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        permutation_source_sha256=hashlib.sha256(Path('sim/hand_slot_augmentation.py').read_bytes()).hexdigest(),
        scope='Legal single-decision slot relabeling, no new physics or optimality claims',
        limitations=['Augmented rows are dependent, not additional independent trajectories',
                     'Local heldout remains limited to source fixture families',
                     'Full-game validation/test still required'],live_games_started=False,quality_accepted=False)
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key]for key in ('examples','added_examples','exact_heldout_overlap_candidates_excluded')}),flush=True)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();build(args.base,args.output)
