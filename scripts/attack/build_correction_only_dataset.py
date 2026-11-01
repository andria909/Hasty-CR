"""Extract an exact, grouped subset for a diagnostic of task fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.attack.sequence_physics import require_current_physics
from scripts.kaggle.local_artifact_transfer import file_sha


def build(source,output):
    source,output=Path(source),Path(output)
    m=json.loads((source/'manifest.json').read_text());require_current_physics(m)
    if (m['split']!='train'or m.get('correction_label_mode')!='outcome_first_pareto_v1'or
            file_sha(source/'examples.npz')!=m['data_sha256']):raise ValueError('Verified correction curriculum required')
    records=[r for r in m['records']if r['accepted']and r['skill']=='onpolicy_defense_response']
    if {r['seed']for r in records}!={352,353,354}:raise ValueError('All three folds required')
    ids=[r['array_index']for r in records];output.mkdir(parents=True,exist_ok=False)
    with np.load(source/'examples.npz',allow_pickle=False)as z:
        np.savez_compressed(output/'examples.npz',**{k:z[k][ids]for k in ('planes','scalars','masks','success_masks')})
    result={k:m[k]for k in ('schema','split','seeds','physics_pins','command_delay_ms','slot_augmentation_contract')}
    result.update(records=[dict(r,array_index=i,source_array_index=r['array_index'])for i,r in enumerate(records)],
        examples=len(records),skills=['onpolicy_defense_response'],data_sha256=file_sha(output/'examples.npz'),
        subset_parent_manifest_sha256=file_sha(source/'manifest.json'),subset_parent_data_sha256=m['data_sha256'],
        sampling='Equal correction contexts; diagnostic only',full_game_training_ready=False,quality_accepted=False,
        scope='Exact subset including heldout fold; unrelated skills intentionally absent for optimization diagnosis')
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(dict(examples=build(a.source,a.output)['examples'])))
