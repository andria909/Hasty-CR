"""Version existing TRAIN data, filtering only audited Fireball success masks."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from scripts.kaggle.local_artifact_transfer import file_sha


def build(source,audit_path,output,delay=1350):
    source,audit_path,output=map(Path,(source,audit_path,output))
    manifest=json.loads((source/'manifest.json').read_text());audit=json.loads(audit_path.read_text())
    original=Path(audit['source_data'])
    if (manifest['split']!='train' or file_sha(source/'examples.npz')!=manifest['data_sha256']
            or file_sha(original/'manifest.json')!=audit['source_manifest_sha256']
            or file_sha(original/'examples.npz')!=audit['source_data_sha256']
            or manifest.get('base_data_sha256')!=audit['source_data_sha256']
            or audit.get('original225ms_outcomes_reproduced') is not True):
        raise ValueError('Matching verified original TRAIN data required')
    with np.load(source/'examples.npz',allow_pickle=False)as data:
        arrays={k:data[k].copy()for k in data.files}
    before={k:v.copy()for k,v in arrays.items()}
    records=deepcopy(manifest['records'])
    targets={r['id']:r for r in records if r.get('skill')=='fireball_group' and r.get('accepted')}
    if len(targets)!=len(audit['rows']) or set(targets)!={r['id']for r in audit['rows']}:
        raise ValueError('Audited Fireball fixture coverage differs')
    changed=[];removed=0
    for row in audit['rows']:
        record=targets[row['id']];index=record['array_index'];old=arrays['success_masks'][index]
        if sorted(np.flatnonzero(old).tolist())!=sorted(row['by_delay']['225']['successful_action_ids']):
            raise ValueError('Original success mask differs from audit')
        keep=row['by_delay'][str(delay)]['successful_action_ids']
        if not keep or len(set(keep))!=len(keep) or not set(keep)<=set(np.flatnonzero(old)):
            raise ValueError('Filtered label must retain a nonempty audited subset')
        updated=np.zeros_like(old);updated[keep]=True
        if np.any(updated & ~arrays['masks'][index]):raise ValueError('Illegal relabelled action')
        removed+=int(old.sum()-updated.sum())
        if not np.array_equal(old,updated):changed.append(index)
        arrays['success_masks'][index]=updated
        record['original225ms_candidate_outcomes']=record.pop('candidate_outcomes')
        record.update(label_command_delay_ms=delay,success_action_ids=keep,
                      label_scope='Filtered old positive targets only; no newly searched targets')
    for key in arrays:
        if key!='success_masks' and not np.array_equal(arrays[key],before[key]):raise ValueError('Inputs changed')
    other=[i for i in range(manifest['examples']) if i not in {r['array_index']for r in targets.values()}]
    if not np.array_equal(arrays['success_masks'][other],before['success_masks'][other]):raise ValueError('Non-Fireball labels changed')
    output.mkdir(parents=True,exist_ok=False);np.savez_compressed(output/'examples.npz',**arrays)
    result=deepcopy(manifest)
    result.update(schema='joint-fireball-latency-filtered-labels-v1',records=records,data_sha256=file_sha(output/'examples.npz'),
        source_dataset=str(source),source_manifest_sha256=file_sha(source/'manifest.json'),source_data_sha256=manifest['data_sha256'],
        audit_sha256=file_sha(audit_path),builder_sha256=file_sha(__file__),command_delay_ms=None,
        label_timing_by_skill=dict(fireball_group=delay,other_labels='Unchanged original nominal225ms / effective250ms'),
        changed_indices=changed,removed_positive_actions=removed,fireball_examples=len(targets),
        source_inputs_bitwise_preserved=True,other_success_masks_bitwise_preserved=True,
        grouping='Original maximum seed held out unchanged; local diagnostic only, not independent evaluation',
        scope='Fireball-only timing correction; air and reserve labels remain unchanged',
        quality_accepted=False,used_in_training=False)
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','audit','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();r=build(a.source,a.audit,a.output)
    print(json.dumps({k:r[k]for k in ('examples','fireball_examples','removed_positive_actions','data_sha256','used_in_training')}))
