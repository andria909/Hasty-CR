"""Verify assigned initializers, real PPO updates and unchanged execution."""
import argparse
import json
from pathlib import Path

from scripts.attack.verify_continuous_focus import verify
from scripts.attack.verify_crown_economy_pilot import audit_rewards
from scripts.attack.verify_global_full_pair import audit_training_execution
from scripts.attack.evaluation_execution import HOG_CLIPPED
from scripts.kaggle.prepare_policy_adaptation_pilot import VARIANT,INITIALS,validate_protocol
from scripts.kaggle.local_artifact_transfer import file_sha


def verify_adaptation(collected,prepared):
    root,prepared=Path(collected),Path(prepared);prep=json.loads((prepared/'preparation.json').read_text())
    if prep.get('study_variant')!=VARIANT:raise ValueError('Wrong adaptation study')
    protocol=json.loads((prepared/'dataset/protocol.json').read_text());validate_protocol(protocol)
    models=prep['config']['models']
    if len(models)!=4 or {m['id']for m in models}!=set(INITIALS):raise ValueError('Four named initializers required')
    for m in models:
        if file_sha(prepared/'dataset'/m['file'])!=m['sha256']:raise ValueError('Assigned initial checkpoint changed')
    result=verify(root,prepared);audits=[]
    reference_contexts=[]
    for model in models:
        reference=json.loads((root/(model['id']+'_reference.json')).read_text())
        reference_contexts.append({(r['scenario_id'],r['episode_seed'],r['match_seed'],r['task'])
            for r in reference['validation']['records']})
    if any(contexts!=reference_contexts[0]for contexts in reference_contexts):
        raise ValueError('Initializer reference contexts differ')
    import torch
    for trial in protocol['cases']:
        for seed in protocol['pilot_seeds']:
            name=f"pilot_{trial['id']}_s{seed}";run=root/name
            config=json.loads((run/'config.json').read_text());episodes=[]
            for line in (run/'metrics.jsonl').read_text().splitlines():
                row=json.loads(line);episodes.extend(row['episodes'])
                if 'validation'in row:episodes.extend(row['validation']['records'])
            audit_rewards(config,episodes,trial)
            selected=next(r for r in result['pilot_models']if r['id']==name)
            blob=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
            expected=next(m for m in models if m['id']==trial['initial_id'])
            if blob['config']['init_sha256']!=expected['sha256']:raise ValueError('Checkpoint initialization differs')
            execution,count=audit_training_execution(config,blob,run/'metrics.jsonl')
            if execution!=HOG_CLIPPED:raise ValueError('Adaptation combat execution differs')
            audits.append(dict(id=name,initial_id=expected['id'],initial_sha256=expected['sha256'],geometry_records=count))
    result['policy_adaptation_audit']=dict(verified=True,runs=audits)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('collected','prepared','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();Path(a.output).write_text(json.dumps(verify_adaptation(a.collected,a.prepared),indent=2)+'\n')
