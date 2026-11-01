"""Run frozen real PPO/resume and assert the new entropy option reached it."""
import argparse
import importlib.util
import json
from pathlib import Path


def child(path):
    file=Path(__file__).with_name('preflight_policy_adaptation_pilot.py')
    spec=importlib.util.spec_from_file_location('shared_adaptation_preflight',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.child(path)
    import torch
    s=json.loads(Path(path).read_text());root=Path(s['root']);weight=s['trial']['spatial_entropy_weight']
    blob=torch.load(root/'whole/update_000002.pt',map_location='cpu',weights_only=False)
    if blob['config']['spatial_entropy_weight']!=weight:raise ValueError('Actual PPO entropy weight differs')
    for line in (root/'whole/metrics.jsonl').read_text().splitlines():
        row=json.loads(line)['entropy_components']
        if row['spatial_weight']!=weight or row['optimizer_batches']<=0:
            raise ValueError('Actual optimizer entropy instrumentation missing')
        if abs(row['objective']-(weight*row['full']+(1-weight)*row['group']))>1e-5:
            raise ValueError('Entropy objective does not match decomposition')
    p=root/'child_preflight.json';r=json.loads(p.read_text());r['spatial_entropy_verified']=weight
    p.write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared');p.add_argument('--child');a=p.parse_args()
    if a.child:child(a.child)
    else:
        from scripts.kaggle.preflight_sequence_pilot import preflight
        preflight(a.prepared,child_script=__file__)
