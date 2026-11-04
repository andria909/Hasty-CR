"""Check actual frozen policy-choice commands, resume and optimizer counters."""
import argparse
import importlib.util
import json
from pathlib import Path


def child(path,*,full=False):
    source=Path(__file__).with_name('preflight_spell_interaction_pilot.py')
    spec=importlib.util.spec_from_file_location('choice_shared',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.child(path,full=full)
    import torch
    from scripts.attack.policy_choice_accounting import audit_update,audit_checkpoint
    from scripts.attack.boundary_state import equal_boundary
    spec=json.loads(Path(path).read_text());root=Path(spec['root']);trial=spec['trial']
    whole=torch.load(root/'whole/update_000002.pt',map_location='cpu',weights_only=False)
    resumed=torch.load(root/'resumed/update_000002.pt',map_location='cpu',weights_only=False)
    config=whole['config'];audit_checkpoint(config,whole,trial)
    for key in ('state_dict','optimiser','rng','sampler','environment_states','observations','infos',
                'episode_returns','episode_counters','step','update'):
        if not equal_boundary(whole[key],resumed[key]):raise ValueError('Choice-aware resume differs: '+key)
    rows=[json.loads(line)for line in (root/'whole/metrics.jsonl').read_text().splitlines()]
    for row in rows:audit_update(row,config,trial)
    p=root/'child_preflight.json';report=json.loads(p.read_text())
    args=report['original_full_command'if full else'original_pilot_command']
    for key in ('policy_reduction','advantage_normalization'):
        if args[args.index('--'+key.replace('_','-'))+1]!=trial[key]:
            raise ValueError('Frozen choice-aware command differs: '+key)
        report['actual_'+key]=trial[key]
    report.update(policy_choice_verified=True,full_environment_resume_equal=True,
                  policy_decisions=[row['policy_decisions']for row in rows])
    if full:report['long_economy_full_verified']=True
    p.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared');p.add_argument('--child');a=p.parse_args()
    if a.child:child(a.child)
    else:
        from scripts.kaggle.preflight_sequence_pilot import preflight
        preflight(a.prepared,child_script=__file__)
