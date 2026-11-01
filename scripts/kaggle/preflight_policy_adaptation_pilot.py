"""Check every frozen initializer command with real CPU PPO and exact resume."""
import argparse
import importlib.util
import json
from pathlib import Path


def child(path):
    source=Path(__file__).with_name('preflight_mana_campaign.py')
    spec=importlib.util.spec_from_file_location('adaptation_physics',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.child(path)
    import torch
    spec=json.loads(Path(path).read_text());root=Path(spec['root'])
    expected=next(m for m in spec['config']['models']if m['id']==spec['trial']['initial_id'])
    blob=torch.load(root/'whole/update_000002.pt',map_location='cpu',weights_only=False)
    if blob['config']['init_sha256']!=expected['sha256']:raise ValueError('Actual PPO did not use its assigned policy')
    p=root/'child_preflight.json';r=json.loads(p.read_text());r['explicit_initial_model']=expected
    p.write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared');p.add_argument('--child');a=p.parse_args()
    if a.child:child(a.child)
    else:
        from scripts.kaggle.preflight_sequence_pilot import preflight
        preflight(a.prepared,child_script=__file__)
