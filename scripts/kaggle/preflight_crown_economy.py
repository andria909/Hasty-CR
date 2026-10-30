"""Exercise exact frozen commands, resumed optimizer and actual new rewards."""
import argparse
import importlib.util
import json
from pathlib import Path


def child(path):
    import torch
    import sim.train_attack_ppo as trainer
    spec=json.loads(Path(path).read_text());trial=spec['trial'];evaluate=trainer.evaluate
    checks=[]
    def configured(*args,**kwargs):
        kwargs['env_options']=dict(kwargs.get('env_options',{}),
            tower_support_weight=.15,spell_waste_weight=.1,
            crown_only_weight=trial['crown_only_weight'],dormant_king_weight=trial['dormant_king_weight'])
        result=evaluate(*args,**kwargs)
        for row in result['records']:
            for name in ('crown_only_weight','dormant_king_weight'):
                if row[name]!=trial[name]:raise ValueError('Validation economy coefficient differs')
            if row['normal_range_mode']!='edge_to_edge_v1' or row['hog_sight_mode']!='side_clip_v1':
                raise ValueError('Validation combat mode differs')
            if row['cannon_outside_center'] or row['tower_damage_reward_mode']!='all_damage':
                raise ValueError('Cannon/tower contract differs')
            if row['dormant_at_request_king_only_elixir']>row['nonlethal_crown_only_elixir']:
                raise ValueError('Invalid disjoint spell accounting')
        checks.append(len(result['records']));return result
    trainer.evaluate=configured
    source=Path(__file__).with_name('preflight_global_full_pair.py')
    definition=importlib.util.spec_from_file_location('crown_preflight_shared',source)
    module=importlib.util.module_from_spec(definition);definition.loader.exec_module(module);module.child(path)
    root=Path(spec['root']);blob=torch.load(root/'whole/update_000002.pt',map_location='cpu',weights_only=False)
    for name in ('crown_only_weight','dormant_king_weight'):
        if blob['config'][name]!=trial[name] or blob['environment_states'][0][name]!=trial[name]:
            raise ValueError('Actual optimizer/environment reward differs')
    if checks!=[12]:raise ValueError('Missing validation coverage')
    p=root/'child_preflight.json';r=json.loads(p.read_text());r.update(crown_economy_verified=True,actual_validation_checks=checks)
    p.write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child');p.add_argument('--prepared');a=p.parse_args()
    if a.child:child(a.child)
    else:
        from scripts.kaggle.preflight_sequence_pilot import preflight
        preflight(a.prepared,child_script=__file__)
