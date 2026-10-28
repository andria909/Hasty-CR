"""Actual frozen optimizer/resume plus mask, cooldown and reward verification."""
import argparse
import importlib.util
import json
from pathlib import Path


def child(path):
    import torch
    import sim.train_attack_ppo as trainer
    spec=json.loads(Path(path).read_text());trial=spec['trial'];evaluate=trainer.evaluate;checks=[]
    def configured(*args,**kwargs):
        profile=dict(kwargs.get('env_options',{}).get('continuous_profile')or{})
        profile.update(action_cooldown_ms=0,cannon_center_only=1,tower_damage_all=1)
        kwargs['env_options']=dict(kwargs.get('env_options',{}),continuous_profile=profile,
            tower_support_weight=trial['tower_support_weight'],spell_waste_weight=.1)
        result=evaluate(*args,**kwargs)
        for row in result['records']:
            assert row['cannon_center_only']==1 and row['cannon_outside_center']==0
            assert row['action_cooldown_ms']==0 and row['tower_damage_reward_mode']=='all_damage'
            assert row['tower_support_weight']==trial['tower_support_weight']
        checks.append(len(result['records']));return result
    trainer.evaluate=configured
    file=Path(__file__).with_name('preflight_global_full_pair.py')
    s=importlib.util.spec_from_file_location('central_shared',file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.child(path)
    root=Path(spec['root']);blob=torch.load(root/'whole/update_000002.pt',map_location='cpu',weights_only=False)
    for key,value in dict(action_cooldown_ms=0,cannon_center_only=1,tower_damage_all=1).items():
        assert blob['config']['continuous_profile'][key]==value
        assert blob['environment_states'][0]['profile'][key]==value
    assert blob['config']['tower_support_weight']==trial['tower_support_weight']and checks==[12]
    p=root/'child_preflight.json';report=json.loads(p.read_text());report.update(priority_contract_verified=True,validation_priority_checks=checks)
    p.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child');p.add_argument('--prepared');a=p.parse_args()
    if a.child:child(a.child)
    else:
        from scripts.kaggle.preflight_sequence_pilot import preflight
        preflight(a.prepared,child_script=__file__)
