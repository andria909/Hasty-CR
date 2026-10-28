"""Verify both shaping terms in actual frozen PPO, resume and validation."""
import argparse
import importlib.util
import json
from pathlib import Path


def child(path):
    import torch
    import sim.train_attack_ppo as trainer
    spec=json.loads(Path(path).read_text());trial=spec['trial'];evaluate=trainer.evaluate;checks=[]
    def configured(*args,**kwargs):
        kwargs['env_options']=dict(kwargs.get('env_options',{}),
            tower_support_weight=trial['tower_support_weight'],spell_waste_weight=trial['spell_waste_weight'])
        result=evaluate(*args,**kwargs)
        for row in result['records']:
            assert row['tower_support_weight']==trial['tower_support_weight']
            if trial['spell_waste_weight']is not None:
                assert row['spell_waste_weight']==trial['spell_waste_weight']
        checks.append(len(result['records']));return result
    trainer.evaluate=configured
    file=Path(__file__).with_name('preflight_global_full_pair.py')
    module_spec=importlib.util.spec_from_file_location('combined_shared_preflight',file)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module);module.child(path)
    root=Path(spec['root']);p=root/'child_preflight.json';report=json.loads(p.read_text())
    command=report['original_full_command']
    checkpoint=torch.load(root/'whole/update_000002.pt',map_location='cpu',weights_only=False)
    for key in ('tower_support_weight','spell_waste_weight'):
        flag='--'+key.replace('_','-');captured=float(command[command.index(flag)+1])if flag in command else None
        if captured!=trial[key] or checkpoint['config'].get(key)!=trial[key]:raise ValueError('Actual combined command/config differs')
        if checkpoint['environment_states'][0].get(key)!=trial[key]:raise ValueError('Actual combined environment differs')
    if checkpoint['config']['continuous_profile']['missed_fireball_penalty']!=trial['missed_fireball_penalty']:
        raise ValueError('Duplicate or missing old spell penalty')
    if checks!=[12]:raise ValueError('Combined validation coverage differs')
    report.update(tower_support_weight=trial['tower_support_weight'],spell_waste_weight=trial['spell_waste_weight'],
        actual_validation_reward_checks=checks)
    p.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared');p.add_argument('--child');a=p.parse_args()
    if a.child:child(a.child)
    else:
        from scripts.kaggle.preflight_sequence_pilot import preflight
        preflight(a.prepared,child_script=__file__)
