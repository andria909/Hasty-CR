"""Actual frozen pilot PPO/resume plus paid dormant-king penalty accounting."""
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
            crown_only_weight=trial['crown_only_weight'],dormant_king_weight=trial['dormant_king_weight'])
        result=evaluate(*args,**kwargs)
        for row in result['records']:
            for name in ('crown_only_weight','dormant_king_weight','spell_waste_weight'):
                if row[name]!=trial[name]:raise ValueError('Actual validation reward differs')
            king=row['dormant_at_request_king_only_elixir'];crown=row['nonlethal_crown_only_elixir']
            if not 0<=king<=crown:raise ValueError('Invalid paid spell bins')
            if abs(row['spell_economy_penalty']-(crown-king)*trial['crown_only_weight']-king*trial['dormant_king_weight'])>1e-7:
                raise ValueError('Actual crown penalty differs from paid bins')
        checks.append(len(result['records']));return result
    trainer.evaluate=configured
    source=Path(__file__).with_name('preflight_air_exposure_pilot.py')
    module_spec=importlib.util.spec_from_file_location('spell_interaction_shared',source)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module);module.child(path)
    root=Path(spec['root']);blob=torch.load(root/'whole/update_000002.pt',map_location='cpu',weights_only=False)
    for key in ('crown_only_weight','dormant_king_weight'):
        if blob['config'][key]!=trial[key]or blob['environment_states'][0][key]!=trial[key]:
            raise ValueError('Actual optimizer/environment coefficient differs')
    if checks!=[12]:raise ValueError('Missing twelve validation games')
    p=root/'child_preflight.json';r=json.loads(p.read_text());args=r['original_pilot_command']
    if float(args[args.index('--dormant-king-weight')+1])!=trial['dormant_king_weight']:
        raise ValueError('Frozen command king coefficient differs')
    r.update(spell_interaction_verified=True,actual_dormant_king_weight=trial['dormant_king_weight'],crown_validation_games=12)
    p.write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared');p.add_argument('--child');a=p.parse_args()
    if a.child:child(a.child)
    else:
        from scripts.kaggle.preflight_sequence_pilot import preflight
        preflight(a.prepared,child_script=__file__)
