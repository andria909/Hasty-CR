"""Exercise exact frozen start-fraction commands before any GPU submission."""
import argparse
import importlib.util
import json
from pathlib import Path


def child(path):
    import torch
    import sim.train_attack_ppo as trainer
    spec=json.loads(Path(path).read_text());trial=spec['trial'];evaluate=trainer.evaluate;checks=[]
    fraction=trial['tactical_start_fraction']
    def configured(*args,**kwargs):
        kwargs['env_options']=dict(kwargs.get('env_options',{}),
            tower_support_weight=trial['tower_support_weight'],spell_waste_weight=trial['spell_waste_weight'],
            tactical_start_fraction=fraction)
        result=evaluate(*args,**kwargs)
        for row in result['records']:
            assert row['tactical_start'] is None and row['tactical_start_fraction']==fraction
            assert row['cannon_center_only']==1 and row['cannon_outside_center']==0
            assert row['action_cooldown_ms']==0 and row['tower_damage_reward_mode']=='all_damage'
        checks.append(len(result['records']));return result
    trainer.evaluate=configured
    file=Path(__file__).with_name('preflight_global_full_pair.py')
    s=importlib.util.spec_from_file_location('tactical_shared',file)
    module=importlib.util.module_from_spec(s);s.loader.exec_module(module);module.child(path)
    root=Path(spec['root'])
    blob=torch.load(root/'whole/update_000002.pt',map_location='cpu',weights_only=False)
    assert blob['config']['tactical_start_fraction']==fraction and checks==[12]
    assert blob['environment_states'][0]['tactical_start_fraction']==fraction
    report_path=root/'child_preflight.json';report=json.loads(report_path.read_text())
    command=report['original_full_command']
    assert float(command[command.index('--tactical-start-fraction')+1])==fraction
    env=trainer.make_training_env(curriculum='continuous_v1',action_delay_ms=1350,
        continuous_profile=blob['config']['continuous_profile'],tactical_start_fraction=fraction,
        tower_support_weight=trial['tower_support_weight'],spell_waste_weight=trial['spell_waste_weight'])
    seen=None
    for seed in range(322000,322100):
        env.reset(seed=seed)
        if env.tactical_start:
            seen=env.tactical_start
            player=env.match.players[-1]
            assert sorted(player.hand+player.queue)==sorted(player.deck)
            assert sum(r['elixir_before']-r['elixir_after'] for r in seen['enemy_setup'])<=10000
            break
    assert bool(seen)==bool(fraction)
    report.update(tactical_fraction_verified=fraction,validation_unchanged_checks=checks,
                  actual_tactical_start_observed=bool(seen),example_start=seen)
    report_path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child');p.add_argument('--prepared');a=p.parse_args()
    if a.child:child(a.child)
    else:
        from scripts.kaggle.preflight_sequence_pilot import preflight
        preflight(a.prepared,child_script=__file__)
