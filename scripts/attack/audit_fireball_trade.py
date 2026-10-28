"""Read-only paid-hit diagnostic in the exact archived simulation package."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def child(spec_path):
    import torch
    from sim.continuous_push import ContinuousPushEnv,CONTRACT
    from sim.train_attack_ppo import initialize_attack_policy,evaluate
    torch.set_num_threads(1);spec=json.loads(Path(spec_path).read_text())
    original=ContinuousPushEnv._metrics
    def metrics(self):
        result=original(self);casts=[]
        for e in self.match.battle.spell_cast_log:
            if e.get('spell')!='fireball' or e.get('caster_side')!=1 or e.get('side')!=1 or not e.get('resolved'):continue
            hits=[h for h in e.get('hits',[]) if h['target_side']==-1 and h['damage']>0]
            units=[h for h in hits if not h['is_tower']]
            removed=sum(max(0,h['elixir_milli'])/1000*min(1.,h['damage']/max(1,h['max_hp']))for h in units)
            casts.append(dict(unit_hits=len(units),tower_hits=sum(h['is_tower']for h in hits),
                removed_unit_elixir=removed,unit_positive_half_cost=bool(units)and removed>2,
                unit_positive_full_cost=bool(units)and removed>4))
        result['fireball_trade_diagnostic']=casts;return result
    ContinuousPushEnv._metrics=metrics
    output=[]
    for model in spec['models']:
        blob=torch.load(model['weights'],map_location='cpu',weights_only=False)
        net,_,_=initialize_attack_policy(blob,num_scalars=309,observation_contract=CONTRACT)
        config=blob['config']
        evaluation=evaluate(net,'cpu',episodes=6,seed=800100,task_mode='joint',env_options=dict(
            curriculum='continuous_v1',validation_selection='worst_family',continuous_profile=config['continuous_profile'],
            geometry=config['geometry'],coordinate_mode=config['coordinate_mode'],action_delay_ms=1350,max_ms=310000))
        rows=evaluation['records'];casts=[c for r in rows for c in r['fireball_trade_diagnostic']]
        output.append(dict(id=model['id'],weights_sha256=model['sha256'],games=len(rows),wins=sum(r['outcome']=='win'for r in rows),
            resolved_casts=len(casts),unit_hit_casts=sum(c['unit_hits']>0 for c in casts),
            positive_half_cost=sum(c['unit_positive_half_cost']for c in casts),positive_full_cost=sum(c['unit_positive_full_cost']for c in casts),
            removed_unit_elixir=sum(c['removed_unit_elixir']for c in casts),
            casts=casts,records=rows))
    Path(spec['result']).write_text(json.dumps(dict(models=output,scope='Paired12development-context diagnostic; not full validation, no training or promotion',quality_accepted=False),indent=2)+'\n')


def run(prepared,output):
    from scripts.kaggle.preflight_global_full_pair import unpack
    from scripts.kaggle.local_artifact_transfer import file_sha
    prepared,root=Path(prepared).absolute(),Path(output).absolute();root.mkdir(parents=True,exist_ok=False)
    prep=json.loads((prepared/'preparation.json').read_text());project=root/'project';unpack(prepared/'dataset',prep['config'],project)
    models=[]
    for name in ('control','slot800'):
        selection=json.loads((prepared/'collected'/f'pilot_{name}_s222_summary.json').read_text())
        weights=prepared/'collected'/selection['checkpoint']
        if file_sha(weights)!=selection['checkpoint_sha256']:raise ValueError('Selected weights changed')
        models.append(dict(id=name,weights=str(weights),sha256=file_sha(weights)))
    spec=root/'spec.json';spec.write_text(json.dumps(dict(models=models,result=str(root/'result.json'))))
    with (root/'run.log').open('x')as log:
        subprocess.run([sys.executable,str(Path(__file__).absolute()),'--child',str(spec)],cwd=project,
            env=dict(os.environ,PYTHONPATH=str(project)),stdout=log,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child');p.add_argument('--prepared');p.add_argument('--output');a=p.parse_args()
    if a.child:child(a.child)
    else:run(a.prepared,a.output)
