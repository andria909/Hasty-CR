"""Measure previously unlogged control Zap waste on the original frozen validation.

No optimization, new test selection, reward changes, or live games.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def child(spec_path):
    import torch
    from sim.continuous_push import ContinuousPushEnv,CONTRACT
    from sim.spell_waste import SpellWasteTracker
    from sim.train_attack_ppo import initialize_attack_policy,evaluate
    spec=json.loads(Path(spec_path).read_text());torch.set_num_threads(1)
    reset,metrics=ContinuousPushEnv.reset,ContinuousPushEnv._metrics
    def instrumented_reset(self,*args,**kwargs):
        self._diagnostic_tracker=None
        observation,info=reset(self,*args,**kwargs)
        self._diagnostic_tracker=SpellWasteTracker(self.match).attach()
        return observation,info
    def instrumented_metrics(self):
        result=metrics(self)
        tracker=getattr(self,'_diagnostic_tracker',None)
        if tracker is not None:result.update(tracker.metrics())
        return result
    ContinuousPushEnv.reset=instrumented_reset;ContinuousPushEnv._metrics=instrumented_metrics
    reports=[]
    for name in ('pilot_control_s242','pilot_control_s243'):
        root=Path(spec['collected']);selected=json.loads((root/(name+'_summary.json')).read_text())
        path=root/selected['checkpoint']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=selected['checkpoint_sha256']:raise ValueError('Selected control changed')
        blob=torch.load(path,map_location='cpu',weights_only=False);config=blob['config']
        if config.get('spell_waste_weight')is not None:raise ValueError('Old uninstrumented control required')
        options={k:config[k]for k in ('action_delay_ms','max_ms','deadline_scale','visible_fraction','initial_hand_mode','geometry','coordinate_mode','curriculum','continuous_profile','validation_selection','continuous_focus','history_frames','pending_inputs')}
        options['action_delay_ms']=config['validation_action_delay_ms'] or config['action_delay_ms']
        net,_,_=initialize_attack_policy(blob,device='cpu',num_scalars=309,observation_contract=CONTRACT)
        result=evaluate(net,'cpu',episodes=36,seed=900100,env_options=options,task_mode='joint')
        originals=selected['validation']['records'];records=result['records']
        if len(records)!=len(originals):raise ValueError('Validation context count differs')
        differences=[]
        for old,new in zip(originals,records,strict=True):
            identity=('scenario_id','episode_seed','match_seed')
            if any(old[k]!=new[k]for k in identity):raise ValueError('Validation contexts changed')
            diff=[k for k,v in old.items()if new.get(k)!=v]
            if diff:differences.append(dict(scenario_id=old['scenario_id'],episode_seed=old['episode_seed'],fields=diff))
        report=dict(id=name,checkpoint_sha256=selected['checkpoint_sha256'],old_metrics_reproduced=not differences,
                    differences=differences,validation=result)
        reports.append(report)
        Path(spec['output']).write_text(json.dumps(dict(complete=False,reports=reports),indent=2)+'\n')
        print(json.dumps(dict(model=name,games=len(records),old_metrics_reproduced=not differences)),flush=True)
    rows=[r for model in reports for r in model['validation']['records']]
    minutes=sum(r['elapsed_ms']for r in rows)/60000
    summary=dict(games=len(rows),wins=sum(r['outcome']=='win'for r in rows),
        empty_fireball_per_match=sum(r['empty_spell_counts'].get('fireball',0)for r in rows)/len(rows),
        empty_zap_per_match=sum(r['empty_spell_counts'].get('zap',0)for r in rows)/len(rows),
        empty_elixir_per_match=sum(r['empty_spell_elixir']for r in rows)/len(rows),
        empty_elixir_per_minute=sum(r['empty_spell_elixir']for r in rows)/minutes)
    Path(spec['output']).write_text(json.dumps(dict(complete=True,reports=reports,summary=summary,
        old_metrics_reproduced=all(r['old_metrics_reproduced']for r in reports),
        scope='Same selected control checkpoints and72 validation contexts per seed; observer only, old rewards and physics preserved'),indent=2)+'\n')


def launch(prepared,output):
    from scripts.kaggle.preflight_global_full_pair import unpack
    prepared=Path(prepared).resolve();output=Path(output).resolve();output.mkdir(exist_ok=False)
    prep=json.loads((prepared/'preparation.json').read_text())
    if prep.get('study_variant')!='spell_waste_reward_v1':raise ValueError('Spell-waste study required')
    receipt=json.loads((prepared/'collected/collection_receipt.json').read_text())
    if not receipt.get('complete')or receipt['remote_status']!='COMPLETE':raise ValueError('Completed collection required')
    project=output/'project';unpack(prepared/'dataset',prep['config'],project)
    spec=output/'spec.json';spec.write_text(json.dumps(dict(collected=str(prepared/'collected'),output=str(output/'result.json')),indent=2)+'\n')
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child',str(spec)],cwd=project,
                   env=dict(os.environ,PYTHONPATH=str(project)),check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared');p.add_argument('--output');p.add_argument('--child')
    a=p.parse_args()
    if a.child:child(a.child)
    else:launch(a.prepared,a.output)
