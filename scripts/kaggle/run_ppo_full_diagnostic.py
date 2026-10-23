"""Same-source diagnostic continuation from a saved full-PPO update boundary."""
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import sys
import tarfile

from scripts.kaggle.run_five_round_robin import prepare_source,sha
from scripts.kaggle.run_ppo_selected_full import full_command
from scripts.kaggle.run_ppo_coefficient_study import training_command

LAUNCH_CONFIG=None

TRACE_WRAPPER='''import json,runpy,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from sim.defense_env import DefenseEnv
from sim import arena
original=DefenseEnv._process_enemy
def observed(self):
 result=original(self)
 if self.terminal_reason=='invalid_script_play_timeout':
  event=self.scenario.plays[self._script_index];point=self._target(event);player=self.match.players[-1];spell=self.match.spells.get(event.card)
  row=dict(event='SCRIPT_PLAY_DIAGNOSTIC',scenario=self.scenario.scenario_id,seed=self.seed,elapsed_ms=self.match.elapsed_ms,
           script_index=self._script_index,card=event.card,enemy_hand=list(player.hand),enemy_elixir=player.elixir,
           cost=self.match.cards[event.card].cost,can_play=self.match.can_play(-1,event.card),target=[point.x,point.y],
           in_arena=arena.in_arena(point),spell_available=spell is not None,
           spell_can_place_on_water=getattr(spell,'can_place_on_water',None),
           accepted_enemy_plays=[r for r in self.match.play_log if r['side']==-1],
           defenders=[dict(uid=u.uid,name=u.name,pos=[u.pos.x,u.pos.y],in_arena=arena.in_arena(u.pos),state=str(u.state))
                      for u in self.match.battle.entities.values()if u.alive and u.side==1 and not u.is_tower])
  print(json.dumps(row),flush=True)
 return result
DefenseEnv._process_enemy=observed
runpy.run_module('sim.train_defense_ppo',run_name='__main__')
'''


def execute(config):
    def mount(reference):
        owner,slug=reference.split('/');paths=[p for p in (Path('/kaggle/input')/slug,Path('/kaggle/input/datasets')/owner/slug)if p.is_dir()]
        if len(paths)!=1:raise ValueError('Expected exact diagnostic input mount')
        return paths[0]
    original=config['original'];inputs=mount(original['dataset']);boundary=mount(config['boundary_dataset'])
    for name,digest in original['pins'].items():
        if sha(inputs/name)!=digest:raise ValueError('Original input pin differs')
    for name,digest in config['boundary_pins'].items():
        if sha(boundary/name)!=digest:raise ValueError('Diagnostic boundary pin differs')
    output=Path('/kaggle/working/selected_full');output.mkdir(exist_ok=False)
    project=output/'project';manifest=prepare_source(inputs,project,original['pins']['manifest.json'])
    sys.path[:0]=[str(project),str(project/'scripts')]
    import torch
    if not torch.cuda.is_available()or torch.cuda.device_count()!=2:raise RuntimeError('Same two CUDA devices required')
    blob=torch.load(boundary/'resume.pt',map_location='cpu',weights_only=False)
    if blob['step']!=716800 or blob['update']!=700 or not blob.get('exact_resume_available'):raise ValueError('Expected complete update700 boundary')
    if blob['config']['reward_profile_sha256']!=config['reward_profile_sha256']:raise ValueError('Diagnostic reward profile differs')
    if len(blob['rng']['cuda'])!=2:raise ValueError('Original CUDA RNG device count differs')
    wrapper=output/'trace_run.py';wrapper.write_text(TRACE_WRAPPER)
    command=full_command(project,inputs,output/'train','defense','economy_soft','cuda:0',42,resume=boundary/'resume.pt',completed_updates=700)
    command[command.index('--updates')+1]='100';command[command.index('--checkpoint-every')+1]='1';command[command.index('--wallclock-seconds')+1]='1800'
    command=[sys.executable,str(wrapper)]+command[3:]
    report=dict(schema='selected-ppo-full-diagnostic-v1',status='running',mode='defense',profile='economy_soft',
                target_environment_steps=2048000,diagnostic_updates=100,resume_update=700,resume_step=716800,
                source_sha256=manifest['archive_sha256'],source_unchanged=True,observation_only_wrapper=True,
                actual_cuda=torch.cuda.get_device_name(0),quality_accepted=False,test_used_for_selection=False)
    def save():(output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    save();print(json.dumps(report),flush=True)
    environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    with (output/'train.log').open('x')as log:
        process=subprocess.Popen(command,cwd=project,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            log.write(line);log.flush()
            try:row=json.loads(line)
            except ValueError:continue
            if not isinstance(row,dict):continue
            step=row.get('step',row.get('steps'))
            if isinstance(step,int)and'update'in row:
                print(json.dumps(dict(event='FULL_TRAINING_PROGRESS',mode='defense',profile='economy_soft',step=step,update=row['update'],percent=100*step/2048000)),flush=True)
            if row.get('event')=='SCRIPT_PLAY_DIAGNOSTIC':report['failure_telemetry']=row;save();print(json.dumps(row),flush=True)
        code=process.wait()
    checkpoints=sorted((output/'train').glob('update_*.pt'))
    if checkpoints:
        latest=checkpoints[-1];state=torch.load(latest,map_location='cpu',weights_only=False)
        report.update(checkpoint=str(latest.relative_to(output)),checkpoint_sha256=sha(latest),step=state['step'],update=state['update'])
    expected={r['update']:r for r in json.loads((boundary/'reference_metrics.json').read_text())}
    actual=[json.loads(line)for line in (output/'train/metrics.jsonl').read_text().splitlines()]
    def comparable(row):return {k:v for k,v in row.items()if k not in ('elapsed_seconds','actual_cuda')}
    comparisons={str(r['update']):comparable(r)==comparable(expected[r['update']])for r in actual if r['update']in expected}
    report.update(status='failed'if code else'incomplete_budget',subprocess_exit_code=code,
                  original_prefix_comparisons=comparisons,all_compared_updates_identical=bool(comparisons)and all(comparisons.values()),diagnostic_only=True)
    save();print(json.dumps(report),flush=True)
    if code:raise RuntimeError('Diagnostic continuation reproduced a training error; artifacts preserved')
    return report


if __name__=='__main__':
    if LAUNCH_CONFIG is None:raise RuntimeError('Pinned saved-boundary diagnostic config required')
    execute(LAUNCH_CONFIG)
