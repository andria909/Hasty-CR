import concurrent.futures,json,subprocess,sys,time
from pathlib import Path
from scripts.kaggle.start_pilot_observers import start
jobs=[('tower_support',2,'tower-support-pilot'),('spell_waste',2,'spell-waste-pilot'),('native_fireball_slot',1,'native-fireball-slot-pilot')]
python=str(Path('.venv-live/bin/python').absolute())
sdk='/Users/andriashadrin/.local/share/hastycr-kaggle-cli/bin/python'
def run(job):
 stem,index,profile=job;root=Path('tmp/kaggle/'+stem+'_motionfix_20261028')
 if (root/'launch_state.json').exists():
  return {'root':str(root),'status':'existing_launch_preserved_inspect_state'}
 state={'status':'preflight_running','started_ns':time.time_ns(),'root':str(root)}
 def save(): (root/'launch_state.json').write_text(json.dumps(state,indent=2)+'\n')
 save()
 try:
  with (root/'preflight.log').open('x') as log:
   subprocess.run([python,'-m','scripts.kaggle.preflight_'+stem+'_pilot','--prepared',str(root)],stdout=log,stderr=subprocess.STDOUT,check=True)
  state['status']='submitting';save()
  with (root/'submission.log').open('x') as log:
   subprocess.run([sdk,'-m','scripts.kaggle.submit_prepared_pilot','--root',str(root),'--token-index',str(index)],stdout=log,stderr=subprocess.STDOUT,check=True)
  state['observers']=start(root,stem+'_motionfix','%s'%profile,index)
  state['status']='submitted_observers_started'
 except Exception as exc:state.update(status='failed_requires_inspection',error_type=type(exc).__name__)
 save();return state
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for row in pool.map(run,jobs):print(json.dumps(row),flush=True)
