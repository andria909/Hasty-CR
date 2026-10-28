import json,hashlib
from pathlib import Path
from collections import Counter
import torch
from sim.continuous_push import ContinuousPushEnv,CONTRACT
from sim.train_attack_ppo import initialize_attack_policy
from scripts.attack.summarize_balanced_live_usage import summarize
ROOT=Path('tmp/plan2_research_20261028')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
bal=summarize('tmp/live/balanced_20261027_live10')
assert bal==json.loads(Path('reports/defense_plan/balanced_live_card_usage_20261027.json').read_text())
meta=json.loads(Path('reports/defense_plan/live_balanced_20261027.json').read_text())
tower=json.loads(Path('reports/defense_plan/tower_support_live5_results_20261028.json').read_text())
for p,h in tower['event_sha256'].items():assert sha(p)==h

def stats(actions,games):
 out={}
 for card in sorted({r['card'] for r in actions}):
  a=[r for r in actions if r['card']==card];cells=Counter((r['x'],r['y'])for r in a)
  out[card]=dict(n=len(a),per_game=len(a)/games,left=sum(r['x']<9 for r in a),
      own_river_band_y16_18=sum(16<=r['y']<=18 for r in a),
      central_cannon_region=sum(6<=r['x']<=11 and 18<=r['y']<=23 for r in a),
      top_cells=[dict(x=x,y=y,n=n)for (x,y),n in cells.most_common(5)])
 return out

def live(sources):
 actions=[]
 for path,allowed in sources:
  plays={};seen=set()
  for line in Path(path).read_text().splitlines():
   e=json.loads(line)
   if e.get('match')not in allowed:continue
   if e['event']=='PLAY':plays[e['match'],e['card'],e['tick']]=e
   if e['event']=='PLAY_ACK':
    key=e['match'],e['card'],e['source_tick'];assert key in plays and key not in seen;seen.add(key);p=plays[key]
    actions.append(dict(card=p['card'],x=p['grid'][0],y=p['grid'][1],match=p['match'],run=path,tick=p['tick']))
 return actions
sources={'balanced':[(f'tmp/live/balanced_20261027_live10/{name}/events.jsonl',matches) for name,matches in [('run',[1,2,3]),('recovery_run',[1,2,3,4,5,6])]],
 'tower_large':[(f'tmp/live/tower_support_20261028_live5/{name}/events.jsonl',matches)for name,matches in [('run',[1]),('recovery_run',[1,2,3,4])]]}
models={'balanced':meta['checkpoint'],'tower_large':'tmp/kaggle/tower_support_motionfix_20261028/collected/pilot_large_s232/update_000100.pt'}
result=dict(schema='plan2-research-evidence-v1',live={},sim={},source_hashes={},training_inventory=[],limitations=['Live ACK confirms hand rotation, not spawn or hit. Grid cells are command targets.', 'Simulation scenarios differ from live opponents; this is not paired causal evaluation or a new test benchmark.', 'River band is geometric, not a tactical quality label. All simulations are current corrected-physics TRAIN diagnostics.'])
for name,entries in sources.items():
 actions=live(entries);n=9 if name=='balanced' else 5
 result['live'][name]=dict(games=n,cards=stats(actions,n),actions=actions)
 for p,_ in entries:result['source_hashes'][p]=sha(p)
 if name=='balanced':assert {k:v['n']for k,v in result['live'][name]['cards'].items()}=={k:v['confirmed']for k,v in bal['cards'].items()}
for p in sorted(Path('reports/defense_plan').glob('*summary*.json')):
 try:r=json.loads(p.read_text())
 except ValueError:continue
 rows=r.get('rows')
 if isinstance(rows,list) and rows and isinstance(rows[0],dict)and 'games'in rows[0]and 'wins'in rows[0]:
  result['training_inventory'].append(dict(path=str(p),sha256=sha(p),kernel=r.get('kernel'),eligible_for_full=r.get('eligible_for_full'),rows=[dict(id=x.get('id'),wins=x['wins'],games=x['games'])for x in rows]))
torch.set_num_threads(1)
for name,path in models.items():
 blob=torch.load(path,map_location='cpu',weights_only=False);net,_,_=initialize_attack_policy(blob,device='cpu',num_scalars=309,observation_contract=CONTRACT);net.eval();games=[];actions=[]
 with torch.inference_mode():
  for index,scenario in enumerate([f'{task}::{family}/push'for task in ('attack','defense')for family in ('air','hog','swarm')]):
   env=ContinuousPushEnv(split='train',action_delay_ms=1350,continuous_profile=dict(source_radius_reach=1,target_lock_in_range=0,cannon_center_only=1,action_cooldown_ms=0))
   obs,info=env.reset(seed=8102800+index,scenario_id=scenario);available=below=wait=0
   for step in range(1242):
    mask=info['action_mask'];available+=bool(mask[1:].any());below+=float(obs['scalars'][0])*10<4
    logits,_=net(torch.from_numpy(obs['planes'])[None],torch.from_numpy(obs['scalars'])[None]);a=int(logits.masked_fill(~torch.from_numpy(mask)[None],-torch.inf).argmax(-1));wait+=bool(mask[1:].any())and a==0
    obs,_,term,trunc,info=env.step(a)
    if term or trunc:break
   else:raise ValueError('Nonterminal diagnostic')
   games.append(dict(scenario=scenario,seed=8102800+index,metrics=info['metrics'],decisions=step+1,below4=below,can_play=available,voluntary_wait=wait,actions=env.actions));actions.extend(env.actions)
   print(name,scenario,'done',flush=True)
 result['sim'][name]=dict(checkpoint=path,checkpoint_sha256=sha(path),cards=stats(actions,len(games)),games=games)
result['script_sha256']=sha(__file__)
(ROOT/'evidence.json').write_text(json.dumps(result,indent=2)+'\n')
print('complete',len(result['training_inventory']),flush=True)
