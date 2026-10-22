"""Bounded deterministic diagnostics, not a runtime change."""
import json
from pathlib import Path
from sim.defense_env import DefenseEnv
from sim.defense_audit_policy import defender_action
from sim.defense_scenarios import build_catalog
CATALOG={s.scenario_id:s for s in build_catalog()[0]}

OUT=Path(__file__).resolve().parent
ids=[]
for line in Path('reports/defense_plan/catalog_cannon_audit_v11.jsonl').open():
    row=json.loads(line)
    if row['metrics'].get('terminal_reason')=='timeout':ids.append(row['scenario_id'])
    if len(ids)==3:break

def entities(env):
    result=[]
    for e in env.match.battle.entities.values():
        if not e.alive:continue
        r={k:getattr(e,k,None)for k in ('uid','name','side','hitpoints','state','target_uid','flying','is_tower','is_building','speed_mt_per_sec','damage','range_mt','attacks_air','attacks_ground','target_only_buildings','walk_target_uid')}
        r['pos']=[e.pos.x,e.pos.y]
        target=env.match.battle.entities.get(e.target_uid)
        r['target_alive']=getattr(target,'alive',None)
        r['target_name']=getattr(target,'name',None)
        result.append(r)
    return result
def compact(value):
    if hasattr(value,'uid'):return {'entity_uid':value.uid,'name':value.name}
    if isinstance(value,(list,tuple)):return [compact(v)for v in value]
    if isinstance(value,dict):return {str(k):compact(v)for k,v in value.items()}
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return str(value)
results=[]
for scenario_id in ids:
    scenario=CATALOG[scenario_id]
    env=DefenseEnv(split=scenario.split,scenario=scenario,max_ms=240000)
    _,info=env.reset(seed=42)
    history=[]
    for step in range(1000):
        _,_,terminated,truncated,info=env.step(defender_action(env,info,'cannon_cycle'))
        if step%20==19 or terminated or truncated:
            history.append(dict(at_ms=env.match.elapsed_ms,entities=entities(env)))
        if terminated or truncated:break
    battle=env.match.battle
    pending={k:compact(v)for k,v in vars(battle).items()if any(t in k for t in ('projectile','area','pending','effect','flight','summon','impacts','drop_events','barrage','paratrooper','forced_moves','active_pulls'))}
    result=dict(scenario_id=scenario_id,seed=42,step_count=step+1,metrics=info['metrics'],phase=info['phase'],
        script_index=env._script_index,script_length=len(env.scenario.plays),
        pending_action=env._pending,battle_pending=pending,history=history,
        entities=entities(env),recent_damage=env.match.battle.damage_log[-30:],
        defender_actions=env._actions,enemy_plays=[e for e in env.match.play_log if e['side']==-1])
    results.append(result)
(OUT/'diagnostics.json').write_text(json.dumps(results,indent=2,default=str)+'\n')
