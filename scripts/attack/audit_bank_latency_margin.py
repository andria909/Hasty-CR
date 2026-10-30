"""Audit nominal frontier plans and an earlier request at two delivery delays."""
import argparse
import json
from pathlib import Path
from scripts.attack.build_bank_counterfactuals import opening
from scripts.attack.build_safe_bank_sequences import verify_search, choose
from scripts.attack.search_late_air_defense import rollout
from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha


def audit(root,output):
    contexts,_=verify_search(root);rows=[]
    for record in contexts:
        plan=choose(record);env=opening(record['seed'],record['lane'],record['bank'],record['enemy_y'])
        hp={k:t.hitpoints for k,t in env.match.towers[1].items()};results={}
        for name,offset,delay in [('late_slow',plan['delay_ms'],2025),
                                  ('margin_nominal',plan['delay_ms']-750,1350),
                                  ('margin_slow',plan['delay_ms']-750,2025)]:
            if offset<0:raise ValueError('Negative safety margin request')
            p=rollout(env.match,hp,env.cards,env.spells,record['lane'],
                      second=dict(plan['candidate'],offset_ms=offset),delay_ms=delay)
            results[name]=dict(p,delay_ms=offset,candidate=plan['candidate'],command_delay_ms=delay)
        rows.append(dict(seed=record['seed'],lane=record['lane'],bank=record['bank'],enemy_y=record['enemy_y'],
                         original_plan=plan,results=results))
        if len(rows)%6==0:print(json.dumps(dict(contexts=len(rows),total=len(contexts))),flush=True)
    result=dict(schema='bank-latency-margin-audit-v1',contexts=len(rows),rows=rows,margin_ms=750,
        source_contexts_sha256=file_sha(Path(root)/'collected/contexts.jsonl'),physics_pins=physics_pins(),
        source_sha256=file_sha(__file__),
        safe_counts={name:sum(r['results'][name]['valid']and r['results'][name]['clear_without_damage']for r in rows)
                     for name in ('late_slow','margin_nominal','margin_slow')},
        delays_tested_ms=[1350,2025],quality_accepted=False,
        limitations=['Only two delivery delays; not a proof for every intermediate delay or arbitrary tails',
                     '1350/2025ms requests are delivered on50ms physics ticks;2025rounds up to2050',
                     'Single paid Balloon; not full-game performance'])
    with Path(output).open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=audit(a.root,a.output);print(json.dumps(r['safe_counts']))
