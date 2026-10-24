"""Train-only paired control: finishing opportunity versus an empty full-HP tower.

This is a waste/retention diagnostic, not a new held-out selection metric.
Full-HP chip can be strategically intentional, so report casts rather than
automatically labelling every such action an error.
"""
import argparse
import json
from pathlib import Path

import torch

from scripts.attack.evaluate_fireball_refinement import load_policy
from scripts.attack.train_fireball_refinement import PARENT_SHA
from sim.fireball_curriculum import FireballEnv


def probe(parent,candidate,candidate_sha,count=48):
    torch.set_num_threads(2)
    nets={'parent':load_policy(parent,'cpu',PARENT_SHA),
          'candidate':load_policy(candidate,'cpu',candidate_sha,True)}
    rows=[]
    for index in range(count):
        lane=('left','right')[index%2];seed=850000+index
        for control in ('low_hp_target','full_hp_no_units'):
            for name,net in nets.items():
                env=FireballEnv(split='train',kind='tower_finish',lane=lane)
                env.reset(seed=seed)
                if control=='full_hp_no_units':
                    tower=env.match.towers[-1][lane];tower.hitpoints=tower.max_hitpoints
                obs=env._observe();info=env._info();first_card=None
                with torch.no_grad():
                    for step in range(8):
                        logits,_=net(torch.as_tensor(obs['planes'])[None],torch.as_tensor(obs['scalars'])[None])
                        action=int(logits.masked_fill(~torch.as_tensor(info['action_mask'])[None],-1e9).argmax(-1))
                        if step==0:first_card=env.match.players[1].hand[(action-1)//576]if action else 'wait'
                        obs,_,done,truncated,info=env.step(action)
                        if done or truncated:break
                casts=sum(e.get('caster_side',e.get('side'))==1 and e.get('spell')=='fireball'
                          for e in env.match.battle.spell_cast_log)
                rows.append(dict(policy=name,control=control,seed=seed,lane=lane,first_card=first_card,
                    fireball_casts_first_4s=casts,task_success=bool(info['metrics']['fireball_success'])))
    summary={}
    for name in nets:
        summary[name]={}
        for control in ('low_hp_target','full_hp_no_units'):
            subset=[r for r in rows if r['policy']==name and r['control']==control]
            summary[name][control]=dict(episodes=len(subset),first_action_fireball=sum(r['first_card']=='fireball'for r in subset),
                episodes_with_fireball_first_4s=sum(r['fireball_casts_first_4s']>0 for r in subset),
                task_success_first_4s=sum(r['task_success']for r in subset))
    return dict(schema='fireball-train-control-probe-v1',parent_sha256=PARENT_SHA,candidate_sha256=candidate_sha,
                split='train',training_performed=False,used_for_checkpoint_selection=False,
                interpretation='Counts measure cast preference, not proof that all full-HP tower chip is a mistake.',summary=summary,rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','candidate','candidate-sha','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();r=probe(a.parent,a.candidate,a.candidate_sha)
    Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['summary']))
