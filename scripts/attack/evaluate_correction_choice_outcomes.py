"""Measure chosen first actions under the original fixed continuation policy.

This distinguishes a disagreement with sparse search labels from actual loss.
Fold354 stays evaluation-only; no generated outcome is used for training here.
"""
import argparse
import json
from pathlib import Path
import pickle

import torch

from scripts.attack.search_bank_match_actions import actor_action,branch,quality,DELAYS
from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.env import ClashEnv
from sim.train_attack_ppo import initialize_attack_policy


def evaluate(verified,choices,search,output,limit=None):
    torch.set_num_threads(1)
    verified,choices,search,output=map(Path,(verified,choices,search,output))
    p=json.loads(verified.read_text());d=json.loads(choices.read_text());s=json.loads(search.read_text())
    if (p.get('schema')!='verified-bank-fit-depth-v1'or not p['actual_cuda_verified']or
            p['supervised_updates_verified']!=35200 or d['verified_sha256']!=file_sha(verified)
            or d['data_sha256']!=p['common_data_sha256']or s['contexts_verified']!=106):
        raise ValueError('Verified fitting choices and complete original search required')
    root=Path(s['root']);m=json.loads((root/'manifest.json').read_text());pins=physics_pins()
    index={r['snapshot']:i for i,r in enumerate(m['records'])}
    accepted={r['index']:r for r in s['accepted']if r['fold']==354}
    if len(accepted)!=15:raise ValueError('All fifteen reused heldout contexts required')
    def network(path):
        net,_,_=initialize_attack_policy(torch.load(path,map_location='cpu',weights_only=False),device='cpu',
            num_planes=20,num_scalars=309,observation_contract='joint_public_presence_continuous_slots_v2')
        return net.eval()
    continuation=network(m['weights']);output.mkdir(parents=True,exist_ok=False)
    (output/'driver.py').write_bytes(Path(__file__).read_bytes())
    reports=[];computed={};baseline_checked=set()
    for name in ('mixed8000_s402','mixed8000_s403'):
        candidate=next(r for r in p['runs']if r['id']==name)
        if file_sha(candidate['checkpoint'])!=candidate['checkpoint_sha256']:raise ValueError('Candidate changed')
        policy=network(candidate['checkpoint'])
        selected=[r for r in next(r for r in d['models']if r['id']==name)['rows']if r['seed']==354]
        if len(selected)!=15 or len({r['context']for r in selected})!=15:raise ValueError('Heldout choice coverage differs')
        for record in selected[:limit]:
            snapshot=record['context'].removeprefix('match:');i=index[snapshot];entry=accepted[i]
            context=Path(entry['context_path'])
            if file_sha(context)!=entry['context_sha256']:raise ValueError('Cached context changed')
            saved=json.loads(context.read_text());protocol=json.loads((context.parent/'protocol.json').read_text())
            if (protocol['physics_pins']!=pins or protocol['manifest_sha256']!=file_sha(root/'manifest.json')
                    or protocol['weights_sha256']!=file_sha(m['weights'])or
                    protocol['source_sha256']!=file_sha('scripts/attack/search_bank_match_actions.py')):
                raise ValueError('Continuation or physical execution differs')
            state=m['records'][i]
            if state!=saved['state']or file_sha(root/snapshot)!=state['sha256']:raise ValueError('Snapshot changed')
            env=pickle.loads((root/snapshot).read_bytes())
            action=actor_action(policy,env)
            if action!=record['choice']:raise ValueError('Actual snapshot decision differs from label diagnostic')
            if action:
                slot,x,y=ClashEnv.decode(action)
                if slot=='ability':raise ValueError('Unsupported ability in fixed deck')
                first=dict(action=action,card=env.match.players[1].hand[slot],cell=[x,y])
            else:first=dict(action=0,card='wait',cell=None)
            baseline={r['delay_ms']:r for r in saved['finals']if r['action']==0}
            trials=[]
            for delay in DELAYS:
                # One independently repeated control per used context/delay;
                # shared across both policy seeds, never silently assumed.
                if (i,delay)not in baseline_checked:
                    control=branch(env,continuation,dict(action=0,card='wait',cell=None),delay)
                    if control!=baseline[delay]:raise ValueError('Actual control replay differs')
                    baseline_checked.add((i,delay))
                key=(i,action,delay)
                if key not in computed:
                    result=branch(env,continuation,first,delay)
                    computed[key]=result
                    (output/f'context_{i:04d}_action_{action}_delay_{delay}.json').write_text(json.dumps(result,indent=2)+'\n')
                trial=computed[key]
                trials.append(dict(delay_ms=delay,outcome=trial['metrics']['outcome'],own_damage=trial['own_damage'],
                    first_command_accepted=trial['first_command_accepted'],
                    baseline_outcome=baseline[delay]['metrics']['outcome'],baseline_damage=baseline[delay]['own_damage'],
                    better_than_wait=quality(trial)>quality(baseline[delay]),worse_than_wait=quality(trial)<quality(baseline[delay])))
            row=dict(id=name,index=i,context=record['context'],action=action,category=record['category'],trials=trials,
                better_both=all(r['better_than_wait']and r['first_command_accepted']for r in trials),
                worse_any=any(r['worse_than_wait']for r in trials))
            reports.append(row)
            (output/'progress.json').write_text(json.dumps(dict(choices=len(reports),target=30,partial=limit is not None))+'\n')
            print(json.dumps(dict(id=name,context=i,better_both=row['better_both'],worse_any=row['worse_any'])),flush=True)
    if physics_pins()!=pins or file_sha(output/'driver.py')!=file_sha(__file__):raise ValueError('Evaluator changed during execution')
    result=dict(schema='fit-depth-chosen-action-outcomes-v1',partial=limit is not None,
        verified_sha256=file_sha(verified),choices_sha256=file_sha(choices),search_sha256=file_sha(search),physics_pins=pins,
        rows=reports,unique_action_branches=len(computed),control_branches_replayed=len(baseline_checked),
        full_training_accepted=False,scope='Reused heldout snapshots, chosen first action then original policy; not new-policy match win rate')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('verified','choices','search','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--limit',type=int);args=parser.parse_args();evaluate(**vars(args))
