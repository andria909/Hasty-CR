"""Experimental deterministic decoder on cached validation contexts.

One flat-greedy control block must replay exactly before testing the changed
decoder. This does not modify training, saved weights, or live execution.
"""
import argparse
import json
from pathlib import Path

import torch

from scripts.attack.compare_ppo_pilot_behavior import compare_records
from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.train_attack_ppo import initialize_attack_policy, make_training_env
from sim.train_defense_ppo import tensor_observations, action_masks


def choose(logits, mask, mode):
    if mode not in ('flat','card_mass') or logits.shape != mask.shape or logits.shape[0] != 1:
        raise ValueError('One supported action distribution required')
    masked=logits.masked_fill(~mask,-torch.inf)
    if mode=='flat':return int(masked.argmax(-1).item())
    if mask[:,2305:].any():raise ValueError('Fixed deck has no ability actions')
    cells=masked[:,1:2305].reshape(1,4,576)
    groups=torch.cat((masked[:,:1],torch.logsumexp(cells,dim=-1)),dim=-1)
    group=int(groups.argmax(-1).item())
    if group==0:return 0
    return 1+(group-1)*576+int(cells[0,group-1].argmax().item())


def episode(net, row, options, mode):
    env=make_training_env(task_mode='joint',split='validation',seed=row['episode_seed'],**options)
    obs,info=env.reset(seed=row['episode_seed'],scenario_id=row['scenario_id'])
    total=0.;changed=0;decisions=0
    with torch.inference_mode():
        for _ in range(int(env.max_ms/env.profile['decision_interval_ms'])+2):
            logits,_=net(*tensor_observations([obs],'cpu'));mask=action_masks([info],'cpu')
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite logits')
            action=choose(logits,mask,mode)
            changed+=action!=choose(logits,mask,'flat');decisions+=1
            obs,reward,terminal,truncated,info=env.step(action);total+=reward
            if terminal or truncated:
                return dict(info['metrics'],episode_return=total),dict(decisions=decisions,changed_actions=changed)
    raise RuntimeError('Finite validation bound exceeded')


def run(validation,proof,output,smoke=False):
    validation,proof,root=map(Path,(validation,proof,output))
    verified=json.loads(proof.read_text());raw=json.loads(validation.read_text())
    if (verified.get('schema')!='verified-fit-depth-full-games-v1' or verified['games_verified']!=360
            or verified['raw_validation_sha256']!=file_sha(validation)
            or raw['protocol']['physics_pins']!=physics_pins()):
        raise ValueError('Verified current 360-game control required')
    baseline=next(r for r in raw['reports']if Path(r['model']['checkpoint']).parent.name=='mixed800_s402')
    weights=Path(baseline['model']['checkpoint'])
    if file_sha(weights)!=baseline['model']['sha256']:raise ValueError('Baseline weights changed')
    root.mkdir(parents=True,exist_ok=False)
    pins=physics_pins();source=file_sha(__file__)
    options=raw['protocol']['options']
    protocol=dict(validation_sha256=file_sha(validation),proof_sha256=file_sha(proof),weights_sha256=file_sha(weights),
                  source_sha256=source,physics_pins=pins,options=options,mode='card_mass',
                  smoke=smoke,live_games=False,training_changed=False,control_replays=1 if smoke else 12,
                  target_games=1 if smoke else 72)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (root/'driver.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    net,_,_=initialize_attack_policy(torch.load(weights,map_location='cpu',weights_only=False),device='cpu',
        num_planes=20,num_scalars=309,observation_contract='joint_public_presence_continuous_slots_v2')
    net.eval();controls=[];rows=[];audit=[]
    for index,row in enumerate(baseline['records'][:protocol['control_replays']]):
        repeated,_=episode(net,row,options,'flat')
        if repeated!=row:raise ValueError('Flat control does not exactly replay cached metrics')
        controls.append(dict(index=index,all_terminal_metrics_exact=True))
        (root/'controls.json').write_text(json.dumps(controls,indent=2)+'\n')
        print(json.dumps(dict(event='CONTROL_REPLAY_EXACT',completed=len(controls))),flush=True)
    for index,row in enumerate(baseline['records'][:protocol['target_games']]):
        result,counts=episode(net,row,options,'card_mass');rows.append(result);audit.append(counts)
        (root/f'game_{index:03d}.json').write_text(json.dumps(dict(metrics=result,decoder=counts),indent=2)+'\n')
        (root/'progress.json').write_text(json.dumps(dict(games=len(rows),target=protocol['target_games'],
            wins=sum(r['outcome']=='win'for r in rows),smoke=smoke))+'\n')
        print(json.dumps(dict(event='DECODER_GAME',completed=len(rows),outcome=result['outcome'])),flush=True)
    if pins!=physics_pins()or source!=file_sha(__file__):raise ValueError('Evaluation source changed')
    result=dict(schema='card-mass-decoder-evaluation-v1',protocol=protocol,controls=controls,records=rows,
                decoder_counts=audit,comparison=None if smoke else compare_records(baseline['records'],rows),
                quality_accepted=False,live_checkpoint_changed=False)
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('validation','proof','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args();run(**vars(a))
