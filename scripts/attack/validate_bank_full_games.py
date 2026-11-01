"""Paired full-game validation of BC candidates on all six enemy families."""
import argparse
import json
from pathlib import Path

import torch

from scripts.attack.select_continuous_focus import validation_score
from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.train_attack_ppo import initialize_attack_policy, evaluate


def run(weights, config, output, device='cpu'):
    torch.set_num_threads(1)
    if device.startswith('cuda')and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    config=Path(config);c=json.loads(config.read_text());root=Path(output);root.mkdir(parents=True,exist_ok=False)
    keys=('curriculum','continuous_profile','validation_selection','continuous_focus','geometry','coordinate_mode',
        'max_ms','tower_support_weight','spell_waste_weight','crown_only_weight','dormant_king_weight')
    options={k:c[k]for k in keys};options['action_delay_ms']=1350
    profile=options['continuous_profile']
    if (options['curriculum']!='continuous_v1'or options['validation_selection']!='worst_family'or
            profile['action_cooldown_ms']!=0 or profile['decision_interval_ms']!=250 or
            any(profile[k]!=1 for k in ('cannon_center_only','source_radius_reach','hog_sight_clipping'))):
        raise ValueError('Current physical execution contract required')
    pins=physics_pins();source=file_sha(__file__)
    protocol=dict(schema='bank-full-validation-protocol-v1',config_sha256=file_sha(config),options=options,
        blocks=6,episodes_per_task_per_block=6,seed_base=202611010,seed_stride=12,
        models=[dict(checkpoint=str(w),sha256=file_sha(w))for w in weights],physics_pins=pins,
        source_sha256=source,split='validation',independent_test=False,live_games=False)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (root/'driver.py').write_bytes(Path(__file__).read_bytes())
    reports=[]
    for index,path in enumerate(weights):
        blob=torch.load(path,map_location='cpu',weights_only=False)
        net,_,_=initialize_attack_policy(blob,device=device,num_planes=20,num_scalars=309,
            observation_contract='joint_public_presence_continuous_slots_v2')
        rows=[]
        for block in range(6):
            result=evaluate(net,device,episodes=6,seed=202611010+12*block,env_options=options,task_mode='joint')
            if len(result['records'])!=12:raise ValueError('Incomplete validation block')
            rows.extend(result['records'])
            (root/f'model_{index}_block_{block}.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(event='VALIDATION_PROGRESS',model=index,games=len(rows),target_games=72)),flush=True)
        score=validation_score(dict(records=rows))
        report=dict(model=protocol['models'][index],records=rows,score=score,
            wins=sum(r['outcome']=='win'for r in rows),games=len(rows))
        (root/f'model_{index}.json').write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
    if pins!=physics_pins()or source!=file_sha(__file__):raise ValueError('Source changed during validation')
    (root/'result.json').write_text(json.dumps(dict(protocol=protocol,reports=reports,
        full_training_accepted=False),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--weights',nargs='+',required=True)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    a=p.parse_args();run(a.weights,a.config,a.output,a.device)
