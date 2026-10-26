"""Small paired validation screen of transfer weights; not source promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch

from sim.train_attack_ppo import initialize_attack_policy,evaluate
from sim.continuous_push import CONTRACT

MODELS=(('defense_economy_transfer','tmp/kaggle/ppo_selected_final_evaluations_20261023/vertel0/selected_dataset/policy.pt',
    'b9b34f2b7d55b9e37080790df60ee9903a769b7037d154c3248773ee4876f6ba'),
    ('parent','checkpoints/ppo_joint_economy_soft_20261023/policy.pt',
    '9d6ac7d1d41ff00df6a9c74354fa8447793e0be8d69f723b27ff43b23c4c73e8'))


def screen():
    torch.set_num_threads(1);started=time.monotonic();results=[]
    options=dict(curriculum='continuous_v1',validation_selection='worst_family',continuous_focus='uniform',
        geometry='native_anchors_candidate_v1',coordinate_mode='physical_live_cells_v2',action_delay_ms=225,max_ms=310000)
    for name,path,pin in MODELS:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=pin:raise ValueError('Transfer source changed')
        blob=torch.load(path,map_location='cpu',weights_only=False)
        net,_,migration=initialize_attack_policy(blob,device='cpu',num_scalars=309,observation_contract=CONTRACT)
        report=evaluate(net,'cpu',episodes=6,seed=900100,env_options=options,task_mode='joint')
        rows=report['records']
        if len(rows)!=12 or len({r['match_seed']for r in rows})!=12:raise ValueError('Missing paired validation coverage')
        results.append(dict(id=name,sha256=pin,source_coordinate_mode=blob['config'].get('coordinate_mode'),migration=migration,validation=report))
    return dict(schema='controlled-warmstart-transfer-screen-v1',models=results,elapsed_seconds=time.monotonic()-started,
        scope='Twelve paired validation contexts per source; diagnostic only, not full warmstart selection',
        limits=['Defense source additionally changes physical_live/legacy geometry into new v2/native inputs; transfer may lose skills',
            'No training or checkpoint promotion; both source files SHA preserved'],live_games_started=False,test_used_for_selection=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();r=screen()
    Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({m['id']:m['validation']['wins']for m in r['models']}))
