"""Real short optimizer runs: pending replay, uninterrupted versus resumed."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch
import random
import numpy as np


def equal_state(a,b,seen=None):
    if type(a) is not type(b):return False
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,random.Random):return a.getstate()==b.getstate()
    seen=set() if seen is None else seen
    if isinstance(a,(dict,list,tuple)) or hasattr(a,'__dict__'):
        if (id(a),id(b)) in seen:return True
        seen.add((id(a),id(b)))
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal_state(a[k],b[k],seen) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_state(x,y,seen) for x,y in zip(a,b))
        return equal_state(vars(a),vars(b),seen)
    return a==b


def smoke(initial,data,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    common=['-m','sim.train_attack_ppo','--device','cpu','--curriculum','continuous_v1',
        '--task-mode','joint','--pending-inputs','--supervised-replay-data',str(data),
        '--supervised-replay-weight','1','--envs','1','--rollout','16','--minibatch-size','16',
        '--epochs','1','--eval-every','0','--checkpoint-every','1','--seed','112',
        '--action-delay-ms','1350','--geometry','native_anchors_candidate_v1',
        '--coordinate-mode','physical_live_cells_v2','--max-ms','310000']
    for name,steps,source in [('first',16,['--init',str(initial)]),('whole',32,['--init',str(initial)]),
                             ('resumed',32,['--resume',str(root/'first/update_000001.pt')])]:
        command=[sys.executable,*common,*source,'--output',str(root/name),
                 '--updates',str(steps//16),'--target-environment-steps',str(steps)]
        with (root/(name+'.log')).open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    return verify(root)


def verify(root):
    root=Path(root)
    a=torch.load(root/'whole/update_000002.pt',map_location='cpu',weights_only=False)
    b=torch.load(root/'resumed/update_000002.pt',map_location='cpu',weights_only=False)
    keys=('state_dict','optimiser','rng','sampler','environment_states','observations','infos',
          'episode_returns','episode_counters','step','update')
    checks={k:equal_state(a[k],b[k]) for k in keys}
    if not all(checks.values()):raise ValueError('Resume state differs: '+str(checks))
    metrics=[json.loads(line) for line in (root/'whole/metrics.jsonl').read_text().splitlines()]
    if not all(row['supervised_replay']['optimizer_batches']>0 for row in metrics):
        raise ValueError('Replay optimizer batches absent')
    from sim.train_attack_ppo import validate_resume
    changed=dict(a['config']);changed['supervised_replay_metadata']=dict(changed['supervised_replay_metadata'],sequence_wait_fraction=.5)
    try:validate_resume(a,changed)
    except ValueError as exc:
        if 'supervised_replay_metadata' not in str(exc):raise
    else:raise ValueError('Changed replay sampler accepted on resume')
    result=dict(exact_resume=checks,replay_metrics=[r['supervised_replay'] for r in metrics],
        metadata=a['config']['supervised_replay_metadata'],num_planes=a['config']['num_planes'],
        num_scalars=a['config']['num_scalars'],changed_sampler_rejected=True,
        actual_cuda=False,quality_accepted=False,source_code_sha256=a['config']['source_code_sha256'])
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--init',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=smoke(a.init,a.data,a.output);print(json.dumps(r))
