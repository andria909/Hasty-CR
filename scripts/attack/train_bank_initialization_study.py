"""Finite BC-only initialization study; no PPO steps or automatic promotion."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from queue import Queue
import subprocess
import sys

from scripts.kaggle.local_artifact_transfer import file_sha


def run(data,initial,protocol,output,smoke=False):
    import torch
    p=json.loads(Path(protocol).read_text());m=json.loads((Path(data)/'manifest.json').read_text())
    if (p['schema']!='bank-initialization-study-v1' or p['data_sha256']!=m['data_sha256'] or
            file_sha(Path(data)/'examples.npz')!=m['data_sha256'] or file_sha(initial)!=p['initial_sha256']):
        raise ValueError('Pinned initialization data/parent required')
    if not smoke and (not torch.cuda.is_available() or torch.cuda.device_count()<2):
        raise RuntimeError('Two CUDA devices required; no CPU fallback')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    devices=Queue()
    for d in (['cpu']if smoke else['cuda:0','cuda:1']):devices.put(d)
    updates=2 if smoke else p['updates_each'];seeds=p['seeds'][:1]if smoke else p['seeds']
    print(json.dumps(dict(event='CUDA_READY',gpus=[]if smoke else[torch.cuda.get_device_name(i)for i in range(2)],smoke=smoke)),flush=True)
    def child(case,seed):
        device=devices.get();name=case['id']+'_s'+str(seed)
        try:
            command=[sys.executable,'-u','-m','scripts.attack.train_air_position_warmstart',
                '--data',str(data),'--init',str(initial),'--output',str(output/name),
                '--device',device,'--updates',str(updates),'--kl-weight',str(case['kl_weight']),
                '--seed',str(seed),'--sequence-wait-fraction','.5','--slot-augmentation']
            if case['reset_actor_heads']:command.append('--reset-actor-heads')
            with(output/(name+'.log')).open('x')as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
            result=json.loads((output/name/'result.json').read_text())
            if (result['supervised_updates']!=updates or result['environment_steps']!=0 or
                    bool(result['actual_cuda'])!=(not smoke) or result['reset_actor_heads']!=case['reset_actor_heads'] or
                    result['kl_weight']!=case['kl_weight']):
                raise ValueError('Actual BC execution differs')
            row=dict(id=name,case=case,seed=seed,result=result,policy_sha256=file_sha(output/name/'policy.pt'))
            print(json.dumps(dict(event='INITIALIZATION_RUN_COMPLETE',id=name,updates=updates,target_updates=updates,
                                  actual_cuda=result['actual_cuda'])),flush=True)
            return row
        finally:devices.put(device)
    rows=[];pool=ThreadPoolExecutor(max_workers=1 if smoke else 2)
    try:
        futures=[pool.submit(child,case,seed)for case in p['cases']for seed in seeds]
        for future in as_completed(futures):rows.append(future.result())
    except BaseException:
        pool.shutdown(wait=True,cancel_futures=True);raise
    else:pool.shutdown()
    result=dict(schema='bank-initialization-study-result-v1',smoke=smoke,runs=sorted(rows,key=lambda r:r['id']),
        supervised_updates=sum(r['result']['supervised_updates']for r in rows),environment_steps=0,
        protocol_sha256=file_sha(protocol),data_sha256=m['data_sha256'],source_sha256=file_sha(__file__),
        quality_accepted=False,scope='Local skill fitting only; full-match PPO validation required before promotion')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('data','initial','protocol','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.data,a.initial,a.protocol,a.output,a.smoke)
