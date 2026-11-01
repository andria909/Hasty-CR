"""Paired BC: unchanged curriculum versus replayed corrections, at equal budget."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
from queue import Queue
import subprocess
import sys
from threading import Timer

import numpy as np

from scripts.attack.sequence_physics import require_current_physics
from scripts.kaggle.local_artifact_transfer import file_sha

BASE_SKILLS={'safe_bank_sequence','air_support_defense','fireball_group','spell_restraint'}


def protocol(base,corrections,initial,technical_identical=False):
    datasets={}
    for name,path in (('control',Path(base)),('corrections',Path(corrections))):
        manifest=json.loads((path/'manifest.json').read_text());require_current_physics(manifest)
        if (manifest['schema']!='mixed-bank-public-sequences-v1'or manifest['split']!='train'or
                manifest['seeds']!=[352,353,354]or file_sha(path/'examples.npz')!=manifest['data_sha256']):
            raise ValueError('Pinned public three-fold curriculum required')
        datasets[name]=dict(data_sha256=manifest['data_sha256'],manifest_sha256=file_sha(path/'manifest.json'),
            skills=sorted(manifest['skills']),examples=manifest['examples'])
    if set(datasets['control']['skills'])!=BASE_SKILLS:raise ValueError('Four-skill control required')
    if technical_identical:
        if datasets['control']!=datasets['corrections']:raise ValueError('Technical null comparison requires identical data')
    else:
        m=json.loads((Path(corrections)/'manifest.json').read_text())
        if (set(m['skills'])!=BASE_SKILLS|{'onpolicy_defense_response'}or
                m.get('correction_label_mode')!='outcome_first_pareto_v1'or
                m.get('base_manifest_sha256')!=datasets['control']['manifest_sha256']or
                m.get('correction_examples',0)<3 or
                any(m.get('correction_counts_by_fold',{}).get(str(f),0)<1 for f in (352,353,354))):
            raise ValueError('Verified correction treatment covering all folds required')
        count=datasets['control']['examples']
        with np.load(Path(base)/'examples.npz',allow_pickle=False)as a,np.load(Path(corrections)/'examples.npz',allow_pickle=False)as b:
            for key in ('planes','scalars','masks','success_masks'):
                if not np.array_equal(a[key],b[key][:count]):raise ValueError('Treatment changes existing examples')
    return dict(schema='bank-correction-paired-study-v1',technical_identical=technical_identical,
        initial_sha256=file_sha(initial),datasets=datasets,seeds=[402,403],updates_each=800,
        cases=[dict(id=f'{name}_kl{tag}',dataset=name,kl_weight=kl)for name in ('control','corrections')
            for tag,kl in (('001',.01),('01',.1))],sequence_wait_fraction=.5,slot_augmentation=True,
        reset_actor_heads=False,full_training_accepted=False,
        scope='Paired dataset treatment at two KL values; equal updates and initial weights, BC only')


def run(base,corrections,initial,protocol_file,output,smoke=False):
    import torch
    p=json.loads(Path(protocol_file).read_text())
    if p!=protocol(base,corrections,initial,p.get('technical_identical',False)):
        raise ValueError('Actual paired study differs from pinned protocol')
    if p['technical_identical']and not smoke:raise ValueError('Technical identical-data study is CPU preflight only')
    if not smoke and(not torch.cuda.is_available()or torch.cuda.device_count()<2):
        raise RuntimeError('Two CUDA devices required; no fallback')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);devices=Queue()
    for device in (['cpu']if smoke else['cuda:0','cuda:1']):devices.put(device)
    updates=2 if smoke else p['updates_each'];seeds=p['seeds'][:1]if smoke else p['seeds']
    print(json.dumps(dict(event='CUDA_READY',smoke=smoke,gpus=[]if smoke else[torch.cuda.get_device_name(i)for i in range(2)])),flush=True)
    paths=dict(control=base,corrections=corrections)
    def child(case,seed):
        device=devices.get();name=case['id']+'_s'+str(seed)
        try:
            command=[sys.executable,'-u','-m','scripts.attack.train_air_position_warmstart',
                '--data',str(paths[case['dataset']]),'--init',str(initial),'--output',str(root/name),
                '--device',device,'--updates',str(updates),'--kl-weight',str(case['kl_weight']),
                '--seed',str(seed),'--sequence-wait-fraction','.5','--slot-augmentation','--progress-interval','25']
            with(root/(name+'.log')).open('x')as log:
                process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                timer=Timer(7200,process.kill);timer.daemon=True;timer.start()
                try:
                    for line in process.stdout:
                        log.write(line);log.flush()
                        try:event=json.loads(line)
                        except ValueError:continue
                        if isinstance(event,dict)and event.get('event')=='WARMSTART_PROGRESS':
                            print(json.dumps(dict(event='CORRECTION_PROGRESS',id=name,updates=event['update'],
                                target_updates=event['target_updates'],actual_cuda=event['actual_cuda'])),flush=True)
                    code=process.wait()
                    if code:raise subprocess.CalledProcessError(code,command)
                finally:
                    timer.cancel()
                    if process.poll()is None:process.kill();process.wait()
                    process.stdout.close()
            result=json.loads((root/name/'result.json').read_text());data=p['datasets'][case['dataset']]
            if (result['supervised_updates']!=updates or result['environment_steps']or
                    bool(result['actual_cuda'])!=(not smoke)or result['seed']!=seed or
                    result['train_seeds']!=[352,353]or result['heldout_seed']!=354 or
                    result['data_sha256']!=data['data_sha256']or result['initial_sha256']!=p['initial_sha256']or
                    result['reset_actor_heads']or result['kl_weight']!=case['kl_weight']or
                    set(result['final']['local_heldout']['by_skill'])!=set(data['skills'])):
                raise ValueError('Executed paired update differs')
            row=dict(id=name,case=case,seed=seed,result=result,policy_sha256=file_sha(root/name/'policy.pt'))
            print(json.dumps(dict(event='CORRECTION_RUN_COMPLETE',id=name,updates=updates,actual_cuda=result['actual_cuda'])),flush=True)
            return row
        finally:devices.put(device)
    with ThreadPoolExecutor(max_workers=1 if smoke else 2)as pool:
        futures=[pool.submit(child,case,seed)for case in p['cases']for seed in seeds]
        rows=[future.result()for future in as_completed(futures)]
    result=dict(schema='bank-correction-paired-result-v1',smoke=smoke,technical_identical=p['technical_identical'],
        runs=sorted(rows,key=lambda r:r['id']),supervised_updates=sum(r['result']['supervised_updates']for r in rows),
        environment_steps=0,protocol_sha256=file_sha(protocol_file),source_sha256=file_sha(__file__),
        full_training_accepted=False)
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('base','corrections','initial','protocol','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    run(a.base,a.corrections,a.initial,a.protocol,a.output,a.smoke)
