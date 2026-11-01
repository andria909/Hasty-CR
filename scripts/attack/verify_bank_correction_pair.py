"""Verify paired BC artifacts, then score every policy on the same curriculum."""
import argparse
import json
import math
from pathlib import Path
import shutil

from scripts.attack.evaluate_air_position_labels import score_checkpoint
from scripts.attack.train_bank_correction_pair import protocol
from scripts.kaggle.local_artifact_transfer import file_sha


def verify(root,technical_smoke=False):
    import torch
    torch.set_num_threads(1);root=Path(root);prep=json.loads((root/'preparation.json').read_text());c=prep['config']
    p=json.loads((root/'dataset/initialization_protocol.json').read_text())
    if (prep.get('study_variant')!='bank_correction_pair_v1'or
            file_sha(root/'dataset/initialization_protocol.json')!=c['pins']['initialization_protocol.json']):
        raise ValueError('Pinned paired protocol required')
    if p['technical_identical']and not technical_smoke:raise ValueError('Null preflight is not a production training result')
    collected=root/'actual_preflight/result'if technical_smoke else root/'collected'
    seeds=p['seeds'][:1]if technical_smoke else p['seeds'];updates=2 if technical_smoke else 800
    names={case['id']+'_s'+str(seed)for case in p['cases']for seed in seeds}
    if not technical_smoke:
        receipt=json.loads((collected/'collection_receipt.json').read_text())
        required={'result.json'}|{name+'/'+f for name in names for f in ('result.json','updates.json','policy.pt')}
        if (not receipt['complete']or receipt['remote_status']!='COMPLETE'or receipt['kernel']!=prep['kernel']or
                set(receipt['files'])!=required):raise ValueError('Complete paired CUDA collection required')
        for name,row in receipt['files'].items():
            if not (collected/name).resolve().is_relative_to(collected.resolve())or file_sha(collected/name)!=row['sha256']:
                raise ValueError('Collected artifact changed')
    result=json.loads((collected/'result.json').read_text())
    if (result['schema']!='bank-correction-paired-result-v1'or result['smoke']!=technical_smoke or
            result['technical_identical']!=p['technical_identical']or result['environment_steps']!=0 or
            result['supervised_updates']!=updates*len(names)or len(result['runs'])!=len(names)or
            {r['id']for r in result['runs']}!=names or
            result['protocol_sha256']!=c['pins']['initialization_protocol.json']or
            result['source_sha256']!=c['extensions']['scripts/attack/train_bank_correction_pair.py']):
        raise ValueError('Actual paired run budget or source differs')
    data={}
    for name,prefix in (('control','skill'),('corrections','correction')):
        target=root/'paired_verification_data'/name;target.mkdir(parents=True,exist_ok=True)
        for src,dst in ((prefix+'_examples.npz','examples.npz'),(prefix+'_manifest.json','manifest.json')):
            if file_sha(root/'dataset'/src)!=c['pins'][src]:raise ValueError('Input curriculum changed')
            if not(target/dst).exists():shutil.copyfile(root/'dataset'/src,target/dst)
            if file_sha(target/dst)!=c['pins'][src]:raise ValueError('Verification curriculum changed')
        data[name]=target
    if p!=protocol(data['control'],data['corrections'],root/'dataset/parent.pt',p['technical_identical']):
        raise ValueError('Frozen treatment specification differs from actual data')
    rows=[]
    for row in result['runs']:
        case=next((v for v in p['cases']if v['id']==row['case']['id']),None)
        if case!=row['case']or row['seed']not in seeds or row['id']!=case['id']+'_s'+str(row['seed']):
            raise ValueError('Case/seed identity differs')
        path=collected/row['id'];r=json.loads((path/'result.json').read_text());spec=p['datasets'][case['dataset']]
        if (r!=row['result']or r['supervised_updates']!=updates or r['environment_steps']!=0 or
                bool(r['actual_cuda'])==technical_smoke or r['seed']!=row['seed']or
                r['initial_sha256']!=p['initial_sha256']or r['data_sha256']!=spec['data_sha256']or
                r['train_seeds']!=[352,353]or r['heldout_seed']!=354 or r['kl_weight']!=case['kl_weight']or
                r['reset_actor_heads']or not r['slot_augmentation']or r['sequence_wait_fraction']!=.5 or
                r['sampling']!='equal_skill_context_mass_bounded_wait_with_replacement'or
                r['source_sha256']!=c['extensions']['scripts/attack/train_air_position_warmstart.py']or
                set(r['final']['local_heldout']['by_skill'])!=set(spec['skills'])):
            raise ValueError('Executed optimizer contract differs')
        history=json.loads((path/'updates.json').read_text())
        if len(history)!=updates or any(x['update']!=i or not all(math.isfinite(x[k])for k in ('loss','imitation','kl'))for i,x in enumerate(history,1)):
            raise ValueError('Incomplete or nonfinite update history')
        checkpoint=path/'policy.pt'
        if file_sha(checkpoint)!=row['policy_sha256']:raise ValueError('Checkpoint changed')
        blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if blob['config']['supervised_warmstart']!=r or not all(torch.isfinite(t).all()for t in blob['state_dict'].values()):
            raise ValueError('Checkpoint metadata or tensors invalid')
        own=score_checkpoint(checkpoint,data[case['dataset']])
        for skill,metrics in own['by_skill'].items():
            for split in ('train','local_heldout'):
                reported=r['final'][split]['by_skill'][skill];actual=metrics[split]
                if (actual['examples']!=reported['examples']or
                        abs(actual['mean_log_successful_probability']-reported['mean_log_successful_probability'])>1e-3 or
                        abs(actual['fraction']-reported['successful_greedy_fraction'])>1e-6):
                    raise ValueError('Independent local scoring differs')
        common=own if case['dataset']=='corrections'else score_checkpoint(checkpoint,data['corrections'])
        rows.append(dict(id=row['id'],case=case,seed=row['seed'],checkpoint=str(checkpoint),
            checkpoint_sha256=file_sha(checkpoint),own_data_score=own,common_data_score=common))
    differences=[]
    for seed in seeds:
        for tag in ('001','01'):
            control=next(r for r in rows if r['id']==f'control_kl{tag}_s{seed}')
            treatment=next(r for r in rows if r['id']==f'corrections_kl{tag}_s{seed}')
            differences.append(dict(seed=seed,kl=control['case']['kl_weight'],
                local_heldout_skill_deltas={skill:treatment['common_data_score']['by_skill'][skill]['local_heldout']['fraction']-
                    control['common_data_score']['by_skill'][skill]['local_heldout']['fraction']
                    for skill in p['datasets']['corrections']['skills']}))
    return dict(schema='verified-bank-correction-pair-smoke-v1'if technical_smoke else'verified-bank-correction-pair-v1',
        kernel=prep['kernel'],supervised_updates_verified=updates*len(names),environment_steps=0,
        initial_sha256=p['initial_sha256'],
        actual_cuda_verified=not technical_smoke,runs=rows,paired_differences=differences,
        common_data_sha256=p['datasets']['corrections']['data_sha256'],full_training_accepted=False,
        scope='All policies scored on identical corrected curriculum; local labels are not full-match validation')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--technical-smoke',action='store_true');a=p.parse_args()
    Path(a.output).write_text(json.dumps(verify(a.root,a.technical_smoke),indent=2)+'\n')
