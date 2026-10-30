"""Verify eight BC-only CUDA runs and independently score their CPU policies."""
import argparse
import json
import math
from pathlib import Path
import shutil

from scripts.kaggle.local_artifact_transfer import file_sha
from scripts.attack.evaluate_air_position_labels import score_checkpoint


def verify(root):
    import torch
    torch.set_num_threads(1);root=Path(root);collected=root/'collected'
    prep=json.loads((root/'preparation.json').read_text());c=prep['config']
    receipt=json.loads((collected/'collection_receipt.json').read_text())
    p=json.loads((root/'dataset/initialization_protocol.json').read_text())
    if file_sha(root/'dataset/initialization_protocol.json')!=c['pins']['initialization_protocol.json']:
        raise ValueError('Initialization protocol changed')
    if not receipt['complete']or receipt['remote_status']!='COMPLETE'or receipt['kernel']!=prep['kernel']:
        raise ValueError('Complete initialization receipt required')
    for name,row in receipt['files'].items():
        if not (collected/name).resolve().is_relative_to(collected.resolve())or file_sha(collected/name)!=row['sha256']:
            raise ValueError('Initialization output changed')
    result=json.loads((collected/'result.json').read_text())
    if (result['schema']!='bank-initialization-study-result-v1'or result['smoke']or
            result['protocol_sha256']!=c['pins']['initialization_protocol.json']or
            result['source_sha256']!=c['extensions']['scripts/attack/train_bank_initialization_study.py']or
            result['environment_steps']!=0 or result['supervised_updates']!=6400):
        raise ValueError('Actual initialization study differs')
    if (p['seeds']!=[392,393]or p['updates_each']!=800 or
            [(r['id'],r['kl_weight'],r['reset_actor_heads'])for r in p['cases']]!=
            [('retain01',.1,False),('retain001',.01,False),('retain0',0.,False),('zero01',.1,True)]):
        raise ValueError('Wrong initialization budget/objectives')
    if len(result['runs'])!=8 or {r['id']for r in result['runs']}!={case['id']+'_s'+str(seed)for case in p['cases']for seed in p['seeds']}:
        raise ValueError('Missing initialization run')
    data=root/'verification_data';data.mkdir(exist_ok=True)
    for src,dst in [('skill_examples.npz','examples.npz'),('skill_manifest.json','manifest.json')]:
        if file_sha(root/'dataset'/src)!=c['pins'][src]:raise ValueError('Initialization data changed')
        if not(data/dst).exists():shutil.copyfile(root/'dataset'/src,data/dst)
        if file_sha(data/dst)!=c['pins'][src]:raise ValueError('Verification data changed')
    verified=[]
    for row in result['runs']:
        name=row['id'];r=json.loads((collected/name/'result.json').read_text());case=row['case']
        if case!=next(v for v in p['cases']if v['id']==case['id']):raise ValueError('Run case differs from frozen protocol')
        if (r!=row['result']or r['supervised_updates']!=800 or r['environment_steps']!=0 or not r['actual_cuda']or
                r['heldout_seed']!=354 or r['train_seeds']!=[352,353]or r['seed']!=row['seed']or
                r['initial_sha256']!=p['initial_sha256']or r['data_sha256']!=p['data_sha256']or
                r['source_sha256']!=c['extensions']['scripts/attack/train_air_position_warmstart.py']or
                r['reset_actor_heads']!=case['reset_actor_heads']or r['kl_weight']!=case['kl_weight']or
                not r['slot_augmentation']or r['sequence_wait_fraction']!=.5 or
                r['sampling']!='equal_skill_context_mass_bounded_wait_with_replacement'):
            raise ValueError('Actual BC budget, folds, origin or objective differs')
        history=json.loads((collected/name/'updates.json').read_text())
        if len(history)!=800 or any(v['update']!=i or not all(math.isfinite(v[k])for k in ('loss','imitation','kl'))for i,v in enumerate(history,1)):
            raise ValueError('Incomplete/nonfinite initialization optimizer history')
        policy=collected/name/'policy.pt'
        if file_sha(policy)!=row['policy_sha256']:raise ValueError('Initialization policy changed')
        blob=torch.load(policy,map_location='cpu',weights_only=False)
        if blob['config']['supervised_warmstart']!=r or not all(torch.isfinite(t).all()for t in blob['state_dict'].values()):
            raise ValueError('Invalid initialization checkpoint')
        score=score_checkpoint(policy,data)
        for skill,metrics in score['by_skill'].items():
            reported=r['final']['local_heldout']['by_skill'][skill]
            if abs(metrics['local_heldout']['mean_log_successful_probability']-reported['mean_log_successful_probability'])>1e-3:
                raise ValueError('Independent CPU scoring disagrees with training')
        verified.append(dict(id=name,case=case,seed=row['seed'],checkpoint=str(policy),score=score))
    ranking=[]
    for case in p['cases']:
        runs=[r for r in verified if r['case']==case]
        skill=[v['local_heldout']['fraction']for r in runs for v in r['score']['by_skill'].values()]
        decisions=[v['local_heldout']['fraction']for r in runs for v in r['score']['by_decision_type'].values()]
        ranking.append(dict(case=case,worst_skill=min(skill),mean_skill=sum(skill)/len(skill),
            worst_wait_or_deploy=min(decisions),local_fit_gate=min(skill+decisions)>=.5))
    ranking.sort(key=lambda r:(r['worst_skill'],r['worst_wait_or_deploy'],r['mean_skill']),reverse=True)
    return dict(schema='verified-bank-initialization-study-v1',kernel=prep['kernel'],supervised_updates_verified=6400,
        environment_steps=0,runs=verified,ranking=ranking,quality_accepted=False,
        scope='Local teacher fitting only; full-match PPO validation/test required before full continuation')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();Path(a.output).write_text(json.dumps(verify(a.root),indent=2)+'\n')
