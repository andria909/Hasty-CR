"""Verify terminal joint-training artifacts before any heldout evaluation.

No network, training, selection from test, or changes to the collected inputs.
The receipt checks internal consistency; Kaggle metadata provides no remote SHA.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

BASELINE_SHA='d65ab55893592c0e4655a1882db459c4a7fb1ebd9e10afaf2daa4d2b369fc972'
CONTRACT='joint_attack_defense_timer_task_v1'
TARGETS={'ppo':2048000,'grpo':1000000}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_file(root,name):
    relative=PurePosixPath(name)
    if relative.is_absolute()or '..'in relative.parts or '\\'in name or str(relative)!=name:
        raise ValueError('Unsafe collected path')
    root=Path(root).resolve();path=root/name
    if path.is_symlink()or not path.resolve().is_relative_to(root)or not path.is_file():
        raise ValueError('Missing or unsafe collected file')
    return path


def audit(collected,launch_config,source_manifest):
    import torch
    root=Path(collected)
    launch=json.loads(Path(launch_config).read_text())
    manifest=json.loads(Path(source_manifest).read_text())
    if sha(source_manifest)!=launch['pins']['manifest.json']:
        raise ValueError('Pinned source manifest mismatch')
    algorithm=launch['algorithm'];target=TARGETS[algorithm]
    if launch['target_steps']!=target or launch['pins']['v25.pt']!=BASELINE_SHA:
        raise ValueError('Training target or baseline changed')
    receipt=json.loads(exact_file(root,'collection_receipt.json').read_text())
    owner=launch['dataset'].split('/')[0]
    if (receipt.get('complete')is not True or receipt.get('remote_status')!='COMPLETE'
            or receipt.get('kernel')!=f'{owner}/hastycr-joint-{algorithm}-20261023'):
        raise ValueError('Expected completed collection from the exact submitted kernel')
    files=receipt['files']
    for name,row in files.items():
        if sha(exact_file(root,name))!=row['sha256']:
            raise ValueError('Collected artifact SHA mismatch')
    if 'strategy_result.json'not in files:raise ValueError('Unreceipted training result')
    result=json.loads(exact_file(root,'strategy_result.json').read_text())
    expected=dict(schema='joint-strategy-account-v1',status='trainer_completed',
                  owner=owner,algorithm=algorithm,target_steps=target,
                  source_archive_sha256=manifest['archive_sha256'])
    if any(result.get(k)!=v for k,v in expected.items())or not result.get('gpu_names'):
        raise ValueError('Runtime provenance or CUDA missing/mismatched')
    pattern=(r'ppo/update_(\d{6})\.pt'if algorithm=='ppo'else
             r'grpo/final_u(\d{6})_s(\d{6})\.pt')
    candidates=[(tuple(map(int,m.groups())),name)for name in files
                if (m:=re.fullmatch(pattern,name))]
    if not candidates:raise ValueError('Missing final completed-boundary checkpoint')
    name=max(candidates)[1]
    blob=torch.load(exact_file(root,name),map_location='cpu',weights_only=False)
    config=blob['config']
    required=dict(task_mode='joint',observation_contract=CONTRACT,num_scalars=309,
                  init_sha256=BASELINE_SHA,initial_hand_mode='shuffle',deadline_scale=3,
                  target_environment_steps=target,geometry='legacy_v1',
                  coordinate_mode='physical_live_cells_v2',device='cuda:0')
    if any(config.get(k)!=v for k,v in required.items()):
        raise ValueError('Checkpoint training contract mismatch')
    steps=blob.get('step'if algorithm=='ppo'else 'environment_steps')
    if isinstance(steps,bool)or not isinstance(steps,int)or steps<target:
        raise ValueError('Training target not reached')
    state=blob.get('state_dict',{})
    if not state or state['scalar.0.weight'].shape[1]!=309:
        raise ValueError('Joint policy shape mismatch')
    def finite(value):
        if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
        if isinstance(value,dict):return all(finite(v)for v in value.values())
        if isinstance(value,(list,tuple)):return all(finite(v)for v in value)
        return True
    optimizer_key='optimiser'if algorithm=='ppo'else 'optimizer'
    if not blob.get(optimizer_key)or not finite(state)or not finite(blob[optimizer_key]):
        raise ValueError('Missing/nonfinite policy or optimizer')
    if algorithm=='ppo':
        if not blob.get('exact_resume_available'):raise ValueError('PPO resume state missing')
        if 'ppo/training_result.json'not in files:raise ValueError('Missing PPO target result')
        report=json.loads(exact_file(root,'ppo/training_result.json').read_text())
        if report.get('target_met')is not True or report.get('step')!=steps:
            raise ValueError('PPO target report disagrees with checkpoint')
        metrics='ppo/metrics.jsonl'
        if metrics not in files:raise ValueError('Missing task metrics')
        counts={task:0 for task in ('attack','defense')}
        for line in exact_file(root,metrics).read_text().splitlines():
            row=json.loads(line)
            for task in counts:counts[task]+=row.get('by_task',{}).get(task,{}).get('n',0)
        # Attack summaries use episodes rather than n; inspect both explicit forms.
        if not all(counts.values()):
            counts={task:0 for task in counts}
            for line in exact_file(root,metrics).read_text().splitlines():
                for episode in json.loads(line).get('episodes',[]):
                    task=episode.get('task')
                    if task in counts:counts[task]+=1
    else:
        if blob.get('checkpoint_phase')!='boundary'or not blob.get('resume_supported'):
            raise ValueError('GRPO checkpoint contains an unfinished group')
        if not blob.get('reference_state_dict')or not finite(blob['reference_state_dict']):
            raise ValueError('Missing/nonfinite frozen reference')
        counts={task:blob.get('actual_environment_steps',{}).get('train_'+task,0)
                for task in ('attack','defense')}
    if any(isinstance(n,bool)or not isinstance(n,int)or n<=0 for n in counts.values()):
        raise ValueError('Actual training on both tasks not demonstrated')
    return dict(schema='joint-training-audit-v1',algorithm=algorithm,steps=steps,
                target_steps=target,target_met=True,checkpoint=name,
                checkpoint_sha256=sha(root/name),task_counts=counts,
                source_archive_sha256=manifest['archive_sha256'],
                ready_for_validation=True,quality_accepted=False,
                evaluation_completed=False,remote_hash_verified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collected',type=Path,required=True)
    parser.add_argument('--launch-config',type=Path,required=True)
    parser.add_argument('--source-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=audit(args.collected,args.launch_config,args.source_manifest)
    with args.output.open('x')as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report))


if __name__=='__main__':main()
