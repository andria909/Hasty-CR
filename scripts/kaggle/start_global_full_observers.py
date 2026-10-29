"""Start20-minute observers for a matching launched full pair, exactly once."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def commands(root,stem,owner,sdk,python):
    report=f'reports/defense_plan/{stem}_training_20261027.json'
    prepared=json.loads((Path(root)/'preparation.json').read_text())
    return dict(
        stream=[sdk,'-m','scripts.kaggle.stream_training','--kernel',prepared['kernel'],
            '--output',str(Path(root)/'live_stream'),'--api-user',owner,'--reconnect-seconds','1200','--reconnects','100'],
        collector=[sdk,'-m','scripts.kaggle.collect_global_full','--prepared',str(root),
            '--output',str(Path(root)/'collected'),'--api-user',owner,'--poll-seconds','1200'],
        postprocess=[python,'-m','scripts.kaggle.watch_global_full_pair','--root',str(root),'--report',report,
            '--verified-report',f'reports/defense_plan/{stem}_verified_20261027.json'])


def start(root,stem,token_index):
    root=Path(root)
    if not re.fullmatch(r'(global_full_pair[01]|mana_full_pair0_20261029|bank_teacher_full_pair0_20261029)',stem):raise ValueError('Explicit full pair report stem required')
    if token_index not in(0,1,2):raise ValueError('Only the three available accounts are allowed')
    if(root/'monitoring.json').exists():raise ValueError('Inspect existing full observers; no duplicates')
    prep=json.loads((root/'preparation.json').read_text());receipt=json.loads((root/'launch_receipt.json').read_text())
    if receipt.get('ref')!='/code/'+prep['kernel']or receipt.get('versionNumber')!=1:
        raise ValueError('Matching full-pair launch receipt required')
    if prep['full_steps']!=8192000 or len(prep['children'])!=2:
        raise ValueError('Expected two full targets required')
    preflight=json.loads((root/'preflight.json').read_text())
    if not preflight.get('exact_resume_16_to_32')or not preflight.get('frozen_sources_verified'):
        raise ValueError('Verified full-pair preflight required')
    owner=prep['config']['owner'];item=json.loads((Path.home()/'.kaggle/access_tokens.json').read_text())['tokens'][token_index]
    env=dict(os.environ,KAGGLE_API_TOKEN=item if isinstance(item,str)else item['token'])
    sdk='/Users/andriashadrin/.local/share/hastycr-kaggle-cli/bin/python';python=str(Path('.venv-live/bin/python').absolute())
    report=Path(f'reports/defense_plan/{stem}_training_20261027.json')
    if report.exists():raise ValueError('Preserve existing full-pair status report')
    report.write_text(json.dumps(dict(schema='global-full-training-status-v1',kernel=prep['kernel'],root=str(root),
        status='submitted_pending_cuda_proof',launch_receipt=receipt,target_steps=8192000,
        children=[dict(id=f"global{i}_full_{c['selected']['trial']['id']}_s64",trial=c['selected']['trial'],
            origin=c['selected']['study'],target_steps=4096000)for i,c in enumerate(prep['children'])],
        progress_snapshot={},quality_accepted=False,live_checkpoint_changed=False),indent=2)+'\n')
    pids={}
    for label,args in commands(root,stem,owner,sdk,python).items():
        with(root/(label+'.log')).open('a')as log:
            pids[label]=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True).pid
    (root/'monitoring.json').write_text(json.dumps(dict(kernel=prep['kernel'],pids=pids,poll_seconds=1200),indent=2)+'\n')
    return pids


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--stem',required=True);parser.add_argument('--token-index',type=int,required=True)
    args=parser.parse_args();print(json.dumps(start(args.root,args.stem,args.token_index)))
