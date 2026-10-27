"""Start bounded observers after a matching pilot launch receipt exists."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def start(root,stem,profile,token_index):
    root=Path(root)
    if(root/'monitoring.json').exists():raise ValueError('Inspect existing observers; do not duplicate')
    prepared=json.loads((root/'preparation.json').read_text());owner=prepared['config']['owner'];kernel=prepared['kernel']
    receipt=json.loads((root/'launch_receipt.json').read_text())
    if receipt.get('ref')!='/code/'+kernel or receipt.get('versionNumber')!=1:
        raise ValueError('Matching first-submission receipt required')
    item=json.loads((Path.home()/'.kaggle/access_tokens.json').read_text())['tokens'][token_index]
    env=dict(os.environ,KAGGLE_API_TOKEN=item if isinstance(item,str)else item['token'])
    sdk='/Users/andriashadrin/.local/share/hastycr-kaggle-cli/bin/python'
    python=str(Path('.venv-live/bin/python').absolute())
    report=f'reports/defense_plan/{stem}_training_20261027.json'
    Path(report).write_text(json.dumps(dict(schema='continuous-focus-training-status-v1',kernel=kernel,
        status='submitted_pending_cuda_proof',progress_snapshot={},pilot_target_steps=819200,conditional_full_steps=0,
        quality_accepted=False,live_stopped=True,submitted=True,root=str(root),launch_receipt=receipt,
        frozen_bootstrap_sha256=prepared['bootstrap_sha256']),indent=2)+'\n')
    commands={
        'stream':[sdk,'-m','scripts.kaggle.stream_training','--kernel',kernel,'--output',str(root/'live_stream'),
            '--api-user',owner,'--reconnect-seconds','1200','--reconnects','100'],
        'collector':[sdk,'-m','scripts.kaggle.collect_final_evaluation','--kernel',kernel,'--output',str(root/'collected'),
            '--profile',profile,'--api-user',owner,'--poll-seconds','1200'],
        'postprocess':[python,'-m','scripts.kaggle.watch_focus_cohort','--root',str(root),'--status-report',report,
            '--verified-report',f'reports/defense_plan/{stem}_verified_20261027.json',
            '--summary-report',f'reports/defense_plan/{stem}_summary_20261027.json',
            '--markdown-report',f'reports/{stem}_results_20261027.md']}
    pids={}
    for label,args in commands.items():
        with(root/(label+'.log')).open('a')as log:
            pids[label]=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True).pid
    (root/'monitoring.json').write_text(json.dumps(dict(kernel=kernel,pids=pids,poll_seconds=1200),indent=2)+'\n')
    return pids


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stem',required=True)
    p.add_argument('--profile',required=True);p.add_argument('--token-index',type=int,required=True)
    a=p.parse_args();print(json.dumps(start(a.root,a.stem,a.profile,a.token_index)))
