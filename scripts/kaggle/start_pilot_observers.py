"""Start bounded observers after a matching pilot launch receipt exists."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import re
import time


def start(root,stem,profile,token_index,*,report_date='20261027'):
    root=Path(root)
    if not re.fullmatch(r'[a-z0-9_]+',stem)or not re.fullmatch(r'\d{8}',report_date):
        raise ValueError('Safe report stem and YYYYMMDD date required')
    if token_index not in (0,1,2):raise ValueError('Authorized working account required')
    if(root/'monitoring.json').exists():raise ValueError('Inspect existing observers; do not duplicate')
    prepared=json.loads((root/'preparation.json').read_text());owner=prepared['config']['owner'];kernel=prepared['kernel']
    receipt=json.loads((root/'launch_receipt.json').read_text())
    if receipt.get('ref')!='/code/'+kernel or receipt.get('versionNumber')!=1:
        raise ValueError('Matching first-submission receipt required')
    env=dict(os.environ)
    # Each remote observer loads its own token. Local report generation needs none.
    env.pop('KAGGLE_API_TOKEN',None)
    sdk='/Users/andriashadrin/.local/share/hastycr-kaggle-cli/bin/python'
    python=str(Path('.venv-live/bin/python').absolute())
    protocol_path=root/'dataset/protocol.json'
    protocol=json.loads(protocol_path.read_text())if protocol_path.exists()else{}
    budget=protocol.get('total_pilot_steps',819200)
    if type(budget)is not int or budget<=0:raise ValueError('Positive pilot budget required')
    report=f'reports/defense_plan/{stem}_training_{report_date}.json'
    if Path(report).exists():raise ValueError('Inspect existing status report before starting observers')
    Path(report).write_text(json.dumps(dict(schema='continuous-focus-training-status-v1',kernel=kernel,
        status='submitted_pending_cuda_proof',progress_snapshot={},pilot_target_steps=budget,conditional_full_steps=0,
        quality_accepted=False,live_stopped=True,submitted=True,root=str(root),launch_receipt=receipt,
        observer_poll_seconds=1200,actual_cuda_ready=False,created_ns=time.time_ns(),
        frozen_bootstrap_sha256=prepared['bootstrap_sha256']),indent=2)+'\n')
    commands={
        'stream':[sdk,'-u','-m','scripts.kaggle.run_observer','--token-index',str(token_index),'--mode','stream','--',
            '--kernel',kernel,'--output',str(root/'live_stream'),
            '--api-user',owner,'--reconnect-seconds','1200','--reconnects','100'],
        'collector':[sdk,'-u','-m','scripts.kaggle.run_observer','--token-index',str(token_index),'--mode','collect','--',
            '--kernel',kernel,'--output',str(root/'collected'),
            '--profile',profile,'--api-user',owner,'--poll-seconds','1200'],
        'postprocess':[python,'-u','-m','scripts.kaggle.watch_focus_cohort','--root',str(root),'--status-report',report,
            '--verified-report',f'reports/defense_plan/{stem}_verified_{report_date}.json',
            '--summary-report',f'reports/defense_plan/{stem}_summary_{report_date}.json',
            '--markdown-report',f'reports/{stem}_results_{report_date}.md']}
    pids={};launch={}
    monitoring=dict(kernel=kernel,pids=pids,poll_seconds=1200,complete=False)
    with(root/'monitoring.json').open('x')as file:file.write(json.dumps(monitoring,indent=2)+'\n')
    for label,args in commands.items():
        with(root/(label+'.log')).open('a')as log:
            pids[label]=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True).pid
        launch[label]=dict(pid=pids[label],command=args)
        (root/'observer_launch.json').write_text(json.dumps(launch,indent=2)+'\n')
        (root/'monitoring.json').write_text(json.dumps(monitoring,indent=2)+'\n')
    monitoring['complete']=True
    (root/'monitoring.json').write_text(json.dumps(monitoring,indent=2)+'\n')
    return pids


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stem',required=True)
    p.add_argument('--profile',required=True);p.add_argument('--token-index',type=int,required=True)
    p.add_argument('--report-date',default='20261027')
    a=p.parse_args();print(json.dumps(start(a.root,a.stem,a.profile,a.token_index,report_date=a.report_date)))
