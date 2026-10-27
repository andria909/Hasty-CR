"""Observe an already submitted evaluation; no job creation or promotion."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from scripts.kaggle.validate_evaluation_submission import validate_submission


def commands(root,binding,report,owner,sdk,python,kernel):
    return dict(
        stream=[sdk,'-m','scripts.kaggle.stream_training','--kernel',kernel,'--output',str(root/'live_stream'),
                '--api-user',owner,'--reconnect-seconds','1200','--reconnects','100'],
        collector=[sdk,'-m','scripts.kaggle.collect_full_candidate_evaluation','--prepared',str(root),
                   '--output',str(root/'collected'),'--api-user',owner,'--poll-seconds','1200'],
        postprocess=[python,'-m','scripts.attack.report_full_candidate_evaluation','--prepared',str(root),
                     '--binding',str(binding),'--collected',str(root/'collected'),'--report',str(report),'--watch'])


def start(root,binding,report,token_index):
    root,binding,report=map(Path,(root,binding,report))
    if token_index not in (0,1,2):raise ValueError('Only three working accounts allowed')
    if (root/'monitoring.json').exists():raise ValueError('Inspect existing observers; no duplicates')
    prep=json.loads((root/'preparation.json').read_text());validate_submission(root,prep)
    receipt=json.loads((root/'launch_receipt.json').read_text())
    if receipt.get('ref')!='/code/'+prep['kernel'] or receipt.get('versionNumber')!=1:
        raise ValueError('Matching first-version evaluation launch required')
    from scripts.kaggle.local_artifact_transfer import file_sha
    if file_sha(binding/'protocol.json')!=prep['config']['files']['protocol.json']:
        raise ValueError('Observer binding differs from submitted evaluation')
    if report.exists() or report.with_suffix('.md').exists():raise ValueError('Preserve existing evaluation reports')
    owner=prep['config']['owner']
    if owner!=('andriashadrin','vertel0','andreiperelaiko')[token_index]:
        raise ValueError('Token index does not match prepared account')
    item=json.loads((Path.home()/'.kaggle/access_tokens.json').read_text())['tokens'][token_index]
    env=dict(os.environ,KAGGLE_API_TOKEN=(item if isinstance(item,str)else item['token']).strip())
    sdk='/Users/andriashadrin/.local/share/hastycr-kaggle-cli/bin/python'
    python=str(Path('.venv-live/bin/python').absolute())
    state=dict(kernel=prep['kernel'],pids={},poll_seconds=1200,report=str(report),remote_job_changed=False)
    marker=root/'monitoring.json'
    with marker.open('x')as stream:json.dump(state,stream,indent=2)
    # Persist each PID immediately; a partial local start is inspectable and
    # must never lead to a duplicate remote evaluation or duplicate observers.
    for name,args in commands(root,binding,report,owner,sdk,python,prep['kernel']).items():
        with(root/(name+'.log')).open('a')as log:
            state['pids'][name]=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,
                env=env if name!='postprocess' else dict(os.environ),start_new_session=True).pid
        marker.write_text(json.dumps(state,indent=2)+'\n')
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('root','binding','report'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--token-index',type=int,required=True)
    args=parser.parse_args();print(json.dumps(start(args.root,args.binding,args.report,args.token_index)))
