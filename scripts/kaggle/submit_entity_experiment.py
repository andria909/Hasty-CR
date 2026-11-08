"""Private, one-shot transport and 20-minute collection for entity pilots."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from scripts.kaggle.entity_experiment import ACCOUNTS, ARCHS, SEEDS, STEPS, sha


BOOTSTRAP = '''import hashlib,json,tarfile,sys
from pathlib import Path
config=CONFIG
paths=[p for p in Path('/kaggle/input').rglob('manifest.json') if p.parent.name==config['slug']]
if len(paths)!=1:raise ValueError('Expected unique private input dataset')
root=paths[0].parent
for name,digest in config['pins'].items():
 if name=='source.tar.gz'and not(root/name).exists():continue
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen input differs: '+name)
manifest=json.loads((root/'manifest.json').read_text())
target=Path('/kaggle/working/entity_project');target.mkdir(exist_ok=False)
if (root/'source.tar.gz').exists():
 with tarfile.open(root/'source.tar.gz')as stream:
  members=stream.getmembers()
  if len(members)!=len(manifest['files'])or {m.name for m in members}!=set(manifest['files']):raise ValueError('Archive membership differs')
  if any(not m.isfile()or Path(m.name).is_absolute()or '..'in Path(m.name).parts for m in members):raise ValueError('Unsafe member')
  stream.extractall(target,filter='data')
else:
 import shutil
 expanded=root/'source'
 if {str(p.relative_to(expanded))for p in expanded.rglob('*')if p.is_file()}!=set(manifest['files']):raise ValueError('Expanded source membership differs')
 for name in manifest['files']:
  p=expanded/name
  if p.is_symlink()or not p.resolve().is_relative_to(expanded.resolve()):raise ValueError('Unsafe expanded source')
  dst=target/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
for name,digest in manifest['files'].items():
 if hashlib.sha256((target/name).read_bytes()).hexdigest()!=digest:raise ValueError('Extracted source differs')
import os
os.chdir(target);sys.path.insert(0,str(target))
print(json.dumps(dict(event='INPUTS_VERIFIED',protocol_sha256=config['pins']['protocol.json'],source_sha256=config['pins']['manifest.json'])),flush=True)
from scripts.kaggle.entity_experiment import remote
remote(root,config['case'])
'''


def client(case):
    from scripts.kaggle.collect_final_evaluation import make_api_client, configure_http_timeout
    index,owner=ACCOUNTS[case]
    item=json.loads((Path.home()/'.kaggle/access_tokens.json').read_text())['tokens'][index]
    os.environ['KAGGLE_API_TOKEN']=(item if isinstance(item,str)else item['token']).strip()
    configure_http_timeout()
    return make_api_client(owner)


def check_preflight(root):
    prep=json.loads((root/'preparation.json').read_text())
    proof=json.loads((root/'preflight.json').read_text())
    if proof['pins']!=prep['pins']or {r['model']for r in proof['proof']}!=set(ARCHS):
        raise ValueError('All three frozen architectures require actual optimizer/resume/eval preflight')
    for row in proof['proof']:
        if row['steps']!=32 or row['validation_games']!=2 or row['architecture']!=ARCHS[row['model']]:
            raise ValueError('Incomplete preflight')
        cp=root/'preflight'/(row['model']+'_whole')/'update_000002.pt'
        if sha(cp)!=row['checkpoint_sha256']:raise ValueError('Preflight checkpoint changed')
    for name,digest in prep['pins'].items():
        if sha(root/'dataset'/name)!=digest:raise ValueError('Input changed after preflight')
    return prep


def submit(root,case):
    from scripts.kaggle.submit_prepared_pilot import wait_for_dataset, validate_launch_receipt
    root=Path(root).resolve();prep=check_preflight(root)
    _,owner=ACCOUNTS[case];slug='hastycr-'+case+'-1108';kernel=owner+'/'+slug
    launch=root/case;launch.mkdir(exist_ok=True)
    if (launch/'kernel_push_requested.json').exists():raise RuntimeError('Existing request; inspect exact kernel, never duplicate')
    dataset=launch/'dataset';job=launch/'job'
    if not dataset.exists():
        shutil.copytree(root/'dataset',dataset)
        (dataset/'dataset-metadata.json').write_text(json.dumps(dict(id=owner+'/'+slug+'-inputs',title=slug+'-inputs',licenses=[{'name':'CC0-1.0'}]),indent=2)+'\n')
    job.mkdir(exist_ok=True)
    cfg=dict(pins=prep['pins'],slug=slug+'-inputs',case=case)
    (job/'train.py').write_text(BOOTSTRAP.replace('CONFIG',repr(cfg)))
    (job/'kernel-metadata.json').write_text(json.dumps(dict(id=kernel,title=slug,code_file='train.py',language='python',
        kernel_type='script',is_private=True,enable_gpu=True,enable_internet=False,
        dataset_sources=[owner+'/'+slug+'-inputs'],competition_sources=[],kernel_sources=[]),indent=2)+'\n')
    (launch/'manifest.json').write_text(json.dumps(dict(kernel=kernel,case=case,owner=owner,
        architecture=ARCHS[case],seeds=SEEDS,steps_each=STEPS,pins=prep['pins'],bootstrap_sha256=sha(job/'train.py')),indent=2)+'\n')
    api=client(case)
    for item in api.kernels_list(mine=True,page_size=1,sort_by='dateRun')or []:
        ref=item.to_dict().get('ref')
        if ref and not ref.startswith(owner+'/'):raise ValueError('Credential owner differs')
    if not(launch/'dataset_create_requested.json').exists():
        (launch/'dataset_create_requested.json').write_text(json.dumps(dict(requested=True)))
        reply=api.dataset_create_new(str(dataset),public=False,quiet=True,convert_to_csv=False,dir_mode='tar')
        (launch/'dataset_receipt.json').write_text(json.dumps(reply.to_dict(),indent=2)+'\n')
    wait_for_dataset(api,owner+'/'+slug+'-inputs')
    (launch/'kernel_push_requested.json').write_text(json.dumps(dict(kernel=kernel,requested=True,host_ns=time.time_ns())))
    reply=api.kernels_push(str(job),acc='NvidiaTeslaT4').to_dict()
    (launch/'launch_receipt.json').write_text(json.dumps(reply,indent=2)+'\n')
    validate_launch_receipt(reply,kernel)
    print(json.dumps(dict(event='SUBMITTED',case=case,kernel=kernel,version=reply['versionNumber'])),flush=True)


def repair_transport(root,case):
    """Repair a proven pre-PPO failure or still-queued broken bootstrap in place."""
    from scripts.kaggle.recover_defense_outputs import terminal_state
    root=Path(root).resolve();prep=check_preflight(root);launch=root/case
    meta=json.loads((launch/'manifest.json').read_text());api=client(case)
    events=launch/'stream/events.jsonl'
    logs=events.read_text()if events.exists()else ''
    state=terminal_state(api.kernels_status(meta['kernel']))
    if state in ('ERROR','FAILED') and 'FileNotFoundError'not in logs:
        raw=api.kernels_logs(meta['kernel']+'/1')
        (launch/'remote_v1_log.txt').write_text(raw);logs+=raw
    if state=='QUEUED':
        original=(launch/'job/train.py').read_text()
        if (sha(launch/'job/train.py')!=meta['bootstrap_sha256'] or
                "expanded=root/'source'"in original or 'TRIAL_START'in logs or
                json.loads((launch/'launch_receipt.json').read_text()).get('versionNumber')!=1):
            raise ValueError('Queued job is not the known broken first bootstrap')
    elif state not in ('ERROR','FAILED') or ('FileNotFoundError'not in logs or 'source.tar.gz'not in logs or
                                            'TRIAL_START'in logs or 'CUDA_READY'in logs):
        raise ValueError('Need queued initial job or actual archive failure before training: '+state)
    if (launch/'transport_repair_requested.json').exists():raise ValueError('Repair already requested')
    job=launch/'job';shutil.copyfile(job/'train.py',launch/'bootstrap_v1.py')
    cfg=dict(pins=prep['pins'],slug=meta['kernel'].split('/')[1]+'-inputs',case=case)
    (job/'train.py').write_text(BOOTSTRAP.replace('CONFIG',repr(cfg)))
    (launch/'transport_repair_requested.json').write_text(json.dumps(dict(
        kernel=meta['kernel'],previous_status=state,reason='Kaggle expanded source archive; initial run queued or failed before PPO',
        old_bootstrap_sha256=sha(launch/'bootstrap_v1.py'),new_bootstrap_sha256=sha(job/'train.py')),
        indent=2)+'\n')
    receipt=api.kernels_push(str(job),acc='NvidiaTeslaT4').to_dict()
    (launch/'launch_receipt_v2.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if receipt.get('error')or receipt.get('ref')!='/code/'+meta['kernel']or receipt.get('versionNumber')!=2:
        raise RuntimeError('Version2 not confirmed; inspect receipt before retry')
    print(json.dumps(dict(event='TRANSPORT_REPAIRED',kernel=meta['kernel'],version=2)),flush=True)


def collect(api,kernel,launch):
    from scripts.kaggle.recover_defense_outputs import list_outputs
    from scripts.kaggle.local_artifact_transfer import download_preserving
    files,_=list_outputs(api,kernel);out=launch/'collected';out.mkdir(exist_ok=True)
    case=json.loads((launch/'manifest.json').read_text())['case']
    names=['complete.json']+[f'{case}_s{s}_summary.json'for s in SEEDS]
    def download(name):
        if Path(name).is_absolute()or '..'in Path(name).parts:raise ValueError('Unsafe artifact')
        key='entity_runs/'+name
        if key not in files:raise ValueError('Missing terminal artifact: '+name)
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():download_preserving(files[key].url,p)
        return p
    for name in names:download(name)
    results=[]
    for seed in SEEDS:
        identity=f'{case}_s{seed}'
        summary=json.loads((out/(identity+'_summary.json')).read_text())
        if summary['step']!=STEPS or summary['architecture']!=ARCHS[case]:raise ValueError('Terminal budget/architecture differs')
        cp=download(summary['checkpoint'])
        if sha(cp)!=summary['checkpoint_sha256']:raise ValueError('Checkpoint checksum differs')
        for name in ('config.json','training_result.json','metrics.jsonl'):download(identity+'/'+name)
        metrics=[json.loads(l)for l in (out/identity/'metrics.jsonl').read_text().splitlines()]
        if metrics[-1]['step']!=STEPS or not all(r.get('actual_cuda')for r in metrics):raise ValueError('CUDA budget not verified')
        selected=[r for r in metrics if r['update']==summary['selected_update']]
        if len(selected)!=1 or selected[0]['validation']!=summary['validation']:raise ValueError('Selected validation differs')
        results.append(dict(id=identity,wins=summary['validation']['wins'],games=summary['validation']['games'],
                            score=summary['validation']['selection_score'],checkpoint=summary['checkpoint'],sha256=sha(cp)))
    (out/'collection_receipt.json').write_text(json.dumps(dict(kernel=kernel,steps=2*STEPS,results=results,
        files={str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()and p.name!='collection_receipt.json'}),indent=2)+'\n')
    return results


def watch(root,case):
    from scripts.kaggle.recover_defense_outputs import terminal_state, TERMINAL
    launch=Path(root).resolve()/case;meta=json.loads((launch/'manifest.json').read_text());api=client(case)
    while True:
        try:
            status=terminal_state(api.kernels_status(meta['kernel']))
            row=dict(kernel=meta['kernel'],status=status,checked_host_ns=time.time_ns(),poll_seconds=1200)
            (launch/'status.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
            if status=='COMPLETE':
                print(json.dumps(dict(event='COLLECTED',results=collect(api,meta['kernel'],launch))),flush=True);return
            if status in TERMINAL:return
        except Exception as exc:
            print(json.dumps(dict(event='POLL_OR_COLLECTION_ERROR',type=type(exc).__name__)),flush=True)
        deadline=time.time()+1200
        while time.time()<deadline:
            time.sleep(min(30,max(0,deadline-time.time())))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('submit','watch','repair-transport'));p.add_argument('--root',required=True);p.add_argument('--case',choices=tuple(ARCHS),required=True)
    a=p.parse_args();{'submit':submit,'watch':watch,'repair-transport':repair_transport}[a.mode](a.root,a.case)
