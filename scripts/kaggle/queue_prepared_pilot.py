"""Run a frozen independent experiment on the first VERIFIED free account.

The unavailable fourth account is never retried. No remote polling while
waiting: existing collectors supply terminal, verified predecessor proofs.
"""
import argparse
import ast
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rebind(source,output,owner):
    source=Path(source);output=Path(output)
    if not re.fullmatch(r'[a-z0-9_]{3,50}',owner):raise ValueError('Invalid explicit owner')
    original=json.loads((source/'preparation.json').read_text());config=copy.deepcopy(original['config'])
    code=(source/'job/train.py').read_text()
    if sha(source/'job/train.py')!=original['bootstrap_sha256']:raise ValueError('Frozen bootstrap changed')
    if not json.loads((source/'preflight.json').read_text()).get('exact_resume_16_to_32'):
        raise ValueError('Exact frozen-package preflight required')
    for name,digest in config['pins'].items():
        if sha(source/'dataset'/name)!=digest:raise ValueError('Frozen dataset input changed')
    tree=ast.parse(code);last=tree.body[-1]
    if (not isinstance(last,ast.Expr)or not isinstance(last.value,ast.Call)or
            not isinstance(last.value.func,ast.Name)or last.value.func.id!='execute' or
            len(last.value.args)!=1 or ast.literal_eval(last.value.args[0])!=config):
        raise ValueError('Unexpected frozen entrypoint')
    config['owner']=owner;kernel=owner+'/'+original['kernel'].split('/')[1]
    changed=''.join(code.splitlines(keepends=True)[:last.lineno-1])+'execute('+repr(config)+')\n'
    compiled=ast.parse(changed)
    if [ast.dump(n)for n in tree.body[:-1]]!=[ast.dump(n)for n in compiled.body[:-1]]:
        raise ValueError('Rebinding changed the algorithm')
    output.mkdir(parents=True,exist_ok=False);dataset=output/'dataset';dataset.mkdir();job=output/'job';job.mkdir()
    for name in set(config['pins'])|{'source.tar.gz'}:shutil.copyfile(source/'dataset'/name,dataset/name)
    metadata=json.loads((source/'dataset/dataset-metadata.json').read_text());metadata['id']=owner+'/'+config['slug']
    (dataset/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    metadata=json.loads((source/'job/kernel-metadata.json').read_text());metadata.update(id=kernel,dataset_sources=[owner+'/'+config['slug']])
    (job/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');(job/'train.py').write_text(changed)
    result=dict(original,config=config,kernel=kernel,bootstrap_sha256=sha(job/'train.py'),
        rebound_from=str(source),algorithm_ast_unchanged=True,submitted=False)
    (output/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    shutil.copyfile(source/'preflight.json',output/'preflight.json')
    return result


def first_verified_free(slots,reserve_eligible_pairs=False,busy_owners=()):
    ready=[]
    for slot in slots:
        if slot['owner']in busy_owners:continue
        path=Path(slot['verified_report'])
        if not path.exists():continue
        try:proof=json.loads(path.read_text())
        except(ValueError,OSError):continue
        if proof.get('kernel')!=slot['kernel']or proof.get('pilot_steps_verified')!=819200:continue
        if reserve_eligible_pairs and proof.get('pilot_selection',{}).get('gate_met'):
            continue
        receipt=Path(slot['root'])/'collected/collection_receipt.json'
        if not receipt.exists():continue
        try:receipt=json.loads(receipt.read_text())
        except(ValueError,OSError):continue
        if receipt.get('kernel')!=slot['kernel']or receipt.get('remote_status')!='COMPLETE'or not receipt.get('complete'):continue
        ready.append((path.stat().st_mtime_ns,slot))
    return min(ready,key=lambda r:r[0])[1]if ready else None


def full_training_reservations(reports):
    busy=set();models=0
    for path in Path(reports).glob('global_full_pair[01]_training_20261027.json'):
        try:state=json.loads(path.read_text())
        except(ValueError,OSError):continue
        if not state.get('launch_receipt'):continue
        models+=len(state.get('children',[]))
        if state.get('status')not in('full_complete_budgets_verified','remote_terminal_error_requires_inspection'):
            busy.add(state['kernel'].split('/')[0])
    return busy,models


def watch(config_path):
    config=json.loads(Path(config_path).read_text());state_path=Path(config['state_file'])
    state_path.parent.mkdir(parents=True,exist_ok=True)
    lock=(state_path.parent/'coordinator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if state_path.exists():raise ValueError('Inspect existing queue state before restarting')
    def save(**values):
        state=dict(values,checked_host_ns=time.time_ns());state_path.write_text(json.dumps(state,indent=2)+'\n')
        return state
    save(status='waiting_verified_free_account',source=config['source'],slots=config['slots'])
    deadline=time.monotonic()+172800
    while time.monotonic()<deadline:
        busy,full_models=full_training_reservations(config.get('full_reports','reports/defense_plan'))
        reserve=config.get('reserve_eligible_pairs',False)and full_models<config.get('full_model_limit',4)
        slot=first_verified_free(config['slots'],reserve,busy_owners=busy)
        if slot is None:time.sleep(30);continue
        root=Path(config['output']);save(status='binding_same_algorithm_to_free_owner',owner=slot['owner'])
        result=rebind(config['source'],root,slot['owner'])
        save(status='submitting_on_verified_free_account',owner=slot['owner'],kernel=result['kernel'],root=str(root))
        sdk='/Users/andriashadrin/.local/share/hastycr-kaggle-cli/bin/python'
        with(root/'submit.log').open('a')as log:
            child=subprocess.run([sdk,'-m','scripts.kaggle.submit_prepared_pilot','--root',str(root),
                '--token-index',str(slot['token_index'])],stdout=log,stderr=subprocess.STDOUT)
        if child.returncode:
            save(status='submission_requires_inspection_no_duplicate',root=str(root),owner=slot['owner']);return
        from scripts.kaggle.start_pilot_observers import start
        pids=start(root,config['stem'],config['profile'],slot['token_index'])
        save(status='submitted_observers_started',kernel=result['kernel'],root=str(root),pids=pids);return
    save(status='queue_wait_expired_remote_jobs_unchanged')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();watch(a.config)
