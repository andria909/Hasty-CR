"""Preserve bounded recovery artifacts from a terminal full-pair job; never resume it."""
import argparse
import json
import math
from pathlib import Path
import re

from scripts.kaggle.local_artifact_transfer import download_preserving,copy_verified_cache,file_sha
from scripts.kaggle.recover_defense_outputs import list_outputs,terminal_state,TERMINAL


def logged_prefix(raw):
    lines=raw.decode().splitlines(keepends=True);rows=[]
    for i,line in enumerate(lines):
        try:row=json.loads(line)
        except ValueError:
            if i==len(lines)-1 and not line.endswith('\n'):break
            raise ValueError('Corrupt interior metrics line')
        if not isinstance(row,dict)or type(row.get('update'))is not int or row['update']!=len(rows)+1:
            raise ValueError('Nonconsecutive saved metrics')
        rows.append(row)
    return rows


def choose_boundaries(names,prefix,rows):
    candidates={int(m.group(1)):name for name in names
        if (m:=re.fullmatch(re.escape(prefix)+r'update_(\d{6})\.pt',name)) and 1<=int(m.group(1))<=4000}
    usable=[u for u in candidates if u<=len(rows)]
    if not usable:return None
    latest=max(usable);validations=[r for r in rows[:latest]if 'validation'in r]
    for row in validations:
        v=row['validation']
        if not isinstance(v,dict)or any(type(v.get(k))not in(int,float)or not math.isfinite(v[k])for k in ('selection_score','worst_archetype_points')):
            raise ValueError('Invalid saved validation score')
    eligible=[r for r in validations if r['validation']['worst_archetype_points']>=.1]
    selected=max(eligible or validations,key=lambda r:r['validation']['selection_score'])if validations else None
    if selected and selected['update']not in candidates:
        raise ValueError('Historical selected checkpoint missing')
    return dict(latest_update=latest,latest_checkpoint=candidates[latest],
        selected_update=selected['update']if selected else None,
        selected_checkpoint=candidates[selected['update']]if selected else None,
        selected_with_family_guard=bool(eligible),metrics_rows=len(rows),
        metrics_rows_after_boundary=len(rows)-latest)


def collect(api,prepared,destination,cache=None):
    prepared,destination=Path(prepared),Path(destination)
    prep=json.loads((prepared/'preparation.json').read_text());kernel=prep['kernel']
    state=terminal_state(api.kernels_status(kernel))
    if state not in TERMINAL:raise ValueError('Authoritative terminal state required for recovery collection')
    files,_=list_outputs(api,kernel)
    if terminal_state(api.kernels_status(kernel))!=state:raise ValueError('Remote state changed')
    cached={}
    if cache is not None:
        cache=Path(cache);previous=json.loads((cache/'collection_receipt.json').read_text())
        if (previous.get('schema')!='global-full-recovery-artifacts-v1'or previous['kernel']!=kernel
                or previous['remote_status']!=state or previous.get('prepared_sha256')!=file_sha(prepared/'preparation.json')):
            raise ValueError('Recovery cache identity differs')
        cached=previous['files']
    if len(prep['children'])!=2:raise ValueError('Exactly two isolated children required')
    destination.mkdir(parents=True,exist_ok=False)
    receipt=dict(schema='global-full-recovery-artifacts-v1',kernel=kernel,remote_status=state,
        complete=False,artifacts_complete=False,files={},children=[],
        prepared_sha256=file_sha(prepared/'preparation.json'),training_budget_verified=False,
        automatic_resume=False,quality_accepted=False,remote_job_changed=False)
    def fetch(name):
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        if name not in receipt['files']:
            if not copy_verified_cache(cache,cached,name,target):download_preserving(files[name].url,target,stream=False)
            receipt['files'][name]=dict(sha256=file_sha(target),bytes=target.stat().st_size)
        return target
    try:
        for index,child in enumerate(prep['children']):
            if child.get('index')!=index or child.get('remote_output')!=f'global_full_child_{index}/continuous_focus':
                raise ValueError('Unsafe child namespace')
            trial=child['selected']['trial']['id']
            if not re.fullmatch(r'[a-z0-9_]+',trial):raise ValueError('Unsafe profile id')
            run='full_'+trial+'_s64';base=child['remote_output']+'/';prefix=base+run+'/'
            entry=dict(index=index,run=run,prefix=prefix,recovery_boundary_available=False)
            for name in (base+run+'_summary.json',base+'result.json',f'global_full_child_{index}_driver.log'):
                if name in files:fetch(name)
            for name in ('config.json','metrics.jsonl','training_result.json'):
                if prefix+name in files:fetch(prefix+name)
            if child['selected']['trial'].get('warmstart_updates',0):
                for name in ('policy.pt','result.json','updates.json'):
                    if base+run+'_warmstart/'+name in files:fetch(base+run+'_warmstart/'+name)
            if prefix+'metrics.jsonl'in files and prefix+'config.json'in files:
                rows=logged_prefix(fetch(prefix+'metrics.jsonl').read_bytes())
                choice=choose_boundaries(files,prefix,rows)
                if choice:
                    fetch(choice['latest_checkpoint'])
                    if choice['selected_checkpoint']:fetch(choice['selected_checkpoint'])
                    entry.update(choice,recovery_boundary_available=True)
            receipt['children'].append(entry)
        receipt['artifacts_complete']=True
    finally:
        (destination/'collection_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    from scripts.kaggle.collect_final_evaluation import configure_http_timeout,make_api_client
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',required=True);p.add_argument('--output',required=True);p.add_argument('--api-user',required=True)
    a=p.parse_args();configure_http_timeout();api=make_api_client(a.api_user)
    print(json.dumps(collect(api,a.prepared,a.output)))
