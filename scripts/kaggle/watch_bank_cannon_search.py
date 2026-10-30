"""Observe one CPU-only search every20min and collect only its two result files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def watch(root):
    import requests
    from scripts.kaggle.collect_final_evaluation import make_api_client,configure_http_timeout
    from scripts.kaggle.recover_defense_outputs import terminal_state,list_outputs
    root=Path(root);prep=json.loads((root/'preparation.json').read_text());kernel=prep['kernel']
    ref=kernel+'/'+str(prep.get('expected_version',1))
    item=json.loads((Path.home()/'.kaggle/access_tokens.json').read_text())['tokens'][1]
    os.environ['KAGGLE_API_TOKEN']=item if isinstance(item,str)else item['token']
    configure_http_timeout();api=make_api_client('vertel0');deadline=time.monotonic()+86400
    while time.monotonic()<deadline:
        try:
            state=terminal_state(api.kernels_status(ref))
            (root/'status.json').write_text(json.dumps(dict(kernel=kernel,status=state,checked_host_ns=time.time_ns()),indent=2)+'\n')
            print(state,flush=True)
            if state=='COMPLETE':
                files,logs=list_outputs(api,ref);out=root/'collected';out.mkdir(exist_ok=True)
                receipt=dict(kernel=kernel,complete=False,files={})
                for name in ('bank_cannon_search/contexts.jsonl','bank_cannon_search/result.json'):
                    if name not in files:raise ValueError('Missing search output '+name)
                    response=requests.get(files[name].url,timeout=(30,120));response.raise_for_status()
                    raw=response.content;target=out/Path(name).name;target.write_bytes(raw)
                    receipt['files'][name]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
                result=json.loads((out/'result.json').read_text())
                expected_schema='bank-timing-search-v1' if prep.get('timing') else 'bank-central-cannon-search-v1'
                if result['schema']!=expected_schema:raise ValueError('Unexpected search protocol')
                if bool(result.get('support',False))!=bool(prep.get('support',False)):
                    raise ValueError('Support condition differs from prepared study')
                if result['contexts']!=36 or hashlib.sha256((out/'contexts.jsonl').read_bytes()).hexdigest()!=result['raw_sha256']:
                    raise ValueError('Incomplete or changed CPU search')
                receipt['complete']=True;(out/'collection_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
                print(json.dumps(result.get('summary',result)),flush=True);return
            if state in ('ERROR','CANCELLED','CANCELED'):
                files,logs=list_outputs(api,ref)
                (root/'terminal_logs.json').write_text(json.dumps(logs,indent=2)+'\n');return
        except Exception as exc:
            print(json.dumps(dict(observation_error=type(exc).__name__,remote_job_changed=False)),flush=True)
        time.sleep(1200)
    print('Observer deadline expired; remote job unchanged',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();watch(a.root)
