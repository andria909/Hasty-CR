"""Collect only selected/latest weights from two isolated terminal full runs."""
import argparse
import json
from pathlib import Path
import re
import time

from scripts.kaggle.collect_final_evaluation import make_api_client, configure_http_timeout, preserve_partial
from scripts.kaggle.local_artifact_transfer import download_preserving, copy_verified_cache, file_sha
from scripts.kaggle.recover_defense_outputs import list_outputs, terminal_state, TERMINAL


def collect(api, prepared, destination, cache=None):
    prepared, destination = Path(prepared), Path(destination)
    prep = json.loads((prepared/'preparation.json').read_text())
    kernel = prep['kernel']; state = terminal_state(api.kernels_status(kernel))
    if state != 'COMPLETE':
        raise ValueError('Successful authoritative terminal state required')
    files, _ = list_outputs(api, kernel)
    if terminal_state(api.kernels_status(kernel)) != state:
        raise ValueError('Remote state changed')
    cached = {}
    if cache is not None:
        cache = Path(cache); row = json.loads((cache/'collection_receipt.json').read_text())
        if row['kernel'] != kernel or row['remote_status'] != state:
            raise ValueError('Cache belongs to another job/state')
        cached = row['files']
    allowed = {'global_full_result.json'}; prefetched = {}
    if len(prep['children']) != 2:
        raise ValueError('Exactly two isolated children required')
    for index, child in enumerate(prep['children']):
        if child.get('index') != index or child.get('remote_output') != f'global_full_child_{index}/continuous_focus':
            raise ValueError('Unsafe or changed child output namespace')
        prefix = child['remote_output']+'/'
        run = 'full_'+child['selected']['trial']['id']+'_s64'
        summary = prefix+run+'_summary.json'
        if summary not in files:
            raise ValueError('Selected child summary missing')
        # Prefetch only the bounded JSON needed to identify the chosen weight.
        if cache is not None and summary in cached:
            path = cache/summary
            if path.is_symlink() or not path.resolve().is_relative_to(cache.resolve()) or file_sha(path) != cached[summary]['sha256']:
                raise ValueError('Unsafe or changed cached summary')
            raw = path.read_bytes()
        else:
            import requests
            response = requests.get(files[summary].url, timeout=(30,300)); response.raise_for_status(); raw = response.content
        row = json.loads(raw); checkpoint = row.get('checkpoint')
        if row.get('id') != run or not isinstance(checkpoint,str) or not re.fullmatch(re.escape(run)+r'/update_\d{6}\.pt',checkpoint):
            raise ValueError('Cross-child or malformed checkpoint reference')
        if not 1 <= int(checkpoint[-9:-3]) <= 4000:
            raise ValueError('Selected checkpoint exceeds full budget')
        prefetched[summary] = raw
        allowed.update({summary, prefix+'result.json', prefix+checkpoint,
                        prefix+run+'/update_004000.pt'})
        allowed.update(prefix+run+'/'+name for name in ('config.json','metrics.jsonl','training_result.json'))
        if child['selected']['trial'].get('warmstart_updates',0):
            allowed.update(prefix+run+'_warmstart/'+name for name in ('policy.pt','result.json','updates.json'))
    if not allowed <= set(files):
        raise ValueError('Required paired full artifacts missing')
    destination.mkdir(parents=True, exist_ok=False)
    receipt = dict(kernel=kernel, remote_status=state, complete=False, files={}, remote_job_changed=False)
    try:
        for name in sorted(allowed):
            target = destination/name; target.parent.mkdir(parents=True, exist_ok=True)
            if copy_verified_cache(cache,cached,name,target):
                receipt['files'][name] = dict(sha256=file_sha(target),bytes=target.stat().st_size)
        for name in sorted(allowed):
            if name in receipt['files']:
                continue
            target = destination/name
            if name in prefetched:
                target.write_bytes(prefetched[name])
            else:
                download_preserving(files[name].url,target,stream=False)
            receipt['files'][name] = dict(sha256=file_sha(target),bytes=target.stat().st_size)
        receipt['complete'] = True
    finally:
        (destination/'collection_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--api-user',required=True); parser.add_argument('--poll-seconds',type=int,default=1200)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error('Positive polling interval required')
    configure_http_timeout(); api = make_api_client(args.api_user)
    kernel = json.loads((Path(args.prepared)/'preparation.json').read_text())['kernel']
    cache = None
    while True:
        try:
            state = terminal_state(api.kernels_status(kernel))
            print(json.dumps(dict(kernel=kernel,remote_status=state,remote_job_changed=False)),flush=True)
            if state in TERMINAL:
                if state != 'COMPLETE':
                    raise ValueError('Full job ended unsuccessfully; inspect outputs')
                if Path(args.output).exists():
                    cache = preserve_partial(args.output,kernel,state)
                receipt = collect(api,args.prepared,args.output,cache=cache)
                print(json.dumps(dict(complete=receipt['complete'],files=len(receipt['files']))),flush=True)
                break
        except Exception as error:
            status = getattr(getattr(error,'response',None),'status_code',None)
            if type(error).__name__ not in ('ConnectionError','SSLError','ReadTimeout','Timeout') and status not in (404,408,429,500,502,503,504):
                raise
            print(json.dumps(dict(observation_error=type(error).__name__,remote_job_changed=False)),flush=True)
        time.sleep(args.poll_seconds)
