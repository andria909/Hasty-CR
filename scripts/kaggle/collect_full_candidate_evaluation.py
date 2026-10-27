"""Collect allowlisted evaluation results, retaining completed parts after failure.

SDK-only process: game/metric validation runs separately in the project venv.
No polling or observation failure ever changes or restarts the remote job.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import tarfile
import time

from scripts.kaggle.collect_final_evaluation import make_api_client, configure_http_timeout, preserve_partial
from scripts.kaggle.local_artifact_transfer import download_preserving, copy_verified_cache, file_sha
from scripts.kaggle.recover_defense_outputs import list_outputs, terminal_state, TERMINAL

PREFIX = 'full_candidate_evaluation/'


def plan(prepared):
    prepared = Path(prepared)
    prep = json.loads((prepared / 'preparation.json').read_text())
    if file_sha(prepared / 'job/train.py') != prep['bootstrap_sha256']:
        raise ValueError('Evaluation bootstrap changed')
    config = prep['config']; archive_path = prepared / 'evaluation_inputs.tar.gz'
    if file_sha(archive_path) != config['archive_sha256']:
        raise ValueError('Prepared evaluation archive changed')
    with tarfile.open(archive_path) as archive:
        def read(name):
            raw = archive.extractfile(name).read()
            if hashlib.sha256(raw).hexdigest() != config['files'][name]:
                raise ValueError('Prepared evaluation payload changed')
            return json.loads(raw)
        protocol = read('protocol.json'); blueprint = read('weights/evaluation_blueprint.json')
    parts = {f"test__{test['id']}__{model['id']}.json" for model in protocol['models'] for test in blueprint['test_sets']}
    parts.update(f"duel__{a['id']}__{b['id']}.json" for a,b in itertools.combinations(protocol['models'], 2))
    if any(Path(n).name != n or '\\' in n for n in parts):
        raise ValueError('Unsafe evaluation part names')
    return prep, parts


def collect(api, prepared, destination, cache=None):
    prep, parts = plan(prepared); destination = Path(destination)
    kernel = prep['kernel']; state = terminal_state(api.kernels_status(kernel))
    if state not in TERMINAL:
        raise ValueError('Authoritative terminal state required')
    remote, _ = list_outputs(api, kernel)
    if terminal_state(api.kernels_status(kernel)) != state:
        raise ValueError('Remote evaluation state changed')
    metadata = {'run_identity.json', 'source_manifest.json'}
    all_names = metadata | {'result.json'} | parts | {n + '.receipt.json' for n in parts}
    available = {n for n in all_names if PREFIX + n in remote}
    if state == 'COMPLETE':
        if available != all_names: raise ValueError('Successful evaluation is missing required outputs')
        selected = all_names
    else:
        # Failed startup may have no identity and no reusable parts at all.
        selected = metadata if metadata <= available else set()
        if selected:
            for name in parts:
                if {name, name + '.receipt.json'} <= available:
                    selected.update((name, name + '.receipt.json'))
    cached = {}
    if cache is not None:
        row = json.loads((Path(cache) / 'collection_receipt.json').read_text())
        if row['kernel'] != kernel or row['remote_status'] != state or row['protocol_sha256'] != prep['config']['files']['protocol.json']:
            raise ValueError('Recovery cache belongs to another evaluation')
        cached = row['files']
    destination.mkdir(parents=True, exist_ok=False)
    receipt = dict(kernel=kernel, remote_status=state, complete=False, transfer_complete=False,
                   protocol_sha256=prep['config']['files']['protocol.json'],
                   source_manifest_sha256=prep['config']['source_manifest_sha256'], files={},
                   remote_job_changed=False, parts_validated=False)
    try:
        for name in sorted(selected):
            path = destination / name
            if copy_verified_cache(cache, cached, name, path):
                receipt['files'][name] = dict(sha256=file_sha(path), bytes=path.stat().st_size)
        for name in sorted(selected):
            if name in receipt['files']: continue
            path = destination / name
            download_preserving(remote[PREFIX + name].url, path, stream=False)
            receipt['files'][name] = dict(sha256=file_sha(path), bytes=path.stat().st_size)
        if selected:
            identity = json.loads((destination / 'run_identity.json').read_text())
            if (identity['protocol_sha256'] != receipt['protocol_sha256']
                    or identity['source_manifest_sha256'] != receipt['source_manifest_sha256']
                    or file_sha(destination / 'source_manifest.json') != receipt['source_manifest_sha256']):
                raise ValueError('Downloaded evaluation identity differs')
            for name in parts & selected:
                part_receipt = json.loads((destination / (name + '.receipt.json')).read_text())
                if (part_receipt['file'] != name or part_receipt['sha256'] != file_sha(destination / name)
                        or part_receipt['protocol_sha256'] != receipt['protocol_sha256']
                        or part_receipt['source_manifest_sha256'] != receipt['source_manifest_sha256']):
                    raise ValueError('Downloaded part receipt differs')
        receipt['transfer_complete'] = True
        receipt['complete'] = state == 'COMPLETE'
        receipt['downloaded_parts'] = len(parts & selected)
    finally:
        (destination / 'collection_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('prepared', 'output', 'api-user'): parser.add_argument('--' + name, required=True)
    parser.add_argument('--poll-seconds', type=int, default=1200)
    args = parser.parse_args()
    if args.poll_seconds < 1200: parser.error('Remote observations must be at least 20 minutes apart')
    configure_http_timeout(); api = make_api_client(args.api_user)
    kernel = json.loads((Path(args.prepared) / 'preparation.json').read_text())['kernel']
    while True:
        try:
            state = terminal_state(api.kernels_status(kernel))
            status = dict(kernel=kernel, remote_status=state, checked_host_ns=time.time_ns(), remote_job_changed=False)
            path = Path(args.output).with_suffix('.status.json'); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(status, indent=2) + '\n'); print(json.dumps(status), flush=True)
            if state in TERMINAL:
                cache = preserve_partial(args.output, kernel, state) if Path(args.output).exists() else None
                result = collect(api, args.prepared, args.output, cache)
                print(json.dumps(result), flush=True); break
        except Exception as error:
            code = getattr(getattr(error, 'response', None), 'status_code', None)
            if type(error).__name__ not in ('ConnectionError', 'SSLError', 'ReadTimeout', 'Timeout') and code not in (404,408,429,500,502,503,504): raise
            print(json.dumps(dict(observation_error=type(error).__name__, remote_job_changed=False)), flush=True)
        time.sleep(args.poll_seconds)
