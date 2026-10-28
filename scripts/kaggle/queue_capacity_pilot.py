"""Retry one capacity-rejected GPU job after one of our two jobs finishes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

CAPACITY = 'Maximum batch GPU session count of 2 reached.'


def capacity_available(blockers):
    for root in map(Path, blockers):
        prep = json.loads((root/'preparation.json').read_text())
        status = root/'collected.status.json'
        if not status.exists():
            continue
        try:
            row = json.loads(status.read_text())
        except ValueError:
            continue
        if row.get('kernel') == prep['kernel'] and row.get('remote_status') in ('COMPLETE', 'ERROR', 'CANCELLED', 'CANCELED'):
            return True
    return False


def retry(root, token_index):
    from scripts.kaggle.collect_final_evaluation import make_api_client, configure_http_timeout
    from scripts.kaggle.submit_prepared_pilot import validate_launch_receipt
    root = Path(root)
    receipt = json.loads((root/'launch_receipt.json').read_text())
    if receipt != {'error': CAPACITY}:
        raise ValueError('Only a confirmed capacity rejection can be retried')
    prep = json.loads((root/'preparation.json').read_text())
    if token_index != 1 or prep['config']['owner'] != 'vertel0':
        raise ValueError('This campaign queue is bound to the available authorized account')
    preflight = json.loads((root/'preflight.json').read_text())
    if not preflight.get('frozen_sources_verified') or not preflight.get('exact_resume_16_to_32'):
        raise ValueError('Verified preflight required')
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    if sha(root/'job/train.py') != prep['bootstrap_sha256']:
        raise ValueError('Controller changed while queued')
    for name, digest in prep['config']['pins'].items():
        if sha(root/'dataset'/name) != digest:
            raise ValueError('Frozen input changed while queued')
    with (root/'capacity_rejection.json').open('x') as out:
        json.dump(receipt, out)
    with (root/'capacity_retry_requested.json').open('x') as out:
        json.dump(dict(kernel=prep['kernel'], requested_ns=time.time_ns()), out)
    token = json.loads((Path.home()/'.kaggle/access_tokens.json').read_text())['tokens'][token_index]
    os.environ['KAGGLE_API_TOKEN'] = token if isinstance(token, str) else token['token']
    configure_http_timeout(); api = make_api_client(prep['config']['owner'])
    response = api.kernels_push(str(root/'job'), acc='NvidiaTeslaT4').to_dict()
    (root/'capacity_retry_receipt.json').write_text(json.dumps(response, indent=2)+'\n')
    validate_launch_receipt(response, prep['kernel'])
    (root/'launch_receipt.json').write_text(json.dumps(response, indent=2)+'\n')
    return response


def watch(root, blockers):
    from scripts.kaggle.start_pilot_observers import start
    root = Path(root)
    state_file = root/'capacity_queue.json'
    if state_file.exists():
        raise FileExistsError('Inspect existing queue; do not duplicate')
    state = dict(status='waiting_for_free_gpu_slot', blockers=blockers, poll_seconds=1200)
    def save():state_file.write_text(json.dumps(state, indent=2)+'\n')
    save()
    deadline = time.monotonic()+86400
    try:
        while time.monotonic() < deadline:
            if capacity_available(blockers):
                state.update(status='retrying_once'); save()
                response = retry(root, 1)
                pids = start(root, 'mana_wait_pilot', 'mana-decision-pilot', 1)
                state.update(status='submitted_pending_cuda_proof', receipt=response, observers=pids); save()
                return
            time.sleep(1200)
        state['status'] = 'local_wait_expired'; save()
    except Exception as exc:
        state.update(status='inspection_required', error_type=type(exc).__name__, message=str(exc)); save()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--blockers', nargs=2, required=True)
    a = p.parse_args(); watch(a.root, a.blockers)
