"""Reverify comparable completed cohorts and rank2–4 profiles locally.

No remote requests, training submission or live promotion. Completed reports
are immutable snapshots; a small public pointer identifies the latest one.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.attack.select_pilot_pool import select


def ready_studies(config):
    ready = []
    for study in config['studies']:
        proof = Path(study['proof'])
        receipt = Path(study['prepared']) / 'collected/collection_receipt.json'
        if not proof.exists() or not receipt.exists():
            continue
        try:
            row = json.loads(proof.read_text())
            collection = json.loads(receipt.read_text())
        except (ValueError, OSError):
            continue
        if (row.get('pilot_steps_verified') == 819200 and collection.get('complete') is True
                and collection.get('remote_status') == 'COMPLETE'
                and collection.get('kernel') == row.get('kernel')):
            ready.append(study)
    return ready


def refresh(config, previous_signature=None):
    ready = ready_studies(config)
    signature = [(r['prepared'], hashlib.sha256(Path(r['proof']).read_bytes()).hexdigest()) for r in ready]
    if not ready or signature == previous_signature:
        return signature, None
    # Do not trust the saved proofs alone: select() reruns complete budget,
    # checkpoint and source/context verification before scoring anything.
    result = select([r['prepared'] for r in ready], limit=config.get('limit', 4))
    snapshot = Path(config['reports']) / f'global_pilot_pool_auto_cohorts_{len(ready)}_20261027.json'
    with snapshot.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    pointer = dict(report=str(snapshot), report_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        verified_cohorts=len(ready), expected_cohorts=len(config['studies']), verified_models=8*len(ready),
        eligible_profiles=result['eligible_profiles'], selected_profiles=len(result['selected']),
        ready_for_full_preflight=result['gate_met'], training_submitted=False, quality_accepted=False)
    Path(config['pointer']).write_text(json.dumps(pointer, indent=2) + '\n')
    return signature, pointer


def watch(config_path, state_file):
    config = json.loads(Path(config_path).read_text())
    state_file = Path(state_file)
    if state_file.exists():
        raise FileExistsError('Inspect existing global selector before restarting')
    roots = [r['prepared'] for r in config['studies']]
    if len(set(roots)) != len(roots):
        raise ValueError('Duplicate prepared cohort')
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state = dict(status='waiting_for_verified_cohorts', api_requests=False,
                 training_submitted=False, quality_accepted=False)
    def save():
        state_file.write_text(json.dumps(state, indent=2) + '\n')
    save()
    try:
        os.nice(10)
    except OSError:
        pass
    signature = None
    deadline = time.monotonic() + 172800
    try:
        while time.monotonic() < deadline:
            signature, pointer = refresh(config, signature)
            if pointer:
                state.update(latest=pointer)
                save()
                subprocess.run([sys.executable, '-m', 'scripts.defense.update_done'], check=True)
                print(json.dumps(pointer), flush=True)
                if pointer['verified_cohorts'] == pointer['expected_cohorts']:
                    state['status'] = ('eligible_candidates_ready_for_full_preflight'
                                       if pointer['ready_for_full_preflight'] else 'completed_no_eligible_pair')
                    save()
                    return
            time.sleep(30)
        state['status'] = 'local_wait_expired_remote_unchanged'
        save()
    except Exception as error:
        state.update(status='error_requires_inspection', error_type=type(error).__name__, message=str(error))
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--state-file', required=True)
    args = parser.parse_args()
    watch(args.config, args.state_file)
