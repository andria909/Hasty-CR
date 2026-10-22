"""Recover bounded terminal defense artifacts without relying on stage2_result.

Run in the Kaggle CLI environment. No torch/checkpoint deserialization, uploads,
restarts, or automatic evaluation. A downloaded checkpoint is not quality accepted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

TERMINAL = {'COMPLETE', 'ERROR', 'FAILED', 'CANCELLED', 'CANCELED'}
PATTERNS = {
    'ppo_latest': r'ppo/update_(\d{6})\.pt',
    'ppo_best_validation': r'ppo/best_validation_(\d{6})\.pt',
    'grpo_boundary': r'grpo/boundary_u(\d{6})_s(\d{6})\.pt',
    'grpo_best_validation': r'grpo/best_validation_u(\d{6})_s(\d{6})\.pt',
}
AUXILIARY = {'stage2_result.json', 'ppo/metrics.jsonl', 'grpo/updates.jsonl',
             'ppo/config.json', 'grpo/config.json'}


def terminal_state(status):
    value = getattr(status, 'status', status)
    return getattr(value, 'name', str(value)).upper().split('.')[-1]


def select_outputs(names):
    """Exact anchored allowlist; numeric completed-update ordering, not mtime."""
    selected = {}
    for role, pattern in PATTERNS.items():
        candidates = [(tuple(map(int, match.groups())), name) for name in names
                      if (match := re.fullmatch(pattern, name))]
        if candidates:
            selected[role] = max(candidates)[1]
    return selected, sorted(AUXILIARY.intersection(names))


def list_outputs(api, kernel, max_pages=100):
    """Read metadata only, following 500-item pages; never serialize signed URLs."""
    from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
    owner, slug, version = api.parse_kernel_string(kernel)
    token, seen, items, logs = '', set(), {}, []
    with api.build_kaggle_client() as client:
        for _ in range(max_pages):
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = owner, slug
            request.page_size, request.page_token = 500, token
            if version:
                request.version_label = version
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            if getattr(response, 'log', '') and response.log not in logs:
                logs.append(response.log)
            for item in response.files or []:
                if item.file_name in items:
                    raise ValueError('Duplicate output metadata path')
                items[item.file_name] = item
            token = response.next_page_token
            if not token:
                return items, logs
            if token in seen:
                raise ValueError('Repeated output pagination token')
            seen.add(token)
    raise ValueError('Output listing exceeded bounded page limit; nothing downloaded')


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def remote_sha256(item):
    # Do not reinterpret a generic checksum (which may be MD5) as SHA256.
    value = getattr(item, 'sha256', None)
    return value.lower() if isinstance(value, str) and re.fullmatch('[a-fA-F0-9]{64}', value) else None


def recover(api, kernel, output, *, fetch=None):
    state = terminal_state(api.kernels_status(kernel))
    if state not in TERMINAL:
        raise ValueError(f'Kernel is not confirmed terminal: {state}; download forbidden')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    items, logs = list_outputs(api, kernel)
    roles, auxiliary = select_outputs(items)
    planned = sorted(set(roles.values()) | set(auxiliary))
    # A second terminal check closes the listing/download window conservatively.
    final_state = terminal_state(api.kernels_status(kernel))
    if final_state not in TERMINAL or final_state != state:
        raise ValueError('Remote status changed during listing; download forbidden')
    if fetch is None:
        import requests
        def fetch(url, destination):
            with requests.get(url, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                with destination.open('xb') as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            stream.write(chunk)
    receipt = dict(kernel=kernel, remote_status=state, selected_roles=roles,
                   missing_roles=sorted(set(PATTERNS)-set(roles)), files={},
                   missing_metrics=sorted({'ppo/metrics.jsonl','grpo/updates.jsonl'}-set(items)),
                   selection_basis='anchored frozen-trainer filenames; numeric update/sequence, not content validation',
                   remote_hash_note='Installed SDK output metadata exposes only file_name/url; SHA absent means unverified',
                   quality_accepted=False, checkpoint_contents_verified=False,
                   remote_job_changed=False, listing_file_count=len(items))
    # Receipt written even after a partial failed transfer; never masquerades as complete.
    receipt['complete'] = False
    try:
        for name in planned:
            destination = output/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            fetch(items[name].url, destination)
            digest = file_digest(destination)
            expected = remote_sha256(items[name])
            receipt['files'][name] = dict(sha256=digest, bytes=destination.stat().st_size,
                remote_sha256=expected, remote_hash_verified=bool(expected and expected == digest))
            if expected and expected != digest:
                raise ValueError(f'Remote SHA256 mismatch: {name}')
        log_path = output/'kernel_log_receipt.json'
        log_path.write_text(json.dumps(logs)+'\n')
        receipt['files'][log_path.name] = dict(sha256=file_digest(log_path),
                                              remote_hash_verified=False)
        receipt['complete'] = True
    finally:
        with (output/'recovery_receipt.json').open('x') as stream:
            json.dump(receipt, stream, indent=2)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--watch-seconds', type=int, default=0)
    args = parser.parse_args(argv)
    if args.watch_seconds < 0:
        parser.error('watch-seconds must be nonnegative')
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    if args.watch_seconds:
        deadline = time.monotonic() + args.watch_seconds
        while time.monotonic() < deadline:
            try:
                if terminal_state(api.kernels_status(args.kernel)) in TERMINAL:
                    break
            except Exception as error:
                print(json.dumps({'observation_error': type(error).__name__, 'remote_job_changed': False}), flush=True)
            time.sleep(30)
        else:
            raise TimeoutError('Observation window ended; remote job unchanged')
    print(json.dumps(recover(api, args.kernel, args.output), indent=2))


if __name__ == '__main__':
    main()
