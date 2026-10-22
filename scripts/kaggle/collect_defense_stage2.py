"""Collect terminal stage2 outputs and explicitly selected weights, never all snapshots."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time


def selected_files(result):
    files = {}
    for name in ('ppo', 'grpo'):
        row = result.get(name, {})
        for field, hash_field in (('checkpoint', 'sha256'),
                                  ('selected_checkpoint', 'selected_sha256'),
                                  ('resume_checkpoint', None)):
            value = row.get(field)
            if value is None:
                continue
            path = PurePosixPath(value)
            if ('\\' in value or path.is_absolute() or '..' in path.parts
                    or len(path.parts) != 2 or path.parts[0] != name
                    or path.suffix != '.pt' or str(path) != value):
                raise ValueError('Unsafe explicitly reported checkpoint path')
            digest = row.get(hash_field) if hash_field else None
            if hash_field and (not isinstance(digest, str) or not re.fullmatch('[a-f0-9]{64}', digest)):
                raise ValueError('Missing reported checkpoint SHA256')
            previous = files.get(value)
            if previous is not None and digest is not None and previous != digest:
                raise ValueError('Conflicting checkpoint hashes')
            files[value] = previous or digest
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernel', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--watch-seconds', type=int, default=28800)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    deadline = time.monotonic() + args.watch_seconds
    while time.monotonic() < deadline:
        try:
            status = api.kernels_status(args.kernel)
            value = getattr(status, 'status', None)
            state = getattr(value, 'name', str(value)).upper().split('.')[-1]
            if state not in ('COMPLETE', 'ERROR', 'CANCELLED', 'CANCELED'):
                time.sleep(30)
                continue
            # JSONL files are complete only after the remote kernel is terminal.
            api.kernels_output(args.kernel, path=str(args.output), page_size=500,
                file_pattern=r'^(stage2_result\.json|ppo/metrics\.jsonl|grpo/updates\.jsonl)$')
            result_path = args.output/'stage2_result.json'
            if not result_path.exists():
                raise ValueError('Terminal kernel has no stage2 result')
            result = json.loads(result_path.read_text())
            planned = selected_files(result)
            if planned:
                pattern = '^(?:' + '|'.join(re.escape(path) for path in planned) + ')$'
                api.kernels_output(args.kernel, path=str(args.output), page_size=500, file_pattern=pattern)
            receipt = dict(kernel=args.kernel, remote_status=state, result_status=result.get('status'),
                           files={}, quality_accepted=False)
            for relative, expected in planned.items():
                path = args.output/relative
                if not path.is_file():
                    raise ValueError('Explicitly reported checkpoint missing from output')
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if expected and actual != expected:
                    raise ValueError('Downloaded checkpoint SHA256 mismatch')
                receipt['files'][relative] = dict(sha256=actual, remote_hash_verified=expected is not None)
            for relative in ('stage2_result.json', 'ppo/metrics.jsonl', 'grpo/updates.jsonl'):
                path = args.output/relative
                if path.exists():
                    receipt['files'][relative] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            (args.output/'collection_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
            print(json.dumps(receipt), flush=True)
            return
        except Exception as error:
            print(json.dumps(dict(observation_error=type(error).__name__,
                                  remote_job_restarted=False)), flush=True)
            time.sleep(30)
    print('Collection observation window ended; remote job unchanged.', flush=True)


if __name__ == '__main__':
    main()
