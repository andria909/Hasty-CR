"""Summarize complete trainer JSONL; train outcomes are not held-out quality."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.defense_log_summary import summarize_episodes, summarize_update


def summarize_file(path, algorithm, window_updates=100):
    if algorithm not in ('PPO', 'GRPO') or window_updates < 1:
        raise ValueError('Expected PPO/GRPO and positive window size')
    digest = hashlib.sha256()
    episodes, returns, windows, validations = [], [], {}, []
    last = 0
    with Path(path).open('rb') as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            if not line.strip():
                continue
            record = json.loads(line)
            update = record['update']
            if not isinstance(update, int) or update <= last:
                raise ValueError(f'Non-increasing update at line {line_number}')
            last = update
            batch = record.get('episodes' if algorithm == 'PPO' else 'metrics', [])
            rewards = record.get('returns', []) if algorithm == 'GRPO' else None
            if rewards is not None and len(rewards) != len(batch):
                raise ValueError(f'Return/episode mismatch at line {line_number}')
            key = (update - 1) // window_updates
            bucket = windows.setdefault(key, dict(first_update=update, last_update=update,
                                                   episodes=[], returns=[]))
            bucket['last_update'] = update
            bucket['episodes'].extend(batch)
            episodes.extend(batch)
            if rewards is not None:
                returns.extend(rewards)
                bucket['returns'].extend(rewards)
            validation = summarize_update(record, algorithm)['validation']
            if validation is not None:
                validations.append(dict(update=update, summary=validation))
    return dict(algorithm=algorithm, source=str(path), source_sha256=digest.hexdigest(),
                scope='Completed stochastic training episodes; repeated scenarios and changing policies. Not paired evaluation or test acceptance.',
                last_update=last,
                train=summarize_episodes(episodes, returns if algorithm == 'GRPO' else None),
                windows=[dict(first_update=b['first_update'], last_update=b['last_update'],
                              train=summarize_episodes(b['episodes'], b['returns'] if algorithm == 'GRPO' else None))
                         for b in windows.values()], validation=validations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--algorithm', choices=('PPO', 'GRPO'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window-updates', type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new path')
    result = summarize_file(args.input, args.algorithm, args.window_updates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result['train'], indent=2))


if __name__ == '__main__':
    main()
