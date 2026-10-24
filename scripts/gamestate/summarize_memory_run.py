"""Summarize memory-run timings without conflating input, ACK and images."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.summarize_latency import describe


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(directory):
    events = rows(directory / 'events.jsonl')
    plays = [r for r in events if r['event'] == 'PLAY']
    acks = [r for r in events if r['event'] == 'PLAY_ACK']
    images = rows(directory / 'screenshots.jsonl')
    stages = sorted({k for r in plays for k in r.get('timing', {})})
    result = {
        'directory': str(directory), 'units': 'milliseconds',
        'counts': dict(Counter(r['event'] for r in events)),
        'input_boundary': 'ADB target tap process returned, not measured touch processing time',
        'ack_boundary': 'First sampled native own-hand rotation, not exact troop appearance',
        'image_boundary': 'PNG received on Mac; capture exposure and display lag unknown',
        'model_input': 'Native memory snapshot; screenshots are diagnostic only',
        'stages': {k: describe([r['timing'][k] for r in plays if k in r.get('timing', {})]) for k in stages},
        'confirmation': {k: describe([r[k] for r in acks if k in r]) for k in
                         ('delay_ms', 'input_return_to_ack_ms', 'source_to_ack_game_ms')},
        'images': {},
        'by_match': {},
        'warnings': sorted({r['reason'] for r in events if r['event'] == 'INPUT_WARNING'}),
    }
    for phase in sorted({r['phase'] for r in images}):
        selected = [r for r in images if r['phase'] == phase]
        result['images'][phase] = {k: describe([r[k] for r in selected if k in r]) for k in
                                   ('capture_ms', 'input_to_png_ms', 'ack_to_png_ms')}
        # Engine-tick bounds surround capture, not pixel-derived clock readings.
        for boundary in ('before', 'after'):
            key = f'engine_tick_{boundary}_capture'
            result['images'][phase][f'source_to_capture_{boundary}_game_ms'] = describe(
                [(r[key]-r['source_tick'])*50 for r in selected if r.get(key) is not None])
    for number in sorted({r['match'] for r in plays}):
        p = [r for r in plays if r['match'] == number]
        a = [r for r in acks if r['match'] == number]
        result['by_match'][str(number)] = {
            'attempts': len(p), 'confirmations': len(a),
            'ack_ms': describe([r['delay_ms'] for r in a]),
            'request_to_input_ms': describe([r['timing']['parser_request_to_input_return_ms']
                                            for r in p if 'parser_request_to_input_return_ms' in r.get('timing', {})]),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('counts', 'stages', 'confirmation', 'images')}, indent=2))


if __name__ == '__main__':
    main()
