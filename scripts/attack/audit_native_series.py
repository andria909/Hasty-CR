"""Audit a finite native series across restarts, excluding interrupted games.

Commands and native hand confirmations are separate; no inferred spell hits.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.gamestate.memory_play import VERIFIED_DECK
from scripts.summarize_latency import describe


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(series_file, checkpoint_sha256):
    series_file = Path(series_file)
    series = json.loads(series_file.read_text())
    matches = series['matches']
    if series['status'] != 'complete' or len(matches) != series['requested']:
        raise ValueError('Completed finite series required')
    if sorted(m['number'] for m in matches) != list(range(1, series['requested'] + 1)):
        raise ValueError('Missing or duplicated global match')
    excluded = [m['number'] for m in matches if m.get('interrupted') or m.get('exclude_from_uninterrupted_model_winrate')]
    sources = {str(series_file): sha(series_file)}
    cache = {}; rows = []; confirmed = []; attempts = []
    for match in matches:
        if match['number'] in excluded:
            continue
        path = Path(match['run']) / 'events.jsonl'
        if path not in cache:
            cache[path] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            sources[str(path)] = sha(path)
        events = [e for e in cache[path] if e.get('match') == match['local_match']]
        starts = [e for e in events if e['event'] == 'MATCH_START']
        ends = [e for e in events if e['event'] == 'MATCH_END']
        if len(starts) != 1 or starts[0]['checkpoint_sha256'] != checkpoint_sha256 or len(ends) != 1:
            raise ValueError('Incomplete match or changed checkpoint')
        plays = {}; seen = set(); cards = Counter(); lanes = Counter(); positions = Counter()
        for e in events:
            if e['event'] == 'PLAY':
                key = e['card'], e['tick']
                if key in plays:
                    raise ValueError('Ambiguous command identity')
                plays[key] = e
                attempts.append(e)
            elif e['event'] == 'PLAY_ACK':
                key = e['card'], e['source_tick']
                if key not in plays or key in seen:
                    raise ValueError('Duplicate or unmatched hand confirmation')
                seen.add(key); play = plays[key]
                lane = 'left' if play['grid'][0] < 9 else 'right'
                cards[e['card']] += 1; lanes[lane] += 1
                positions[tuple(play['grid'])] += 1
                confirmed.append(dict(match=match['number'], card=e['card'], grid=play['grid'], lane=lane,
                                      timing=play.get('timing', {}), ack=e))
        rows.append(dict(number=match['number'], result=match['result'], commands=len(plays),
                         confirmations=sum(cards.values()), cards=dict(cards), lanes=dict(lanes),
                         top_positions=[dict(grid=list(k), count=v) for k, v in positions.most_common(5)]))
    if not rows:
        raise ValueError('No uninterrupted matches')
    totals = {}
    for card in VERIFIED_DECK:
        uses = [r for r in confirmed if r['card'] == card]
        positions = Counter(tuple(r['grid']) for r in uses)
        totals[card] = dict(confirmed=len(uses), mean_per_match=len(uses)/len(rows),
            lanes=dict(Counter(r['lane'] for r in uses)),
            top_positions=[dict(grid=list(k), count=v) for k, v in positions.most_common(5)])
    stages = sorted({k for e in attempts for k in e.get('timing', {})})
    return dict(schema='native-finite-series-audit-v1', checkpoint_sha256=checkpoint_sha256,
        total_matches=len(matches), excluded_interrupted_matches=excluded, uninterrupted_matches=len(rows),
        wins=sum(r['result']=='win' for r in rows), losses=sum(r['result']=='loss' for r in rows),
        unknown=sum(r['result'] not in ('win','loss') for r in rows),
        cards=totals, lanes=dict(Counter(e['lane'] for e in confirmed)), matches=rows,
        command_timings_ms={k:describe([e['timing'][k] for e in attempts if k in e.get('timing',{})]) for k in stages},
        source_to_hand_confirmation_game_ms=describe([e['ack']['source_to_ack_game_ms'] for e in confirmed]),
        commands_before_tick20=sum(e['tick']<20 for e in attempts), sources=sources,
        limitations=['Hand rotation confirms card consumption, not damage or spell value.',
                     'Requested target cell is not a measured unit spawn location.',
                     'Lane counts alone do not account for enemy threats.',
                     'Interrupted matches remain in total series count and are excluded from autonomous metrics.'])


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--series',required=True)
    p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=audit(a.series,a.checkpoint_sha256)
    with Path(a.output).open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
