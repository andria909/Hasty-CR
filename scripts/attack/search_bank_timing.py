"""Compare paid immediate and delayed defenses; no reward for waiting by itself.

All central Cannon cells and three Musketeer cells are searched at each delay.
Only zero additional tower damage with all enemy bodies cleared is safe.
This bounded search produces evidence, not automatic expert labels.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import hashlib
import json
from pathlib import Path

from scripts.attack.build_bank_counterfactuals import opening
from scripts.attack.search_late_air_defense import rollout, rank
from sim.cannon_placement import CENTRAL_CELLS


def context(spec):
    seed, lane, bank, enemy_y, delays = spec
    env = opening(seed, lane, bank, enemy_y)
    hp = {k: t.hitpoints for k, t in env.match.towers[1].items()}
    records = []
    wait = rollout(env.match, hp, env.cards, env.spells, lane)
    records.append(dict(wait, delay_ms=None, candidate=None))
    x = 3 if lane == 'left' else 14
    candidates = [dict(card='cannon', cell=list(c)) for c in CENTRAL_CELLS]
    candidates += [dict(card='musketeer', cell=list(c)) for c in
                   ((x, 20), (6 if lane == 'left' else 11, 22), (x, 24))]
    for delay, candidate in product(delays, candidates):
        # Same legal-hand/cost/reservation checks and measured delivery delay
        # for both branches. Deferred requests do not reserve mana early.
        row = rollout(env.match, hp, env.cards, env.spells, lane,
                      second=dict(candidate, offset_ms=delay))
        records.append(dict(row, delay_ms=delay, candidate=candidate))
    by_delay = []
    for delay in delays:
        valid = [r for r in records if r['delay_ms'] == delay and r['valid']]
        safe = [r for r in valid if r['clear_without_damage']]
        by_delay.append(dict(delay_ms=delay, legal=len(valid), safe=len(safe),
                             best=min(valid, key=rank) if valid else None))
    safe_delays = [r['delay_ms'] for r in by_delay if r['safe']]
    return dict(seed=seed, lane=lane, bank=bank, enemy_y=enemy_y,
                wait_forever=wait, by_delay=by_delay,
                latest_tested_safe_request_ms=max(safe_delays) if safe_delays else None,
                cannon_first_audited=True, trials=records,
                normal_range_mode=env.match.battle.normal_range_mode,
                hog_sight_mode=env.match.battle.hog_sight_mode)


def search(output, smoke=False, workers=2):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    delays = (0, 1500, 3000, 6000, 9000, 12000)
    cases = [(*case, delays) for case in product(
        (352, 353, 354), ('left', 'right'), (2000, 3000, 6000), (9, 15))]
    if smoke:
        cases = [(352, 'left', 3000, 15, (0, 3000))]
    rows = []
    with (output/'contexts.jsonl').open('x') as stream:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(context, cases):
                if (row['normal_range_mode'] != 'edge_to_edge_v1' or
                        row['hog_sight_mode'] != 'side_clip_v1'):
                    raise ValueError('Unexpected physics')
                rows.append(row)
                stream.write(json.dumps(row)+'\n'); stream.flush()
                print(json.dumps(dict(completed=len(rows), total=len(cases))), flush=True)
    result = dict(schema='bank-timing-search-v1', contexts=len(rows), smoke=smoke,
                  command_delay_ms=1350, cannon_first_audited=True,
                  delayed_safe_contexts=sum(any(d['safe'] and d['delay_ms'] > 0
                                               for d in r['by_delay']) for r in rows),
                  immediate_safe_contexts=sum(any(d['safe'] and d['delay_ms'] == 0
                                                 for d in r['by_delay']) for r in rows),
                  raw_sha256=hashlib.sha256((output/'contexts.jsonl').read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  quality_accepted=False, live_games=False,
                  limitations=['Synthetic single paid Balloon, no reinforcements',
                               'Fixed hand; only Cannon and Musketeer single-card defenses',
                               'Latest safe tested delay is not an optimal deadline',
                               'Waiting safely does not prove waiting is preferable',
                               'No labels or policy promotion without subsequent trajectory audit'])
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    print(json.dumps(search(a.output, a.smoke, a.workers)))
