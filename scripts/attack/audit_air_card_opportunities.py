"""Paid single-card counterfactuals on the existing TRAIN policy fixtures.

Search results are hindsight opportunities, not model quality or training labels.
"""
import argparse
import json
from pathlib import Path

from scripts.attack.probe_central_defense_policies import contexts, opening
from scripts.attack.search_late_air_defense import rank, rollout
from scripts.attack.sequence_execution import EXECUTION
from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.cannon_placement import CENTRAL_CELLS
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26
from sim.runner import resolve_deck


def choices(hand, lane):
    yield None
    for card in ('cannon', 'musketeer'):
        if card not in hand:
            continue
        cells = CENTRAL_CELLS if card == 'cannon' else tuple(
            (x if lane == 'left' else 17-x, y)
            for x in range(1, 9) for y in range(18, 29, 2))
        for x, y in cells:
            yield dict(card=card, cell=[x, y])


def audit(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cards, spells, *_ = load_defense_assets()
    cards.update(resolve_deck(cards, list(DECK_26)))
    suite = [r for r in contexts() if r['threat'] == 'balloon']
    protocol = dict(schema='air-card-opportunities-v1', contexts=suite,
        execution=EXECUTION, physics_pins=physics_pins(), horizon_ms=60000,
        split='TRAIN development',
        sources={p:file_sha(p) for p in (__file__,
            'scripts/attack/probe_central_defense_policies.py',
            'scripts/attack/search_late_air_defense.py')},
        policy_source='tmp/combined_central_probe_20261028',
        training_data_modified=False, quality_accepted=False)
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    groups = []
    attempted = 0
    with (output/'trials.jsonl').open('x') as stream:
        for index, record in enumerate(suite):
            match, hp = opening(record, cards, spells)
            hand = list(match.players[1].hand)
            rows = []
            for first in choices(hand, record['lane']):
                row = rollout(match, hp, cards, spells, record['lane'], first)
                row.update(context=index, first=first)
                stream.write(json.dumps(row)+'\n')
                rows.append(row)
            stream.flush()
            best = {}
            for card in ('cannon', 'musketeer'):
                valid = [r for r in rows if r['valid'] and r['first'] and r['first']['card'] == card]
                best[card] = min(valid, key=rank) if valid else None
            groups.append(dict(context=index, opening=record, hand=hand,
                no_card=rows[0], best=best, attempted=len(rows),
                invalid=sum(not r['valid'] for r in rows)))
            attempted += len(rows)
            print(json.dumps(dict(contexts_complete=index+1, total=len(suite), trials=attempted)), flush=True)
    if protocol['physics_pins'] != physics_pins():
        raise ValueError('Physics changed during audit')
    result = dict(protocol=protocol, groups=groups, trials=attempted,
        trials_sha256=file_sha(output/'trials.jsonl'),
        limitations=['Only two correlated hand seeds and one unsupported Balloon.',
            'Best placement selected with hindsight from a bounded grid.',
            'Single cards only; this cannot rank complete policies or defense combinations.',
            'Simulator opportunities are not validated real-game tactics.'])
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    audit(p.parse_args().output)
