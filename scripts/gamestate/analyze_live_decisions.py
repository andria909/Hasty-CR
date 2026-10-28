"""Audit saved commands, exact policy masks and native mana; never uses ADB.

An ACK is an observed hand rotation, not an independently verified spell hit.
Decision counts are conditional on recorded inputs, not counterfactual games.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.gamestate.audit_targeting_history import clock_frame
from scripts.gamestate.hand_readiness import HandReadiness
from scripts.gamestate.replay_decisions import load_capture
from scripts.gamestate.summarize_memory_run import summarize
from sim.env import CARD_ACTIONS, DECK_26, NUM_SCALARS, TILES

ROOT = Path(__file__).resolve().parents[2]

def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_audit(events):
    commands, pending, previous_ack = {}, {}, {}
    for event in events:
        kind = event['event']
        if kind == 'PLAY':
            if (event['card'] not in DECK_26 or event['slot'] not in range(4)
                    or len(event['grid']) != 2 or not 0 <= event['grid'][0] < 18
                    or not 0 <= event['grid'][1] < 32):
                raise ValueError('Invalid command for this deck/board')
            key = event['match'], event['tick'], event['card']
            slot_key = event['match'], event['slot']
            if key in commands or slot_key in pending:
                raise ValueError('Duplicate command or reuse before ACK/timeout')
            prior = previous_ack.get(slot_key)
            commands[key] = dict(event, outcome='unresolved',
                previous_slot_ack_tick=prior,
                ticks_since_slot_ack=None if prior is None else event['tick']-prior)
            pending[slot_key] = key
        elif kind in ('PLAY_ACK', 'PLAY_NOT_CONFIRMED'):
            candidates = [key for key in pending.values()
                          if key[0] == event['match'] and key[2] == event['card']
                          and ('source_tick' not in event or key[1] == event['source_tick'])]
            if len(candidates) != 1:
                raise ValueError('Missing or ambiguous command for ACK/timeout')
            key = candidates[0]
            command = commands[key]
            if event['tick'] < key[1]:
                raise ValueError('ACK/timeout precedes command')
            if 'slot' in event and event['slot'] != command['slot']:
                raise ValueError('ACK/timeout names a different slot')
            command.update(outcome='confirmed' if kind == 'PLAY_ACK' else 'not_confirmed',
                           resolution_tick=event['tick'])
            slot_key = command['match'], command['slot']
            del pending[slot_key]
            if kind == 'PLAY_ACK':
                previous_ack[slot_key] = event['tick']
    result = list(commands.values())
    by_card = {}
    for card in DECK_26:
        selected = [r for r in result if r['card'] == card]
        confirmed = [r for r in selected if r['outcome'] == 'confirmed']
        by_card[card] = dict(attempts=len(selected),
            **{outcome: sum(r['outcome'] == outcome for r in selected)
               for outcome in ('confirmed', 'not_confirmed', 'unresolved')},
            confirmed_left=sum(r['grid'][0] < 9 for r in confirmed),
            confirmed_right=sum(r['grid'][0] >= 9 for r in confirmed),
            confirmed_cells=[dict(grid=list(cell), count=count) for cell, count in
                             sorted(Counter(tuple(r['grid']) for r in confirmed).items())])
    return dict(counts=dict(Counter(r['outcome'] for r in result)), by_card=by_card,
        same_tick_as_previous_slot_ack=dict(Counter(
            r['outcome'] for r in result if r['ticks_since_slot_ack'] == 0)), commands=result)


def in_window(tick, minimum_tick, maximum_tick):
    return tick >= minimum_tick and (maximum_tick is None or tick < maximum_tick)


def decision_audit(files, *, expected_sha, minimum_tick=0, maximum_tick=None):
    counts = Counter()
    by_card = {card: Counter() for card in DECK_26}
    identity = set()
    pins = hashlib.sha256()
    selected = []
    contracts = set()
    for path in sorted(map(Path, files)):
        pins.update(path.name.encode()+b'\0'+bytes.fromhex(sha(path)))
        with np.load(path, allow_pickle=False) as data:
            joint = data['scalars'].shape == (NUM_SCALARS+3,)
        _, scalars, mask, metadata = load_capture(path, joint=joint)
        if metadata['checkpoint_sha256'] != expected_sha:
            raise ValueError('Mixed checkpoints in captured decisions')
        key = metadata['match'], metadata['tick']
        if key in identity:
            raise ValueError('Duplicate decision tick within a match')
        identity.add(key)
        contracts.add((metadata['coordinate_mode'], metadata['geometry']))
        hand = metadata['hand']
        if len(hand) != 4 or len(set(hand)) != 4 or not set(hand) <= set(DECK_26):
            raise ValueError('Invalid captured Hasty-CR hand')
        action = metadata['decision']['action']
        if action >= CARD_ACTIONS:
            raise ValueError('Ability actions are not supported in this deck audit')
        legal = mask[1:CARD_ACTIONS].reshape(4, TILES).any(axis=1)
        available = float(scalars[0])*10
        reserved = float(metadata['reserved_elixir'])
        native = float(metadata.get('native_elixir', available+reserved))
        if (not 0 <= reserved <= 10 or not -1e-5 <= available <= 10+1e-5
                or not -1e-5 <= native <= 10+1e-5
                or abs(native-available-reserved) > 1e-4):
            raise ValueError('Invalid native/available/reserved balance')
        if action:
            slot = (action-1)//TILES
            selected.append(dict(match=key[0], tick=key[1], card=hand[slot], slot=slot,
                                 file=str(path)))
        if not in_window(key[1], minimum_tick, maximum_tick):
            counts['excluded_by_tick_window'] += 1
            continue
        counts['decisions'] += 1
        counts['available_below4'] += int(available < 4-1e-5)
        counts['native_below4'] += int(native < 4-1e-5)
        counts['reservation_makes_below4'] += int(available < 4-1e-5 <= native)
        counts['play' if action else 'voluntary_wait' if legal.any() else 'forced_wait'] += 1
        for slot, card in enumerate(hand):
            by_card[card]['in_hand'] += 1
            by_card[card]['legal'] += int(legal[slot])
            by_card[card]['selected'] += int(action > 0 and (action-1)//TILES == slot)
    if not identity or len(contracts) != 1:
        raise ValueError('Missing captures or mixed coordinate contracts')
    return dict(total_captures=len(identity), capture_manifest_sha256=pins.hexdigest(),
        coordinate_contract=list(next(iter(contracts))), counts=dict(counts),
        by_card={card: dict(count) for card, count in by_card.items()}, selected=selected,
        mana_note='scalars[0]*10 is spendable mana, not raw native mana; older captures reconstruct raw as spendable+reserved.')


def native_audit(frames, matches, commands, *, minimum_tick=0, maximum_tick=None):
    by_world = {m['world']: m['number'] for m in matches}
    if len(by_world) != len(matches):
        raise ValueError('Reused world pointers require separate run analysis')
    command_map = {(p['match'], p['tick']): p for p in commands}
    if len(command_map) != len(commands):
        raise ValueError('Multiple inputs on the same native tick')
    guards, previous, stats = {}, None, Counter()
    deferred, matched_commands = {}, set()
    for index, row in enumerate(frames):
        frame = clock_frame(row)
        own = row.get('own_state', {})
        match = by_world.get(own.get('world'))
        if frame is None or match is None:
            stats['invalid_or_other_world_frames'] += 1
            previous = None
            continue
        tick = frame['tick']
        if not 0 <= own.get('elixir_raw', -1) <= 100000:
            raise ValueError('Invalid native mana')
        guard = guards.setdefault(match, HandReadiness())
        if guard.tick is not None and tick < guard.tick:
            stats['stale_frames'] += 1
            previous = None
            continue
        blocked = guard.observe(tick, own['hand_indices'])
        key = match, tick
        command = command_map.get(key)
        # Use only snapshots received before the command was emitted. Repeated
        # ticks after input cannot retrospectively change its observed hand.
        if command and row['received_host_ns'] <= command['host_ns']:
            matched_commands.add(key)
            if command['slot'] in blocked:
                deferred[key] = dict(match=match, tick=tick, card=command['card'],
                    slot=command['slot'], outcome=command['outcome'], source_frame=index)
        current = dict(match=match, tick=tick, world=frame['world'], mana=own['elixir_raw']/10000)
        stats['valid_frames'] += 1
        if previous and previous['world'] == current['world'] and previous['match'] == match:
            delta = tick-previous['tick']
            if (0 < delta <= 5 and in_window(previous['tick'], minimum_tick, maximum_tick)
                    and in_window(tick, minimum_tick, maximum_tick)):
                seconds = delta/20
                stats['covered_game_seconds'] += seconds
                if previous['mana'] < 4:
                    stats['left_sample_below4_seconds'] += seconds
                if previous['mana'] >= 9.99:
                    stats['left_sample_full_seconds'] += seconds
            elif delta > 5:
                stats['gaps_over_5_ticks'] += 1
        previous = current
    return dict(**dict(stats), commands_with_native_source=len(matched_commands),
        fresh_tick_guard_would_defer=dict(Counter(p['outcome'] for p in deferred.values())),
        deferred_commands=list(deferred.values()),
        timing_note='Native 20Hz; mana is held from the left sample for intervals <=5 ticks. Invalid frames/world changes/large gaps split coverage. This is sampled time, not exact continuous mana.')


def hand_guard_fixture(root, report):
    """Portable three-observation examples, linked to exact raw JSONL lines."""
    path = Path(root)/'frames.jsonl'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != report['sources']['frames.jsonl']:
        raise ValueError('Native source changed since analysis')
    lines = raw.splitlines()
    cases = []
    for command in report['native']['deferred_commands']:
        index = command['source_frame']
        source = clock_frame(json.loads(lines[index]))
        before = clock_frame(json.loads(lines[index-1])) if index else None
        if before is None or before['world'] != source['world'] or not 0 <= source['tick']-before['tick'] <= 5:
            raise ValueError('No continuous pre-rotation anchor')
        after_index = index+1
        while after_index < len(lines):
            after = clock_frame(json.loads(lines[after_index]))
            if after is None or after['world'] != source['world'] or not 0 <= after['tick']-source['tick'] <= 5:
                raise ValueError('No continuous post-rotation anchor')
            if after['tick'] > source['tick']:
                break
            after_index += 1
        if after_index == len(lines):
            raise ValueError('Source ended before fresh observation')
        observations = []
        ready = HandReadiness()
        for row_index in (index-1, index, after_index):
            row = json.loads(lines[row_index])
            own = row['own_state']
            blocked = ready.observe(own['tick'], own['hand_indices'])
            observations.append(dict(source_frame=row_index,
                source_line_sha256=hashlib.sha256(lines[row_index]).hexdigest(),
                tick=own['tick'], hand=own['hand_indices'], blocked_slots=list(blocked)))
        if command['slot'] not in observations[1]['blocked_slots'] or command['slot'] in observations[2]['blocked_slots']:
            raise ValueError('Window does not isolate a single observed rotation')
        cases.append(dict(command=command, observations=observations))
    return dict(schema='recorded-hand-rotation-guard-v1', source=str(path),
        source_sha256=report['sources']['frames.jsonl'], cases=cases,
        limitation='Observed successful and unconfirmed inputs are both delayed by this guard. No counterfactual acceptance or win rate is asserted.')


def audit_run(root, *, minimum_tick=0, maximum_tick=None):
    root = Path(root)
    summary = json.loads((root/'summary.json').read_text())
    captures = json.loads((root/'decisions/summary.json').read_text())
    commands = command_audit(rows(root/'events.jsonl'))
    decisions = decision_audit((root/'decisions').glob('*.npz'),
        expected_sha=summary['checkpoint_sha256'], minimum_tick=minimum_tick, maximum_tick=maximum_tick)
    if decisions['total_captures'] != captures['written']:
        raise ValueError('Capture manifest/file count mismatch')
    emitted = {(p['match'], p['tick'], p['card'], p['slot']) for p in commands['commands']}
    chosen = {(p['match'], p['tick'], p['card'], p['slot']) for p in decisions['selected']}
    if not chosen <= emitted:
        raise ValueError('Captured chosen play has no matching emitted command')
    native = native_audit(rows(root/'frames.jsonl'), summary['matches'], commands['commands'],
        minimum_tick=minimum_tick, maximum_tick=maximum_tick)
    report = dict(schema='saved-live-decision-audit-v1', root=str(root),
        checkpoint_sha256=summary['checkpoint_sha256'],
        sources={name: sha(root/name) for name in
                 ('events.jsonl', 'frames.jsonl', 'summary.json', 'decisions/summary.json')},
        code_sha256={name: sha(ROOT/name) for name in (str(Path(__file__).resolve().relative_to(ROOT)),
            'scripts/gamestate/hand_readiness.py', 'scripts/gamestate/replay_decisions.py',
            'scripts/gamestate/audit_targeting_history.py', 'scripts/gamestate/summarize_memory_run.py',
            'sim/env.py')},
        decision_window=dict(minimum_tick_inclusive=minimum_tick, maximum_tick_exclusive=maximum_tick),
        capture_status=captures, commands=commands, decisions=decisions, native=native,
        timings=summarize(root),
        limitations=['No new games or Android connections.',
            'Native hand ACK does not prove body location, spell impact or successful defence.',
            'Unconfirmed and unresolved commands are separate from ACKs; no result/win rate is inferred.',
            'Per-card legal/selection counts use recorded masks, not changed-policy rollouts.',
            'The readiness guard replay only identifies deferred original commands, not their counterfactual outcomes.',
            'Frames and decisions from a single match are correlated, not independent quality trials.'])
    for name, path in [('series', root.parent/'series/series.json'),
                       ('user_stop', root.parent/'user_stop.json'),
                       ('exact_replay', root.parent/'exact_replay.json')]:
        if path.exists():
            payload = json.loads(path.read_text())
            report[name] = {k: v for k, v in payload.items() if k != 'rows'}
            report['sources'][str(path)] = sha(path)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--minimum-tick', type=int, default=0)
    p.add_argument('--maximum-tick', type=int)
    p.add_argument('--fixture-output', type=Path)
    p.add_argument('--comparison-events', type=Path, nargs='*', default=[])
    args = p.parse_args()
    if args.minimum_tick < 0 or (args.maximum_tick is not None and args.maximum_tick <= args.minimum_tick):
        p.error('Invalid tick window')
    result = audit_run(args.run, minimum_tick=args.minimum_tick, maximum_tick=args.maximum_tick)
    result['historical_command_comparisons'] = []
    for path in args.comparison_events:
        comparison = command_audit(rows(path))
        result['historical_command_comparisons'].append(dict(path=str(path), sha256=sha(path),
            counts=comparison['counts'], same_tick_as_previous_slot_ack=comparison['same_tick_as_previous_slot_ack']))
    if args.fixture_output:
        fixture = hand_guard_fixture(args.run, result)
        with args.fixture_output.open('x') as stream:
            json.dump(fixture, stream, indent=2)
            stream.write('\n')
        result['hand_guard_fixture'] = dict(path=str(args.fixture_output), sha256=sha(args.fixture_output))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(commands=result['commands']['counts'], decisions=result['decisions']['counts'],
        deferred=result['native']['fresh_tick_guard_would_defer'],
        coverage_seconds=result['native'].get('covered_game_seconds', 0))))
