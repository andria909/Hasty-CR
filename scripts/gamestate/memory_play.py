"""Finite v25 trial using ARM64 memory only. Battle queue is MANUAL.

No screenshots, CV detectors, network/model training or game-memory writes.
The owner must authorize matches. This runner never presses Battle or menus.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time

from scripts.gamestate.elixir import ElixirEstimate
from scripts.gamestate.model_bridge import (
    MemoryPolicy, MissingModelInput, NativeModelAdapter, OwnState)
from scripts.gamestate.query_arm64 import CurrentStateReader
from sim.env import DECK_26
from scripts.gamestate.placement_budget import PlacementBudget

# Explicit deck-screen order, visually checked on this account, slot 1.
# This is NOT the seen-card list at KingTower+0x288, which is play-history order.
VERIFIED_DECK = ('hog_rider', 'cannon', 'zap', 'musketeer',
                 'electro_spirit', 'fireball', 'skeletons', 'knight')

def diagnostic_screenshot(reader, adb, output, name, *, phase, input_host_ns,
                          source_tick, ack_host_ns=None):
    """Diagnostic only: screenshots NEVER become model input."""
    before = reader.query()
    start = time.time_ns()
    png = subprocess.check_output(adb + ['exec-out','screencap','-p'], timeout=8)
    end = time.time_ns()
    after = reader.query()
    (output / (name+'.png')).write_bytes(png)
    result = dict(name=name,phase=phase,source_tick=source_tick,
                  input_host_ns=input_host_ns,capture_request_host_ns=start,
                  capture_received_host_ns=end,capture_ms=(end-start)/1e6,
                  input_to_capture_request_ms=(start-input_host_ns)/1e6,
                  input_to_png_ms=(end-input_host_ns)/1e6,
                  engine_tick_before_capture=before.get('frame',{}).get('battle_tick'),
                  engine_tick_after_capture=after.get('frame',{}).get('battle_tick'),
                  note='Engine tick brackets around capture; no pixel-clock recognition or exact image exposure timestamp.')
    if ack_host_ns is not None:
        result['ack_to_png_ms']=(end-ack_host_ns)/1e6
    with (output/'screenshots.jsonl').open('a') as stream:
        stream.write(json.dumps(result)+'\n')
    return result


def own_context(state: dict, deck=VERIFIED_DECK, *, destroyed=(), events=()):
    if state.get('status') != 'ok':
        raise MissingModelInput('Native private hand is unavailable or transitioning')
    h, cycle = state['hand_indices'], state['cycle_indices']
    if sorted(h + cycle) != list(range(8)) or len(h) != 4 or len(cycle) != 4:
        raise MissingModelInput('Invalid private hand/cycle partition')
    tick, mana, side = state['tick'], state['elixir_raw'], state['own_side']
    if not (isinstance(tick, int) and 0 <= tick <= 12000 and
            isinstance(mana, int) and 0 <= mana <= 100000 and side in (0, 1)):
        raise MissingModelInput('Invalid native own state')
    seconds, value = tick / 20, mana / 10000
    balance = ElixirEstimate(seconds, value, value, True, source='arm64_memory')
    return OwnState(seconds, tuple(deck[i] for i in h), deck[cycle[0]], balance,
                    tuple(events), public_history_complete=False,
                    battle_active=True, destroyed_towers=tuple(destroyed))


def tap_points(slot, x, y, width, height):
    if not 0 <= slot < 4 or not 0 <= x < 18 or not 0 <= y < 32:
        raise ValueError('Invalid placement')
    # Same empirically measured screen calibration as the approved v25 runner.
    scale_x, scale_y = width / 1080, height / 1920
    select = (round((334 + 204 * slot) * scale_x), round(1711 * scale_y))
    target = (round((103.5 + 51 * x) * scale_x), round((171.9 + 41.4 * y) * scale_y))
    return select, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adb', default='/tmp/Hasty-android-tools/platform-tools/adb')
    parser.add_argument('--serial', required=True)
    parser.add_argument('--binary', default=None)
    parser.add_argument('--su', default=None)
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/portable_v25/portable_v25_cannon_economy_best.pt'))
    parser.add_argument('--max-matches', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Send approved card taps; never starts battles')
    parser.add_argument('--attach-current', action='store_true', help='Resume the already authorized current first match')
    parser.add_argument('--seconds', type=float, default=2400)
    parser.add_argument('--width', type=int, default=1440)
    parser.add_argument('--height', type=int, default=2560)
    parser.add_argument('--deck-order', default=','.join(VERIFIED_DECK))
    parser.add_argument('--screenshot-every', type=int, default=0,
                        help='Async diagnostic image after input/ACK on every Nth play; 0 disables')
    parser.add_argument('--result-reference',type=Path,help='Require verified result button before treating a frozen tick as match end')
    parser.add_argument('--capture-every',type=int,default=0,help='Record every Nth exact policy input; 0 disables')
    parser.add_argument('--capture-limit',type=int,default=2000)
    args = parser.parse_args()
    if args.capture_every<0 or not 1<=args.capture_limit<=10000:
        parser.error('Invalid bounded decision capture')
    deck = tuple(args.deck_order.split(','))
    if (not 1 <= args.max_matches <= 50 or not 1 <= args.seconds <= 21600 or
            len(deck) != 8 or set(deck) != set(DECK_26) or
            (args.width, args.height) not in ((1440, 2560),(1080,1920)) or not 0 <= args.screenshot_every <= 100):
        parser.error('Invalid finite limit, verified deck or calibrated resolution')
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(1)
    policy = MemoryPolicy(args.checkpoint)
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    reader_kwargs = dict(own_state=True)
    if args.su: reader_kwargs['su'] = args.su
    if args.binary: reader_kwargs['binary'] = args.binary
    reader = CurrentStateReader(args.adb, args.serial, **reader_kwargs)
    adb = [args.adb, '-s', args.serial]
    report = {'requested': args.max_matches, 'started': 0, 'finished': 0,
              'checkpoint': str(args.checkpoint), 'checkpoint_sha256': checkpoint_sha256, 'perception': 'arm64_memory_v8',
              'cv': False, 'queue': False, 'execute': args.execute,
              'deck_order': deck, 'warning': 'Experimental incomplete opponent play history', 'matches': []}
    report['coordinate_mode']=policy.coordinate_mode
    report['geometry']=policy.geometry
    report['joint_policy']=policy.joint
    report['screenshot_every']=args.screenshot_every
    from sim.cannon_placement import CENTRAL_MODE
    report['cannon_placement_mode']=CENTRAL_MODE
    log = (args.output / 'events.jsonl').open('x')
    frames = (args.output / 'frames.jsonl').open('x')
    pool = ThreadPoolExecutor(max_workers=1)
    capture_job = None
    decision_capture=None
    if args.capture_every:
        from scripts.gamestate.decision_capture import DecisionCapture
        decision_capture=DecisionCapture(args.output/'decisions',every=args.capture_every,limit=args.capture_limit)

    def emit(event, **values):
        row = dict(event=event, host_ns=time.time_ns(), **values)
        log.write(json.dumps(row) + '\n'); log.flush()
        print(json.dumps(row), flush=True)
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')

    emit('READY', max_matches=args.max_matches, queue=False, cv=False, execute=args.execute)
    active = None
    waiting_for_lobby = False
    deadline = time.monotonic() + args.seconds
    result_reference=args.result_reference.read_bytes()if args.result_reference else None
    last_result_check=0.;result_visible=False
    def frozen_result_confirmed():
        nonlocal last_result_check,result_visible
        if result_reference is None:return True
        now=time.monotonic()
        if now-last_result_check>=2:
            from scripts.gamestate.memory_series import lobby_button_matches
            png=subprocess.check_output(adb+['exec-out','screencap','-p'],timeout=10)
            result_visible=lobby_button_matches(result_reference,png,box=(800,2240,1100,2400))
            last_result_check=now
        return result_visible
    try:
        while time.monotonic() < deadline and report['finished'] < args.max_matches:
            began = time.monotonic()
            if capture_job is not None and capture_job.done():
                try: capture_job.result()
                except Exception as exc: emit('SCREENSHOT_ERROR',detail=str(exc))
                capture_job=None
            snapshot = reader.query()
            frames.write(json.dumps(snapshot) + '\n'); frames.flush()
            private = snapshot.get('own_state', {})
            if active is not None and snapshot['status'] == 'ok':
                dead_kings = [u['side'] for u in snapshot['units']
                              if u['name'] == 'KingTower' and u['hp'] == 0]
                if dead_kings:
                    report['finished'] += 1
                    emit('MATCH_END', match=active['number'], reason='king_destroyed',
                         dead_king_sides=dead_kings, plays=active['plays'])
                    active = None
                    waiting_for_lobby = True
                    continue
                if private.get('status') != 'ok':
                    tick = snapshot['frame'].get('battle_tick')
                    if tick is not None and tick != active['last_tick']:
                        active['last_tick'] = tick
                        active['last_progress'] = time.monotonic()
                    elif tick is not None and time.monotonic()-active['last_progress'] >= 4:
                        if not frozen_result_confirmed():
                            time.sleep(.1)
                            continue
                        report['finished'] += 1
                        emit('MATCH_END', match=active['number'], reason='tick_frozen_hand_unavailable',
                             tick=tick, plays=active['plays'])
                        active = None
                        waiting_for_lobby = True
                        continue
            if snapshot['status'] == 'no_battle':
                if active is not None:
                    report['finished'] += 1
                    emit('MATCH_END', match=active['number'], reason='world_unloaded', plays=active['plays'])
                    active = None
                waiting_for_lobby = False
                time.sleep(.25)
                continue
            if snapshot['status'] != 'ok' or private.get('status') != 'ok':
                time.sleep(.1)
                continue
            if waiting_for_lobby:
                time.sleep(.2)
                continue
            now = time.monotonic()
            if active is None:
                if private['tick'] > 200 and not (args.attach_current and report['started'] == 0):
                    emit('BLOCKED', reason='not_a_new_match', tick=private['tick'])
                    waiting_for_lobby = True
                    continue
                report['started'] += 1
                adapter = NativeModelAdapter(own_side=private['own_side'], allow_incomplete_history=True,
                                             allow_unknown_profiles=True, coordinate_mode=policy.coordinate_mode,
                                             geometry=policy.geometry,cannon_placement_mode=CENTRAL_MODE)
                active = {'number': report['started'], 'world': private['world'], 'last_tick': -1,
                          'last_progress': now, 'last_decide': -100, 'last_idle': now,
                          'plays': 0, 'cards': {}, 'seen_towers': set(), 'missing_towers': {},
                          'destroyed': set(), 'pending': {}, 'slot_cooldowns': {},
                          'budget': PlacementBudget(),
                          'warnings': set()}
                report['matches'].append({'number': active['number'], 'world': active['world'], 'plays': []})
                emit('MATCH_START', match=active['number'], checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_sha256, side=private['own_side'], tick=private['tick'], hand=own_context(private, deck).hand)
            if (private['world'] == active['world'] and
                    0 < active['last_tick'] - private['tick'] <= 20):
                emit('STALE_FRAME_SKIPPED', match=active['number'],
                     tick=private['tick'], last_tick=active['last_tick'])
                time.sleep(.1)
                continue
            if private['world'] != active['world'] or private['tick'] < active['last_tick']:
                raise RuntimeError('Unexpected world/tick change during match; manual review required')
            if private['tick'] != active['last_tick']:
                active['last_progress'] = now
                active['last_tick'] = private['tick']
            elif now - active['last_progress'] >= 4:
                if not frozen_result_confirmed():
                    time.sleep(.1)
                    continue
                report['finished'] += 1
                emit('MATCH_END', match=active['number'], reason='tick_frozen_needs_visual_result_confirmation',
                     plays=active['plays'], tick=private['tick'], cards=active['cards'])
                active = None
                waiting_for_lobby = True
                continue
            # A frozen tick is never a reason to repeat input.
            if now - active['last_progress'] > .3:
                time.sleep(.1)
                continue
            active['budget'].observe(private['tick'],private['elixir_raw']/10000)
            for pending_slot,pending in list(active['pending'].items()):
                if private['hand_indices'][pending_slot] != pending['hand'][pending_slot]:
                    ack_ns=snapshot['received_host_ns']
                    emit('PLAY_ACK', match=active['number'], card=pending['card'], tick=private['tick'],
                         confirmation='native_hand_rotation', delay_ms=(now-pending['sent'])*1000,
                         input_return_to_ack_ms=(ack_ns-pending['input_return_ns'])/1e6,
                         source_tick=pending['source_tick'],
                         source_to_ack_game_ms=(private['tick']-pending['source_tick'])*50)
                    if pending['sample'] and (capture_job is None or capture_job.done()):
                        capture_job=pool.submit(diagnostic_screenshot,reader,adb,args.output,
                            pending['sample_name']+'_ack',phase='after_ack',
                            input_host_ns=pending['input_return_ns'],source_tick=pending['source_tick'],
                            ack_host_ns=ack_ns)
                    del active['pending'][pending_slot]
                elif now - pending['sent'] > 4:
                    emit('PLAY_NOT_CONFIRMED', match=active['number'], card=pending['card'],
                         tick=private['tick'], detail='fresh hand unchanged; reconsider other slots')
                    active['slot_cooldowns'][pending['slot']] = now + 2
                    del active['pending'][pending_slot]
            active['budget'].reconcile(private['elixir_raw']/10000,all_confirmed=not active['pending'])
            if private['tick'] <= active['last_decide']:
                time.sleep(.1)
                continue
            active['last_decide'] = private['tick']
            # Persistent removal of an initially observed static crown body in
            # the same guarded world is the trial's destruction evidence.
            current = set()
            rotated = next(u['y'] for u in snapshot['units'] if u['name']=='KingTower' and u['side']==private['own_side']) < 16000
            for u in snapshot['units']:
                if u['name'] == 'PrincessTower':
                    x = 18000-u['x'] if rotated else u['x']
                    key = (u['side'], 'left' if x < 9000 else 'right')
                    current.add(key); active['seen_towers'].add(key)
                    if u['hp'] == 0: active['destroyed'].add(key)
            for key in active['seen_towers'] - current:
                since = active['missing_towers'].setdefault(key, now)
                if now-since >= .5: active['destroyed'].add(key)
            for key in current: active['missing_towers'].pop(key, None)
            try:
                prepare_start=time.perf_counter_ns()
                own = own_context(private, deck, destroyed=active['destroyed'])
                own = replace(own,elixir=replace(own.elixir,reserved=max(0.,
                    own.elixir.minimum-active['budget'].value),pending_actions=len(active['pending'])))
                view = adapter.encode(snapshot, own)
                prepare_ms=(time.perf_counter_ns()-prepare_start)/1e6
            except MissingModelInput as exc:
                if now-active['last_idle'] > 2:
                    emit('INPUT_BLOCKED', match=active['number'], reason=str(exc)); active['last_idle']=now
                continue
            for warning in view.metadata['warnings']:
                if warning not in active['warnings']:
                    emit('INPUT_WARNING', match=active['number'], reason=warning)
                    active['warnings'].add(warning)
            for slot, until in active['slot_cooldowns'].items():
                if now < until:
                    view.action_mask[1+slot*576:1+(slot+1)*576] = False
            for slot in active['pending']:
                view.action_mask[1+slot*576:1+(slot+1)*576] = False
            inference_start=time.perf_counter_ns()
            capture=None
            if decision_capture is not None:
                context=dict(match=active['number'],tick=private['tick'],host_ns=time.time_ns(),
                    checkpoint_sha256=checkpoint_sha256,hand=list(own.hand),
                    reserved_elixir=own.elixir.reserved,pending_slots=list(active['pending']),
                    coordinate_mode=policy.coordinate_mode,geometry=policy.geometry)
                capture=lambda planes,scalars,mask,result: decision_capture.capture(
                    planes,scalars,mask,result,context)
            decision = policy.decide(view,capture=capture)
            inference_ms=(time.perf_counter_ns()-inference_start)/1e6
            placement = decision['placement']
            if placement is None:
                if now-active['last_idle'] > 5:
                    emit('IDLE', match=active['number'], tick=private['tick'], elixir=own.elixir.value,
                         hand=own.hand, next_card=own.next_card, entities=len(snapshot['units']),
                         reason='policy_hold' if view.action_mask[1:].any() else 'mask_hold',
                         legal_actions_per_slot=[int(view.action_mask[1+i*576:1+(i+1)*576].sum()) for i in range(4)])
                    active['last_idle']=now
                continue
            slot = placement['slot']; card = own.hand[slot]
            select, target = tap_points(slot, placement['grid_x'], placement['grid_y'], args.width, args.height)
            timing={'parser_ms':snapshot['round_trip_ms'],
                    'native_read_ms':snapshot['frame']['read_us']/1000,
                    'prepare_ms':prepare_ms,'inference_ms':inference_ms}
            sample=args.screenshot_every and (active['plays'] % args.screenshot_every == 0)
            sample_name=f"m{active['number']:02d}_p{active['plays']+1:03d}"
            if args.execute:
                # Only Android input, exactly two taps. No Battle/menu actions.
                select_start=time.perf_counter_ns()
                subprocess.run(adb+['shell', 'input', 'tap', str(select[0]), str(select[1])],check=True,timeout=3)
                timing['adb_select_ms']=(time.perf_counter_ns()-select_start)/1e6
                target_start=time.perf_counter_ns()
                target_request_ns=time.time_ns()
                subprocess.run(adb+['shell', 'input', 'tap', str(target[0]), str(target[1])],check=True,timeout=3)
                target_return_ns=time.time_ns()
                timing['adb_target_ms']=(time.perf_counter_ns()-target_start)/1e6
                timing['parser_request_to_input_return_ms']=(target_return_ns-snapshot['request_host_ns'])/1e6
                timing['parser_response_to_input_start_ms']=(target_request_ns-snapshot['received_host_ns'])/1e6
                active['budget'].spend(adapter.cards[card].cost)
                active['pending'][slot] = {'hand': list(private['hand_indices']), 'card': card, 'slot': slot, 'sent': time.monotonic(),
                    'input_return_ns':target_return_ns,'source_tick':private['tick'],
                    'sample':bool(sample),'sample_name':sample_name}
                if sample and (capture_job is None or capture_job.done()):
                    capture_job=pool.submit(diagnostic_screenshot,reader,adb,args.output,
                        sample_name+'_input',phase='after_input',input_host_ns=target_return_ns,
                        source_tick=private['tick'])
            active['plays'] += 1
            active['cards'][card] = active['cards'].get(card,0)+1
            play = {'tick': private['tick'], 'card': card, 'slot': slot, 'grid': [placement['grid_x'],placement['grid_y']], 'target': target, 'elixir': own.elixir.value,'timing':timing}
            report['matches'][-1]['plays'].append(play)
            emit('PLAY' if args.execute else 'WOULD_PLAY', match=active['number'], **play)
            time.sleep(max(0,.1-(time.monotonic()-began)))
        emit('STOP', reason='max_matches' if report['finished'] >= args.max_matches else 'time_limit', finished=report['finished'])
    except (Exception, KeyboardInterrupt) as exc:
        emit('STOP', reason=type(exc).__name__, detail=str(exc), finished=report['finished'])
        raise
    finally:
        if decision_capture is not None:decision_capture.close()
        pool.shutdown(wait=True)
        log.close(); frames.close()


if __name__ == '__main__':
    main()
