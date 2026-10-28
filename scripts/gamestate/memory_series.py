"""Finite authorized series supervisor for the verified MuMu UI calibration.

Continues an existing memory_play run, then starts only the remaining count.
No model training, CV inference, game-memory writes or data deletion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from io import BytesIO


def lobby_button_matches(reference, current, box=(480,1870,970,2000)):
    """Compare a verified lobby button only; this never supplies policy input."""
    from PIL import Image, ImageChops, ImageStat
    def crop(blob):
        im=Image.open(BytesIO(blob)).convert('RGB')
        if im.size not in ((1440,2560),(1080,1920)):
            raise ValueError('Unexpected Android resolution')
        if im.size != (1440,2560): im=im.resize((1440,2560))
        return im.crop(box).resize((98,26))
    return sum(ImageStat.Stat(ImageChops.difference(crop(reference),crop(current))).mean)/3 < 12

from scripts.gamestate.query_arm64 import CurrentStateReader
from scripts.gamestate.summarize_memory_run import rows, summarize


def terminal_result(snapshot, side, initial_towers, reason):
    """Only infer results supported by crown-body evidence; otherwise unknown."""
    units = snapshot.get('units', [])
    kings = {u['side']: u for u in units if u['name'] == 'KingTower'}
    if len(kings) == 2 and all(u['hp'] == 0 for u in kings.values()):
        return {'result': 'unknown', 'evidence': 'both_kings_zero'}
    for s, king in kings.items():
        if king['hp'] == 0:
            return {'result': 'loss' if s == side else 'win', 'evidence': 'king_hp_zero'}
    if len(initial_towers) != 4 or reason not in (
            'tick_frozen_needs_visual_result_confirmation', 'tick_frozen_hand_unavailable'):
        return {'result': 'unknown', 'evidence': 'no_confirmed_terminal_crown_comparison'}
    alive = {u['entity_id'] for u in units if u['name'] == 'PrincessTower' and u['hp'] > 0}
    lost = {s: sum(s0 == s and identity not in alive for identity, s0 in initial_towers.items()) for s in (0, 1)}
    if lost[side] != lost[1-side]:
        return {'result': 'win' if lost[1-side] > lost[side] else 'loss',
                'evidence': 'frozen_terminal_crown_body_comparison', 'lost_towers': lost}
    return {'result': 'unknown', 'evidence': 'equal_crowns_tiebreak_not_read', 'lost_towers': lost}


def remaining_runner_command(python, *, adb, serial, checkpoint, remaining, output):
    """Every continuation uses the explicitly selected checkpoint, never defaults."""
    if not Path(checkpoint).is_file() or not 1 <= remaining <= 50:
        raise ValueError('Missing selected checkpoint or invalid remaining count')
    return [python, '-m', 'scripts.gamestate.memory_play', '--adb', adb,
            '--serial', serial, '--checkpoint', str(Path(checkpoint).resolve()),
            '--max-matches', str(remaining), '--execute', '--seconds', '18000',
            '--screenshot-every', '5', '--output', str(output)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--existing', type=Path, required=True)
    p.add_argument('--total', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--serial', required=True)
    p.add_argument('--binary', default=None)
    p.add_argument('--su', default=None)
    p.add_argument('--width',type=int,default=1440)
    p.add_argument('--height',type=int,default=2560)
    p.add_argument('--adb', default='/tmp/Hasty-android-tools/platform-tools/adb')
    p.add_argument('--resume', action='store_true', help='Resume saved count after manual recovery; never reset series')
    p.add_argument('--lobby-reference', type=Path, help='Human-verified lobby screenshot for bounded queue retries')
    p.add_argument('--result-reference', type=Path, help='Human-verified result OK button; wait for animation and retry only verified OK')
    a = p.parse_args()
    if not 1 <= a.total <= 50:
        p.error('Finite total must be 1..50')
    existing = json.loads((a.existing/'summary.json').read_text())
    checkpoint = Path(existing['checkpoint']).resolve()
    if not checkpoint.is_file():
        p.error('Existing runner checkpoint is unavailable; refusing fallback')
    if existing['requested'] > a.total:
        p.error('Existing run limit exceeds the authorized total')
    a.output.mkdir(parents=True, exist_ok=a.resume)
    adb = [a.adb, '-s', a.serial]
    reader_kwargs=dict(own_state=True)
    if a.su:reader_kwargs['su']=a.su
    if a.binary:reader_kwargs['binary']=a.binary
    reader = CurrentStateReader(a.adb, a.serial, **reader_kwargs)
    lobby_reference=a.lobby_reference.read_bytes() if a.lobby_reference else None
    result_reference=a.result_reference.read_bytes() if a.result_reference else None
    state = {'requested': a.total, 'finished': 0, 'wins': 0, 'losses': 0, 'unknown': 0,
             'status': 'running', 'runs': [str(a.existing)], 'matches': [],
             'checkpoint': str(checkpoint), 'scope': 'Only matches belonging to this explicitly authorized finite series'}
    if a.resume:
        state=json.loads((a.output/'series.json').read_text())
        if state['requested'] != a.total or state['runs'][-1] != str(a.existing):
            p.error('Resume count/run does not match saved series')
        state['status']='running'
    log = (a.output/'supervisor.jsonl').open('a' if a.resume else 'x')

    def emit(event, **extra):
        row = dict(event=event, host_ns=time.time_ns(), **extra)
        log.write(json.dumps(row)+'\n'); log.flush()
        state['updated_host_ns'] = row['host_ns']
        (a.output/'series.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps(row), flush=True)

    def screenshot(name):
        (a.output/name).write_bytes(subprocess.check_output(adb+['exec-out','screencap','-p'],timeout=10))

    def tap(x, y):
        subprocess.run(adb+['shell','input','tap',str(round(x*a.width/1440)),str(round(y*a.height/2560))],check=True,timeout=5)

    run = a.existing
    offset = sum(m['run'] != str(run) for m in state['matches'])
    seen_ends = {(m['run'],m['local_match']) for m in state['matches']}
    child = None
    child_log = None
    heartbeat = 0
    deadline = time.monotonic()+6*3600
    emit('SERIES_READY', total=a.total, existing=str(run))
    try:
        while time.monotonic() < deadline and state['finished'] < a.total:
            events = rows(run/'events.jsonl')
            starts = {e['match']: e for e in events if e['event']=='MATCH_START'}
            ends = [e for e in events if e['event']=='MATCH_END' and (str(run), e['match']) not in seen_ends]
            if not ends:
                stops = [e for e in events if e['event']=='STOP']
                if stops:
                    raise RuntimeError(f'Runner stopped before expected match end: {stops[-1]}')
                if time.monotonic()-heartbeat > 30:
                    emit('HEARTBEAT', finished=state['finished'], latest=events[-1] if events else None)
                    heartbeat = time.monotonic()
                time.sleep(1)
                continue
            end = ends[0]
            number = offset+end['match']
            if number != state['finished']+1:
                raise RuntimeError('Unexpected match sequence; refusing to queue')
            start = starts[end['match']]
            # Use the last accepted frame from this match at/before MATCH_END.
            initial_towers = {}
            final = {}
            with (run/'frames.jsonl').open() as stream:
                for line in stream:
                    frame = json.loads(line)
                    if not start['host_ns']-100_000_000 <= frame['received_host_ns'] <= end['host_ns']:
                        continue
                    if frame['status'] != 'ok':
                        continue
                    if not initial_towers:
                        towers = {u['entity_id']:u['side'] for u in frame['units'] if u['name']=='PrincessTower'}
                        if len(towers)==4:
                            initial_towers=towers
                    final=frame
            result = terminal_result(final,start['side'],initial_towers,end['reason'])
            state['finished'] += 1
            state[{'win':'wins','loss':'losses','unknown':'unknown'}[result['result']]] += 1
            state['winrate_all_matches'] = state['wins']/state['finished'] if not state['unknown'] else None
            screenshot_name=f'match_{number:02d}_result.png'
            state['matches'].append(dict(number=number,run=str(run),local_match=end['match'],
                                          result_screenshot=screenshot_name,**result))
            seen_ends.add((str(run),end['match']))
            emit('RESULT',number=number,**result)
            time.sleep(6)
            screenshot(screenshot_name)
            (a.output/f'timings_run_{offset:02d}.json').write_text(json.dumps(summarize(run),indent=2)+'\n')
            if state['finished'] >= a.total:
                break
            run_summary=json.loads((run/'summary.json').read_text())
            if run_summary['finished'] >= run_summary['requested']:
                if child is not None:
                    child.wait(timeout=15)
                    child_log.close()
                offset=state['finished']
                run=a.output/'remaining'
                remaining=a.total-offset
                child_log=(a.output/'runner_stdout.log').open('x')
                command=remaining_runner_command(sys.executable,
                    adb=a.adb,serial=a.serial,checkpoint=checkpoint,remaining=remaining,output=run)
                command+=['--width',str(a.width),'--height',str(a.height)]
                if a.binary:command+=['--binary',a.binary]
                if a.su:command+=['--su',a.su]
                if a.result_reference:command+=['--result-reference',str(a.result_reference)]
                child=subprocess.Popen(command,
                    stdout=child_log,stderr=subprocess.STDOUT)
                state['runs'].append(str(run))
                ready_deadline=time.monotonic()+40
                while time.monotonic()<ready_deadline:
                    if child.poll() is not None:
                        raise RuntimeError('Remaining runner failed to start; see runner_stdout.log')
                    if any(e['event']=='READY' for e in rows(run/'events.jsonl')):
                        break
                    time.sleep(.5)
                else:
                    raise RuntimeError('Remaining runner READY timeout')
                emit('RUNNER_READY',remaining=remaining,pid=child.pid)
            if result_reference is not None:
                result_deadline=time.monotonic()+40
                while time.monotonic()<result_deadline:
                    png=subprocess.check_output(adb+['exec-out','screencap','-p'],timeout=10)
                    if lobby_button_matches(result_reference,png,box=(800,2240,1100,2400)):
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError('Result OK did not become visible')
            tap(950,2320)  # verified blue OK on result screen, never Play Again
            lobby_deadline=time.monotonic()+30
            next_ok_retry=time.monotonic()+5
            while time.monotonic()<lobby_deadline:
                if reader.query()['status']=='no_battle':
                    time.sleep(5)  # world unload precedes completion of lobby animations
                    if reader.query()['status']=='no_battle':
                        png=subprocess.check_output(adb+['exec-out','screencap','-p'],timeout=10)
                        if lobby_reference is None or lobby_button_matches(lobby_reference,png):
                            break
                if result_reference is not None and time.monotonic()>=next_ok_retry:
                    png=subprocess.check_output(adb+['exec-out','screencap','-p'],timeout=10)
                    if lobby_button_matches(result_reference,png,box=(800,2240,1100,2400)):
                        tap(950,2320)
                        emit('RESULT_OK_RETRY_VERIFIED',match=number)
                    next_ok_retry=time.monotonic()+5
                time.sleep(.5)
            else:
                raise RuntimeError('Did not reach lobby after result OK; refusing blind Battle input')
            screenshot(f'before_match_{number+1:02d}.png')
            foreground=subprocess.check_output(adb+['shell','dumpsys','activity','activities'],text=True,timeout=10)
            if not any('com.supercell.clashroyale' in line and ('mResumedActivity' in line or 'topResumedActivity' in line) for line in foreground.splitlines()):
                raise RuntimeError('Clash Royale is not resumed; refusing menu input')
            tap(720,1940)
            emit('QUEUE',next_match=number+1)
            start_deadline=time.monotonic()+60
            expected_local=number+1-offset
            next_retry=time.monotonic()+8
            retries=0
            while time.monotonic()<start_deadline:
                if any(e['event']=='MATCH_START' and e['match']==expected_local for e in rows(run/'events.jsonl')):
                    emit('MATCH_CONFIRMED',number=number+1)
                    break
                if lobby_reference is not None and retries<3 and time.monotonic()>=next_retry:
                    snap=reader.query()
                    if snap['status']=='no_battle':
                        png=subprocess.check_output(adb+['exec-out','screencap','-p'],timeout=10)
                        if lobby_button_matches(lobby_reference,png):
                            (a.output/f'retry_match_{number+1:02d}_{retries+1}.png').write_bytes(png)
                            tap(720,1940)
                            retries+=1
                            emit('QUEUE_RETRY_VERIFIED_LOBBY',next_match=number+1,retry=retries)
                    next_retry=time.monotonic()+8
                time.sleep(.5)
            else:
                raise RuntimeError('No MATCH_START after Battle; no repeated menu taps')
        if state['finished'] != a.total:
            raise RuntimeError('Series time limit reached')
        state['status']='complete'
        emit('SERIES_COMPLETE',finished=state['finished'],wins=state['wins'],losses=state['losses'],unknown=state['unknown'])
    except BaseException as exc:
        state['status']='stopped_error'
        emit('SERIES_ERROR',detail=str(exc))
        # Stop only our child. Never kill the game or emulator.
        if child is not None and child.poll() is None:
            child.send_signal(2)
        raise
    finally:
        log.close()


if __name__=='__main__':
    main()
