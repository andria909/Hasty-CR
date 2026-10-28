"""Run an explicitly configured finite sequence of model series, one at a time.

Only verified lobby/result controls are tapped. Explicit resume preserves counts
and requires all prior matches to have ended and their workers to have stopped.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from scripts.gamestate.memory_series import lobby_button_matches
from scripts.gamestate.query_arm64 import CurrentStateReader


def remaining_matches(saved, old):
    if saved['requested']!=10 or not 0<=saved['finished']<=10:
        raise ValueError('Invalid authorized count')
    if len(saved['matches'])!=saved['finished'] or sum(saved[k] for k in ('wins','losses','unknown'))!=saved['finished']:
        raise ValueError('Inconsistent saved results')
    if old['started']!=old['finished']:
        raise ValueError('Unfinished match requires inspection')
    if sum(x['run']==saved['runs'][-1] for x in saved['matches'])!=old['finished']:
        raise ValueError('Uncounted results')
    return 10-saved['finished']


def run(config_path, resume=False):
    c=json.loads(Path(config_path).read_text());root=Path(c['root']);state_path=root/'comparison.json'
    prior=json.loads(state_path.read_text()) if state_path.exists() else None
    if prior and not resume:raise ValueError('Existing comparison must be inspected, never reset')
    if resume and (not prior or prior['status']!='stopped_error'):raise ValueError('Only stopped comparisons may resume')
    if prior:
        pids=[prior['pid']]+[row[key] for row in prior['series'] for key in ('runner_pid','supervisor_pid')]
        for pid in pids:
            try:os.kill(pid,0)
            except ProcessLookupError:continue
            raise ValueError(f'Previous worker PID {pid} still exists; inspect before resume')
        (root/f'comparison_before_resume_{time.time_ns()}.json').write_text(json.dumps(prior,indent=2)+'\n')
    models=c['models']
    if len(models)!=2 or any(m['matches']!=10 for m in models):raise ValueError('Explicit two-by-ten comparison required')
    for m in models:
        if hashlib.sha256(Path(m['checkpoint']).read_bytes()).hexdigest()!=m['sha256']:raise ValueError('Checkpoint changed')
        if (root/m['label']).exists() and not resume:raise ValueError('Preserve existing model series')
    adb=[c['adb'],'-s',c['serial']]
    reader=CurrentStateReader(c['adb'],c['serial'],own_state=True,su=c['su'],binary=c['binary'])
    lobby=Path(c['lobby_reference']).read_bytes();result=Path(c['result_reference']).read_bytes()
    state=prior or dict(status='waiting_verified_lobby',models=models,series=[])
    if state['models']!=models:raise ValueError('Resume models changed')
    state['pid']=os.getpid()
    state.pop('error_type',None);state.pop('detail',None)
    children=[]
    def save(**kw):
        state.update(kw,updated_host_ns=time.time_ns());state_path.write_text(json.dumps(state,indent=2)+'\n')
    def screenshot():return subprocess.check_output(adb+['exec-out','screencap','-p'],timeout=10)
    def tap(x,y):subprocess.run(adb+['shell','input','tap',str(x),str(y)],check=True,timeout=5)
    def stop(signum,frame):raise KeyboardInterrupt('Comparison interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    save(status='waiting_verified_lobby')
    try:
        for index,m in enumerate(models):
            folder=root/m['label'];saved=None
            if (folder/'series/series.json').exists():
                saved=json.loads((folder/'series/series.json').read_text())
                if saved['requested']!=10 or Path(saved['checkpoint']).resolve()!=Path(m['checkpoint']).resolve():raise ValueError('Resume series mismatch')
                old=json.loads((Path(saved['runs'][-1])/'summary.json').read_text())
                if remaining_matches(saved,old)==0:continue
            remaining=10-(saved['finished'] if saved else 0)
            suffix=f"resume_{time.time_ns()}" if saved else 'run'
            run_path=folder/suffix
            # First series waits without touching any pre-existing human match.
            # Only after our completed series may we dismiss its result screen.
            deadline=time.monotonic()+1800;last_ok=0
            while time.monotonic()<deadline:
                snap=reader.query();png=screenshot()
                if snap['status']=='no_battle' and lobby_button_matches(lobby,png):break
                if index and time.monotonic()-last_ok>10 and lobby_button_matches(result,png,box=(800,2240,1100,2400)):
                    tap(712,1740);last_ok=time.monotonic()
                time.sleep(5)
            else:raise TimeoutError('No verified lobby; no blind queue')
            folder.mkdir(exist_ok=bool(saved));(folder/f'before_{suffix}.png').write_bytes(png)
            base=['--adb',c['adb'],'--serial',c['serial'],'--su',c['su'],'--binary',c['binary'],'--width','1080','--height','1920','--result-reference',c['result_reference']]
            runner_cmd=[sys.executable,'-m','scripts.gamestate.memory_play',*base,'--checkpoint',m['checkpoint'],'--max-matches',str(remaining),'--execute','--seconds','18000','--screenshot-every','5','--output',str(run_path)]
            with (folder/f'{suffix}_runner.log').open('x') as log:runner=subprocess.Popen(runner_cmd,stdout=log,stderr=subprocess.STDOUT)
            children.append(runner)
            for _ in range(100):
                if runner.poll() is not None:raise RuntimeError('Runner stopped before READY')
                if (run_path/'summary.json').exists():break
                time.sleep(.1)
            else:raise TimeoutError('No runner READY')
            summary=json.loads((run_path/'summary.json').read_text())
            if summary['checkpoint_sha256']!=m['sha256'] or summary['started']!=0:raise ValueError('Unexpected runner state before queue')
            supervisor_cmd=[sys.executable,'-m','scripts.gamestate.memory_series',*base,'--existing',str(run_path),'--total','10','--output',str(folder/'series'),'--lobby-reference',c['lobby_reference']]
            if saved:
                (folder/f'series_before_{suffix}.json').write_text(json.dumps(saved,indent=2)+'\n')
                saved['runs'].append(str(run_path))
                (folder/'series/series.json').write_text(json.dumps(saved,indent=2)+'\n')
                supervisor_cmd.append('--resume')
            with (folder/f'{suffix}_supervisor.log').open('x') as log:supervisor=subprocess.Popen(supervisor_cmd,stdout=log,stderr=subprocess.STDOUT)
            children.append(supervisor)
            state['series'].append(dict(model=m['label'],runner_pid=runner.pid,supervisor_pid=supervisor.pid,path=str(folder/'series/series.json')))
            save(status='running',active_model=m['label'])
            png=screenshot()
            if reader.query()['status']!='no_battle' or not lobby_button_matches(lobby,png):raise RuntimeError('Lobby changed before initial queue')
            tap(540,1455)
            for _ in range(120):
                if runner.poll() is not None or supervisor.poll() is not None:raise RuntimeError('Live worker stopped during queue')
                if json.loads((run_path/'summary.json').read_text())['started']>0:break
                time.sleep(.5)
            else:raise TimeoutError('No match after one initial Battle tap')
            while supervisor.poll() is None:
                if runner.poll() is not None:
                    summary=json.loads((run_path/'summary.json').read_text())
                    if summary['finished']!=remaining:raise RuntimeError('Runner stopped early')
                time.sleep(5)
            series=json.loads((folder/'series/series.json').read_text())
            if supervisor.returncode or series['status']!='complete' or series['finished']!=10:raise RuntimeError('Series incomplete; preserve counts')
            runner.wait(timeout=30)
            state['series'][-1]['result']=series;save()
        save(status='complete')
    except BaseException as error:
        save(status='stopped_error',error_type=type(error).__name__,detail=str(error))
        raise
    finally:
        for child in reversed(children):
            if child.poll() is None:child.send_signal(signal.SIGINT)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.config,a.resume)
