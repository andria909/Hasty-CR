"""Run an explicitly configured finite sequence of model series, one at a time.

Only verified lobby/result controls are tapped. Does not attach to a human's
ongoing match, retry failed series, or count a pre-existing match.
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


def run(config_path):
    c=json.loads(Path(config_path).read_text());root=Path(c['root']);state_path=root/'comparison.json'
    if state_path.exists():raise ValueError('Existing comparison must be inspected, never reset')
    models=c['models']
    if len(models)!=2 or any(m['matches']!=10 for m in models):raise ValueError('Explicit two-by-ten comparison required')
    for m in models:
        if hashlib.sha256(Path(m['checkpoint']).read_bytes()).hexdigest()!=m['sha256']:raise ValueError('Checkpoint changed')
        if (root/m['label']).exists():raise ValueError('Preserve existing model series')
    adb=[c['adb'],'-s',c['serial']]
    reader=CurrentStateReader(c['adb'],c['serial'],own_state=True,su=c['su'],binary=c['binary'])
    lobby=Path(c['lobby_reference']).read_bytes();result=Path(c['result_reference']).read_bytes()
    state=dict(pid=os.getpid(),status='waiting_verified_lobby',models=models,series=[])
    children=[]
    def save(**kw):
        state.update(kw,updated_host_ns=time.time_ns());state_path.write_text(json.dumps(state,indent=2)+'\n')
    def screenshot():return subprocess.check_output(adb+['exec-out','screencap','-p'],timeout=10)
    def tap(x,y):subprocess.run(adb+['shell','input','tap',str(x),str(y)],check=True,timeout=5)
    def stop(signum,frame):raise KeyboardInterrupt('Comparison interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    save()
    try:
        for index,m in enumerate(models):
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
            folder=root/m['label'];folder.mkdir();(folder/'before_first.png').write_bytes(png)
            base=['--adb',c['adb'],'--serial',c['serial'],'--su',c['su'],'--binary',c['binary'],'--width','1080','--height','1920','--result-reference',c['result_reference']]
            runner_cmd=[sys.executable,'-m','scripts.gamestate.memory_play',*base,'--checkpoint',m['checkpoint'],'--max-matches','10','--execute','--seconds','18000','--screenshot-every','5','--output',str(folder/'run')]
            with (folder/'runner.log').open('x') as log:runner=subprocess.Popen(runner_cmd,stdout=log,stderr=subprocess.STDOUT)
            children.append(runner)
            for _ in range(100):
                if runner.poll() is not None:raise RuntimeError('Runner stopped before READY')
                if (folder/'run/summary.json').exists():break
                time.sleep(.1)
            else:raise TimeoutError('No runner READY')
            summary=json.loads((folder/'run/summary.json').read_text())
            if summary['checkpoint_sha256']!=m['sha256'] or summary['started']!=0:raise ValueError('Unexpected runner state before queue')
            supervisor_cmd=[sys.executable,'-m','scripts.gamestate.memory_series',*base,'--existing',str(folder/'run'),'--total','10','--output',str(folder/'series'),'--lobby-reference',c['lobby_reference']]
            with (folder/'supervisor.log').open('x') as log:supervisor=subprocess.Popen(supervisor_cmd,stdout=log,stderr=subprocess.STDOUT)
            children.append(supervisor)
            state['series'].append(dict(model=m['label'],runner_pid=runner.pid,supervisor_pid=supervisor.pid,path=str(folder/'series/series.json')))
            save(status='running',active_model=m['label'])
            png=screenshot()
            if reader.query()['status']!='no_battle' or not lobby_button_matches(lobby,png):raise RuntimeError('Lobby changed before initial queue')
            tap(540,1455)
            for _ in range(120):
                if runner.poll() is not None or supervisor.poll() is not None:raise RuntimeError('Live worker stopped during queue')
                if json.loads((folder/'run/summary.json').read_text())['started']>0:break
                time.sleep(.5)
            else:raise TimeoutError('No match after one initial Battle tap')
            while supervisor.poll() is None:
                if runner.poll() is not None:
                    summary=json.loads((folder/'run/summary.json').read_text())
                    if summary['finished']!=10:raise RuntimeError('Runner stopped early')
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
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(a.config)
