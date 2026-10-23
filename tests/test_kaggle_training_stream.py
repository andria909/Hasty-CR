import json
from pathlib import Path
import runpy
import sys
import types

import pytest


@pytest.mark.parametrize('error_name,status', [('SSLError', None), ('HTTPError', 404)])
def test_transient_stream_failure_reconnects_without_losing_or_regressing_progress(
        tmp_path, monkeypatch, error_name, status):
    error_type = type(error_name, (Exception,), {})
    calls = []

    class API:
        def authenticate(self):
            pass

        def kernels_logs_stream(self, kernel):
            calls.append(kernel)
            if len(calls) == 1:
                yield dict(data='ppo: {"update": 2, "step": 2048}', time=2)
                error = error_type()
                error.response = types.SimpleNamespace(status_code=status)
                raise error
            # Replay is normal when reconnecting; it must not regress counters.
            yield dict(data='ppo: {"update": 1, "step": 1024}', time=1)
            yield dict(data='ppo: {"update": 3, "step": 3072}', time=3)

    module = types.ModuleType('kaggle.api.kaggle_api_extended')
    module.KaggleApi = API
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr('time.sleep', lambda seconds: None)
    output = tmp_path/'stream'
    monkeypatch.setattr(sys, 'argv', ['stream_training.py', '--kernel', 'owner/job',
                                     '--output', str(output), '--reconnects', '1'])
    script = Path(__file__).resolve().parents[1]/'scripts/kaggle/stream_training.py'
    runpy.run_path(str(script), run_name='__main__')
    progress = json.loads((output/'progress.json').read_text())
    assert calls == ['owner/job', 'owner/job']
    assert progress['trainers']['ppo']['steps'] == 3072
    assert progress['stream_state'] == 'ended'
    assert 'error_type' not in progress
    assert len((output/'events.jsonl').read_text().splitlines()) == 3


def test_normal_connection_end_is_reobserved_at_twenty_minute_interval(tmp_path,monkeypatch):
    calls=[];sleeps=[]
    class API:
        def authenticate(self):pass
        def kernels_logs_stream(self,kernel):
            calls.append(kernel)
            yield dict(data='grpo: '+json.dumps(dict(update=len(calls),steps=100*len(calls))),time=len(calls))
    module=types.ModuleType('kaggle.api.kaggle_api_extended');module.KaggleApi=API
    monkeypatch.setitem(sys.modules,module.__name__,module)
    monkeypatch.setattr('time.sleep',lambda seconds:sleeps.append(seconds))
    output=tmp_path/'stream'
    monkeypatch.setattr(sys,'argv',['stream_training.py','--kernel','owner/job','--output',str(output),'--reconnects','1'])
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/kaggle/stream_training.py'),run_name='__main__')
    assert len(calls)==2 and sleeps==[1200]
    state=json.loads((output/'progress.json').read_text())
    assert state['trainers']['grpo']['steps']==200 and state['stream_state']=='ended'


def test_distinct_coefficient_trials_have_actual_percent_without_replay_regression():
    from scripts.kaggle.stream_training import record_trial_event
    progress={}
    def event(trial,step,update,percent=999):return json.dumps(dict(event='TRIAL_PROGRESS',trial_id=trial,step=step,update=update,percent=percent))
    assert record_trial_event(progress,event('defense_baseline_s42',2048,2),10)
    assert record_trial_event(progress,event('joint_baseline_s42',1024,1),11)
    assert progress['trials']['defense_baseline_s42']['percent']==2
    assert progress['trials']['joint_baseline_s42']['percent']==1
    assert not record_trial_event(progress,event('defense_baseline_s42',1024,1),9)
    assert not record_trial_event(progress,event('defense_baseline_s42',204800,3),12)
    assert record_trial_event(progress,json.dumps(dict(event='FULL_TRAINING_PROGRESS',mode='joint',profile='baseline',step=102400,update=100)),20)
    assert progress['trials']['full_joint_baseline']['percent']==5
