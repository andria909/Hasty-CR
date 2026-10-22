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
