import json
from types import SimpleNamespace

import pytest

from scripts.attack import watch_bank_match_search as module


def test_missing_search_stops_without_creating_dataset(tmp_path,monkeypatch):
    state=tmp_path/'state.json'
    state.write_text(json.dumps(dict(pid=123,root='job',shards=[str(tmp_path/'job')])))
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='',returncode=1))
    monkeypatch.setattr(module,'verify',lambda *a,**k:pytest.fail('Incomplete search must not be verified'))
    with pytest.raises(RuntimeError,match='absent'):
        module.watch(state,'base',tmp_path/'verified.json',tmp_path/'replays',tmp_path/'dataset')
    assert json.loads(state.read_text())['postprocess_phase']=='error'
    assert not (tmp_path/'dataset').exists()


def test_completed_search_advances_once_and_keeps_promotion_false(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    shard=tmp_path/'job';shard.mkdir();(shard/'result.json').write_text('{}')
    state=tmp_path/'state.json';state.write_text(json.dumps(dict(pid=123,root=str(shard),shards=[str(shard)])))
    calls=[]
    def verify(*args):
        calls.append('verify');return dict(root='collection',accepted=[dict(index=5,context_path='context')])
    def replay(context,output):calls.append('replay');output.write_text('{}')
    def assemble(*args):
        calls.append('assemble');return dict(examples=1061,correction_examples=1,data_sha256='fixture')
    monkeypatch.setattr(module,'verify',verify);monkeypatch.setattr(module,'replay',replay)
    monkeypatch.setattr(module,'assemble',assemble)
    monkeypatch.setattr(module,'replayed_example',lambda *a:calls.append('check_example'))
    monkeypatch.setattr(module,'render',lambda *a:'dashboard')
    module.watch(state,'base',tmp_path/'verified.json',tmp_path/'replays',tmp_path/'dataset')
    assert calls==['verify','replay','check_example','assemble']
    result=json.loads(state.read_text())
    assert result['postprocess_phase']=='complete'and result['full_training_accepted']is False
    assert (tmp_path/'done.md').read_text()=='dashboard'


def test_existing_observer_is_not_duplicated(tmp_path,monkeypatch):
    state=tmp_path/'state.json';state.write_text(json.dumps(dict(postprocess_pid=321)))
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='python -m scripts.attack.watch_bank_match_search'))
    with pytest.raises(RuntimeError,match='already running'):
        module.watch(state,'base','verified','replays','dataset')
