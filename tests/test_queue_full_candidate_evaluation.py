import json
from pathlib import Path
import pytest
from scripts.kaggle import queue_full_candidate_evaluation as module


def config(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'manifest.json').write_text('{}')
    pairs=[]
    for i in (0,1):
        root=tmp_path/f'full{i}';root.mkdir()
        pairs.append(dict(prepared=str(root),verified_report=str(tmp_path/f'proof{i}.json'),kernel=f'owner/full{i}'))
    return dict(schema='queued-full-candidate-evaluation-v1',owner='andriashadrin',token_index=0,
        source=str(source),source_manifest_sha256=module.file_sha(source/'manifest.json'),code_pins={},full_pairs=pairs,
        **{n:str(tmp_path/n) for n in ('binding','prepared','preflight_output')})


def finish(entry, guard=False):
    root=Path(entry['prepared']);(root/'collected').mkdir()
    (root/'collected/collection_receipt.json').write_text(json.dumps(dict(kernel=entry['kernel'],remote_status='COMPLETE',complete=True)))
    Path(entry['verified_report']).write_text(json.dumps(dict(schema='verified-global-selected-full-pair-v1',full_steps_verified=8192000,kernel=entry['kernel'],ready_for_independent_evaluation=guard)))


def test_waits_for_both_full_pairs_but_keeps_failed_family_guards(tmp_path):
    c=config(tmp_path);module.check_config(c);assert not module.ready(c)
    finish(c['full_pairs'][0]);assert not module.ready(c)
    finish(c['full_pairs'][1]);assert module.ready(c)


def test_foreign_terminal_proof_rejected(tmp_path):
    c=config(tmp_path);finish(c['full_pairs'][0]);finish(c['full_pairs'][1])
    p=Path(c['full_pairs'][0]['verified_report']);r=json.loads(p.read_text());r['kernel']='foreign';p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='proof/terminal'):module.ready(c)


def test_partial_previous_launch_is_not_retried(tmp_path):
    c=config(tmp_path);finish(c['full_pairs'][0]);finish(c['full_pairs'][1]);Path(c['binding']).mkdir()
    with pytest.raises(ValueError,match='no duplicate launch'):module.execute(c,lambda *a,**kw:None)


def test_changed_code_prevents_submission(tmp_path):
    c=config(tmp_path);code=tmp_path/'code.py';code.write_text('original');c['code_pins']={str(code):module.file_sha(code)}
    code.write_text('changed')
    with pytest.raises(ValueError,match='implementation changed'):module.check_config(c)


def test_ordered_pipeline_launches_once_then_starts_observers(tmp_path,monkeypatch):
    from scripts.attack import bind_full_candidate_evaluation as binder
    from scripts.kaggle import prepare_full_candidate_evaluation as packer
    from scripts.kaggle import preflight_full_candidate_evaluation as preflight
    from scripts.kaggle import start_full_evaluation_observers as observers
    c=config(tmp_path);finish(c['full_pairs'][0]);finish(c['full_pairs'][1])
    c.update(slug='hastycr-eval',report=str(tmp_path/'report.json'))
    calls=[];stages=[]
    def bind(*args):calls.append('bind');return dict(candidates=4,total_games=6300)
    monkeypatch.setattr(binder,'bind',bind)
    monkeypatch.setattr(packer,'prepare',lambda *args:calls.append('prepare'))
    monkeypatch.setattr(preflight,'preflight',lambda *args:calls.append('preflight'))
    monkeypatch.setattr(preflight,'enable_submission',lambda *args:calls.append('enable'))
    def submit(command,**kwargs):
        assert '--evaluation' in command and kwargs['check'];calls.append('submit')
    monkeypatch.setattr(module.subprocess,'run',submit)
    def start(*args):calls.append('observers');return dict(kernel='andriashadrin/hastycr-eval',pids={'collector':123})
    monkeypatch.setattr(observers,'start',start)
    module.execute(c,lambda stage,**kw:stages.append(stage))
    assert calls==['bind','prepare','preflight','enable','submit','observers']
    assert stages[-1]=='evaluation_submitted_observers_running'
