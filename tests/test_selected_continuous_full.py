import ast
import inspect
from pathlib import Path
import pytest
from scripts.kaggle.prepare_continuous_focus import execute
from scripts.kaggle import prepare_selected_continuous_full as module


def test_full_controller_preserves_training_and_skips_reference_pilot_calls():
    source=inspect.getsource(execute)
    before=ast.parse(source).body[0]
    old_run=next(n for n in before.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    changed=module.full_only_bootstrap(source)
    compile(changed,'full_controller.py','exec')
    after=ast.parse(changed).body[0]
    new_run=next(n for n in after.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    assert ast.dump(old_run)==ast.dump(new_run)
    calls=[n.func.id for n in ast.walk(after)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)]
    assert 'select_profiles'not in calls and 'select_initial'not in calls
    assert not any(isinstance(n,ast.FunctionDef)and n.name in ('pilots','ref_group')for n in after.body)
    assert "inputs / 'selection.json'"in changed
    assert 'pilot_steps=0'in changed


def test_failed_gate_cannot_create_a_full_package(monkeypatch,tmp_path):
    monkeypatch.setattr(module,'verify',lambda *a:dict(pilot_selection=dict(gate_met=False,selected=[])))
    target=tmp_path/'uncreated'
    with pytest.raises(ValueError,match='eligible'):
        module.prepare(tmp_path/'pilot',target,'owner')
    assert not target.exists()


def test_full_verifier_rejects_failed_original_gate_before_artifacts(monkeypatch,tmp_path):
    import json
    from scripts.attack import verify_selected_continuous_full as verifier
    proof=dict(pilot_selection=dict(gate_met=False,selected=[]))
    prepared=tmp_path/'prepared';(prepared/'dataset').mkdir(parents=True)
    (prepared/'preparation.json').write_text(json.dumps(dict(config={},pilot_prepared=str(tmp_path/'pilot'))))
    (prepared/'dataset/pilot_verified.json').write_text(json.dumps(proof))
    monkeypatch.setattr(verifier,'verify_pilot',lambda *a:proof)
    with pytest.raises(ValueError,match='eligible'):
        verifier.verify(tmp_path/'missing_artifacts',prepared)
