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


def test_frozen_controller_runs_only_two_selected_full_trials(monkeypatch,tmp_path):
    """Exercise real controller flow with GPU/trainer replaced by explicit spies."""
    import json
    import torch
    import sys
    monkeypatch.setattr(sys,'path',list(sys.path))
    inputs=tmp_path/'input/study';inputs.mkdir(parents=True)
    working=tmp_path/'working';working.mkdir()
    trials=[dict(trial=dict(id=name,focus='uniform',entropy_coef=.01))for name in ('a','b')]
    (inputs/'initial_selection.json').write_text(json.dumps(dict(selected='parent')))
    (inputs/'selection.json').write_text(json.dumps(dict(gate_met=True,selected=trials)))
    (inputs/'protocol.json').write_text('{}')
    tree=ast.parse(module.full_only_bootstrap(inspect.getsource(execute)))
    func=tree.body[0]
    # This spy makes no model updates. Reported counters below are routing
    # assertions inside a test, never evidence of actual training completion.
    index=next(i for i,n in enumerate(func.body)if isinstance(n,ast.FunctionDef)and n.name=='run')
    func.body[index]=ast.parse("def run(trial,device,seed,full=False):\n return dict(trial=trial,device=device,seed=seed,full=full)").body[0]
    class TestPaths(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value=='/kaggle/input':return ast.copy_location(ast.Constant(str(tmp_path/'input')),node)
            if node.value=='/kaggle/working/focus_project':return ast.copy_location(ast.Constant(str(working/'project')),node)
            if node.value=='/kaggle/working/continuous_focus':return ast.copy_location(ast.Constant(str(working/'output')),node)
            return node
    tree=TestPaths().visit(tree);ast.fix_missing_locations(tree)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'TEST_ONLY_NO_CUDA')
    namespace=dict(Path=Path,json=json,sha=module.sha,prepare_source=lambda source,out,pin:out.mkdir())
    exec(compile(tree,'test_full_routing.py','exec'),namespace)
    namespace['execute'](dict(owner='owner',slug='study',source_slug='study',pins={'protocol.json':module.sha(inputs/'protocol.json')},
        replaced={},extensions={},source_manifest_sha256='test',models=[dict(id='parent',file='parent.pt')],allow_full_training=True))
    result=json.loads((working/'output/result.json').read_text())
    assert result['pilot_steps']==0
    assert len(result['full_runs'])==2
    assert [(r['trial']['id'],r['device'],r['seed'],r['full'])for r in result['full_runs']]==[
        ('a','cuda:0',64,True),('b','cuda:1',64,True)]
