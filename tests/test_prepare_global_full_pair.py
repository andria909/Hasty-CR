import ast
import json
import inspect
from pathlib import Path

import pytest
from scripts.kaggle import prepare_global_full_pair as module


def frozen_source():
    from scripts.kaggle.prepare_continuous_focus import execute
    return inspect.getsource(execute)


def execute_tree(source):
    return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='execute')


def test_isolation_keeps_training_function_exact_and_assigns_disjoint_paths_and_gpu():
    source=frozen_source()
    original=next(n for n in execute_tree(source).body if isinstance(n,ast.FunctionDef)and n.name=='run')
    for index in (0,1):
        result=module.isolated_child(source,index)
        compile(result,'child.py','exec')
        child=execute_tree(result)
        run=next(n for n in child.body if isinstance(n,ast.FunctionDef)and n.name=='run')
        assert ast.dump(run)==ast.dump(original)
        assert f'global_full_child_{index}/project' in result
        assert f'global_full_child_{index}/continuous_focus' in result
        assert f"'cuda:{index}'" in result
        assert "len(selection['selected']) == 1" in result
        assert "config['bundle']" in result
        assert 'def pilots('not in result
    with pytest.raises(ValueError):module.isolated_child(source,2)


def test_failed_global_gate_creates_no_full_bundle(monkeypatch,tmp_path):
    monkeypatch.setattr(module,'select',lambda *a,**kw:dict(gate_met=False))
    with pytest.raises(ValueError,match='independently eligible'):
        module.prepare([],tmp_path/'full','owner')
    assert not(tmp_path/'full').exists()


def test_pair_never_uses_a_lone_third_candidate(monkeypatch,tmp_path):
    monkeypatch.setattr(module,'select',lambda *a,**kw:dict(gate_met=True,selected=[{}, {}, {}]))
    with pytest.raises(ValueError,match='two selected'):
        module.prepare([],tmp_path/'full','owner',pair_index=1)
    assert not(tmp_path/'full').exists()


def test_cross_cohort_duplicate_profile_names_keep_their_own_dataset(monkeypatch,tmp_path):
    chosen=[];proofs={}
    trial=dict(id='joint_replay100',warmstart_updates=800,supervised_replay_weight=1.)
    for index in (0,1):
        root=tmp_path/f'pilot{index}';(root/'dataset').mkdir(parents=True);(root/'job').mkdir()
        for name,data in [('parent.pt',b'parent'),('source.tar.gz',b'archive'),
                          ('manifest.json',b'{}'),('protocol.json',b'{}'),
                          ('air_examples.npz',f'different-data-{index}'.encode())]:
            (root/'dataset'/name).write_bytes(data)
        pins={name:module.sha(root/'dataset'/name)for name in ('parent.pt','manifest.json','protocol.json','air_examples.npz')}
        config=dict(owner='old',slug='old',source_slug='old',pins=pins,
                    models=[dict(id='parent',file='parent.pt',sha256=module.sha(root/'dataset/parent.pt'))])
        (root/'job/train.py').write_text(frozen_source()+'\nexecute({})\n')
        (root/'preparation.json').write_text(json.dumps(dict(config=config,bootstrap_sha256=module.sha(root/'job/train.py'))))
        proofs[root]=dict(initial_selection=dict(selected='parent'),pilot_selection=dict(ranking=[dict(trial=trial,eligible=True)]))
        chosen.append(dict(prepared=str(root),trial=trial))
    monkeypatch.setattr(module,'select',lambda *a,**kw:dict(gate_met=True,selected=chosen))
    monkeypatch.setattr(module,'verify',lambda collected,root:proofs[root])
    output=tmp_path/'full';result=module.prepare([],output,'newowner')
    assert result['full_steps']==8192000 and not result['submitted']
    assert result['requires_exact_package_preflight']
    assert result['requires_dedicated_collector_and_verifier']
    assert not(output/'job/kernel-metadata.json').exists()
    assert (output/'dataset/bundle_0/air_examples.npz').read_bytes()==b'different-data-0'
    assert (output/'dataset/bundle_1/air_examples.npz').read_bytes()==b'different-data-1'
    for child in result['children']:
        assert child['selected']['trial']==trial
        assert child['config']['owner']=='newowner'
        assert child['config']['bundle']==f"bundle_{child['index']}"
