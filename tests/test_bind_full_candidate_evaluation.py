from copy import deepcopy
import json
from pathlib import Path

import pytest
from scripts.attack import bind_full_candidate_evaluation as module
from tests.test_full_candidate_evaluator import blueprint


def test_corrected_blueprint_rejects_old_training_before_creating_output(tmp_path,monkeypatch):
    from scripts.attack.evaluation_execution import CORRECTED
    bp,roots,proofs,calls=inputs(tmp_path,monkeypatch,1)
    spec=json.loads(bp.read_text());spec.update(schema='precommitted-full-candidate-evaluation-v3',execution=CORRECTED)
    spec['evaluation_runtime'].update({k:v for k,v in CORRECTED.items()if k!='schema'})
    bp.write_text(json.dumps(spec));out=tmp_path/'corrected'
    with pytest.raises(ValueError,match='training physics'):module.bind(roots,out,blueprint_path=bp)
    assert not out.exists()
    for proof in proofs.values():
        for model in proof['models']:model['training_execution']=deepcopy(CORRECTED)
    module.bind(roots,out,blueprint_path=bp)
    p=json.loads((out/'protocol.json').read_text())
    assert p['schema']=='bound-full-candidate-evaluation-v3'
    module.validate_protocol(p,out/'weights')
    p['models'][0]['training_execution']=dict(CORRECTED,normal_range_mode='legacy_target_edge_v1')
    with pytest.raises(ValueError,match='training physics'):module.validate_protocol(p,out/'weights')


def inputs(tmp_path, monkeypatch, pair_count=2):
    spec = blueprint()
    for entry in spec['baselines']:
        path = tmp_path / (entry['id'] + '.pt'); path.write_bytes(entry['id'].encode())
        entry.update(checkpoint=str(path), checkpoint_sha256=module.sha(path))
    bp = tmp_path / 'blueprint.json'; bp.write_text(json.dumps(spec))
    roots, proofs = [], {}
    for index in range(pair_count):
        root = tmp_path / f'training{index}'; root.mkdir(); roots.append(root)
        models = []
        for child in range(2):
            path = root / f'model{child}.pt'; path.write_bytes(f'{index}/{child}'.encode())
            models.append(dict(id=f'global{child}_full', checkpoint=str(path), checkpoint_sha256=module.sha(path),
                               family_guard=child == 0, selection_score=.4))
        proofs[root] = dict(schema='verified-global-selected-full-pair-v1', kernel=f'owner/full{index}',
                            full_steps_verified=8192000, models=models, quality_accepted=False, test_used_for_selection=False)
    calls = []
    def verify(collected, prepared):
        assert collected == prepared / 'collected'
        calls.append(prepared)
        return deepcopy(proofs[prepared])
    monkeypatch.setattr(module, 'verify_full', verify)
    return bp, roots, proofs, calls


@pytest.mark.parametrize('pairs,total', [(1,4200),(2,6300)])
def test_binds_every_completed_model_including_failed_family_guard(tmp_path, monkeypatch, pairs, total):
    bp, roots, proofs, calls = inputs(tmp_path, monkeypatch, pairs)
    output = tmp_path / 'bound'
    result = module.bind(list(reversed(roots)), output, blueprint_path=bp)
    assert result['total_games'] == total and result['submitted'] is False
    assert len(result['failed_family_guard']) == pairs
    assert set(calls) == set(roots)
    protocol = json.loads((output / 'protocol.json').read_text())
    assert len({m['id'] for m in protocol['models']}) == pairs * 2 + 3
    assert protocol['models'][0]['origin_kernel'] == 'owner/full0'
    module.validate_protocol(protocol, output / 'weights')
    assert len(json.loads((output / 'schedule.json').read_text())) == len(module.schedule(protocol, json.loads(bp.read_text())))


def test_unfinished_training_cannot_create_package(tmp_path, monkeypatch):
    bp, roots, proofs, calls = inputs(tmp_path, monkeypatch)
    def unfinished(*args): raise ValueError('Matching successful terminal collection required')
    monkeypatch.setattr(module, 'verify_full', unfinished)
    with pytest.raises(ValueError, match='terminal'): module.bind(roots, tmp_path / 'bound', blueprint_path=bp)
    assert not (tmp_path / 'bound').exists()


@pytest.mark.parametrize('corruption', ['baseline', 'candidate', 'kernel'])
def test_changed_provenance_rejected_before_creating_package(tmp_path, monkeypatch, corruption):
    bp, roots, proofs, calls = inputs(tmp_path, monkeypatch)
    if corruption == 'baseline': (tmp_path / 'balanced.pt').write_bytes(b'changed')
    if corruption == 'candidate': (roots[0] / 'model0.pt').write_bytes(b'changed')
    if corruption == 'kernel': proofs[roots[1]]['kernel'] = proofs[roots[0]]['kernel']
    with pytest.raises(ValueError): module.bind(roots, tmp_path / 'bound', blueprint_path=bp)
    assert not (tmp_path / 'bound').exists()


def test_existing_binding_is_never_overwritten(tmp_path, monkeypatch):
    bp, roots, proofs, calls = inputs(tmp_path, monkeypatch)
    output = tmp_path / 'bound'; output.mkdir(); (output / 'keep').write_text('preserved')
    with pytest.raises(ValueError, match='already exists'): module.bind(roots, output, blueprint_path=bp)
    assert (output / 'keep').read_text() == 'preserved'
    assert not calls


def test_new_execution_is_bound_and_cannot_silently_become_legacy(tmp_path,monkeypatch):
    from scripts.attack.evaluation_execution import CURRENT
    bp,roots,_,_=inputs(tmp_path,monkeypatch,1)
    spec=json.loads(bp.read_text())
    spec.update(schema='precommitted-full-candidate-evaluation-v2',execution=CURRENT)
    spec['evaluation_runtime'].update({k:v for k,v in CURRENT.items()if k!='schema'})
    bp.write_text(json.dumps(spec))
    output=tmp_path/'bound_new'
    module.bind(roots,output,blueprint_path=bp)
    protocol=json.loads((output/'protocol.json').read_text())
    assert protocol['schema']=='bound-full-candidate-evaluation-v2'
    assert protocol['execution']==CURRENT
    module.validate_protocol(protocol,output/'weights')
