from copy import deepcopy
import json
from pathlib import Path

import pytest
from scripts.attack import full_evaluation_recovery as module
from tests.test_full_candidate_verifier import setup, part


def ready(tmp_path):
    protocol, blueprint, jobs = setup(tmp_path)
    path = tmp_path / 'protocol.json'; path.write_text(json.dumps(protocol))
    identity = module.identity(path, 'source-digest', jobs)
    source = tmp_path / 'interrupted'; source.mkdir()
    module.write_json(source / 'run_identity.json', identity)
    module.commit_part(source, jobs[0], part(jobs[0], protocol, blueprint), identity)
    output = tmp_path / 'recovered'; output.mkdir()
    return protocol, blueprint, jobs, identity, source, output


def test_reuses_only_receipted_complete_part_preserving_original(tmp_path):
    protocol, blueprint, jobs, identity, source, output = ready(tmp_path)
    (source / jobs[1]['file']).write_text('{interrupted write')
    restored = module.restore_parts(source, output, identity, protocol, blueprint)
    assert set(restored) == {jobs[0]['file']}
    assert sum(len(r['rows']) for r in restored.values()) == 360
    assert module.sha(source / jobs[0]['file']) == module.sha(output / jobs[0]['file'])
    assert not (output / jobs[1]['file']).exists()
    assert (source / jobs[1]['file']).read_text() == '{interrupted write'


@pytest.mark.parametrize('change', ['source', 'protocol', 'schedule'])
def test_cannot_reuse_other_runtime_or_weights(tmp_path, change):
    protocol, blueprint, jobs, identity, source, output = ready(tmp_path)
    identity = deepcopy(identity)
    if change == 'source': identity['source_manifest_sha256'] = 'changed'
    if change == 'protocol': identity['protocol_sha256'] = 'changed'
    if change == 'schedule': identity['jobs'] = identity['jobs'][1:]
    with pytest.raises(ValueError, match='differs'): module.restore_parts(source, output, identity, protocol, blueprint)
    assert list(output.iterdir()) == []


def test_corrupt_part_does_not_copy_any_recovery_files(tmp_path):
    protocol, blueprint, jobs, identity, source, output = ready(tmp_path)
    (source / jobs[0]['file']).write_text('{}')
    with pytest.raises(ValueError, match='changed'): module.restore_parts(source, output, identity, protocol, blueprint)
    assert list(output.iterdir()) == []


def test_receipt_does_not_override_context_validation(tmp_path):
    protocol, blueprint, jobs, identity, source, output = ready(tmp_path)
    path = source / jobs[0]['file']; report = json.loads(path.read_text())
    report['rows'][0] = report['rows'][1]; path.write_text(json.dumps(report))
    rp = source / (jobs[0]['file'] + '.receipt.json'); receipt = json.loads(rp.read_text())
    receipt['sha256'] = module.sha(path); rp.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='coverage'): module.restore_parts(source, output, identity, protocol, blueprint)
    assert list(output.iterdir()) == []


def test_source_identity_checks_actual_file_contents(tmp_path):
    project = tmp_path / 'project'; project.mkdir(); (project / 'physics.py').write_text('original')
    manifest = tmp_path / 'manifest.json'
    module.write_json(manifest, dict(schema='frozen-full-evaluation-source-v1', files={'physics.py': module.sha(project / 'physics.py')}))
    assert module.source_identity(manifest, project) == module.sha(manifest)
    (project / 'physics.py').write_text('changed')
    with pytest.raises(ValueError, match='source changed'): module.source_identity(manifest, project)


def test_evaluator_finishes_fully_recovered_schedule_without_network_inference(tmp_path, monkeypatch):
    from scripts.attack import evaluate_full_candidates as evaluator
    from scripts.attack.verify_full_candidate_evaluation import verify
    protocol, blueprint, jobs = setup(tmp_path)
    protocol_path = tmp_path / 'protocol.json'; module.write_json(protocol_path, protocol)
    manifest_path = tmp_path / 'source.json'
    filename = 'scripts/attack/evaluate_full_candidates.py'
    module.write_json(manifest_path, dict(schema='frozen-full-evaluation-source-v1', files={filename: module.sha(Path(filename))}))
    run = module.identity(protocol_path, module.sha(manifest_path), jobs)
    source = tmp_path / 'interrupted'; source.mkdir(); module.write_json(source / 'run_identity.json', run)
    for job in jobs: module.commit_part(source, job, part(job, protocol, blueprint), run)
    monkeypatch.setattr(evaluator.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(evaluator.torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(evaluator.torch.cuda, 'get_device_name', lambda i: 'mocked, no GPU computation')
    def forbidden(*args): raise AssertionError('Completed parts must not run inference')
    monkeypatch.setattr(evaluator, 'load', forbidden)
    output = tmp_path / 'result'
    result = evaluator.evaluate(protocol_path, tmp_path, output, manifest_path, resume=source)
    assert result['reused_games'] == result['total_games'] == 4200
    assert verify(protocol_path, tmp_path, output, expected_source_sha256=module.sha(manifest_path))['total_games_verified'] == 4200
    with pytest.raises(ValueError, match='Submitted evaluation source differs'):
        verify(protocol_path, tmp_path, output, expected_source_sha256='wrong')
