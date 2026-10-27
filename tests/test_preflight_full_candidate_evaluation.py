import json
from pathlib import Path

import pytest
from scripts.kaggle import preflight_full_candidate_evaluation as module
from scripts.kaggle.prepare_full_candidate_evaluation import prepare
from tests.test_prepare_full_candidate_evaluation import inputs


def package(tmp_path):
    binding, source = inputs(tmp_path); output = tmp_path / 'package'
    prepare(binding, source, output, 'vertel0', 'hastycr-full-eval-test')
    return output


def test_missing_preflight_cannot_enable_gpu(tmp_path):
    output = package(tmp_path)
    with pytest.raises(FileNotFoundError): module.enable_submission(output)
    assert not json.loads((output / 'job/kernel-metadata.json').read_text())['enable_gpu']


def test_changed_upload_refused_before_decode(tmp_path):
    output = package(tmp_path)
    (output / 'upload_dataset/evaluation_inputs.b64').write_text('changed')
    with pytest.raises(ValueError, match='upload changed'): module.preflight(output, tmp_path / 'check')
    assert not (tmp_path / 'check').exists()


def test_failed_real_full_audit_cannot_produce_preflight_or_enable(tmp_path, monkeypatch):
    output = package(tmp_path)
    def reject(inputs): raise ValueError('Actual full budget not verified')
    monkeypatch.setattr(module, 'audit_candidates', reject)
    with pytest.raises(ValueError, match='full budget'): module.preflight(output, tmp_path / 'check')
    assert not (output / 'preflight.json').exists()
    assert not json.loads((output / 'job/kernel-metadata.json').read_text())['enable_gpu']


def test_exact_subprocess_contract_and_enable_flow_with_stub_audit(tmp_path, monkeypatch):
    # Synthetic control-flow test only; no claim of actual candidate preflight.
    output = package(tmp_path); calls = []
    def audit(inputs):
        calls.append(str(inputs))
        return json.loads((Path(inputs) / 'protocol.json').read_text())
    monkeypatch.setattr(module, 'audit_candidates', audit)
    def child(command, **kwargs):
        spec = json.loads(Path(command[-1]).read_text()); inputs = Path(spec['inputs'])
        protocol = json.loads((inputs / 'protocol.json').read_text())
        assert Path(kwargs['cwd']).name == 'full_eval_project'
        assert kwargs['env']['PYTHONPATH'] == str(kwargs['cwd'])
        assert command[-2] == '--child'
        from scripts.kaggle.local_artifact_transfer import file_sha
        report = dict(schema='full-evaluation-exact-package-preflight-v1',
                      models=[dict(id=m['id'],sha256=m['sha256'],technical_games=2) for m in protocol['models']],
                      technical_games=11, source_manifest_sha256=file_sha(inputs/'source/manifest.json'),
                      protocol_sha256=file_sha(inputs/'protocol.json'), actual_cuda=False,
                      independent_test_run=False, quality_accepted=False)
        Path(spec['report']).write_text(json.dumps(report))
    monkeypatch.setattr(module.subprocess, 'run', child)
    module.preflight(output, tmp_path / 'check')
    result = module.enable_submission(output)
    assert result['enabled'] and result['submitted'] is False and len(calls) == 2
    assert json.loads((output/'job/kernel-metadata.json').read_text())['enable_gpu']
    assert not json.loads((output/'kernel-metadata.preflight-disabled.json').read_text())['enable_gpu']
    with pytest.raises(ValueError, match='already enabled'): module.enable_submission(output)
    from scripts.kaggle.validate_evaluation_submission import validate_submission
    prep=json.loads((output/'preparation.json').read_text())
    assert validate_submission(output,prep)==output/'upload_dataset'
    (output/'upload_dataset/evaluation_inputs.b64').write_text('changed')
    with pytest.raises(ValueError,match='package changed'):validate_submission(output,prep)


def test_submission_bridge_rejects_unenabled_package_before_credentials(tmp_path, monkeypatch):
    from scripts.kaggle.submit_prepared_pilot import submit
    from scripts.kaggle import collect_final_evaluation
    output=package(tmp_path)
    def forbidden(*args):raise AssertionError('API must not be created before package validation')
    monkeypatch.setattr(collect_final_evaluation,'make_api_client',forbidden)
    with pytest.raises(FileNotFoundError):submit(output,1,evaluation=True)
    with pytest.raises(ValueError,match='three authorized'):submit(output,3,evaluation=True)
