import io
import json
from pathlib import Path
import tarfile

import pytest
from scripts.kaggle import prepare_full_candidate_evaluation as module
from tests.test_full_candidate_evaluator import fixture


def inputs(tmp_path):
    binding = tmp_path / 'binding'; weights = binding / 'weights'; weights.mkdir(parents=True)
    protocol = fixture(weights)
    for m in protocol['models']:
        if m['role'] == 'candidate': m['origin_kernel'] = 'owner/full'
    (binding / 'protocol.json').write_text(json.dumps(protocol))
    source = tmp_path / 'source'; source.mkdir()
    content = b'print("technical source fixture")\n'
    with tarfile.open(source / 'source.tar.gz', 'w:gz') as archive:
        info = tarfile.TarInfo('sim/physics.py'); info.size = len(content); archive.addfile(info, io.BytesIO(content))
    import hashlib
    manifest = dict(schema='frozen-full-evaluation-source-v1', files={'sim/physics.py': hashlib.sha256(content).hexdigest()},
                    archive_sha256=module.sha(source / 'source.tar.gz'), training_kernels=['owner/full'])
    (source / 'manifest.json').write_text(json.dumps(manifest))
    return binding, source


def test_exact_flat_transport_and_generated_bootstrap_unpack(tmp_path):
    binding, source = inputs(tmp_path); output = tmp_path / 'package'
    result = module.prepare(binding, source, output, 'vertel0', 'hastycr-full-eval-test')
    assert result['total_games'] == 4200 and result['ready_for_submission'] is False
    assert json.loads((output / 'job/kernel-metadata.json').read_text())['enable_gpu'] is False
    # Execute the actual generated helper definitions, with no remote invocation.
    code = (output / 'job/train.py').read_text().rsplit('\nbootstrap(', 1)[0]
    namespace = {}; exec(compile(code, 'generated_bootstrap', 'exec'), namespace)
    decoded, project = namespace['unpack'](result['config'], output / 'upload_dataset', tmp_path / 'working')
    assert (project / 'sim/physics.py').read_bytes() == b'print("technical source fixture")\n'
    protocol = json.loads((decoded / 'protocol.json').read_text())
    module.validate_protocol(protocol, decoded / 'weights')
    assert set(result['config']['files']) == {str(p.relative_to(decoded)) for p in decoded.rglob('*') if p.is_file()}


def test_changed_transport_cannot_be_unpacked(tmp_path):
    binding, source = inputs(tmp_path); output = tmp_path / 'package'
    result = module.prepare(binding, source, output, 'vertel0', 'hastycr-full-eval-test')
    (output / 'upload_dataset/evaluation_inputs.b64').write_text('changed')
    with pytest.raises(ValueError, match='transport differs'):
        module.unpack(result['config'], output / 'upload_dataset', tmp_path / 'working')
    assert not (tmp_path / 'working').exists()


def test_source_from_other_training_is_rejected(tmp_path):
    binding, source = inputs(tmp_path); p = source / 'manifest.json'; m = json.loads(p.read_text())
    m['training_kernels'] = ['other/full']; p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='candidate training'):
        module.prepare(binding, source, tmp_path / 'package', 'vertel0', 'hastycr-full-eval-test')
    assert not (tmp_path / 'package').exists()


def test_invalid_fourth_account_cannot_be_selected(tmp_path):
    binding, source = inputs(tmp_path)
    with pytest.raises(ValueError, match='working account'):
        module.prepare(binding, source, tmp_path / 'package', 'demjanklimov', 'hastycr-full-eval-test')
