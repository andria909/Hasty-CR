import io
import json
from pathlib import Path
import tarfile

import pytest
from scripts.kaggle import freeze_full_evaluation_source as module


def archive(path, files):
    with tarfile.open(path, 'w:gz') as stream:
        for name, data in files.items():
            info = tarfile.TarInfo(name); info.size = len(data)
            stream.addfile(info, io.BytesIO(data))


def source(tmp_path):
    root = tmp_path / 'training'; children = []
    for index in (0, 1):
        bundle = root / 'dataset' / f'bundle_{index}'; bundle.mkdir(parents=True)
        archive(bundle / 'source.tar.gz', {'sim/physics.py': b'old'})
        import hashlib
        manifest = dict(files={'sim/physics.py': hashlib.sha256(b'old').hexdigest()}, archive_sha256=module.sha(bundle / 'source.tar.gz'))
        (bundle / 'manifest.json').write_text(json.dumps(manifest))
        archive(bundle / 'extensions.tar.gz', {'sim/physics.py': b'new'})
        config = dict(source_manifest_sha256=module.sha(bundle / 'manifest.json'),
                      extensions={'sim/physics.py': hashlib.sha256(b'new').hexdigest()},
                      pins={n: module.sha(bundle / n) for n in ('manifest.json', 'extensions.tar.gz')})
        children.append(dict(index=index, config=config))
    (root / 'preparation.json').write_text(json.dumps(dict(kernel='owner/full', children=children)))
    return root


def test_source_is_training_physics_plus_explicit_evaluators(tmp_path):
    root = source(tmp_path); output = tmp_path / 'output'
    module.freeze([root], output)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert set(manifest['files']) == {'sim/physics.py', *module.EVALUATORS}
    assert (output / 'project/sim/physics.py').read_bytes() == b'new'
    assert manifest['candidate_inference_run'] is False
    with tarfile.open(output / 'source.tar.gz') as stream:
        assert set(stream.getnames()) == set(manifest['files'])


def test_different_candidate_physics_rejected(tmp_path):
    root = source(tmp_path); p = root / 'preparation.json'; prep = json.loads(p.read_text())
    prep['children'][1]['config']['extensions']['sim/physics.py'] = 'different'
    p.write_text(json.dumps(prep))
    with pytest.raises(ValueError, match='simulation contract'): module.freeze([root], tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_changed_training_archive_rejected(tmp_path):
    root = source(tmp_path)
    (root / 'dataset/bundle_1/source.tar.gz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='archive changed'): module.freeze([root], tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_existing_source_is_not_overwritten(tmp_path):
    root = source(tmp_path); output = tmp_path / 'output'; output.mkdir()
    with pytest.raises(ValueError, match='already exists'): module.freeze([root], output)
