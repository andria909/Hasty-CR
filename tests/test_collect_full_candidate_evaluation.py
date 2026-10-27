import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from scripts.kaggle import collect_full_candidate_evaluation as module
from scripts.kaggle.prepare_full_candidate_evaluation import prepare
from tests.test_prepare_full_candidate_evaluation import inputs


def fixture(tmp_path, monkeypatch, state='COMPLETE'):
    binding, source = inputs(tmp_path); package = tmp_path / 'package'
    prep = prepare(binding, source, package, 'vertel0', 'hastycr-full-eval-test')
    _, parts = module.plan(package)
    blobs = {'source_manifest.json': (source / 'manifest.json').read_bytes()}
    blobs['run_identity.json'] = json.dumps(dict(protocol_sha256=prep['config']['files']['protocol.json'],
        source_manifest_sha256=prep['config']['source_manifest_sha256'])).encode()
    for name in parts:
        blobs[name] = b'{"technical_fixture":true}'
        blobs[name + '.receipt.json'] = json.dumps(dict(file=name,sha256=hashlib.sha256(blobs[name]).hexdigest(),
            protocol_sha256=prep['config']['files']['protocol.json'],source_manifest_sha256=prep['config']['source_manifest_sha256'])).encode()
    blobs['result.json'] = b'{}'
    downloads = []
    monkeypatch.setattr(module, 'list_outputs', lambda *args: ({module.PREFIX + n: NS(url=n) for n in blobs} | {'secret.txt': NS(url='forbidden')}, []))
    def download(url, path, **kwargs): downloads.append(url); Path(path).write_bytes(blobs[url])
    monkeypatch.setattr(module, 'download_preserving', download)
    api = NS(kernels_status=lambda kernel: state)
    return package, parts, blobs, downloads, api


def test_complete_download_is_allowlisted_but_not_claimed_metric_verification(tmp_path, monkeypatch):
    package, parts, blobs, downloads, api = fixture(tmp_path, monkeypatch)
    result = module.collect(api, package, tmp_path / 'collected')
    assert result['complete'] and result['transfer_complete']
    assert result['parts_validated'] is False
    assert result['downloaded_parts'] == len(parts) == 20
    assert set(downloads) == set(blobs)


def test_failed_job_keeps_only_receipted_parts(tmp_path, monkeypatch):
    package, parts, blobs, downloads, api = fixture(tmp_path, monkeypatch, 'ERROR')
    name = sorted(parts)[0]; del blobs[name + '.receipt.json']
    result = module.collect(api, package, tmp_path / 'collected')
    assert result['transfer_complete'] and not result['complete']
    assert result['downloaded_parts'] == len(parts) - 1
    assert name not in downloads and 'result.json' not in downloads


def test_failed_startup_without_identity_yields_empty_recoverable_transfer(tmp_path, monkeypatch):
    package, parts, blobs, downloads, api = fixture(tmp_path, monkeypatch, 'ERROR')
    blobs.clear()
    result = module.collect(api, package, tmp_path / 'collected')
    assert result['downloaded_parts'] == 0 and not downloads and not result['complete']


def test_missing_complete_output_refuses_success_before_download(tmp_path, monkeypatch):
    package, parts, blobs, downloads, api = fixture(tmp_path, monkeypatch)
    del blobs['result.json']
    with pytest.raises(ValueError, match='missing'): module.collect(api, package, tmp_path / 'collected')
    assert not downloads and not (tmp_path / 'collected').exists()


def test_cached_results_are_reused_without_download(tmp_path, monkeypatch):
    package, parts, blobs, downloads, api = fixture(tmp_path, monkeypatch)
    module.collect(api, package, tmp_path / 'first')
    downloads.clear()
    result = module.collect(api, package, tmp_path / 'second', cache=tmp_path / 'first')
    assert result['complete'] and not downloads


def test_foreign_remote_identity_cannot_be_marked_complete(tmp_path, monkeypatch):
    package, parts, blobs, downloads, api = fixture(tmp_path, monkeypatch)
    blobs['run_identity.json'] = json.dumps(dict(protocol_sha256='foreign', source_manifest_sha256='foreign')).encode()
    with pytest.raises(ValueError, match='identity differs'): module.collect(api, package, tmp_path / 'collected')
    receipt = json.loads((tmp_path / 'collected/collection_receipt.json').read_text())
    assert not receipt['transfer_complete'] and not receipt['complete']
