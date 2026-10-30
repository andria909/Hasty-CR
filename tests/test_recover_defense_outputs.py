from contextlib import nullcontext
from hashlib import sha256
import json
import sys
from types import SimpleNamespace as NS

import pytest

from scripts.kaggle import recover_defense_outputs as recovery


def install_fake_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, 'kaggle', NS())
    monkeypatch.setitem(sys.modules, 'kaggle.api', NS())
    monkeypatch.setitem(sys.modules, 'kaggle.api.kaggle_api_extended',
                        NS(ApiListKernelSessionOutputRequest=NS))


def fake_api(pages, status='ERROR'):
    calls = []
    def listing(request):
        calls.append(request)
        return pages[request.page_token]
    client = NS(kernels=NS(kernels_api_client=NS(list_kernel_session_output=listing)))
    return NS(kernels_status=lambda kernel: NS(status=status),
              parse_kernel_string=lambda kernel: ('owner', 'slug', None),
              build_kaggle_client=lambda: nullcontext(client)), calls


def test_numeric_selection_and_strict_allowlist():
    names = ['ppo/update_000500.pt', 'ppo/update_000499.pt',
             'ppo/best_validation_000450.pt', 'ppo/best_validation_000010.pt',
             'grpo/boundary_u000375_s000120.pt',
             'grpo/boundary_u000375_s000121.pt',
             'grpo/best_validation_u000350_s000100.pt',
             'grpo/emergency_u000378_s000999.pt',
             'grpo/final_u000378_s000999.pt', '../ppo/update_999999.pt',
             '/ppo/update_999999.pt', 'other/ppo/update_999999.pt',
             'ppo/update_999999.pt.bak', 'ppo/metrics.jsonl', 'grpo/updates.jsonl']
    roles, logs = recovery.select_outputs(names)
    assert roles == dict(ppo_latest=names[0], ppo_best_validation=names[2],
                        grpo_boundary=names[5], grpo_best_validation=names[6])
    assert logs == ['grpo/updates.jsonl', 'ppo/metrics.jsonl']


def test_pagination_500_more_than_1000(monkeypatch):
    install_fake_sdk(monkeypatch)
    pages = {}
    for i, token in enumerate(('', 'second', 'third')):
        pages[token] = NS(files=[NS(file_name=f'other/{500*i+j}') for j in range(500)],
                          log='', next_page_token=['second', 'third', ''][i])
    api, calls = fake_api(pages)
    items, _ = recovery.list_outputs(api, 'owner/slug')
    assert len(items) == 1500
    assert [c.page_size for c in calls] == [500, 500, 500]


def test_running_never_lists_or_downloads(tmp_path):
    api = NS(kernels_status=lambda kernel: NS(status='RUNNING'))
    with pytest.raises(ValueError, match='download forbidden'):
        recovery.recover(api, 'owner/slug', tmp_path/'new')
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('version,expected', [('2', 'v2'), ('v2', 'v2')])
def test_exact_version_output_label(monkeypatch, version, expected):
    install_fake_sdk(monkeypatch)
    api, calls = fake_api({'': NS(files=[], log='', next_page_token='')})
    api.parse_kernel_string = lambda kernel: ('owner', 'slug', version)
    recovery.list_outputs(api, 'owner/slug/'+version)
    assert calls[0].version_label == expected


def test_terminal_failed_result_not_required_and_hash_receipt(monkeypatch, tmp_path):
    install_fake_sdk(monkeypatch)
    payload = b'not deserialized checkpoint'
    rows = [NS(file_name='ppo/update_000500.pt', url='opaque', sha256=sha256(payload).hexdigest()),
            NS(file_name='grpo/boundary_u000375_s000001.pt', url='opaque'),
            NS(file_name='ppo/metrics.jsonl', url='opaque'),
            NS(file_name='grpo/updates.jsonl', url='opaque'),
            NS(file_name='../untrusted.pt', url='do-not-download')]
    api, _ = fake_api({'': NS(files=rows, log='controller failed', next_page_token='')})
    paths = []
    def fetch(url, path):
        assert url == 'opaque'
        paths.append(path)
        with path.open('xb') as stream:
            stream.write(payload)
    output = tmp_path/'new'
    receipt = recovery.recover(api, 'owner/slug', output, fetch=fetch)
    assert len(paths) == 4
    assert receipt['complete'] and not receipt['quality_accepted']
    assert receipt['files'][rows[0].file_name]['remote_hash_verified']
    assert not receipt['files'][rows[1].file_name]['remote_hash_verified']
    assert receipt['files'][rows[1].file_name]['remote_sha256'] is None
    assert json.loads((output/'recovery_receipt.json').read_text()) == receipt
    with pytest.raises(FileExistsError):
        recovery.recover(api, 'owner/slug', output, fetch=fetch)


def test_bad_remote_hash_marks_partial_receipt(monkeypatch, tmp_path):
    install_fake_sdk(monkeypatch)
    item = NS(file_name='ppo/update_000500.pt', url='opaque', sha256='0'*64)
    api, _ = fake_api({'': NS(files=[item], log='', next_page_token='')})
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        recovery.recover(api, 'owner/slug', tmp_path/'new',
                         fetch=lambda url, path: path.write_bytes(b'data'))
    assert not json.loads((tmp_path/'new/recovery_receipt.json').read_text())['complete']
