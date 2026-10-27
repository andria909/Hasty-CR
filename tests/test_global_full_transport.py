import base64
import hashlib
import io
import tarfile
import pytest
from scripts.kaggle.prepare_global_full_pair import transport_inputs


def fixture(tmp_path,member='bundle_0/train.py'):
    raw=io.BytesIO();content=b'child script'
    with tarfile.open(fileobj=raw,mode='w:gz')as stream:
        info=tarfile.TarInfo(member);info.size=len(content);stream.addfile(info,io.BytesIO(content))
    encoded=base64.b64encode(raw.getvalue());(tmp_path/'full_inputs.b64').write_bytes(encoded)
    config=dict(transport_file='full_inputs.b64',transport_sha256=hashlib.sha256(encoded).hexdigest(),
        archive_sha256=hashlib.sha256(raw.getvalue()).hexdigest(),pins={member:hashlib.sha256(content).hexdigest()})
    return config


def test_flat_opaque_transport_restores_exact_bundle_names(tmp_path):
    config=fixture(tmp_path);output=tmp_path/'decoded'
    assert transport_inputs(tmp_path,output,config)==output
    assert(output/'bundle_0/train.py').read_bytes()==b'child script'


def test_transport_tampering_rejected_before_extracting(tmp_path):
    config=fixture(tmp_path);(tmp_path/'full_inputs.b64').write_bytes(b'changed')
    with pytest.raises(ValueError,match='transport changed'):transport_inputs(tmp_path,tmp_path/'decoded',config)
    assert not(tmp_path/'decoded').exists()


def test_parent_traversal_cannot_escape_transport(tmp_path):
    config=fixture(tmp_path,member='../escaped.py')
    with pytest.raises(ValueError,match='Unsafe'):transport_inputs(tmp_path,tmp_path/'decoded',config)
    assert not(tmp_path/'escaped.py').exists()
