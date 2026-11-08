import hashlib
import io
import json
import tarfile
from pathlib import Path
import pytest

from scripts.kaggle.submit_entity_experiment import BOOTSTRAP


@pytest.mark.parametrize('expanded', [False, True])
@pytest.mark.parametrize('tampered', [False, True])
def test_bootstrap_verifies_raw_or_kaggle_expanded_source(tmp_path, expanded, tampered):
    dataset=tmp_path/'inputs'/'private-inputs';dataset.mkdir(parents=True)
    name='sim/example.py';content=b'VALUE = 1\n'
    digest=lambda b:hashlib.sha256(b).hexdigest()
    archive=io.BytesIO()
    with tarfile.open(fileobj=archive,mode='w:gz')as stream:
        info=tarfile.TarInfo(name);info.size=len(content);stream.addfile(info,io.BytesIO(content))
    raw=archive.getvalue()
    if expanded:
        p=dataset/'source'/name;p.parent.mkdir(parents=True);p.write_bytes(content+b'#changed'if tampered else content)
    else:
        (dataset/'source.tar.gz').write_bytes(raw+b'changed'if tampered else raw)
    manifest=json.dumps(dict(files={name:digest(content)},archive_sha256=digest(raw))).encode()
    (dataset/'manifest.json').write_bytes(manifest)
    config=dict(slug=dataset.name,pins={'source.tar.gz':digest(raw),'manifest.json':digest(manifest)})
    code=BOOTSTRAP.split('import os\n')[0].replace('CONFIG',repr(config))
    code=code.replace("'/kaggle/input'",repr(str(tmp_path/'inputs'))).replace("'/kaggle/working/entity_project'",repr(str(tmp_path/'project')))
    if tampered:
        with pytest.raises(ValueError):exec(compile(code,'bootstrap','exec'),{})
    else:
        exec(compile(code,'bootstrap','exec'),{})
        assert (tmp_path/'project'/name).read_bytes()==content
