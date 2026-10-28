"""Reconstruct replay provenance with the exact archived implementation."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from scripts.kaggle.preflight_global_full_pair import unpack


def metadata(dataset,config):
    dataset=Path(dataset).absolute()
    # Recheck all pins even when reusing a result within this verification process.
    for name,digest in config['pins'].items():
        if hashlib.sha256((dataset/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen replay input changed: '+name)
    return dict(_metadata(str(dataset),json.dumps(config,sort_keys=True)))


@lru_cache(maxsize=64)
def _metadata(dataset,encoded):
    dataset=Path(dataset);config=json.loads(encoded)
    root=Path(tempfile.mkdtemp(prefix='hasty-replay-audit-'));project=root/'project'
    unpack(dataset,config,project)
    script="""import json,shutil,sys
from pathlib import Path
from sim.supervised_replay import SuccessfulActionReplay
source=Path(sys.argv[1]);target=Path(sys.argv[2]);target.mkdir()
shutil.copyfile(source/'air_examples.npz',target/'examples.npz')
shutil.copyfile(source/'air_manifest.json',target/'manifest.json')
Path(sys.argv[3]).write_text(json.dumps(SuccessfulActionReplay(target,'cpu').metadata))
"""
    result=root/'metadata.json'
    with (root/'run.log').open('x')as log:
        subprocess.run([sys.executable,'-c',script,str(dataset),str(root/'data'),str(result)],cwd=project,
            env=dict(os.environ,PYTHONPATH=str(project)),stdout=log,stderr=subprocess.STDOUT,check=True)
    value=json.loads(result.read_text())
    if value['source_sha256']!=config['extensions']['sim/supervised_replay.py']:
        raise ValueError('Replay implementation differs from archived source')
    return value
