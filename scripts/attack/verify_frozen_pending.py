"""Verify a completed pending pilot against its own immutable training sources."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.kaggle.preflight_global_full_pair import unpack


def verify(collected,prepared):
    prepared=Path(prepared).resolve();collected=Path(collected).resolve()
    prep=json.loads((prepared/'preparation.json').read_text())
    if prep.get('study_variant')!='own_pending_inputs_v1':raise ValueError('Pending study required')
    if hashlib.sha256((prepared/'job/train.py').read_bytes()).hexdigest()!=prep['bootstrap_sha256']:
        raise ValueError('Frozen bootstrap changed')
    output=prepared/f'frozen_verification_{time.time_ns()}'
    output.mkdir(exist_ok=False);project=output/'project'
    # Verifies base manifest, every input, old replaced sources, and all extensions.
    unpack(prepared/'dataset',prep['config'],project)
    verifier=Path(__file__).with_name('verify_pending_focus.py').resolve()
    report=output/'verified.json'
    with(output/'verification.log').open('x')as log:
        subprocess.run([sys.executable,str(verifier),'--collected',str(collected),
            '--prepared',str(prepared),'--output',str(report)],cwd=project,
            env=dict(os.environ,PYTHONPATH=str(project)),stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads(report.read_text())
    result['verification_sources']=dict(mode='isolated_frozen_training_package',project=str(project),
        bootstrap_sha256=prep['bootstrap_sha256'],extensions_sha256=prep['config']['pins']['extensions.tar.gz'],
        verifier_sha256=hashlib.sha256(verifier.read_bytes()).hexdigest())
    return result
