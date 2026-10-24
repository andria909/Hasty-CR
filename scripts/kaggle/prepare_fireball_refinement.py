"""Package only allowlisted simulator files and the new Fireball refinement."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import tarfile

from scripts.kaggle.run_five_round_robin import sha,prepare_source


def prepare(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    dataset=output/'dataset';dataset.mkdir()
    old=json.loads(Path('tmp/kaggle/ppo_selected_evaluation_20261023_v3/manifest.json').read_text())
    files={name:sha(name) for name in old['files'] if name.startswith(('sim/','scripts/','data/','src/'))}
    for name in ('sim/fireball_curriculum.py','scripts/attack/train_fireball_refinement.py'):
        files[name]=sha(name)
    # Explicit allowlist only; private captures, credentials and checkpoint
    # directories never enter this source dataset.
    for name in files:
        if not name.startswith(('sim/','scripts/','data/','src/')):raise ValueError('Unexpected source path')
    with tarfile.open(dataset/'source.tar.gz','w:gz') as archive:
        for name in sorted(files):archive.add(name,arcname=name,recursive=False)
    manifest=dict(schema='fireball-refinement-inputs-v1',files=files,archive_sha256=sha(dataset/'source.tar.gz'),
                  parent_sha256='9d6ac7d1d41ff00df6a9c74354fa8447793e0be8d69f723b27ff43b23c4c73e8')
    (dataset/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    owner='andreiperelaiko';slug='hastycr-fireball-refine-inputs-20261025'
    (dataset/'dataset-metadata.json').write_text(json.dumps(dict(id=owner+'/'+slug,title=slug,
        licenses=[dict(name='other')],isPrivate=True),indent=2)+'\n')
    job=output/'job';job.mkdir()
    bootstrap='''import hashlib,json,shutil,sys,tarfile,subprocess
from pathlib import Path,PurePosixPath
'''+inspect.getsource(sha)+'\n'+inspect.getsource(prepare_source)+f'''
root=Path('/kaggle/input')
slug={slug!r}
choices=[root/slug,root/'datasets'/{owner!r}/slug]
source=next((p for p in choices if p.is_dir()),None)
if source is None:raise RuntimeError('Source input missing')
project=Path('/kaggle/working/fireball_project')
prepare_source(source,project,{sha(dataset/'manifest.json')!r})
weights_slug='hastycr-ppo-selected-policy-20261023'
weights=next((p for p in [root/weights_slug,root/'datasets'/{owner!r}/weights_slug] if p.is_dir()),None)
if weights is None:raise RuntimeError('Selected weight input missing')
if sha(weights/'policy.pt')!={manifest['parent_sha256']!r}:raise ValueError('Selected checkpoint SHA differs')
import torch
if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
print(json.dumps(dict(event='CUDA_READY',gpu=torch.cuda.get_device_name(0),source_sha256={manifest['archive_sha256']!r})),flush=True)
subprocess.run([sys.executable,'-m','scripts.attack.train_fireball_refinement','--init',str(weights/'policy.pt'),'--output','/kaggle/working/fireball_refinement'],cwd=project,check=True)
'''
    (job/'train_fireball.py').write_text(bootstrap)
    kernel=owner+'/hastycr-fireball-refine-20261025'
    (job/'kernel-metadata.json').write_text(json.dumps(dict(id=kernel,title='hastycr fireball refine 20261025',
        code_file='train_fireball.py',language='python',kernel_type='script',is_private=True,
        enable_gpu=True,enable_internet=False,dataset_sources=[owner+'/'+slug,
        owner+'/hastycr-ppo-selected-policy-20261023'],competition_sources=[],kernel_sources=[]),indent=2)+'\n')
    receipt=dict(kernel=kernel,source_sha256=manifest['archive_sha256'],parent_sha256=manifest['parent_sha256'],
                 bootstrap_sha256=sha(job/'train_fireball.py'),submitted=False)
    (output/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    print(json.dumps(prepare(p.parse_args().output)))
