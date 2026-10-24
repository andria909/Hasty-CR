"""Freeze the train-control correction independently of pending held-out results."""
import argparse
import inspect
import json
from pathlib import Path
import shutil
import tarfile

from scripts.attack.train_fireball_selectivity import INIT_SHA,PARENT_SHA,METHOD
from scripts.kaggle.run_five_round_robin import sha,prepare_source


def prepare(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    dataset=output/'dataset';dataset.mkdir()
    previous=json.loads(Path('tmp/kaggle/fireball_refinement_20261025_v2/dataset/manifest.json').read_text())
    files={name:sha(name)for name in previous['files']}
    for name in ('sim/fireball_decision_pairs.py','scripts/attack/train_fireball_selectivity.py'):
        files[name]=sha(name)
    if any(not n.startswith(('sim/','scripts/','data/','src/'))for n in files):raise ValueError('Unexpected source path')
    with tarfile.open(dataset/'source.tar.gz','w:gz')as archive:
        for name in sorted(files):archive.add(name,arcname=name,recursive=False)
    manifest=dict(schema='fireball-selectivity-source-v1',files=files,archive_sha256=sha(dataset/'source.tar.gz'))
    (dataset/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name,path,digest in [('parent.pt','checkpoints/ppo_joint_economy_soft_20261023/policy.pt',PARENT_SHA),
                             ('init.pt','checkpoints/ppo_joint_economy_soft_fireball_20261025/policy.pt',INIT_SHA)]:
        if sha(path)!=digest:raise ValueError('Checkpoint source differs')
        shutil.copyfile(path,dataset/name)
    owner='andriashadrin';slug='hastycr-fireball-select-inputs-20261025'
    (dataset/'dataset-metadata.json').write_text(json.dumps(dict(id=owner+'/'+slug,title=slug,
        licenses=[dict(name='other')],isPrivate=True),indent=2)+'\n')
    pins={n:sha(dataset/n)for n in ('manifest.json','parent.pt','init.pt')}
    job=output/'job';job.mkdir()
    source='''import hashlib,json,shutil,sys,tarfile,subprocess
from pathlib import Path,PurePosixPath
'''+inspect.getsource(sha)+'\n'+inspect.getsource(prepare_source)+f'''
root=Path('/kaggle/input');slug={slug!r}
paths=[p for p in (root/slug,root/'datasets'/{owner!r}/slug)if p.is_dir()]
if len(paths)!=1:raise ValueError('Ambiguous selectivity inputs')
inputs=paths[0]
for name,digest in {pins!r}.items():
    if sha(inputs/name)!=digest:raise ValueError('Selectivity input SHA differs')
project=Path('/kaggle/working/fireball_selectivity_project')
prepare_source(inputs,project,{pins['manifest.json']!r})
print(json.dumps(dict(event='INPUTS_VERIFIED',source_sha256={manifest['archive_sha256']!r},init_sha256={INIT_SHA!r})),flush=True)
subprocess.run([sys.executable,'-m','scripts.attack.train_fireball_selectivity','--parent',str(inputs/'parent.pt'),'--init',str(inputs/'init.pt'),'--output','/kaggle/working/fireball_selectivity'],cwd=project,check=True)
'''
    (job/'train_selectivity.py').write_text(source)
    kernel=owner+'/hastycr-fireball-select-20261025'
    (job/'kernel-metadata.json').write_text(json.dumps(dict(id=kernel,title='hastycr fireball select 20261025',
        code_file='train_selectivity.py',language='python',kernel_type='script',is_private=True,
        enable_gpu=True,enable_internet=False,dataset_sources=[owner+'/'+slug],competition_sources=[],kernel_sources=[]),indent=2)+'\n')
    report=dict(schema='fireball-selectivity-preparation-v1',kernel=kernel,method=METHOD,pins=pins,
        source_sha256=manifest['archive_sha256'],bootstrap_sha256=sha(job/'train_selectivity.py'),
        init_sha256=INIT_SHA,parent_sha256=PARENT_SHA,basis_report_sha256=sha('reports/defense_plan/fireball_train_controls.json'),
        basis_split='train',test_used_for_selection=False,pending_first_candidate_test_read=False,submitted=False)
    (output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    print(json.dumps(prepare(p.parse_args().output)))
