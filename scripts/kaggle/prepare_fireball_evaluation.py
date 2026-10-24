"""Freeze one audited Fireball candidate for common-source regression testing."""
import argparse
import inspect
import json
from pathlib import Path
import shutil
import tarfile

from scripts.attack.verify_fireball_checkpoint import verify
from scripts.kaggle.run_five_round_robin import sha,prepare_source


def prepare(collected,training,parent,output):
    report=verify(collected,training,parent)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    dataset=output/'dataset';dataset.mkdir()
    shutil.copyfile(Path(collected)/'candidate.pt',dataset/'candidate.pt')
    (dataset/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    names=('sim/fireball_curriculum.py','scripts/attack/train_fireball_refinement.py',
           'scripts/attack/evaluate_fireball_refinement.py')
    training_manifest=json.loads((Path(training)/'dataset/manifest.json').read_text())
    extensions={name:sha(name)for name in names}
    for name in names[:2]:
        if training_manifest['files'][name]!=extensions[name]:raise ValueError('Fireball training extension changed')
    with tarfile.open(dataset/'extensions.tar.gz','w:gz')as archive:
        for name in names:archive.add(name,arcname=name,recursive=False)
    frozen=Path('tmp/kaggle/ppo_selected_evaluation_20261023_v3')
    manifest=json.loads((frozen/'manifest.json').read_text())
    if manifest['archive_sha256']!='1a34b396650d0f0a51c5276cbc40fbb864b4f86db0019fbdc3edb874164453fb':
        raise ValueError('Wrong common evaluation source')
    if set(extensions)&set(manifest['files']):raise ValueError('Evaluation extensions must not replace frozen files')
    owner='andreiperelaiko';slug='hastycr-fireball-eval-inputs-20261025'
    (dataset/'dataset-metadata.json').write_text(json.dumps(dict(id=owner+'/'+slug,title=slug,
        licenses=[dict(name='other')],isPrivate=True),indent=2)+'\n')
    pins={name:sha(dataset/name)for name in ('candidate.pt','selection.json','extensions.tar.gz')}
    job=output/'job';job.mkdir()
    source='''import hashlib,json,shutil,sys,tarfile,subprocess
from pathlib import Path,PurePosixPath
'''+inspect.getsource(sha)+'\n'+inspect.getsource(prepare_source)+f'''
def mount(slug):
    root=Path('/kaggle/input');paths=[p for p in (root/slug,root/'datasets'/{owner!r}/slug) if p.is_dir()]
    if len(paths)!=1:raise ValueError('Ambiguous input mount')
    return paths[0]
source=mount('hastycr-ppo-selected-eval-v3-20261023')
project=Path('/kaggle/working/fireball_eval_project')
prepare_source(source,project,{sha(frozen/'manifest.json')!r})
inputs=mount({slug!r});pins={pins!r}
for name,digest in pins.items():
    if name=='extensions.tar.gz' and not (inputs/name).exists():continue
    if sha(inputs/name)!=digest:raise ValueError('Evaluation input SHA differs')
extensions={extensions!r}
if (inputs/'extensions.tar.gz').exists():
    with tarfile.open(inputs/'extensions.tar.gz')as stream:
        members=stream.getmembers()
        if len(members)!=len(extensions)or{{m.name for m in members}}!=set(extensions)or any(not m.isfile()for m in members):raise ValueError('Extension archive differs')
        stream.extractall(project,filter='data')
else:
    original=(inputs/'extensions').resolve()
    for name in extensions:
        member=original/name
        if member.is_symlink()or not member.resolve().is_relative_to(original):raise ValueError('Unsafe extracted extension')
        target=project/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(member,target)
for name,digest in extensions.items():
    if sha(project/name)!=digest:raise ValueError('Evaluation extension SHA differs')
parent=mount('hastycr-ppo-selected-policy-20261023')/'policy.pt'
if sha(parent)!={report['parent_sha256']!r}:raise ValueError('Parent policy differs')
print(json.dumps(dict(event='INPUTS_VERIFIED',common_source_sha256={manifest['archive_sha256']!r},candidate_sha256={report['checkpoint_sha256']!r})),flush=True)
subprocess.run([sys.executable,'-m','scripts.attack.evaluate_fireball_refinement','--candidate',str(inputs/'candidate.pt'),'--parent',str(parent),'--selection',str(inputs/'selection.json'),'--output','/kaggle/working/fireball_evaluation'],cwd=project,check=True)
'''
    (job/'evaluate.py').write_text(source)
    kernel=owner+'/hastycr-fireball-eval-20261025'
    (job/'kernel-metadata.json').write_text(json.dumps(dict(id=kernel,title='hastycr fireball eval 20261025',
        code_file='evaluate.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,
        enable_internet=False,dataset_sources=[owner+'/'+slug,owner+'/hastycr-ppo-selected-eval-v3-20261023',
        owner+'/hastycr-ppo-selected-policy-20261023'],competition_sources=[],kernel_sources=[]),indent=2)+'\n')
    preparation=dict(schema='fireball-evaluation-preparation-v1',kernel=kernel,pins=pins,extensions=extensions,
        source_sha256=manifest['archive_sha256'],source_manifest_sha256=sha(frozen/'manifest.json'),
        bootstrap_sha256=sha(job/'evaluate.py'),candidate_sha256=report['checkpoint_sha256'],
        fireball_test_episodes_per_model=384,old_test_episodes=804,games=320,submitted=False)
    (output/'preparation.json').write_text(json.dumps(preparation,indent=2)+'\n')
    return preparation


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collected','training','parent','output'):p.add_argument('--'+name,required=True)
    print(json.dumps(prepare(**vars(p.parse_args()))))
