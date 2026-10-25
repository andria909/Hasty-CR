"""Prepare private, independently pinned native evaluation packages; no submission."""
import argparse
import inspect
import json
from pathlib import Path
import shutil
import tarfile

from scripts.kaggle.run_five_round_robin import sha,prepare_source


def bootstrap(config):
    import subprocess,sys
    from pathlib import PurePosixPath
    root=Path('/kaggle/input')
    def mount(slug):
        matches=[p for p in (root/slug,root/'datasets'/config['owner']/slug)if p.is_dir()]
        if len(matches)!=1:raise ValueError('Ambiguous evaluation input')
        return matches[0]
    inputs=mount(config['slug'])
    for name,digest in config['pins'].items():
        if name in ('source.tar.gz','evaluation.tar.gz')and not (inputs/name).exists():continue
        if sha(inputs/name)!=digest:raise ValueError('Evaluation input differs: '+name)
    project=Path('/kaggle/working/native_eval_project')
    prepare_source(inputs,project,config['pins']['manifest.json'])
    if (inputs/'evaluation.tar.gz').exists():
        with tarfile.open(inputs/'evaluation.tar.gz')as stream:
            members=stream.getmembers()
            if {m.name for m in members}!=set(config['extensions'])or len(members)!=len(config['extensions']):
                raise ValueError('Unexpected evaluation extension members')
            for member in members:
                name=PurePosixPath(member.name)
                if not member.isfile()or name.is_absolute()or '..'in name.parts or (project/member.name).exists():
                    raise ValueError('Unsafe evaluation extension')
            stream.extractall(project,filter='data')
    else:
        original=(inputs/'evaluation').resolve()
        for name in config['extensions']:
            member=original/name;target=project/name
            if member.is_symlink()or not member.resolve().is_relative_to(original)or target.exists():
                raise ValueError('Unsafe expanded evaluation extension')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(member,target)
    for name,digest in config['extensions'].items():
        if sha(project/name)!=digest:raise ValueError('Evaluation extension SHA differs')
    print(json.dumps(dict(event='INPUTS_VERIFIED',role=config['role'],source_sha256=config['source_sha256'])),flush=True)
    subprocess.run([sys.executable,'-m','scripts.attack.evaluate_native_fireball',
        '--protocol','data/fireball/native_evaluation_protocol.json','--policy',str(inputs/'policy.pt'),
        '--selection',str(inputs/'selection.json'),'--opponents',str(inputs),
        '--role',config['role'],'--output','/kaggle/working/native_fireball_evaluation'],cwd=project,check=True)


def prepare(output,role,owner):
    protocol_path=Path('data/fireball/native_evaluation_protocol.json')
    protocol=json.loads(protocol_path.read_text())
    selection=Path('reports/defense_plan/fireball_balanced_match_validation.json')
    audit=json.loads(selection.read_text())
    if not audit['candidate_passes_validation_games']:raise ValueError('Candidate validation failed')
    frozen=Path('tmp/kaggle/fireball_balanced_20261025/dataset')
    if sha(frozen/'source.tar.gz')!=protocol['training_source_sha256']or sha(frozen/'manifest.json')!=protocol['training_manifest_sha256']:
        raise ValueError('Training archive changed')
    manifest=json.loads((frozen/'manifest.json').read_text())
    if manifest['files']!=protocol['training_files']:raise ValueError('Training member manifest differs')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    dataset=output/'dataset';dataset.mkdir()
    for name in ('source.tar.gz','manifest.json'):shutil.copyfile(frozen/name,dataset/name)
    policy=(Path('checkpoints/ppo_joint_economy_soft_fireball_balanced_20261025/epoch_50.pt')if role=='candidate'else frozen/'parent.pt')
    digest=audit['checkpoint_sha256']if role=='candidate'else protocol['parent_sha256']
    if sha(policy)!=digest:raise ValueError('Selected checkpoint differs')
    shutil.copyfile(policy,dataset/'policy.pt');shutil.copyfile(selection,dataset/'selection.json')
    pins={n:sha(dataset/n)for n in ('source.tar.gz','manifest.json','policy.pt','selection.json')}
    for opponent in protocol['opponents']:
        path=Path(opponent['path'])
        if sha(path)!=opponent['sha256']:raise ValueError('Opponent changed: '+opponent['id'])
        name=opponent['id']+'.pt';shutil.copyfile(path,dataset/name);pins[name]=opponent['sha256']
    extensions={n:sha(n)for n in ('scripts/attack/evaluate_native_fireball.py',str(protocol_path),protocol['strategy_protocol'])}
    if set(extensions)&set(manifest['files']):raise ValueError('Evaluation must not replace training files')
    with tarfile.open(dataset/'evaluation.tar.gz','w:gz')as archive:
        for name in extensions:archive.add(name,arcname=name,recursive=False)
    pins['evaluation.tar.gz']=sha(dataset/'evaluation.tar.gz')
    slug='hastycr-native-fireball-'+role+'-20261025';kernel=owner+'/'+slug
    config=dict(owner=owner,slug=slug+'-inputs',role=role,pins=pins,extensions=extensions,source_sha256=protocol['training_source_sha256'])
    (dataset/'dataset-metadata.json').write_text(json.dumps(dict(id=owner+'/'+config['slug'],title=config['slug'],licenses=[dict(name='other')],isPrivate=True),indent=2)+'\n')
    job=output/'job';job.mkdir()
    source='import hashlib,json,shutil,tarfile\nfrom pathlib import Path,PurePosixPath\n'+inspect.getsource(sha)+'\n'+inspect.getsource(prepare_source)+'\n'+inspect.getsource(bootstrap)+'\nbootstrap('+repr(config)+')\n'
    compile(source,'evaluate.py','exec')
    (job/'evaluate.py').write_text(source)
    (job/'kernel-metadata.json').write_text(json.dumps(dict(id=kernel,title=slug,code_file='evaluate.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=False,dataset_sources=[owner+'/'+config['slug']],competition_sources=[],kernel_sources=[]),indent=2)+'\n')
    report=dict(schema='native-fireball-evaluation-preparation-v1',kernel=kernel,config=config,bootstrap_sha256=sha(job/'evaluate.py'),submitted=False,training_performed=False)
    (output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    return dict(kernel=kernel,role=role,prepared=True,submitted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--role',choices=('candidate','parent'),required=True)
    parser.add_argument('--owner',required=True)
    print(json.dumps(prepare(**vars(parser.parse_args()))))
