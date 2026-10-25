"""Reuse an existing private weight pool for a common-opponent usage study."""
import argparse
import inspect
import json
from pathlib import Path
import tarfile

from scripts.kaggle.run_five_round_robin import sha,prepare_source


def execute(config):
    import subprocess,sys
    root=Path('/kaggle/input')
    def mount(slug):
        choices=[p for p in (root/slug,root/'datasets'/config['owner']/slug)if p.is_dir()]
        if len(choices)!=1:raise ValueError('Ambiguous usage input')
        return choices[0]
    weights=mount(config['weights_slug']);inputs=mount(config['input_slug'])
    for name,digest in config['pins'].items():
        if name=='extensions.tar.gz'and not (inputs/name).exists():continue
        if sha(inputs/name)!=digest:raise ValueError('Usage input SHA differs')
    project=Path('/kaggle/working/card_usage_project')
    prepare_source(weights,project,config['source_manifest_sha256'])
    if (inputs/'extensions.tar.gz').exists():
        with tarfile.open(inputs/'extensions.tar.gz')as stream:
            members=stream.getmembers()
            if len(members)!=len(config['extensions'])or {m.name for m in members}!=set(config['extensions'])or any(not m.isfile()for m in members):raise ValueError('Usage extension members differ')
            if any((project/m.name).exists()or PurePosixPath(m.name).is_absolute()or '..'in PurePosixPath(m.name).parts for m in members):raise ValueError('Unsafe usage extension')
            stream.extractall(project,filter='data')
    else:
        base=(inputs/'extensions').resolve()
        for name in config['extensions']:
            source=base/name;target=project/name
            if source.is_symlink()or not source.resolve().is_relative_to(base)or target.exists():raise ValueError('Unsafe expanded usage extension')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    for name,digest in config['extensions'].items():
        if sha(project/name)!=digest:raise ValueError('Usage code changed')
    print(json.dumps(dict(event='INPUTS_VERIFIED',protocol_sha256=config['pins']['protocol.json'],source_manifest_sha256=config['source_manifest_sha256'])),flush=True)
    subprocess.run([sys.executable,'-m','scripts.attack.evaluate_card_usage','--protocol',str(inputs/'protocol.json'),
        '--weights',str(weights),'--output','/kaggle/working/card_usage'],cwd=project,check=True)


def prepare(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    dataset=output/'dataset';dataset.mkdir();job=output/'job';job.mkdir()
    original=json.loads(Path('data/fireball/native_evaluation_protocol.json').read_text())
    models=[dict(id=m['id'],file=m['id']+'.pt',sha256=m['sha256'],mode=m['mode'])for m in original['opponents']]
    selected=json.loads(Path('reports/defense_plan/fireball_balanced_match_validation.json').read_text())
    models.append(dict(id='fireball_balanced_epoch50',file='policy.pt',sha256=selected['checkpoint_sha256'],mode='joint_enemy_presence_v1',refined=True))
    protocol=dict(schema='frozen-common-opponent-card-usage-v1',models=models,opponent=next(m for m in models if m['id']=='v25'),
        geometry='native_anchors_candidate_v1',action_delay_ms=150,max_ms=400000,
        seeds=list(range(710001,710011)),seats=[1,-1],games_per_model=20,total_games=400,
        source_sha256=original['training_source_sha256'],source_manifest_sha256=original['training_manifest_sha256'],
        measurement='Accepted card plays; group-card bodies and spell effects are not additional casts',
        comparison_scope='Listed 20 frozen weights; this pool does not claim to include every historical experiment',
        test_used_for_selection=False,training_performed=False,physics_live_parity_accepted=False)
    (dataset/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    names=('scripts/attack/evaluate_card_usage.py',)
    extensions={name:sha(name)for name in names}
    with tarfile.open(dataset/'extensions.tar.gz','w:gz')as stream:
        for name in names:stream.add(name,arcname=name,recursive=False)
    owner='andreiperelaiko';slug='hastycr-card-usage-inputs-20261025'
    (dataset/'dataset-metadata.json').write_text(json.dumps(dict(id=owner+'/'+slug,title=slug,licenses=[dict(name='other')],isPrivate=True),indent=2)+'\n')
    config=dict(owner=owner,input_slug=slug,weights_slug='hastycr-native-fireball-candidate-20261025-inputs',
        source_manifest_sha256=original['training_manifest_sha256'],extensions=extensions,
        pins={name:sha(dataset/name)for name in ('protocol.json','extensions.tar.gz')})
    code='import hashlib,json,shutil,tarfile\nfrom pathlib import Path,PurePosixPath\n'+inspect.getsource(sha)+'\n'+inspect.getsource(prepare_source)+'\n'+inspect.getsource(execute)+'\nexecute('+repr(config)+')\n'
    compile(code,'evaluate.py','exec');(job/'evaluate.py').write_text(code)
    kernel=owner+'/hastycr-card-usage-20261025'
    (job/'kernel-metadata.json').write_text(json.dumps(dict(id=kernel,title='hastycr card usage 20261025',code_file='evaluate.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=False,dataset_sources=[owner+'/'+slug,owner+'/'+config['weights_slug']],competition_sources=[],kernel_sources=[]),indent=2)+'\n')
    report=dict(schema='common-card-usage-preparation-v1',kernel=kernel,config=config,bootstrap_sha256=sha(job/'evaluate.py'),models=20,games=400,submitted=False)
    (output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    return dict(kernel=kernel,models=20,games=400,submitted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    print(json.dumps(prepare(parser.parse_args().output)))
