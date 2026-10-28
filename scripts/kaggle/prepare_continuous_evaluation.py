"""Freeze completed control weights before their independent corrected-slot test."""
import argparse
import inspect
import itertools
import json
import re
from pathlib import Path
import shutil
import tarfile

from scripts.kaggle.run_five_round_robin import sha,prepare_source


def bootstrap(config):
    import subprocess,sys
    def mount(slug):
        root=Path('/kaggle/input');paths=[p for p in(root/slug,root/'datasets'/config['owner']/slug)if p.is_dir()]
        if len(paths)!=1:raise ValueError('Ambiguous mount')
        return paths[0]
    inputs=mount(config['slug']);source=mount(config['source_slug'])
    for name,digest in config['pins'].items():
        if name=='extensions.tar.gz'and not(inputs/name).exists():continue
        if sha(inputs/name)!=digest:raise ValueError('Pinned evaluation input differs')
    project=Path('/kaggle/working/control_eval_project')
    prepare_source(source,project,config['source_manifest_sha256'])
    for name,digest in config['replaced'].items():
        if sha(project/name)!=digest:raise ValueError('Unexpected physics source migration')
    if(inputs/'extensions.tar.gz').exists():
        with tarfile.open(inputs/'extensions.tar.gz')as stream:
            members=stream.getmembers()
            if len(members)!=len(config['extensions'])or{m.name for m in members}!=set(config['extensions']):raise ValueError('Extension list differs')
            if any(not m.isfile()or PurePosixPath(m.name).is_absolute()or '..'in PurePosixPath(m.name).parts for m in members):raise ValueError('Unsafe extension')
            stream.extractall(project,filter='data')
    else:
        base=(inputs/'extensions').resolve()
        for name in config['extensions']:
            original=base/name;target=project/name
            if original.is_symlink()or not original.resolve().is_relative_to(base):raise ValueError('Unsafe extension')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
    for name,digest in config['extensions'].items():
        if sha(project/name)!=digest:raise ValueError('Evaluation source differs')
    print(json.dumps(dict(event='INPUTS_VERIFIED',protocol_sha256=config['pins']['protocol.json'],
        source_manifest_sha256=config['source_manifest_sha256'])),flush=True)
    subprocess.run([sys.executable,'-m','scripts.attack.evaluate_continuous_pool',
        '--protocol',str(inputs/'protocol.json'),'--weights',str(inputs),
        '--output','/kaggle/working/continuous_pool_evaluation'],cwd=project,check=True)


def selected_models(verified,cohort):
    kernels={'control':'andriashadrin/hastycr-continuous-ppo-20261026',
             'slots':'andreiperelaiko/hastycr-continuous-slots-20261026'}
    if (cohort not in kernels or verified.get('kernel')!=kernels[cohort] or
            verified.get('schema')!='verified-continuous-ppo-training-v1' or
            verified.get('all_training_budgets_verified') is not True):
        raise ValueError('Matching verified training cohort required')
    models=verified.get('models',[])
    if (len(models)!=2 or len({m['id'] for m in models})!=2 or
            any(not re.fullmatch(r'full_(control|leak|balanced|strong)_s44',m['id']) or
                m.get('steps')!=4096000 for m in models)):
        raise ValueError('Two distinct full-budget validation-selected models required')
    return models


def prepare(output,cohort='control',training_report=None,owner='andriashadrin'):
    training_report=Path(training_report or ('reports/defense_plan/continuous_slots_verified_20261026.json'
        if cohort=='slots' else 'reports/defense_plan/continuous_training_verified_20261026.json'))
    verified=json.loads(training_report.read_text())
    selected=selected_models(verified,cohort)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    dataset=output/'dataset';dataset.mkdir();job=output/'job';job.mkdir()
    models=[]
    for item in selected:
        path=Path(item['checkpoint']);name=item['id'];digest=item['checkpoint_sha256']
        if sha(path)!=digest:raise ValueError('Verified model changed')
        shutil.copyfile(path,dataset/(name+'.pt'))
        models.append(dict(id=name,file=name+'.pt',sha256=digest,scalars=309,mode='joint_enemy_presence_v1',
            training_executor='independent_slots_v2' if cohort=='slots' else 'legacy_global_ack_block_2100ms'))
    for name,path,mode,count in(
        ('parent','checkpoints/ppo_joint_economy_soft_20261023/policy.pt','joint_enemy_presence_v1',309),
        ('v25','checkpoints/portable_v25/portable_v25_cannon_economy_best.pt','physical_live',306)):
        original=Path(path);shutil.copyfile(original,dataset/(name+'.pt'))
        models.append(dict(id=name,file=name+'.pt',sha256=sha(original),scalars=count,mode=mode))
    protocol=dict(schema='continuous-frozen-pool-test-protocol-v1',models=models,test_seeds=list(range(910001,910011)),
        duel_seeds=list(range(920001,920011)),test_games=480,duel_games=120,total_games=600,
        evaluation_executor='independent_slots_v2',command_delay_ms=225,slot_refill_ms=2100,decision_interval_ms=250,
        selection_used='Only frozen validation-selected weights; this test never chooses coefficients/checkpoints',
        training_cohort=cohort,training_report_sha256=sha(training_report),
        geometry='native_anchors_candidate_v1',physics_fully_validated=False,live_games_started=False)
    (dataset/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    frozen=Path('tmp/kaggle/fireball_balanced_20261025/dataset');manifest=json.loads((frozen/'manifest.json').read_text())
    names=('sim/match.py','sim/train_attack_ppo.py','sim/continuous_push.py','sim/card_balance.py','sim/env.py',
        'sim/kite_geometry.py','sim/projectile_calibration.py','sim/entities.py','sim/gamedata.py','sim/public_gamedata.py',
        'sim/spells.py','sim/cannon_placement.py','scripts/attack/evaluation_execution.py','scripts/attack/evaluate_continuous_pool.py')
    extensions={n:sha(n)for n in names}
    with tarfile.open(dataset/'extensions.tar.gz','w:gz')as stream:
        for name in names:stream.add(name,arcname=name,recursive=False)
    slug=f'hastycr-continuous-{cohort}-eval-inputs-20261026'
    for name in ('manifest.json','source.tar.gz'):shutil.copyfile(frozen/name,dataset/name)
    (dataset/'dataset-metadata.json').write_text(json.dumps(dict(id=owner+'/'+slug,title=slug,licenses=[dict(name='other')],isPrivate=True),indent=2)+'\n')
    config=dict(owner=owner,slug=slug,source_slug=slug,
        source_manifest_sha256=sha(frozen/'manifest.json'),extensions=extensions,
        replaced={n:manifest['files'][n]for n in names if n in manifest['files']},
        pins={n:sha(dataset/n)for n in ('manifest.json','protocol.json','extensions.tar.gz',*[m['file']for m in models])})
    code='import hashlib,json,shutil,tarfile\nfrom pathlib import Path,PurePosixPath\n'+inspect.getsource(sha)+'\n'+inspect.getsource(prepare_source)+'\n'+inspect.getsource(bootstrap)+'\nbootstrap('+repr(config)+')\n'
    compile(code,'train.py','exec');(job/'train.py').write_text(code)
    kernel=owner+f'/hastycr-continuous-{cohort}-eval-20261026'
    (job/'kernel-metadata.json').write_text(json.dumps(dict(id=kernel,title=f'hastycr continuous {cohort} eval 20261026',
        code_file='train.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=False,
        dataset_sources=[owner+'/'+slug],competition_sources=[],kernel_sources=[]),indent=2)+'\n')
    (output/'preparation.json').write_text(json.dumps(dict(kernel=kernel,config=config,bootstrap_sha256=sha(job/'train.py'),submitted=False),indent=2)+'\n')
    return dict(kernel=kernel,total_games=600,submitted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--cohort',choices=('control','slots'),default='control')
    p.add_argument('--training-report')
    p.add_argument('--owner',default='andriashadrin')
    a=p.parse_args();print(json.dumps(prepare(a.output,a.cohort,a.training_report,a.owner)))
