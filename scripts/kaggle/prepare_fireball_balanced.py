"""Package an explicit corrected-physics v3 phase, preserving every old input."""
import argparse
import inspect
import json
from pathlib import Path
import shutil
import tarfile

from scripts.kaggle.run_five_round_robin import sha,prepare_source


def prepare(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    dataset=output/'dataset';dataset.mkdir()
    old=json.loads(Path('tmp/kaggle/fireball_selectivity_20261025/dataset/manifest.json').read_text())
    files={name:sha(name)for name in old['files']}
    for name in ('sim/fireball_balanced.py','scripts/attack/train_fireball_balanced.py','data/fireball/balanced_protocol.json'):
        files[name]=sha(name)
    if any(not name.startswith(('sim/','scripts/','data/','src/'))for name in files):raise ValueError('Unexpected source path')
    changed=[n for n in old['files']if files[n]!=old['files'][n]]
    if set(changed)!={'sim/spells.py','sim/public_gamedata.py','sim/fireball_curriculum.py','sim/eval_defense_duel.py'}:
        raise ValueError('Expected only declared Fireball/public-task/geometry changes from v2')
    with tarfile.open(dataset/'source.tar.gz','w:gz')as archive:
        for name in sorted(files):archive.add(name,arcname=name,recursive=False)
    manifest=dict(schema='fireball-balanced-source-v1',archive_sha256=sha(dataset/'source.tar.gz'),files=files,
        previous_archive_sha256=old['archive_sha256'],migration=dict(schema='native-fireball-public-task-v1',
            changed_files=changed,fireball_speed_mt_per_sec=12000,public_task_rule='joint_enemy_presence_v1',
            default_match_geometry='native_anchors_candidate_v1',old_optimizer_resumed=False))
    (dataset/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    protocol=json.loads(Path('data/fireball/balanced_protocol.json').read_text())
    for name,path,digest in [('parent.pt','checkpoints/ppo_joint_economy_soft_20261023/policy.pt',protocol['init_sha256']),
            ('v25.pt','checkpoints/portable_v25/portable_v25_cannon_economy_best.pt',
             'd65ab55893592c0e4655a1882db459c4a7fb1ebd9e10afaf2daa4d2b369fc972')]:
        if sha(path)!=digest:raise ValueError('Preserved policy differs')
        shutil.copyfile(path,dataset/name)
    owner='andriashadrin';slug='hastycr-fireball-balanced-inputs-20261025'
    (dataset/'dataset-metadata.json').write_text(json.dumps(dict(id=owner+'/'+slug,title=slug,
        licenses=[dict(name='other')],isPrivate=True),indent=2)+'\n')
    pins={name:sha(dataset/name)for name in ('manifest.json','parent.pt','v25.pt')}
    job=output/'job';job.mkdir()
    code='''import hashlib,json,shutil,sys,tarfile,subprocess
from pathlib import Path,PurePosixPath
'''+inspect.getsource(sha)+'\n'+inspect.getsource(prepare_source)+f'''
root=Path('/kaggle/input');slug={slug!r}
paths=[p for p in (root/slug,root/'datasets'/{owner!r}/slug)if p.is_dir()]
if len(paths)!=1:raise ValueError('Ambiguous balanced inputs')
inputs=paths[0]
for name,digest in {pins!r}.items():
    if sha(inputs/name)!=digest:raise ValueError('Balanced input SHA differs')
project=Path('/kaggle/working/fireball_balanced_project')
prepare_source(inputs,project,{pins['manifest.json']!r})
print(json.dumps(dict(event='INPUTS_VERIFIED',source_sha256={manifest['archive_sha256']!r},protocol_sha256={sha('data/fireball/balanced_protocol.json')!r})),flush=True)
subprocess.run([sys.executable,'-m','scripts.attack.train_fireball_balanced','--init',str(inputs/'parent.pt'),'--v25',str(inputs/'v25.pt'),'--protocol',str(project/'data/fireball/balanced_protocol.json'),'--source-sha256',{manifest['archive_sha256']!r},'--output','/kaggle/working/fireball_balanced'],cwd=project,check=True)
'''
    (job/'train_balanced.py').write_text(code)
    kernel=owner+'/hastycr-fireball-balanced-20261025'
    (job/'kernel-metadata.json').write_text(json.dumps(dict(id=kernel,title='hastycr fireball balanced 20261025',
        code_file='train_balanced.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,
        enable_internet=False,dataset_sources=[owner+'/'+slug],competition_sources=[],kernel_sources=[]),indent=2)+'\n')
    report=dict(schema='fireball-balanced-preparation-v1',kernel=kernel,pins=pins,
        source_sha256=manifest['archive_sha256'],protocol_sha256=sha('data/fireball/balanced_protocol.json'),
        bootstrap_sha256=sha(job/'train_balanced.py'),previous_source_sha256=old['archive_sha256'],
        migration=manifest['migration'],init_sha256=protocol['init_sha256'],protocol=protocol,submitted=False,
        native_speed_evidence_sha256=sha('reports/defense_plan/recorded_native_fireball_velocity.json'))
    (output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    print(json.dumps(prepare(p.parse_args().output)))
