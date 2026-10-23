"""Package an explicit one-expression environment migration from recovered update796."""
import argparse
import inspect
import json
from pathlib import Path
import shutil
import tarfile

from scripts.kaggle import run_five_round_robin as helpers
from scripts.kaggle.run_ppo_coefficient_study import training_command
from scripts.kaggle.run_ppo_selected_full import verify_script_target_migration


def original_full_config(diagnostic):
    previous=json.loads((Path(diagnostic)/'launch_config.json').read_text())['original']
    if (previous.get('mode'),previous.get('profile'))!=('defense','economy_soft')or previous.get('selection',{}).get('stage')!='confirmed':
        raise ValueError('Expected original confirmed full-training slot')
    return previous


def prepare(study,diagnostic,output):
    study=Path(study);diagnostic=Path(diagnostic);output=Path(output)
    receipt=json.loads((diagnostic/'collected/receipt.json').read_text())
    result=json.loads((diagnostic/'collected/result.json').read_text())
    comparisons=result.get('original_prefix_comparisons',{})
    if (receipt['status']!='ERROR' or result['status']!='failed' or
        set(comparisons)!={str(n)for n in range(701,797)} or not all(comparisons.values()) or
        result.get('all_compared_updates_identical')is not True):
        raise ValueError('Exact failing prefix not verified')
    telemetry=result['failure_telemetry']
    if (telemetry['card'],telemetry['target'],telemetry['can_play'],telemetry['in_arena'])!=('zap',[-24,21468],True,False):
        raise ValueError('Unexpected rejected script target')
    checkpoint=diagnostic/'collected'/result['checkpoint']
    if helpers.sha(checkpoint)!=result['checkpoint_sha256']:
        raise ValueError('Recovered checkpoint hash differs')
    previous=original_full_config(diagnostic)
    original=json.loads((study/'shared/manifest.json').read_text())
    if original['archive_sha256']!=result['source_sha256']:
        raise ValueError('Diagnostic and original source differ')
    output.mkdir(parents=True,exist_ok=False)
    inputs=output/'inputs';inputs.mkdir()
    project=output/'verified_source'
    helpers.prepare_source(study/'shared',project,helpers.sha(study/'shared/manifest.json'))
    path=project/'sim/defense_env.py';text=path.read_text()
    if text.count('return chosen.pos')!=1:raise ValueError('Unexpected original target expression')
    path.write_text(text.replace('return chosen.pos','return arena.clamp_to_arena(chosen.pos)'))
    manifest=dict(original);manifest['files']=dict(original['files'])
    manifest['files']['sim/defense_env.py']=helpers.sha(path)
    with tarfile.open(inputs/'source.tar.gz','w:gz')as stream:
        for name in sorted(manifest['files']):stream.add(project/name,arcname=name,recursive=False)
    manifest['archive_sha256']=helpers.sha(inputs/'source.tar.gz')
    (inputs/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copy2(study/'shared/manifest.json',inputs/'original_manifest.json')
    shutil.copy2(checkpoint,inputs/'resume.pt')
    for name in ('teacher_defense.pt','teacher_joint.pt'):shutil.copy2(study/'shared'/name,inputs/name)
    migration=dict(schema='script-target-clamp-migration-v1',
                   original_manifest_sha256=helpers.sha(inputs/'original_manifest.json'),
                   original_source_sha256=original['archive_sha256'],
                   replacement='arena.clamp_to_arena(chosen.pos)',resume_update=796,resume_step=815104,
                   resume_checkpoint_sha256=helpers.sha(inputs/'resume.pt'),
                   diagnostic_kernel=receipt['kernel'],exact_prefix_updates=96,
                   semantics='Clamp scripted body-target cast point to board; preserve body position, rewards, schedule and invalid-play guard')
    dataset='vertel0/hastycr-ppo-target-recovery-20261023'
    config=dict(previous,dataset=dataset,migration=migration,
                pins={name:helpers.sha(inputs/name)for name in ('manifest.json','original_manifest.json','resume.pt','teacher_defense.pt','teacher_joint.pt')})
    verify_script_target_migration(config,inputs,project,manifest)
    source=Path(__file__).with_name('run_ppo_selected_full.py').read_text()
    source=source.replace('from scripts.kaggle.run_five_round_robin import sha,prepare_source',inspect.getsource(helpers.sha)+'\n'+inspect.getsource(helpers.prepare_source))
    source=source.replace('from scripts.kaggle.run_ppo_coefficient_study import training_command',inspect.getsource(training_command))
    if source.count('LAUNCH_CONFIG=None')!=1:raise ValueError('Ambiguous controller configuration')
    source=source.replace('LAUNCH_CONFIG=None','LAUNCH_CONFIG='+repr(config));compile(source,'recover.py','exec')
    job=output/'job';job.mkdir();(job/'recover.py').write_text(source)
    kernel='vertel0/hastycr-ppo-def796-recovery-20261023'
    metadata=dict(id=kernel,title=kernel.split('/')[1].replace('-',' '),code_file='recover.py',language='python',kernel_type='script',
                  is_private=True,enable_gpu=True,enable_internet=False,dataset_sources=[dataset],competition_sources=[],kernel_sources=[])
    (job/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (inputs/'dataset-metadata.json').write_text(json.dumps(dict(id=dataset,title='HastyCR PPO target recovery 20261023',licenses=[{'name':'CC0-1.0'}]),indent=2)+'\n')
    (output/'launch_config.json').write_text(json.dumps(config,indent=2)+'\n')
    return dict(kernel=kernel,dataset=dataset,migration=migration,source_sha256=manifest['archive_sha256'],launched=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--diagnostic',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())),indent=2))
