"""Prepare four full PPO jobs only after screening and second-seed verification."""
import argparse
import inspect
import json
from pathlib import Path

from scripts.defense.select_ppo_coefficients import select,sha
from scripts.kaggle import run_five_round_robin as helpers
from scripts.kaggle.run_ppo_coefficient_study import training_command
from scripts.kaggle.prepare_ppo_confirmation import OWNERS


def prepare(study,confirmation,output):
    study=Path(study);confirmation=Path(confirmation)
    selected=select([study/o/'collected' for o in OWNERS]+[confirmation/o/'collected' for o in OWNERS[:3]],
                    Path('data/ppo_tuning/protocol.json'),Path('data/ppo_tuning/validation_contexts.json'),
                    study/'shared/manifest.json',Path('data/ppo_tuning/profiles'),'confirmed')
    choices=[(mode,profile) for mode in ('defense','joint') for profile in selected['modes'][mode]['full_training_profiles']]
    if len(choices)!=4 or any(len(selected['modes'][m]['full_training_profiles'])!=2 for m in ('defense','joint')):
        raise ValueError('Exactly two selected profiles per mode required')
    output=Path(output)
    if output.exists():raise ValueError('Use a new full-training package directory')
    source=Path(__file__).with_name('run_ppo_selected_full.py').read_text()
    source=source.replace('from scripts.kaggle.run_five_round_robin import sha,prepare_source',inspect.getsource(helpers.sha)+'\n'+inspect.getsource(helpers.prepare_source))
    source=source.replace('from scripts.kaggle.run_ppo_coefficient_study import training_command',inspect.getsource(training_command))
    if source.count('LAUNCH_CONFIG=None')!=1:raise ValueError('Ambiguous full-training configuration')
    prepared=[]
    for owner,(mode,profile) in zip(OWNERS,choices):
        previous=json.loads((study/owner/'launch_config.json').read_text())
        for name,digest in previous['pins'].items():
            if sha(study/'shared'/name)!=digest:raise ValueError('Original full-training inputs changed')
        config=dict(dataset=previous['dataset'],pins=previous['pins'],mode=mode,profile=profile,selection=selected)
        code=source.replace('LAUNCH_CONFIG=None','LAUNCH_CONFIG='+repr(config));compile(code,'train_full.py','exec')
        kernel=owner+'/hastycr-ppo-coef-full-20261023'
        metadata=dict(id=kernel,title=kernel.split('/')[1].replace('-',' '),code_file='train_full.py',language='python',kernel_type='script',is_private=True,
                      enable_gpu=True,enable_internet=False,dataset_sources=[config['dataset']],competition_sources=[],kernel_sources=[])
        prepared.append((owner,config,code,metadata))
    output.mkdir(parents=True,exist_ok=False)
    (output/'confirmed_selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    jobs=[]
    for owner,config,code,metadata in prepared:
        folder=output/owner;job=folder/'job';job.mkdir(parents=True)
        (folder/'launch_config.json').write_text(json.dumps(config,indent=2)+'\n')
        (job/'train_full.py').write_text(code);(job/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        jobs.append(dict(owner=owner,kernel=metadata['id'],mode=config['mode'],profile=config['profile'],code_sha256=sha(job/'train_full.py')))
    report=dict(schema='selected-full-ppo-preparation-v1',source_sha256=selected['source_sha256'],
                selection_sha256=sha(output/'confirmed_selection.json'),jobs=jobs,target_environment_steps_per_model=2048000,
                submitted=False,test_used_for_selection=False,quality_accepted=False)
    (output/'preparation_receipt.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','confirmation','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();print(json.dumps(prepare(args.study,args.confirmation,args.output),indent=2))


if __name__=='__main__':main()
