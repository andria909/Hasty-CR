"""Prepare private heldout evaluation only from audited completed full PPO output."""
import argparse
import inspect
import json
from pathlib import Path
import shutil

from scripts.defense.pack_evaluation import verified_frozen_package
from scripts.kaggle.bootstrap_defense_recovery import dataset_parts
from scripts.kaggle.select_completed_ppo_full import select,sha
from scripts.kaggle import run_strategy_evaluation,run_five_round_robin


def verify_evaluation_lineage(manifest,source):
    migration=manifest.get('environment_migration')
    if migration is None:
        if manifest.get('training_source_archive_sha256')!=source['archive_sha256']:
            raise ValueError('Evaluation source differs from full training')
        return
    original=manifest.get('original_training_files',{})
    if not original:raise ValueError('Original training file pins missing')
    expected=dict(original)
    if source['archive_sha256']==migration.get('migrated_training_source_sha256'):
        expected['sim/defense_env.py']=migration['evaluation_defense_env_sha256']
    elif source['archive_sha256']!=manifest['training_source_archive_sha256']:
        raise ValueError('Evaluation source differs from full training')
    if source['files']!=expected:raise ValueError('Evaluation/training file lineage differs')


def standalone_source(config):
    code=Path(__file__).with_name('run_selected_ppo_evaluation.py').read_text()
    replacements={
        'from scripts.kaggle.run_strategy_evaluation import prepare_source,sha':inspect.getsource(run_strategy_evaluation.sha)+'\n'+inspect.getsource(run_strategy_evaluation.prepare_source),
        'from scripts.kaggle.run_five_round_robin import skill_policy,skill_environment':inspect.getsource(run_five_round_robin.skill_policy)+'\n'+inspect.getsource(run_five_round_robin.skill_environment),
        'LAUNCH_CONFIG=None':'LAUNCH_CONFIG='+repr(config)}
    for old,new in replacements.items():
        if code.count(old)!=1:raise ValueError('Ambiguous standalone evaluation template')
        code=code.replace(old,new,1)
    compile(code,'evaluate.py','exec');return code


def prepare(collected,launch_config,source_manifest,evaluation_package,output,*,kernel,evaluation_dataset,selected_dataset):
    references=[dataset_parts(ref) for ref in (kernel,evaluation_dataset,selected_dataset)]
    if len({row[0]for row in references})!=1:raise ValueError('Private evaluation inputs must belong to the kernel owner')
    output=Path(output)
    if output.exists():raise ValueError('Use a new selected-PPO evaluation directory')
    _,manifest,_,_=verified_frozen_package(evaluation_package)
    source=json.loads(Path(source_manifest).read_text());launch=json.loads(Path(launch_config).read_text())
    verify_evaluation_lineage(manifest,source)
    if launch['mode'] not in manifest.get('supported_training_modes',[]):raise ValueError('Evaluation package does not support this policy mode')
    if 'sim/evaluation_terminal_damage.py' not in manifest['files']:raise ValueError('Terminal attribution evaluator required')
    selection=select(collected,launch_config,source_manifest,output/'selected_dataset')
    selected=output/'selected_dataset';evaluation=output/'evaluation_dataset';evaluation.mkdir()
    for name in ('source.tar.gz','manifest.json'):shutil.copyfile(Path(evaluation_package)/name,evaluation/name)
    def metadata(reference):return dict(id=reference,title=reference.split('/')[1].replace('-',' '),licenses=[dict(name='other')],isPrivate=True)
    (selected/'dataset-metadata.json').write_text(json.dumps(metadata(selected_dataset),indent=2)+'\n')
    (evaluation/'dataset-metadata.json').write_text(json.dumps(metadata(evaluation_dataset),indent=2)+'\n')
    config=dict(evaluation_dataset=evaluation_dataset,selected_dataset=selected_dataset,
                manifest_sha256=sha(evaluation/'manifest.json'),protocol_sha256=manifest['protocol_sha256'],
                selection_sha256=sha(selected/'selection.json'),policy_sha256=sha(selected/'policy.pt'))
    job=output/'job';job.mkdir();(job/'evaluate.py').write_text(standalone_source(config))
    (job/'kernel-metadata.json').write_text(json.dumps(dict(id=kernel,title=kernel.split('/')[1].replace('-',' '),
        code_file='evaluate.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=False,
        dataset_sources=[evaluation_dataset,selected_dataset],competition_sources=[],kernel_sources=[]),indent=2)+'\n')
    report=dict(schema='selected-ppo-final-evaluation-preparation-v1',kernel=kernel,mode=selection['mode'],profile=selection['profile'],
                config=config,code_sha256=sha(job/'evaluate.py'),training_steps=2048000,
                common_test_episodes=804,defense_test_episodes=708,attack_test_episodes=96,opponent_games=300,
                terminal_attribution_enabled=True,submitted=False,test_used_for_selection=False,quality_accepted=False)
    report['input_pins']=dict(collection_receipt_sha256=sha(Path(collected)/'receipt.json'),
                             launch_config_sha256=sha(launch_config),source_manifest_sha256=sha(source_manifest),
                             evaluation_manifest_sha256=sha(Path(evaluation_package)/'manifest.json'))
    (output/'preparation_receipt.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collected','launch-config','source-manifest','evaluation-package','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('kernel','evaluation-dataset','selected-dataset'):p.add_argument('--'+name,required=True)
    args=p.parse_args();print(json.dumps(prepare(args.collected,args.launch_config,args.source_manifest,args.evaluation_package,args.output,
        kernel=args.kernel,evaluation_dataset=args.evaluation_dataset,selected_dataset=args.selected_dataset),indent=2))


if __name__=='__main__':main()
