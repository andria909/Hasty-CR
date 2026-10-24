"""Prepare descriptive train metrics using already published final-evaluation inputs."""
import argparse
import inspect
import json
from pathlib import Path

from scripts.kaggle import run_strategy_evaluation,run_five_round_robin
from scripts.kaggle.run_selected_ppo_evaluation import verify_selection


def standalone(config):
    source=Path(__file__).with_name('run_selected_ppo_train_evaluation.py').read_text()
    replacements={
        'from scripts.kaggle.run_strategy_evaluation import prepare_source,sha':inspect.getsource(run_strategy_evaluation.prepare_source)+'\n'+inspect.getsource(run_strategy_evaluation.sha),
        'from scripts.kaggle.run_five_round_robin import skill_policy,skill_environment':inspect.getsource(run_five_round_robin.skill_policy)+'\n'+inspect.getsource(run_five_round_robin.skill_environment),
        'from scripts.kaggle.run_selected_ppo_evaluation import verify_selection':inspect.getsource(verify_selection),
        'LAUNCH_CONFIG=None':'LAUNCH_CONFIG='+repr(config)}
    for old,new in replacements.items():
        if source.count(old)!=1:raise ValueError('Ambiguous train-evaluation controller')
        source=source.replace(old,new,1)
    compile(source,'train_evaluate.py','exec');return source


def prepare(prepared,output):
    prepared=Path(prepared);output=Path(output)
    if output.exists():raise ValueError('Use a new train-evaluation package directory')
    previous=json.loads((prepared/'preparation_receipt.json').read_text());config=previous['config']
    if (run_strategy_evaluation.sha(prepared/'selected_dataset/policy.pt')!=config['policy_sha256']or
        run_strategy_evaluation.sha(prepared/'selected_dataset/selection.json')!=config['selection_sha256']):
        raise ValueError('Predetermined final evaluation inputs changed')
    manifest=json.loads((prepared/'evaluation_dataset/manifest.json').read_text())
    if run_strategy_evaluation.sha(prepared/'evaluation_dataset/manifest.json')!=config['manifest_sha256']:
        raise ValueError('Common evaluation manifest changed')
    selected=json.loads((prepared/'selected_dataset/selection.json').read_text());verify_selection(selected,config,manifest)
    owner=previous['kernel'].split('/')[0];kernel=owner+'/hastycr-ppo-selected-train-eval-20261023'
    job=output/'job';job.mkdir(parents=True);(job/'train_evaluate.py').write_text(standalone(config))
    (job/'kernel-metadata.json').write_text(json.dumps(dict(id=kernel,title=kernel.split('/')[1].replace('-',' '),code_file='train_evaluate.py',
        language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=False,
        dataset_sources=[config['evaluation_dataset'],config['selected_dataset']],competition_sources=[],kernel_sources=[]),indent=2)+'\n')
    (output/'launch_config.json').write_text(json.dumps(config,indent=2)+'\n')
    report=dict(kernel=kernel,mode=selected['mode'],profile=selected['profile'],selected_sha256=selected['selected_sha256'],
                expected_episodes=4452,train_scenarios=1484,seeds=[42,43,44],submitted=False,training_performed=False,
                test_used_for_selection=False,quality_accepted=False)
    (output/'preparation_receipt.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())),indent=2))
