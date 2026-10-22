"""Prepare a private, self-contained evaluation job from completed recovery.

No uploads, training, test execution or checkpoint selection happen here.
"""
import argparse
import json
from pathlib import Path
import shutil

from scripts.defense.evaluate_stage2 import frozen_members, selected_recovery_candidates, selected_account2_candidate, sha
from scripts.kaggle.bootstrap_defense_recovery import dataset_parts


BOOTSTRAP='''
EVALUATION_LAUNCH_CONFIG = __CONFIG__

def launch_evaluation(config, input_root=Path('/kaggle/input'), working=Path('/kaggle/working')):
    def mounted(reference):
        owner, slug = reference.split('/')
        candidates = [input_root/slug, input_root/'datasets'/owner/slug]
        found = [p for p in candidates if p.is_dir()]
        if len(found) != 1:
            raise ValueError('Expected exactly one explicit evaluation input mount')
        return found[0]
    evaluation = mounted(config['evaluation_dataset'])
    source = mounted(config['selected_dataset'])
    if sha((evaluation/'manifest.json').read_bytes()) != config['evaluation_manifest_sha256']:
        raise ValueError('Pinned evaluation manifest SHA mismatch')
    result_file = config.get('result_file','recovery_result.json')
    raw = (source/result_file).read_bytes()
    if sha(raw) != config['recovery_result_sha256']:
        raise ValueError('Pinned recovery selection SHA mismatch')
    report = json.loads(raw)
    staged = working/'selected_input'
    # Verify all files before creating the staged input. The evaluation
    # controller additionally validates counters, provenance and selection.
    weights = {}
    for name in config.get('algorithms',('ppo','grpo')):
        artifact = report['selected_artifact'] if config.get('result_mode')=='account2' else report['selected_artifacts'][name]
        if artifact['path'] != 'selected/'+name+'.pt':
            raise ValueError('Unexpected selected checkpoint path')
        data = (source/(name+'.pt')).read_bytes()
        if sha(data) != artifact['sha256']:
            raise ValueError('Selected input SHA mismatch')
        weights[name] = data
    (staged/'selected').mkdir(parents=True,exist_ok=False)
    (staged/result_file).write_bytes(raw)
    for name,data in weights.items():
        (staged/'selected'/(name+'.pt')).write_bytes(data)
    return main(['--evaluation-input',str(evaluation),
        '--extracted-root',str(evaluation/'source'),
        '--stage2-input',str(staged),
        '--account2-result-sha256' if config.get('result_mode')=='account2' else '--recovery-result-sha256',config['recovery_result_sha256'],
        '--output',str(working/'evaluation')])

if __name__ == '__main__':
    launch_evaluation(EVALUATION_LAUNCH_CONFIG)
'''


def prepare(recovery_output,evaluation_input,destination,*,kernel,selected_dataset,
            evaluation_dataset='andriashadrin/hastycr-defense-eval-repaired-20261022',result_mode='recovery'):
    if result_mode not in ('recovery','account2'):raise ValueError('Unknown result mode')
    for reference in (kernel,selected_dataset,evaluation_dataset):dataset_parts(reference)
    recovery_output,evaluation_input=Path(recovery_output),Path(evaluation_input)
    _,manifest,_=frozen_members(evaluation_input)
    result_file='account2_result.json' if result_mode=='account2' else 'recovery_result.json'
    result_path=recovery_output/result_file
    result_sha=sha(result_path.read_bytes())
    selector=selected_account2_candidate if result_mode=='account2' else selected_recovery_candidates
    selected,_=selector(recovery_output,result_sha,manifest)
    controller=Path(__file__).resolve().parents[1]/'defense/evaluate_stage2.py'
    text=controller.read_text()
    marker="if __name__=='__main__':main()"
    if text.count(marker)!=1:raise ValueError('Ambiguous evaluation controller entry point')
    config=dict(evaluation_dataset=evaluation_dataset,selected_dataset=selected_dataset,
        evaluation_manifest_sha256=sha((evaluation_input/'manifest.json').read_bytes()),
        recovery_result_sha256=result_sha,evaluation_controller_sha256=sha(text.encode()),
        result_mode=result_mode,result_file=result_file,algorithms=list(selected))
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    dataset=destination/'selected_dataset';dataset.mkdir()
    for name,row in selected.items():shutil.copyfile(row['path'],dataset/(name+'.pt'))
    shutil.copyfile(result_path,dataset/result_file)
    (dataset/'dataset-metadata.json').write_text(json.dumps(dict(id=selected_dataset,
        title=selected_dataset.split('/')[1].replace('-',' '),licenses=[dict(name='other')],
        isPrivate=True),indent=2)+'\n')
    job=destination/'job';job.mkdir()
    source=text.replace(marker,BOOTSTRAP.replace('__CONFIG__',repr(config)),1)
    (job/'evaluate.py').write_text(source)
    (job/'kernel-metadata.json').write_text(json.dumps(dict(id=kernel,
        title=kernel.split('/')[1].replace('-',' '),code_file='evaluate.py',language='python',
        kernel_type='script',is_private=True,enable_gpu=True,enable_internet=False,
        dataset_sources=[evaluation_dataset,selected_dataset],competition_sources=[],kernel_sources=[]),indent=2)+'\n')
    config['generated_code_sha256']=sha(source.encode())
    (destination/'preparation_receipt.json').write_text(json.dumps(config,indent=2)+'\n')
    return config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery-output',type=Path,required=True)
    parser.add_argument('--evaluation-input',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--kernel',required=True)
    parser.add_argument('--selected-dataset',required=True)
    parser.add_argument('--result-mode',choices=('recovery','account2'),default='recovery')
    args=parser.parse_args()
    print(json.dumps(prepare(args.recovery_output,args.evaluation_input,args.destination,
        kernel=args.kernel,selected_dataset=args.selected_dataset,result_mode=args.result_mode),indent=2))


if __name__=='__main__':main()
