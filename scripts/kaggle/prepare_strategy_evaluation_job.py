"""Prepare one private joint evaluation from audited completed training.

No uploads, test execution, training, or checkpoint selection from test.
"""
import argparse
import json
from pathlib import Path
import shutil

from scripts.defense.pack_evaluation import verified_frozen_package
from scripts.kaggle.select_strategy_checkpoint import select, sha
from scripts.kaggle.bootstrap_defense_recovery import dataset_parts


def prepare(collected,launch_config,source_manifest,evaluation_package,output,*,kernel,evaluation_dataset,selected_dataset):
    for reference in (kernel,evaluation_dataset,selected_dataset):dataset_parts(reference)
    output=Path(output);evaluation_package=Path(evaluation_package)
    if output.exists():raise ValueError('Use a new joint evaluation package directory')
    _,manifest,_,_=verified_frozen_package(evaluation_package)
    original=json.loads(Path(source_manifest).read_text())
    if manifest.get('training_source_archive_sha256')!=original['archive_sha256']:
        raise ValueError('Evaluation package descends from a different training archive')
    selection=select(collected,launch_config,source_manifest,output/'selected_dataset')
    selected=output/'selected_dataset'
    def metadata(reference):
        return dict(id=reference,title=reference.split('/')[1].replace('-',' '),licenses=[dict(name='other')],isPrivate=True)
    (selected/'dataset-metadata.json').write_text(json.dumps(metadata(selected_dataset),indent=2)+'\n')
    evaluation=output/'evaluation_dataset';evaluation.mkdir()
    for name in ('source.tar.gz','manifest.json'):shutil.copyfile(evaluation_package/name,evaluation/name)
    (evaluation/'dataset-metadata.json').write_text(json.dumps(metadata(evaluation_dataset),indent=2)+'\n')
    config=dict(evaluation_dataset=evaluation_dataset,selected_dataset=selected_dataset,
        manifest_sha256=sha(evaluation/'manifest.json'),protocol_sha256=manifest['protocol_sha256'],
        selection_sha256=sha(selected/'selection.json'),policy_sha256=sha(selected/'policy.pt'))
    driver=Path(__file__).with_name('run_strategy_evaluation.py').read_text()
    if driver.count('LAUNCH_CONFIG=None')!=1:raise ValueError('Ambiguous joint runner entry point')
    code=driver.replace('LAUNCH_CONFIG=None','LAUNCH_CONFIG='+repr(config))
    compile(code,'evaluate.py','exec')
    job=output/'job';job.mkdir();(job/'evaluate.py').write_text(code)
    (job/'kernel-metadata.json').write_text(json.dumps(dict(id=kernel,title=kernel.split('/')[1].replace('-',' '),
        code_file='evaluate.py',language='python',kernel_type='script',is_private=True,
        enable_gpu=True,enable_internet=False,dataset_sources=[evaluation_dataset,selected_dataset],
        competition_sources=[],kernel_sources=[]),indent=2)+'\n')
    report=dict(schema='joint-evaluation-preparation-v1',algorithm=selection['algorithm'],
        kernel=kernel,config=config,job_sha256=sha(job/'evaluate.py'),
        selection_source=selection['selection_source'],test_used_for_selection=False,quality_accepted=False)
    (output/'preparation_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('collected','launch-config','source-manifest','evaluation-package','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('kernel','evaluation-dataset','selected-dataset'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.collected,args.launch_config,args.source_manifest,args.evaluation_package,args.output,
        kernel=args.kernel,evaluation_dataset=args.evaluation_dataset,selected_dataset=args.selected_dataset)))


if __name__=='__main__':main()
