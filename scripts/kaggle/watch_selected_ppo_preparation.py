"""Prepare final evaluation from complete local collections, without remote calls."""
import argparse
import json
from pathlib import Path
import time

from scripts.kaggle.prepare_selected_ppo_evaluation import prepare,sha


def input_pins(run,package):
    return dict(collection_receipt_sha256=sha(Path(run['collected'])/'receipt.json'),
                launch_config_sha256=sha(run['launch_config']),source_manifest_sha256=sha(run['source_manifest']),
                evaluation_manifest_sha256=sha(Path(package)/'manifest.json'))


def observe(config):
    rows=[]
    for run in config['runs']:
        row=dict(owner=run['owner'],state='awaiting_collection')
        receipt=Path(run['collected'])/'receipt.json';output=Path(run['output'])
        if not receipt.exists():rows.append(row);continue
        try:collection=json.loads(receipt.read_text())
        except json.JSONDecodeError:rows.append(row);continue
        if collection.get('complete')is not True or collection.get('status')!='COMPLETE':
            row.update(state='training_not_complete',training_status=collection.get('training_status'),
                       kernel_status=collection.get('status'))
            rows.append(row);continue
        if output.exists():
            artifact=output/'preparation_receipt.json'
            if artifact.exists():
                previous=json.loads(artifact.read_text())
                if (previous.get('kernel')!=run['kernel']or previous.get('training_steps')!=2048000 or
                    previous.get('input_pins')!=input_pins(run,config['evaluation_package'])):
                    row.update(state='incomplete_preparation_requires_review',error='Prepared inputs changed');rows.append(row);continue
                row.update(state='prepared',receipt=str(artifact));rows.append(row);continue
            row.update(state='incomplete_preparation_requires_review',output=str(output));rows.append(row);continue
        try:
            prepare(run['collected'],run['launch_config'],run['source_manifest'],config['evaluation_package'],output,
                    kernel=run['kernel'],evaluation_dataset=run['evaluation_dataset'],selected_dataset=run['selected_dataset'])
        except Exception as error:
            # Preserve any partial package. A future observation must not repeat
            # writes over it or reinterpret it as an evaluated model.
            row.update(state='preparation_failed',error_type=type(error).__name__,error=str(error))
        else:row.update(state='prepared',receipt=str(output/'preparation_receipt.json'))
        rows.append(row)
    return dict(schema='selected-ppo-preparation-observation-v1',checked_host_ns=time.time_ns(),runs=rows,
                all_prepared=all(r['state']=='prepared'for r in rows)and len(rows)==4,
                remote_calls_performed=False,gpu_jobs_submitted=False,quality_accepted=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--poll-seconds',type=int,default=1200)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    if args.poll_seconds<1200:parser.error('Observation interval must be at least20minutes')
    config=json.loads(args.config.read_text())
    if len(config.get('runs',[]))!=4 or len({r['owner']for r in config['runs']})!=4:
        parser.error('Exactly four distinct training owners required')
    while True:
        result=observe(config);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
        if not args.wait or result['all_prepared']:break
        if any(r['state']in ('preparation_failed','incomplete_preparation_requires_review')for r in result['runs']):break
        time.sleep(args.poll_seconds)


if __name__=='__main__':main()
