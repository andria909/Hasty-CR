"""Verify fixed-weight train evaluation separately from heldout results."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.defense.pack_evaluation import verified_frozen_package
from scripts.defense.summarize_selected_ppo import summarize_split,sha
from scripts.kaggle.run_selected_ppo_evaluation import verify_selection


def train_contexts(members):
    names=[]
    for task,path in (('defense','data/defense/scenarios.json'),('attack','data/attack/scenarios.json')):
        names.extend(task+'::'+r['scenario_id']for r in json.loads(members[path])['scenarios']if r['split']=='train')
    if len(names)!=len(set(names)):raise ValueError('Duplicate published train scenario')
    return [dict(scenario_id=name,seed=seed)for name in sorted(names)for seed in (42,43,44)]


def validate_records(records,contexts,mode):
    expected={(r['scenario_id'],r['seed'])for r in contexts}
    actual=[(r.get('strategy_scenario_id'),r.get('seed'))for r in records]
    if len(actual)!=len(contexts)or len(expected)!=len(contexts)or set(actual)!=expected:
        raise ValueError('Incomplete, duplicate or off-schedule train episode')
    coordinate='physical_live_cells_v2'if mode=='joint'else'physical_live'
    if any(r.get('split')!='train'or r.get('coordinate_mode')!=coordinate or r.get('geometry')!='legacy_v1'or
           r.get('task')not in ('attack','defense')or not r['strategy_scenario_id'].startswith(r['task']+'::')for r in records):
        raise ValueError('Train split/input/task contract differs')


def verify(collected,prepared,evaluation_package):
    root=Path(collected);prepared=Path(prepared)
    members,manifest,_,_=verified_frozen_package(evaluation_package)
    original=json.loads((prepared/'preparation_receipt.json').read_text());config=original['config']
    if sha(Path(evaluation_package)/'manifest.json')!=config['manifest_sha256']:raise ValueError('Common source differs')
    weights=prepared/'selected_dataset'
    if sha(weights/'selection.json')!=config['selection_sha256']or sha(weights/'policy.pt')!=config['policy_sha256']:
        raise ValueError('Fixed final weights differ')
    selected=json.loads((weights/'selection.json').read_text());mode=verify_selection(selected,config,manifest)
    receipt=json.loads((root/'collection_receipt.json').read_text())
    owner=original['kernel'].split('/')[0]
    if (receipt.get('remote_status')!='COMPLETE'or receipt.get('complete')is not True or
        receipt.get('kernel')!=owner+'/hastycr-ppo-selected-train-eval-20261023'):
        raise ValueError('Complete terminal train collection required')
    for name in ('result.json','train.json'):
        if receipt['files'].get(name,{}).get('sha256')!=sha(root/name):raise ValueError('Collected train artifact SHA differs')
    result=json.loads((root/'result.json').read_text());train=json.loads((root/'train.json').read_text())
    contexts=train_contexts(members);context_sha=hashlib.sha256(json.dumps(contexts,sort_keys=True).encode()).hexdigest()
    if (len(contexts)!=4452 or result.get('status')!='completed'or result.get('training_steps')!=2048000 or
        result.get('selected_sha256')!=config['policy_sha256']or result.get('selection_sha256')!=config['selection_sha256']or
        result.get('mode')!=mode or result.get('profile')!=selected['profile']or result.get('training_performed')is not False or
        result.get('source_sha256')!=manifest['archive_sha256']or result.get('context_schedule_sha256')!=context_sha or
        result.get('seeds')!=[42,43,44]or result.get('train_sha256')!=sha(root/'train.json')or result.get('episodes')!=4452):
        raise ValueError('Fixed train evaluation provenance/budget differs')
    if (train.get('checkpoint_sha256')!=config['policy_sha256']or train.get('source_sha256')!=manifest['archive_sha256']or
        train.get('context_schedule_sha256')!=context_sha or train.get('terminal_attribution_enabled')is not True or
        train.get('checkpoint_selection_performed')is not False or train.get('training_performed')is not False):
        raise ValueError('Train skill report contract differs')
    validate_records(train['records'],contexts,mode)
    return dict(schema='verified-selected-ppo-train-summary-v1',mode=mode,profile=selected['profile'],split='train',episodes=4452,
        checkpoint_sha256=config['policy_sha256'],source_sha256=manifest['archive_sha256'],seeds=[42,43,44],
        context_schedule_sha256=context_sha,metrics=summarize_split(train['records']),training_performed=False,
        used_for_checkpoint_selection=False,quality_accepted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('collected','prepared','evaluation-package','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();report=verify(args.collected,args.prepared,args.evaluation_package)
    with args.output.open('x')as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
