"""Audit two isolated full runs against the reverified global pilot selection."""
import argparse
import json
from pathlib import Path

from scripts.attack.select_pilot_pool import select, verify_study
from scripts.attack.verify_continuous_focus import verify as verify_pilot, audit_run, sha


def audit_training_execution(config,blob,metrics_path):
    from scripts.attack.evaluation_execution import training_execution,combat_modes
    execution=training_execution(config)
    checked=0
    if execution is not None:
        if training_execution(blob['config'])!=execution:
            raise ValueError('Checkpoint training geometry differs from run config')
        states=blob.get('environment_states',[])
        if not states:raise ValueError('Missing actual training environment states')
        for state in states:
            battle=state['match'].battle
            if training_execution(dict(config,continuous_profile=state.get('profile')))!=execution:
                raise ValueError('Actual environment profile differs')
            if any(getattr(battle,k,None)!=v for k,v in combat_modes(execution).items()):
                raise ValueError('Actual checkpoint battle geometry differs')
        checked=0
        for line in Path(metrics_path).read_text().splitlines():
            validation=json.loads(line).get('validation')
            if not validation:continue
            for record in validation['records']:
                if any(record.get(k)!=v for k,v in combat_modes(execution).items()):
                    raise ValueError('Actual validation geometry differs')
                checked+=1
        if not checked:raise ValueError('No geometry-tagged validation records')
    return execution,checked


def verify(collected, prepared):
    root, prepared = Path(collected), Path(prepared)
    prep = json.loads((prepared/'preparation.json').read_text())
    if sha(prepared/'job/train.py') != prep['bootstrap_sha256']:
        raise ValueError('Global full driver changed')
    for name, digest in prep['config']['pins'].items():
        path = prepared/'dataset'/name
        if path.is_symlink() or not path.resolve().is_relative_to((prepared/'dataset').resolve()) or sha(path) != digest:
            raise ValueError('Paired full input changed')
    ranking = json.loads((prepared/'dataset/global_selection.json').read_text())
    current = select([r['prepared'] for r in ranking['studies']], limit=4)
    if current != ranking or not current['gate_met']:
        raise ValueError('Global pilot selection changed or failed')
    pair_index = prep['pair_index']
    expected = current['selected'][2*pair_index:2*pair_index+2]
    if pair_index not in (0,1) or len(expected) != 2 or [r['selected'] for r in prep['children']] != expected:
        raise ValueError('Wrong selected candidate pair')
    receipt = json.loads((root/'collection_receipt.json').read_text())
    if not receipt.get('complete') or receipt.get('remote_status') != 'COMPLETE' or receipt['kernel'] != prep['kernel']:
        raise ValueError('Matching successful terminal collection required')
    for name, meta in receipt['files'].items():
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or sha(path) != meta['sha256']:
            raise ValueError('Full artifact changed')
    if 'global_full_result.json' not in receipt['files']:
        raise ValueError('Required full result missing from receipt')
    result = json.loads((root/'global_full_result.json').read_text())
    if (result.get('status') != 'complete' or result.get('children') != 2
            or result.get('full_steps') != 8192000 or result.get('pilot_steps_repeated') != 0):
        raise ValueError('Global full budget/result differs')
    models = []
    import torch
    torch.set_num_threads(1)
    for index, child in enumerate(prep['children']):
        if child['index'] != index or child['remote_output'] != f'global_full_child_{index}/continuous_focus':
            raise ValueError('Child output namespace differs')
        dataset = prepared/'dataset'/f'bundle_{index}'
        if sha(dataset/'train.py') != child['bootstrap_sha256']:
            raise ValueError('Full child controller changed')
        c = child['config']
        for name, digest in c['pins'].items():
            if sha(dataset/name) != digest:
                raise ValueError('Child dataset changed')
        origin = Path(child['selected']['prepared'])
        pilot = verify_study(origin, verify_pilot)
        if pilot != json.loads((dataset/'pilot_verified.json').read_text()):
            raise ValueError('Selected pilot proof changed')
        trial = child['selected']['trial']
        initial = next(m for m in c['models'] if m['id'] == pilot['initial_selection']['selected'])
        reference = json.loads((origin/'collected/parent_reference.json').read_text())
        contexts = [(r['scenario_id'],r['episode_seed'],r['match_seed']) for r in reference['validation']['records']]
        child_root = root/child['remote_output']
        run_id = 'full_'+trial['id']+'_s64'
        required = {child['remote_output']+'/'+name for name in ('result.json',run_id+'_summary.json',run_id+'/config.json',run_id+'/metrics.jsonl',run_id+'/training_result.json',run_id+'/update_004000.pt')}
        if trial.get('warmstart_updates',0):
            required.update(child['remote_output']+'/'+run_id+'_warmstart/'+name for name in ('policy.pt','result.json','updates.json'))
        if not required <= set(receipt['files']):
            raise ValueError('Required child artifacts missing from receipt')
        child_result = json.loads((child_root/'result.json').read_text())
        if (child_result.get('status') != 'complete' or child_result.get('pilot_steps') != 0
                or child_result.get('full_steps') != 4096000
                or child_result.get('initial_selection') != pilot['initial_selection']
                or child_result.get('protocol_sha256') != c['pins']['protocol.json']):
            raise ValueError('Child result budget/initialization differs')
        run_id = 'full_'+trial['id']+'_s64'
        row = audit_run(child_root,c,initial,contexts,run_id,trial,64,4096000,
                        dataset=dataset,remote_output='/kaggle/working/'+child['remote_output'])
        if child['remote_output']+'/'+row['checkpoint'] not in receipt['files']:
            raise ValueError('Selected full weight missing from receipt')
        if child_result['full_runs'] != [row]:
            raise ValueError('Child final selection differs')
        config = json.loads((child_root/run_id/'config.json').read_text())
        if config['device'] != f'cuda:{index}':
            raise ValueError('Full children did not use their assigned devices')
        blob = torch.load(child_root/row['checkpoint'],map_location='cpu',weights_only=False)
        if (blob['state_dict']['scalar.0.weight'].shape[1] != 309
                or blob['state_dict']['conv.0.weight'].shape[1] != 20
                or not all(torch.isfinite(t).all() for t in blob['state_dict'].values())):
            raise ValueError('Invalid selected full weights')
        execution,checked=audit_training_execution(config,blob,child_root/run_id/'metrics.jsonl')
        study_variant=json.loads((dataset/'protocol.json').read_text()).get('study_variant')
        if study_variant in ('mana_decision_v1','bank_counterfactual_v1'):
            from scripts.attack.verify_tactical_starts_pilot import audit_exposure
            from scripts.attack.verify_mana_campaign import audit_economy
            from scripts.attack.evaluation_execution import HOG_CLIPPED
            training=[];validation=[]
            for line in (child_root/run_id/'metrics.jsonl').read_text().splitlines():
                metric=json.loads(line);training.extend(metric['episodes'])
                if 'validation' in metric:validation.extend(metric['validation']['records'])
            audit_exposure(config,training,validation,0.,.15)
            audit_economy(training+validation)
            if execution!=HOG_CLIPPED or any(blob['config'][k]!=config[k]
                    for k in ('tower_support_weight','spell_waste_weight','tactical_start_fraction')):
                raise ValueError('Full economy checkpoint reward/physics differs')
            if study_variant=='bank_counterfactual_v1' and trial.get('warmstart_updates'):
                warm=json.loads((child_root/(run_id+'_warmstart/result.json')).read_text())
                if warm.get('sequence_wait_fraction')!=.5:raise ValueError('Full BC sampler differs')
        models.append(dict(id=f'global{index}_'+run_id,origin=child['selected']['study'],trial=trial,
            checkpoint=str(child_root/row['checkpoint']),checkpoint_sha256=row['checkpoint_sha256'],
            family_guard=row['selected_with_family_guard'],selection_score=row['validation']['selection_score']))
        if execution is not None:
            models[-1].update(training_execution=execution,training_geometry_records_verified=checked)
    return dict(schema='verified-global-selected-full-pair-v1',kernel=prep['kernel'],
        full_steps_verified=8192000,pilot_steps_repeated=0,models=models,
        ready_for_independent_evaluation=all(m['family_guard'] for m in models),
        quality_accepted=False,test_used_for_selection=False,live_checkpoint_changed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collected',required=True); parser.add_argument('--prepared',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    result = verify(args.collected,args.prepared)
    with Path(args.output).open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
