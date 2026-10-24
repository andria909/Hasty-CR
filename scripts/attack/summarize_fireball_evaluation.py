"""Verify complete paired Fireball and regression evidence, preserving limits."""
import argparse
import json
from pathlib import Path

from scripts.attack.verify_fireball_checkpoint import sha
from scripts.attack.train_fireball_refinement import contexts,KINDS,PARENT_SHA
from scripts.defense.pack_evaluation import verified_frozen_package
from scripts.defense.summarize_selected_ppo import summarize_split,verify as verify_parent
from sim.eval_defense_duel import summarize,outcome,JOINT_MODE


def fireball_result(report,count=384):
    expected=set(contexts('test',count))
    result={}
    for label in ('before','after'):
        value=report[label];rows=value['rows']
        actual=[(r['kind'],r['lane'],r['seed'])for r in rows]
        if len(actual)!=count or len(expected)!=count or set(actual)!=expected:
            raise ValueError('Missing, duplicate or off-split Fireball context')
        if any(type(r.get('success'))is not bool for r in rows):raise ValueError('Unknown Fireball outcome')
        if any(r.get('reason')not in ('goal_success','goal_window_missed','own_king_destroyed',
                    'enemy_king_destroyed','technical_timeout')for r in rows):
            raise ValueError('Nonterminal Fireball result')
        if any(r['success']!=(r.get('reason')=='goal_success')for r in rows):
            raise ValueError('Fireball success disagrees with terminal event')
        if value['episodes']!=count or value['successes']!=sum(r['success']for r in rows):
            raise ValueError('Fireball aggregate differs')
        result[label]={}
        for kind in KINDS:
            subset=[r for r in rows if r['kind']==kind]
            summary=dict(episodes=len(subset),successes=sum(r['success']for r in subset))
            if value['by_kind'][kind]!=summary:raise ValueError('Fireball kind aggregate differs')
            result[label][kind]=dict(summary,rate=summary['successes']/summary['episodes'])
    return result


def duel_result(report,baselines):
    rows=report['matches'];expected={(b,s,seat)for b in baselines for s in range(710001,710011)for seat in (1,-1)}
    actual=[(r.get('baseline_id'),r.get('seed'),r.get('candidate_side'))for r in rows]
    if len(actual)!=len(expected)or set(actual)!=expected:raise ValueError('Missing or duplicate paired game')
    for r in rows:
        b=baselines[r['baseline_id']]
        if r.get('baseline_sha256')!=b['sha256']or r.get('baseline_mode')!=b['mode']or r.get('candidate_mode')!=JOINT_MODE:
            raise ValueError('Duel policy identity or observation mode differs')
        if r.get('outcome')!=outcome(r.get('result'),r['candidate_side'],r.get('result')in ('bottom','top','draw')):
            raise ValueError('Duel outcome disagrees with match result')
    groups=report['baselines']
    if len(groups)!=len(baselines)or {g['id']for g in groups}!=set(baselines):raise ValueError('Duel groups differ')
    summaries={}
    for g in groups:
        subset=[r for r in rows if r['baseline_id']==g['id']]
        if g['sha256']!=baselines[g['id']]['sha256']or g['matches']!=subset or g['summary']!=summarize(subset):
            raise ValueError('Duel group aggregate differs')
        summaries[g['id']]=summarize(subset)
    historical=summarize([r for r in rows if r['baseline_id']!='parent_joint_economy_soft'])
    parent=summarize([r for r in rows if r['baseline_id']=='parent_joint_economy_soft'])
    if report['historical_summary']!=historical or report['parent_summary']!=parent:
        raise ValueError('Combined duel aggregate differs')
    return dict(historical=historical,parent=parent,by_opponent=summaries)


def verify(collected,prepared,evaluation_package,parent_prepared,parent_collected):
    root=Path(collected);prepared=Path(prepared)
    prep=json.loads((prepared/'preparation.json').read_text())
    receipt=json.loads((root/'collection_receipt.json').read_text())
    if receipt.get('remote_status')!='COMPLETE'or receipt.get('complete')is not True or receipt.get('kernel')!=prep['kernel']:
        raise ValueError('Complete terminal evaluation collection required')
    def checked(name):
        path=root/name
        if receipt.get('files',{}).get(name,{}).get('sha256')!=sha(path):raise ValueError('Collected artifact SHA differs')
        return json.loads(path.read_text())
    result=checked('result.json')
    selection=prepared/'dataset/selection.json';candidate=prepared/'dataset/candidate.pt'
    if sha(selection)!=prep['pins']['selection.json']or sha(candidate)!=prep['candidate_sha256']:
        raise ValueError('Frozen selected candidate changed')
    for name,digest in prep['pins'].items():
        if sha(prepared/'dataset'/name)!=digest:raise ValueError('Prepared input SHA differs')
    if sha(prepared/'job/evaluate.py')!=prep['bootstrap_sha256']:raise ValueError('Evaluation bootstrap changed')
    selected=json.loads(selection.read_text())
    if (result.get('schema')!='fireball-independent-evaluation-v1'or result.get('status')!='complete'or
            result.get('checkpoint_sha256')!=prep['candidate_sha256']or result.get('parent_sha256')!=PARENT_SHA or
            result.get('selected_epoch')!=selected['epoch']or result.get('selection_sha256')!=sha(selection)):
        raise ValueError('Evaluation candidate identity differs')
    members,manifest,_,_=verified_frozen_package(evaluation_package)
    if prep['source_sha256']!=manifest['archive_sha256']or sha(Path(evaluation_package)/'manifest.json')!=prep['source_manifest_sha256']:
        raise ValueError('Frozen common evaluation source differs')
    protocol=json.loads(members['data/strategy/evaluation_protocol.json'])
    import hashlib
    protocol_sha=hashlib.sha256(members['data/strategy/evaluation_protocol.json']).hexdigest()
    if result.get('protocol_sha256')!=protocol_sha:raise ValueError('Common protocol differs')
    reports={name:checked(name+'.json')for name in ('fireball_test','regression_test','duels')}
    for name in reports:
        if result.get('parts',{}).get(name,{}).get('sha256')!=sha(root/(name+'.json')):
            raise ValueError('Internal part SHA differs')
    fireball=reports['fireball_test']
    if fireball.get('candidate_sha256')!=prep['candidate_sha256']or fireball.get('parent_sha256')!=PARENT_SHA or fireball.get('checkpoint_selection_performed')is not False:
        raise ValueError('Fireball paired policy identity differs')
    fireball_summary=fireball_result(fireball)
    regression=reports['regression_test'];records=regression['records']
    expected={(sid,seed)for sid in protocol['scenario_ids']['test']for seed in protocol['seeds']['test']}
    actual=[(r.get('strategy_scenario_id'),r.get('seed'))for r in records]
    if len(actual)!=804 or len(expected)!=804 or set(actual)!=expected:raise ValueError('Common test contexts differ')
    if (regression.get('checkpoint_sha256')!=prep['candidate_sha256']or regression.get('protocol_sha256')!=protocol_sha or
            regression.get('terminal_attribution_enabled')is not True or regression.get('checkpoint_selection_performed')is not False):
        raise ValueError('Common regression test identity/attribution differs')
    if any(r.get('split')!='test'or r.get('geometry')!='legacy_v1'or r.get('coordinate_mode')!='physical_live_cells_v2'
           or r.get('task')not in ('attack','defense')or not r['strategy_scenario_id'].startswith(r['task']+'::')for r in records):
        raise ValueError('Common test environment differs')
    regression_summary=summarize_split(records)
    baseline_rows=json.loads(members['data/defense/baselines.json'])['checkpoints']
    baselines={r['id']:dict(sha256=r['sha256'],mode=r['observation_mode'])for r in baseline_rows}
    baselines['parent_joint_economy_soft']=dict(sha256=PARENT_SHA,mode=JOINT_MODE)
    duels=reports['duels']
    if duels.get('candidate_sha256')!=prep['candidate_sha256']:raise ValueError('Duel candidate differs')
    matches=duel_result(duels,baselines)
    original=verify_parent(parent_collected,parent_prepared,evaluation_package)
    if original['checkpoint_sha256']!=PARENT_SHA:raise ValueError('Wrong parent regression reference')
    original_records=json.loads((Path(parent_collected)/'test.json').read_text())['records']
    by_context={(r['strategy_scenario_id'],r['seed']):r for r in original_records}
    better=worse=0
    for r in records:
        if r['task']!='defense':continue
        old=by_context[(r['strategy_scenario_id'],r['seed'])]
        better+=not old['no_tower_damage'] and r['no_tower_damage']
        worse+=old['no_tower_damage'] and not r['no_tower_damage']
    return dict(schema='verified-fireball-evaluation-summary-v1',checkpoint_sha256=prep['candidate_sha256'],parent_sha256=PARENT_SHA,
        evaluation_source_sha256=manifest['archive_sha256'],selected_epoch=selected['epoch'],
        fireball_test=fireball_summary,old_test_before=original['test'],old_test_after=regression_summary,
        paired_defense_no_damage=dict(improved=better,degraded=worse,unchanged=708-better-worse),
        matches=matches,old_opponents_before=original['opponents'],test_used_for_selection=False,
        quality_accepted=False,simulation_only=True,live_checkpoint_changed=False,
        quality_note='Descriptive fixed-checkpoint tests; simulator/live physics and all plan acceptance gates remain unverified.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collected','prepared','evaluation-package','parent-prepared','parent-collected','output'):
        p.add_argument('--'+name,required=True,type=Path)
    args=p.parse_args();output=args.output;kw=vars(args);del kw['output']
    output.write_text(json.dumps(verify(**kw),indent=2)+'\n')
