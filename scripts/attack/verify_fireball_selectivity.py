"""Audit the new selectivity phase and optionally replay its validation controls.

Replaying selected negative controls checks actual weights, not just reported
cast counts. It is validation verification, never an independent test result.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from scripts.attack.verify_fireball_checkpoint import sha
from scripts.attack.train_fireball_refinement import contexts,KINDS
from scripts.attack.train_fireball_selectivity import PARENT_SHA,INIT_SHA,METHOD
from sim.fireball_decision_pairs import CONTRACT


def positive_score(report):
    expected=set(contexts('validation',96));rows=report['rows']
    actual=[(r.get('kind'),r.get('lane'),r.get('seed'))for r in rows]
    if len(actual)!=96 or set(actual)!=expected:raise ValueError('Positive validation contexts differ')
    if any(type(r.get('success'))is not bool or r['success']!=(r.get('reason')=='goal_success')for r in rows):
        raise ValueError('Positive validation outcome differs')
    if any(r.get('reason')not in ('goal_success','goal_window_missed','technical_timeout',
                'own_king_destroyed','enemy_king_destroyed')for r in rows):raise ValueError('Nonterminal validation outcome')
    successes=sum(r['success']for r in rows)
    if report['episodes']!=96 or report['successes']!=successes:raise ValueError('Positive aggregate differs')
    for kind in KINDS:
        group=[r for r in rows if r['kind']==kind]
        if report['by_kind'][kind]!=dict(episodes=len(group),successes=sum(r['success']for r in group)):
            raise ValueError('Positive group aggregate differs')
    return successes/96


def negative_rate(report):
    if report.get('episodes')!=96:raise ValueError('Negative validation size differs')
    rate=report.get('first_action_fireball_rate');mass=report.get('mean_fireball_probability')
    if any(isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)or not 0<=v<=1 for v in (rate,mass)):
        raise ValueError('Invalid negative validation metric')
    if abs(rate-round(rate*96)/96)>1e-7:raise ValueError('Negative rate does not represent an integer count')
    return rate


def select_epoch(rows,parent_negative):
    if [r.get('epoch')for r in rows]!=list(range(1,41)):raise ValueError('Incomplete epoch history')
    best=None;score=-float('inf')
    parent_rate=negative_rate(parent_negative)
    for row in rows:
        if not math.isfinite(row.get('loss',float('nan'))):raise ValueError('Nonfinite epoch loss')
        if row.get('percent')!=row['epoch']*2.5:raise ValueError('Epoch progress differs')
        if row['epoch']%5:
            if 'validation'in row:raise ValueError('Unexpected validation boundary')
            continue
        current=positive_score(row['validation'])-.25*negative_rate(row['negative_validation'])
        agreement=row.get('rehearsal_action_agreement')
        if isinstance(agreement,bool)or not isinstance(agreement,(int,float))or not math.isfinite(agreement)or not 0<=agreement<=1:
            raise ValueError('Invalid rehearsal agreement')
        eligible=agreement>=.90 and negative_rate(row['negative_validation'])<=parent_rate+.05
        reported=row.get('selection_score')
        if (isinstance(reported,bool)or not isinstance(reported,(int,float))or not math.isfinite(reported)or
                row.get('eligible')is not eligible or abs(current-reported)>1e-12):
            raise ValueError('Selection score or guard differs')
        if eligible and current>score:best=row;score=current
    return best


def verify(collected,prepared,replay=True):
    root=Path(collected);prepared=Path(prepared)
    prep=json.loads((prepared/'preparation.json').read_text())
    receipt=json.loads((root/'collection_receipt.json').read_text())
    if receipt.get('remote_status')!='COMPLETE'or receipt.get('complete')is not True or receipt.get('kernel')!=prep['kernel']:
        raise ValueError('Complete terminal selectivity collection required')
    def checked(name):
        path=root/name
        if receipt.get('files',{}).get(name,{}).get('sha256')!=sha(path):raise ValueError('Collected artifact SHA differs')
        return path
    for name,digest in prep['pins'].items():
        if sha(prepared/'dataset'/name)!=digest:raise ValueError('Prepared input differs')
    if (prep['init_sha256']!=INIT_SHA or prep['parent_sha256']!=PARENT_SHA or
            sha(prepared/'job/train_selectivity.py')!=prep['bootstrap_sha256']):raise ValueError('Training lineage differs')
    manifest=json.loads((prepared/'dataset/manifest.json').read_text())
    if sha(prepared/'dataset/source.tar.gz')!=manifest['archive_sha256']or manifest['archive_sha256']!=prep['source_sha256']:
        raise ValueError('Source archive differs')
    config=json.loads(checked('config.json').read_text())
    expected=dict(method=METHOD,parent_sha256=PARENT_SHA,initial_checkpoint_sha256=INIT_SHA,num_scalars=309,
        observation_contract='joint_attack_defense_timer_task_v1',control_contract=CONTRACT,epochs=40,lr=3e-5,
        positive_weight=1.,negative_weight=1.,retention_weight=2.,demonstrations=3072,negative_pairs=1024,rehearsal=1536,
        validation_positive=96,validation_negative=96,min_rehearsal_agreement=.90,negative_cast_margin=.05,
        selection_negative_penalty=.25,test_used_for_selection=False,protocol_fixed_before_first_candidate_test_results=True)
    if any(config.get(k)!=v for k,v in expected.items()):raise ValueError('Fixed training protocol differs')
    controls=json.loads(checked('control_contexts.json').read_text())
    for split,count,offset in (('train',1024,860000),('validation',96,960000)):
        rows=controls[split]
        if len(rows)!=count or [r.get('seed')for r in rows]!=list(range(offset,offset+count)):
            raise ValueError('Control context schedule differs')
        for i,r in enumerate(rows):
            if (r.get('contract')!=CONTRACT or r.get('split')!=split or r.get('lane')!=('left','right')[i%2]or
                    r.get('positive_reference_impact_verified')is not True or
                    not 0<r['positive_hp']<=r['fireball_crown_damage']<r['negative_hp']):
                raise ValueError('Control pair semantics differ')
    audit=json.loads(checked('demonstration_audit.json').read_text())
    if sum(audit['accepted'].values())+len(audit['rejected'])!=3072:raise ValueError('Positive training accounting differs')
    baseline=json.loads(checked('baseline_validation.json').read_text())
    positive_score(baseline['initial_positive']);negative_rate(baseline['initial_negative'])
    rows=[json.loads(line)for line in checked('metrics.jsonl').read_text().splitlines()]
    chosen=select_epoch(rows,baseline['parent_negative'])
    result=json.loads(checked('result.json').read_text())
    if result.get('status')!='complete'or result.get('candidate_available')is not (chosen is not None):
        raise ValueError('Completed result disagrees with selection')
    report=dict(schema='verified-fireball-selectivity-v1',source_sha256=prep['source_sha256'],
        parent_sha256=PARENT_SHA,initial_checkpoint_sha256=INIT_SHA,candidate_available=chosen is not None,
        baseline_validation=baseline,quality_accepted=False,independent_test_pending=chosen is not None,
        live_checkpoint_changed=False)
    if chosen is None:
        if result.get('best_epoch')is not None or result.get('best_score')is not None:raise ValueError('Unexpected selected candidate')
        report['reason']='No epoch met the predeclared validation/rehearsal guards'
        return report
    if result.get('best_epoch')!=chosen['epoch']or result.get('best_score')!=chosen['selection_score']:
        raise ValueError('Selected epoch differs')
    path=checked('candidate.pt');blob=torch.load(path,map_location='cpu',weights_only=False)
    if (blob.get('config')!=config or blob.get('epoch')!=chosen['epoch']or blob.get('parent_sha256')!=PARENT_SHA or
            blob.get('initial_checkpoint_sha256')!=INIT_SHA or blob.get('validation')!=chosen['validation']or
            blob.get('negative_validation')!=chosen['negative_validation']or
            blob.get('rehearsal_action_agreement')!=chosen['rehearsal_action_agreement']or
            blob.get('selection_score')!=chosen['selection_score']):raise ValueError('Selected weight boundary differs')
    if (blob['state_dict']['scalar.0.weight'].shape[1]!=309 or
            not all(torch.isfinite(t).all()for t in blob['state_dict'].values())):raise ValueError('Invalid candidate weights')
    report.update(checkpoint_sha256=sha(path),epoch=chosen['epoch'],validation=chosen['validation']['by_kind'],
        negative_validation=chosen['negative_validation'],rehearsal_action_agreement=chosen['rehearsal_action_agreement'],
        negative_weight_replay_verified=False)
    if replay:
        # Only replay the precise checked source. A changed workspace requires
        # an isolated extraction, not silently using different mechanics.
        for name,digest in manifest['files'].items():
            if sha(name)!=digest:
                raise ValueError('Validation replay source differs: '+name)
        from scripts.attack.train_fireball_selectivity import negative_pairs,negative_metric
        from sim.train_attack_ppo import initialize_attack_policy
        from sim.train_defense_ppo import tensor_observations
        torch.set_num_threads(2)
        obs,mask,fireball,_=negative_pairs(96,'validation')
        planes,scalars=tensor_observations(obs,'cpu');mask=torch.as_tensor(mask);fireball=torch.as_tensor(fireball)
        for checkpoint,expected_metric in ((path,chosen['negative_validation']),
                (prepared/'dataset/parent.pt',baseline['parent_negative'])):
            net,_,_=initialize_attack_policy(torch.load(checkpoint,map_location='cpu',weights_only=False),device='cpu',num_scalars=309)
            net.eval().requires_grad_(False)
            measured=negative_metric(net,planes,scalars,mask,fireball)
            if abs(measured['first_action_fireball_rate']-expected_metric['first_action_fireball_rate'])>1e-7 or abs(measured['mean_fireball_probability']-expected_metric['mean_fireball_probability'])>1e-5:
                raise ValueError('Actual selected/parent negative validation differs')
        report['negative_weight_replay_verified']=True
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collected','prepared','output'):p.add_argument('--'+name,required=True,type=Path)
    args=p.parse_args();args.output.write_text(json.dumps(verify(args.collected,args.prepared),indent=2)+'\n')
