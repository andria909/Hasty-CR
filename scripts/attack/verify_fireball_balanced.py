"""Verify v3's source, per-skill floors, shortlist and held-out validation games.

This audits validation selection only. A candidate passing these guards still
needs independent tests and cannot claim live/simulator physics acceptance.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import tarfile

import torch

from scripts.attack.verify_fireball_checkpoint import sha
from scripts.attack.verify_fireball_selectivity import positive_score,negative_rate
from scripts.attack.train_fireball_balanced import match_score
from sim.fireball_balanced import skill_eligibility
from sim.eval_defense_duel import summarize,outcome,JOINT_MODE


def verify_games(report,protocol,opponent_mode):
    rows=report['matches']
    expected={(seed,side)for seed in protocol['validation_match_seeds']for side in (1,-1)}
    actual=[(r.get('seed'),r.get('candidate_side'))for r in rows]
    if len(actual)!=len(expected)or set(actual)!=expected:raise ValueError('Validation game schedule differs')
    for r in rows:
        if r.get('candidate_mode')!=JOINT_MODE or r.get('baseline_mode')!=opponent_mode:
            raise ValueError('Validation game observation modes differ')
        observed=outcome(r.get('result'),r['candidate_side'],r.get('result')in ('bottom','top','draw'))
        if r.get('outcome')!=observed:raise ValueError('Validation game outcome differs')
    if report['summary']!=summarize(rows):raise ValueError('Validation game aggregate differs')
    return match_score(report['summary'])


def skill_shortlist(rows,teacher_negative,protocol):
    if [r.get('epoch')for r in rows]!=list(range(1,protocol['epochs']+1)):
        raise ValueError('Incomplete training epoch history')
    negative_rate(teacher_negative);eligible=[]
    for row in rows:
        if not math.isfinite(row.get('loss',float('nan'))):raise ValueError('Nonfinite training loss')
        if row.get('percent')!=100*row['epoch']/protocol['epochs']:raise ValueError('Epoch percentage differs')
        if row['epoch']%protocol['validation_every']:
            if 'validation'in row:raise ValueError('Unexpected validation epoch')
            continue
        positive_score(row['validation']);negative_rate(row['negative_validation'])
        agreement=row.get('rehearsal_action_agreement')
        if isinstance(agreement,bool)or not isinstance(agreement,(int,float))or not math.isfinite(agreement)or not 0<=agreement<=1:
            raise ValueError('Invalid rehearsal agreement')
        passed=skill_eligibility(row['validation'],row['negative_validation'],agreement,teacher_negative,protocol)
        rates=[v['successes']/v['episodes']for v in row['validation']['by_kind'].values()]
        score=sum(rates)/len(rates)-.25*row['negative_validation']['first_action_fireball_rate']
        if (row.get('eligible')is not passed or row.get('minimum_skill_rate')!=min(rates)or
                row.get('skill_score')!=score):raise ValueError('Skill guards or ranking differ')
        if passed:eligible.append(row)
    return eligible,sorted(eligible,key=lambda r:(-r['minimum_skill_rate'],-r['skill_score'],r['epoch']))[:protocol['shortlist_size']]


def choose_games(games,shortlist,teacher_reference,protocol,v25_mode):
    if [r['epoch']for r in games]!=[r['epoch']for r in shortlist]:raise ValueError('Game-tested shortlist differs')
    baseline_score=verify_games(teacher_reference,protocol,v25_mode)
    passing=[]
    for row,expected in zip(games,shortlist):
        if any(row.get(k)!=expected[k]for k in ('epoch','path','sha256','minimum_skill_rate','skill_score')):
            raise ValueError('Game-tested weight identity differs')
        parent=verify_games(row['parent'],protocol,JOINT_MODE)
        v25=verify_games(row['v25'],protocol,v25_mode)
        passed=(parent is not None and v25 is not None and baseline_score is not None and
            parent>=protocol['minimum_score_vs_teacher']and v25>=baseline_score-protocol['score_margin_vs_v25_teacher_reference'])
        score=(parent+v25)/2 if parent is not None and v25 is not None else None
        if row.get('match_guards_passed')is not passed or row.get('game_score')!=score:
            raise ValueError('Match guards or ranking differ')
        if passed:passing.append(row)
    return sorted(passing,key=lambda r:(-r['game_score'],-r['minimum_skill_rate'],r['epoch']))[0]if passing else None


def verify(collected,prepared):
    root=Path(collected);prepared=Path(prepared)
    prep=json.loads((prepared/'preparation.json').read_text())
    receipt=json.loads((root/'collection_receipt.json').read_text())
    if receipt.get('remote_status')!='COMPLETE'or receipt.get('complete')is not True or receipt.get('kernel')!=prep['kernel']:
        raise ValueError('Complete terminal v3 collection required')
    def checked(name):
        path=root/name
        if receipt.get('files',{}).get(name,{}).get('sha256')!=sha(path):raise ValueError('Collected artifact SHA differs')
        return path
    for name,digest in prep['pins'].items():
        if sha(prepared/'dataset'/name)!=digest:raise ValueError('Prepared input SHA differs')
    if sha(prepared/'job/train_balanced.py')!=prep['bootstrap_sha256']:raise ValueError('Prepared bootstrap differs')
    manifest=json.loads((prepared/'dataset/manifest.json').read_text())
    archive=prepared/'dataset/source.tar.gz'
    if sha(archive)!=manifest['archive_sha256']or sha(archive)!=prep['source_sha256']:raise ValueError('Source archive differs')
    members={}
    with tarfile.open(archive)as stream:
        items=stream.getmembers()
        if len(items)!=len(manifest['files'])or {m.name for m in items}!=set(manifest['files']):raise ValueError('Source members differ')
        for member in items:
            path=PurePosixPath(member.name)
            if not member.isfile()or path.is_absolute()or '..'in path.parts:raise ValueError('Unsafe source member')
            raw=stream.extractfile(member).read()
            if hashlib.sha256(raw).hexdigest()!=manifest['files'][member.name]:raise ValueError('Source file SHA differs')
            members[member.name]=raw
    protocol=json.loads(members['data/fireball/balanced_protocol.json'])
    if (hashlib.sha256(members['data/fireball/balanced_protocol.json']).hexdigest()!=prep['protocol_sha256']or
            protocol!=prep['protocol']or manifest.get('migration')!=prep['migration']):raise ValueError('Frozen protocol/migration differs')
    if (protocol.get('test_used_for_selection')is not False or set(protocol['validation_match_seeds'])&set(range(710001,710011))or
            len(set(protocol['validation_match_seeds']))!=10 or protocol['epochs']!=80):raise ValueError('Validation/test split contract differs')
    config=json.loads(checked('config.json').read_text())
    if (config.get('protocol')!=protocol or config.get('protocol_sha256')!=prep['protocol_sha256']or
            config.get('source_sha256')!=prep['source_sha256']or config.get('method')!=protocol['method']or
            config.get('parent_sha256')!=prep['init_sha256']or config.get('num_scalars')!=309 or
            config.get('physics_input_migration')is not True or config.get('resume_old_optimizer')is not False):
        raise ValueError('New-phase configuration differs')
    audit=json.loads(checked('demonstration_audit.json').read_text())
    if sum(audit['accepted'].values())+len(audit['rejected'])!=protocol['demonstrations']:
        raise ValueError('Demonstration accounting differs')
    pairs=json.loads(checked('pair_contexts.json').read_text())
    for split,count,offset in (('train',protocol['pairs'],870000),('validation',protocol['validation_negative'],970000)):
        rows=pairs[split]
        if len(rows)!=count or [r['seed']for r in rows]!=list(range(offset,offset+count)):
            raise ValueError('Pair context schedule differs')
        for i,row in enumerate(rows):
            if (row.get('split')!=split or row.get('lane')!=('left','right')[i%2]or row.get('positive_reference_impact_verified')is not True or
                    not 0<row['positive_hp']<=row['fireball_crown_damage']<row['negative_hp']):raise ValueError('Pair semantics differ')
    replay=json.loads(checked('rehearsal_contexts.json').read_text())
    identities=[(r['seed'],r['side'],r['elapsed_ms'])for r in replay]
    if (len(replay)!=protocol['rehearsal_states']or len(set(identities))!=len(identities)or
            any(r['seed']<protocol['rehearsal_match_seed_start']or r['seed']in protocol['validation_match_seeds']or
                r['seed']in range(710001,710011)or r['side']not in (1,-1)for r in replay)):
        raise ValueError('Rehearsal context contract differs')
    baseline=json.loads(checked('baseline_skills.json').read_text())
    positive_score(baseline['positive']);negative_rate(baseline['negative'])
    rows=[json.loads(line)for line in checked('metrics.jsonl').read_text().splitlines()]
    eligible,shortlist=skill_shortlist(rows,baseline['negative'],protocol)
    listed=json.loads(checked('shortlist.json').read_text())
    for name,expected in (('eligible',eligible),('shortlist',shortlist)):
        if [r['epoch']for r in listed[name]]!=[r['epoch']for r in expected]:raise ValueError('Declared skill shortlist differs')
        for item,row in zip(listed[name],expected):
            if item['path']!=f"epoch_{row['epoch']}.pt"or any(item[k]!=row[k]for k in ('minimum_skill_rate','skill_score')):
                raise ValueError('Declared shortlist identity differs')
    games=json.loads(checked('validation_games.json').read_text());chosen=None
    if shortlist:
        reference=json.loads(checked('teacher_validation_games.json').read_text())
        v25_mode=next(r['observation_mode']for r in json.loads(members['data/defense/baselines.json'])['checkpoints']if r['id']=='v25')
        chosen=choose_games(games,listed['shortlist'],reference,protocol,v25_mode)
    elif games:raise ValueError('Games present without eligible skills')
    result=json.loads(checked('result.json').read_text())
    if (result.get('status')!='complete'or result.get('candidate_available')is not (chosen is not None)or
            result.get('source_sha256')!=prep['source_sha256']or result.get('protocol_sha256')!=prep['protocol_sha256']or
            result.get('eligible_skill_epochs')!=len(eligible)or result.get('validation_match_candidates')!=len(shortlist)):
        raise ValueError('Final phase result differs')
    report=dict(schema='verified-fireball-balanced-v1',source_sha256=prep['source_sha256'],protocol_sha256=prep['protocol_sha256'],
        parent_sha256=prep['init_sha256'],candidate_available=chosen is not None,baseline_skills=baseline,
        eligible_skill_epochs=len(eligible),validation_match_candidates=len(shortlist),quality_accepted=False,
        independent_test_required=chosen is not None,live_checkpoint_changed=False)
    if chosen is None:
        if result.get('selected_epoch')is not None or result.get('candidate_sha256')is not None:raise ValueError('Unexpected selected candidate')
        report['reason']='No epoch passed all skill guards'if not shortlist else'No shortlisted epoch passed validation game guards'
        return report
    candidate=checked('candidate.pt')
    if sha(candidate)!=chosen['sha256']or sha(candidate)!=result.get('candidate_sha256')or result.get('selected_epoch')!=chosen['epoch']:
        raise ValueError('Selected weight hash or epoch differs')
    blob=torch.load(candidate,map_location='cpu',weights_only=False)
    selected=next(r for r in rows if r['epoch']==chosen['epoch'])
    if (blob.get('config')!=config or blob.get('epoch')!=chosen['epoch']or
            any(blob.get(k)!=selected[k]for k in ('validation','negative_validation','rehearsal_action_agreement'))or
            blob['state_dict']['scalar.0.weight'].shape[1]!=309 or not all(torch.isfinite(t).all()for t in blob['state_dict'].values())):
        raise ValueError('Selected model boundary/weights differ')
    report.update(checkpoint_sha256=sha(candidate),epoch=chosen['epoch'],validation=selected['validation']['by_kind'],
        negative_validation=selected['negative_validation'],rehearsal_action_agreement=selected['rehearsal_action_agreement'],
        selected_validation_games=dict(parent=chosen['parent']['summary'],v25=chosen['v25']['summary']),
        teacher_validation_games=reference['summary'])
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collected','prepared','output'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();a.output.write_text(json.dumps(verify(a.collected,a.prepared),indent=2)+'\n')
