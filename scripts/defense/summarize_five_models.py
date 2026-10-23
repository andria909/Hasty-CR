"""Verify complete tournament/common-test artifacts and summarize fixed policies.

No selection of new weights, model promotion, training or live-game claims.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.kaggle.run_five_round_robin import schedule, standings
from sim.defense_protocol import V25_SHA256

IDS={'ppo1','grpo2','joint_ppo3','joint_grpo4','v25'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rate(rows,key):
    known=[r[key]for r in rows if isinstance(r.get(key),bool)]
    return dict(known=len(known),unknown=len(rows)-len(known),success=sum(known),
                rate=sum(known)/len(known)if known else None)


def summarize(collected,participants_file,source_manifest,protocol_file):
    root=Path(collected);roster=json.loads(Path(participants_file).read_text())
    names=[r['id']for r in roster]
    if set(names)!=IDS or len(names)!=5:raise ValueError('Exactly the four main candidates plus v25 required')
    if next(p for p in roster if p['id']=='v25')['sha256']!=V25_SHA256:
        raise ValueError('Canonical v25 changed')
    source=json.loads(Path(source_manifest).read_text());protocol=json.loads(Path(protocol_file).read_text())
    if sha(protocol_file)!=source['files']['data/strategy/evaluation_protocol.json']:
        raise ValueError('Protocol differs from pinned common source')
    expected={(sid,seed)for sid in protocol['scenario_ids']['test']for seed in protocol['seeds']['test']}
    if len(expected)!=804:raise ValueError('Expected full804 common skill contexts')
    receipt=json.loads((root/'receipt.json').read_text())
    if receipt.get('status')!='COMPLETE':raise ValueError('Authoritative terminal COMPLETE receipt required')
    required={'result.json',*[name+'_common_skill_test.json'for name in names]}
    for name in required:
        if name not in receipt['files']or sha(root/name)!=receipt['files'][name]['sha256']:
            raise ValueError('Missing or changed collected report')
    tournament=json.loads((root/'result.json').read_text())
    if tournament.get('status')!='complete'or tournament.get('total')!=200:
        raise ValueError('Incomplete tournament')
    if (tournament.get('source_sha256')!=source['archive_sha256']or
            tournament.get('engine_sha256')!=source['files']['sim/match.py']):
        raise ValueError('Tournament uses a different common engine')
    actual_roster=tournament['participants']
    if len(actual_roster)!=5 or {p['id']for p in actual_roster}!=IDS:
        raise ValueError('Unexpected tournament participant')
    for wanted in roster:
        found=[p for p in actual_roster if p['id']==wanted['id']]
        if len(found)!=1 or any(found[0].get(k)!=v for k,v in wanted.items()):
            raise ValueError('Tournament participant identity/contract differs')
    matches=tournament['matches'];plan=schedule(names)
    if len(matches)!=200:raise ValueError('Tournament must contain200 matches')
    for i,(actual,wanted)in enumerate(zip(matches,plan)):
        if actual.get('ordinal')!=i or any(actual.get(k)!=v for k,v in wanted.items()):
            raise ValueError('Duplicate/off-protocol paired match schedule')
        if actual['outcome']in ('win','loss','draw'):
            winner=actual.get('result')
            expected_outcome=('draw'if winner=='draw'else 'win'if winner==('bottom'if actual['candidate_side']==1 else'top')else 'loss')
            if winner not in ('bottom','top','draw')or actual['outcome']!=expected_outcome:
                raise ValueError('Seat/winner/outcome disagreement')
    table=standings(matches,names)
    if table['totals']!=tournament['totals']:raise ValueError('Reported standings disagree with actual matches')
    models={}
    for participant in roster:
        name=participant['id'];path=root/(name+'_common_skill_test.json');report=json.loads(path.read_text())
        if (report.get('participant')!=name or report.get('checkpoint_sha256')!=participant['sha256']or
                report.get('source_sha256')!=source['archive_sha256']or report.get('protocol_sha256')!=sha(protocol_file)):
            raise ValueError('Common skill test provenance mismatch')
        records=report['records'];contexts=[(r.get('strategy_scenario_id'),r.get('seed'))for r in records]
        if len(contexts)!=len(expected)or set(contexts)!=expected or any(r.get('split')!='test'for r in records):
            raise ValueError('Incomplete/duplicate/off-split skill contexts')
        joint=participant['mode']=='joint_enemy_presence_v1'
        coordinate='physical_live_cells_v2'if joint else'physical_live'
        mode='joint_309_cells_v2'if joint else'physical_live_306_timer_task_ignored'
        if report.get('input_mode')!=mode or any(r.get('coordinate_mode')!=coordinate or r.get('geometry')!='legacy_v1'for r in records):
            raise ValueError('Common skill input/geometry contract differs')
        defense=[r for r in records if r.get('task')=='defense'];attack=[r for r in records if r.get('task')=='attack']
        if len(defense)!=708 or len(attack)!=96:raise ValueError('Expected708defense and96attack cases')
        if any(not isinstance(r.get('no_tower_damage'),bool)for r in defense):raise ValueError('Missing defense labels')
        losses=[]
        for row in defense:
            values=[row.get('own_tower_hp_loss',{}).get(t)for t in ('left','right','king')]
            if any(isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)or v<0 for v in values):
                raise ValueError('Missing/nonfinite tower loss metrics')
            losses.append(sum(values))
        hog=[r for r in attack if r.get('goal_kind')=='hog_tower_damage']
        spell=[r for r in attack if r.get('goal_kind')=='spell_finish']
        models[name]=dict(checkpoint_sha256=participant['sha256'],tournament=table['totals'][name],
            defense=dict(episodes=708,no_damage=rate(defense,'no_tower_damage'),mean_hp_loss=sum(losses)/708,
                by_difficulty={difficulty:rate([r for r in defense if r.get('difficulty')==difficulty],'no_tower_damage')
                               for difficulty in sorted({r.get('difficulty')for r in defense})},
                timeouts=sum(r.get('terminal_reason')=='timeout'for r in defense)),
            attack=dict(episodes=96,episode_success=rate(attack,'attack_success'),
                reported_goal_flag=rate(attack,'goal_success'),window_missed=rate(attack,'goal_window_missed'),
                hog_episodes=len(hog),hog_confirmed_hits=sum(r.get('hog_target_tower_damage',0)>0 for r in hog),
                hog_incomplete_ledgers=sum(r.get('hog_damage_ledger_complete')is False for r in hog),
                spell_episodes=len(spell),spell_finish_confirmed=rate(spell,'successful_spell_finish'),
                enemy_king_destroyed=sum(r.get('terminal_reason')=='enemy_king_destroyed'for r in attack)),
            skill_input_mode=report.get('input_mode'),duel_input_mode=participant['mode'],raw_sha256=sha(path))
    return dict(schema='verified-five-model-summary-v1',tournament_matches=200,skill_episodes=4020,
        models=models,matrix=table['matrix'],leaders=table['leaders'],ranking_provisional=table['ranking_provisional'],
        ranking_rule=table['ranking_rule'],source_sha256=source['archive_sha256'],protocol_sha256=sha(protocol_file),
        quality_accepted=False,simulation_only=True,test_used_to_change_weights=False,
        attack_metric_note='Episode success includes enemy king death. Goal flags and confirmed Hog/spell attribution are separate; unknown terminal spell attribution is not a confirmed spell finish.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('collected','participants','source-manifest','protocol','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    report=summarize(args.collected,args.participants,args.source_manifest,args.protocol)
    with args.output.open('x')as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(leaders=report['leaders'],ranking_provisional=report['ranking_provisional'],quality_accepted=False)))


if __name__=='__main__':main()
