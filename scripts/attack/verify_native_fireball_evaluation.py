"""Recompute native held-out metrics and policy/source identity from artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from scripts.attack.verify_fireball_checkpoint import sha
from scripts.defense.summarize_selected_ppo import summarize_split
from sim.eval_defense_duel import summarize,outcome,JOINT_MODE


def verify_skills(value,seed_start,count,*,king=False):
    kinds=('tower_finish','building_finish','tower_unit')
    expected={('king_finish'if king else kinds[i%3],('left','right')[(i//3)%2],seed_start+i)for i in range(count)}
    rows=value['rows'];actual=[(r['kind'],r['lane'],r['seed'])for r in rows]
    if len(actual)!=count or set(actual)!=expected:raise ValueError('Held-out exercise schedule differs')
    for row in rows:
        if type(row.get('success'))is not bool:raise ValueError('Unknown exercise outcome')
        if king:
            events=row.get('terminal_damage_events_v2',[])
            success=any(e.get('source_kind')=='spell'and e.get('source_name')=='fireball'and
                e.get('source_side')==1 and e.get('target_side')==-1 and e.get('killed')is True and
                e.get('damage',0)>0 and e.get('before_hp',0)>0 and e.get('after_hp',1)<=0 for e in events)
            if row['success']!=success:raise ValueError('King success lacks observed Fireball terminal damage')
            if success and row.get('reason')!='enemy_king_destroyed':raise ValueError('King terminal reason differs')
        elif row['success']!=(row.get('reason')=='goal_success'):raise ValueError('Skill outcome differs from actual impact')
        if row.get('reason')not in ('goal_success','goal_window_missed','own_king_destroyed','enemy_king_destroyed','technical_timeout'):
            raise ValueError('Nonterminal exercise')
    keys=('king_finish',)if king else kinds
    computed={k:dict(episodes=sum(r['kind']==k for r in rows),successes=sum(r['success']for r in rows if r['kind']==k))for k in keys}
    if value['episodes']!=count or value['successes']!=sum(r['success']for r in rows)or value['by_kind']!=computed:
        raise ValueError('Exercise aggregate differs')
    return computed


def verify(collected,prepared):
    root=Path(collected);prepared=Path(prepared)
    prep=json.loads((prepared/'preparation.json').read_text());config=prep['config']
    receipt=json.loads((root/'collection_receipt.json').read_text())
    if receipt.get('remote_status')!='COMPLETE'or receipt.get('complete')is not True or receipt.get('kernel')!=prep['kernel']:
        raise ValueError('Complete terminal collection required')
    for name,digest in config['pins'].items():
        if sha(prepared/'dataset'/name)!=digest:raise ValueError('Prepared evaluation input changed')
    if sha(prepared/'job/evaluate.py')!=prep['bootstrap_sha256']:raise ValueError('Evaluation bootstrap changed')
    def checked(name):
        if receipt.get('files',{}).get(name,{}).get('sha256')!=sha(root/name):raise ValueError('Collected artifact SHA differs')
        return json.loads((root/name).read_text())
    with tarfile.open(prepared/'dataset/evaluation.tar.gz')as stream:
        members=stream.getmembers()
        if len(members)!=len(config['extensions'])or {m.name for m in members}!=set(config['extensions'])or any(not m.isfile()for m in members):
            raise ValueError('Evaluation extensions differ')
        extensions={m.name:stream.extractfile(m).read()for m in members}
    if any(hashlib.sha256(raw).hexdigest()!=config['extensions'][name]for name,raw in extensions.items()):
        raise ValueError('Evaluation extension digest differs')
    protocol=json.loads(extensions['data/fireball/native_evaluation_protocol.json'])
    strategy=json.loads(extensions[protocol['strategy_protocol']])
    if hashlib.sha256(extensions[protocol['strategy_protocol']]).hexdigest()!=protocol['strategy_protocol_sha256']:
        raise ValueError('Strategy protocol SHA differs')
    if (sha(prepared/'dataset/source.tar.gz')!=protocol['training_source_sha256']or
            sha(prepared/'dataset/manifest.json')!=protocol['training_manifest_sha256']or
            json.loads((prepared/'dataset/manifest.json').read_text())['files']!=protocol['training_files']):
        raise ValueError('Frozen source identity differs')
    result=checked('result.json');role=config['role']
    if (result.get('schema')!='native-fireball-heldout-result-v1'or result.get('status')!='complete'or
            result.get('role')!=role or result.get('checkpoint_sha256')!=config['pins']['policy.pt']or
            result.get('selection_sha256')!=config['pins']['selection.json']or
            result.get('protocol_sha256')!=config['extensions']['data/fireball/native_evaluation_protocol.json']or
            result.get('training_source_sha256')!=protocol['training_source_sha256']or
            result.get('training_performed')is not False or result.get('checkpoint_selection_performed')is not False or
            result.get('king_finish_test_complete')is not True):raise ValueError('Evaluation identity differs')
    selection=json.loads((prepared/'dataset/selection.json').read_text())
    expected_sha=selection['checkpoint_sha256']if role=='candidate'else protocol['parent_sha256']
    if result['checkpoint_sha256']!=expected_sha:raise ValueError('Frozen policy role differs')
    names=[s['id']+'.json'for s in protocol['fireball_test_strata']]+['king_finish.json','strategy_test.json','duels.json']
    parts={name:checked(name)for name in names}
    if set(result['parts'])!=set(names):raise ValueError('Evaluation parts differ')
    for name in names:
        if result['parts'][name].get('sha256')!=sha(root/name):raise ValueError('Internal output digest differs')
    skills={s['id']:verify_skills(parts[s['id']+'.json'],s['seed_start'],s['episodes_per_policy'])for s in protocol['fireball_test_strata']}
    s=protocol['king_finish_test'];skills['king_finish']=verify_skills(parts['king_finish.json'],s['seed_start'],s['episodes_per_policy'],king=True)
    test=parts['strategy_test.json'];records=test['records']
    expected={(sid,seed)for sid in strategy['scenario_ids']['test']for seed in strategy['seeds']['test']}
    actual=[(r.get('strategy_scenario_id'),r.get('seed'))for r in records]
    if len(actual)!=protocol['strategy_test_episodes_per_policy']or set(actual)!=expected:raise ValueError('Strategy contexts differ')
    if test.get('terminal_attribution_enabled')is not True or test.get('checkpoint_selection_performed')is not False:
        raise ValueError('Strategy attribution or selection contract differs')
    if any(r.get('split')!='test'or r.get('geometry')!=protocol['environment']['geometry']or
            r.get('coordinate_mode')!=protocol['environment']['coordinate_mode']for r in records):raise ValueError('Strategy environment differs')
    groups=parts['duels.json']['groups']
    opponents={o['id']:o for o in protocol['opponents']if role=='candidate'or o['sha256']!=protocol['parent_sha256']}
    if len(groups)!=len(opponents)or {g['id']for g in groups}!=set(opponents):raise ValueError('Opponent pool differs')
    all_rows=[];summaries={}
    for group in groups:
        opponent=opponents[group['id']];rows=group['matches']
        schedule={(seed,seat)for seed in protocol['game_seeds']for seat in protocol['candidate_seats']}
        if len(rows)!=len(schedule)or {(r['seed'],r['candidate_side'])for r in rows}!=schedule:raise ValueError('Game schedule differs')
        for row in rows:
            if (row.get('baseline_sha256')!=opponent['sha256']or row.get('baseline_id')!=group['id']or
                    row.get('baseline_mode')!=opponent['mode']or row.get('candidate_mode')!=JOINT_MODE):raise ValueError('Game policy identity differs')
            expected_outcome=outcome(row.get('result'),row['candidate_side'],row.get('result')in ('bottom','top','draw'))
            if row.get('outcome')!=expected_outcome:raise ValueError('Game outcome differs')
        summaries[group['id']]=summarize(rows)
        if group['summary']!=summaries[group['id']]or group['sha256']!=opponent['sha256']:raise ValueError('Game group aggregate differs')
        all_rows.extend(rows)
    if parts['duels.json']['summary']!=summarize(all_rows):raise ValueError('Game total aggregate differs')
    raw_rows=[json.loads(line)for line in (root/'duels_raw.jsonl').read_text().splitlines()]
    if receipt['files']['duels_raw.jsonl']['sha256']!=sha(root/'duels_raw.jsonl')or raw_rows!=all_rows:raise ValueError('Raw game ledger differs')
    return dict(schema='verified-native-fireball-heldout-v1',role=role,checkpoint_sha256=expected_sha,
        source_sha256=protocol['training_source_sha256'],skills=skills,strategy=summarize_split(records),
        games=dict(overall=summarize(all_rows),by_opponent=summaries),quality_accepted=False,
        physics_live_parity_accepted=False,live_checkpoint_changed=False,test_used_for_selection=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('collected','prepared','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.output.write_text(json.dumps(verify(args.collected,args.prepared),indent=2)+'\n')
