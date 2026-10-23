import hashlib
import json
from pathlib import Path

import pytest

from scripts.defense.summarize_five_models import summarize
from scripts.kaggle.run_five_round_robin import schedule, standings
from sim.defense_protocol import V25_SHA256


def fixture(tmp_path):
    root=tmp_path/'collected';root.mkdir()
    names=['ppo1','grpo2','joint_ppo3','joint_grpo4','v25']
    roster=[dict(id=n,path=n+'.pt',sha256=V25_SHA256 if n=='v25'else str(i)*64,
                 mode='joint_enemy_presence_v1'if n.startswith('joint')else'legacy_training'if n=='v25'else'physical_live')for i,n in enumerate(names)]
    participants=tmp_path/'participants.json';participants.write_text(json.dumps(roster))
    ids=['defense::d'+str(i)for i in range(236)]+['attack::a'+str(i)for i in range(32)]
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps(dict(scenario_ids=dict(test=ids),seeds=dict(test=[810001,810002,810003]))))
    ph=hashlib.sha256(protocol.read_bytes()).hexdigest()
    source=tmp_path/'manifest.json';source.write_text(json.dumps(dict(archive_sha256='source',files={'sim/match.py':'engine','data/strategy/evaluation_protocol.json':ph})))
    matches=[dict(**r,ordinal=i,result='bottom'if r['candidate_side']==1 else'top',outcome='win')for i,r in enumerate(schedule(names))]
    table=standings(matches,names)
    result=dict(status='complete',total=200,source_sha256='source',engine_sha256='engine',participants=roster,matches=matches,**table)
    (root/'result.json').write_text(json.dumps(result))
    for participant in roster:
        joint=participant['id'].startswith('joint');records=[]
        for sid in ids:
            task=sid.split('::')[0]
            for seed in (810001,810002,810003):
                record=dict(strategy_scenario_id=sid,seed=seed,split='test',task=task,geometry='legacy_v1',
                    coordinate_mode='physical_live_cells_v2'if joint else'physical_live')
                if task=='defense':record.update(no_tower_damage=True,own_tower_hp_loss=dict(left=0,right=0,king=0),difficulty='simple')
                else:record.update(goal_kind='spell_finish',attack_success=True,goal_success=False,successful_spell_finish=None,
                                   goal_window_missed=False,terminal_reason='enemy_king_destroyed')
                records.append(record)
        data=dict(participant=participant['id'],checkpoint_sha256=participant['sha256'],source_sha256='source',protocol_sha256=ph,
                  input_mode='joint_309_cells_v2'if joint else'physical_live_306_timer_task_ignored',records=records)
        (root/(participant['id']+'_common_skill_test.json')).write_text(json.dumps(data))
    def receipt():
        (root/'receipt.json').write_text(json.dumps(dict(status='COMPLETE',files={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in root.glob('*.json')if p.name!='receipt.json'})))
    receipt()
    return root,participants,source,protocol,receipt


def test_complete_schedule_and_unknown_spell_attribution_remain_separate(tmp_path):
    root,participants,source,protocol,_=fixture(tmp_path)
    report=summarize(root,participants,source,protocol)
    assert report['tournament_matches']==200 and report['skill_episodes']==4020
    assert all(m['tournament']['matches']==80 for m in report['models'].values())
    attack=report['models']['ppo1']['attack']
    assert attack['episode_success']['rate']==1
    assert attack['spell_finish_confirmed']['unknown']==96
    assert attack['spell_finish_confirmed']['rate']is None
    assert not report['quality_accepted'] and not report['ranking_provisional']


@pytest.mark.parametrize('fault',['duplicate_match','missing_context','wrong_weight','wrong_mode'])
def test_internally_hashed_but_semantically_invalid_reports_fail(tmp_path,fault):
    root,participants,source,protocol,receipt=fixture(tmp_path)
    path=root/('result.json'if fault in ('duplicate_match','wrong_weight')else'ppo1_common_skill_test.json')
    data=json.loads(path.read_text())
    if fault=='duplicate_match':data['matches'][1]=data['matches'][0]
    elif fault=='wrong_weight':data['participants'][0]['sha256']='f'*64
    elif fault=='missing_context':data['records'][-1]=data['records'][0]
    else:data['records'][0]['coordinate_mode']='physical_live_cells_v2'
    path.write_text(json.dumps(data));receipt()
    with pytest.raises(ValueError):summarize(root,participants,source,protocol)


def test_unknown_game_makes_ranking_provisional_and_is_not_a_draw(tmp_path):
    root,participants,source,protocol,receipt=fixture(tmp_path)
    path=root/'result.json';data=json.loads(path.read_text());data['matches'][0].update(outcome='unknown',result=None)
    names=[p['id']for p in data['participants']];data.update(standings(data['matches'],names));path.write_text(json.dumps(data));receipt()
    report=summarize(root,participants,source,protocol)
    assert report['ranking_provisional']
    assert report['models']['ppo1']['tournament']['unknown']==1
    assert report['models']['ppo1']['tournament']['draws']==0
