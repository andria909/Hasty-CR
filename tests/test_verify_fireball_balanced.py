from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.attack.train_fireball_refinement import contexts,KINDS
from scripts.attack.verify_fireball_balanced import skill_shortlist,choose_games,verify_games
from sim.eval_defense_duel import summarize,JOINT_MODE


def protocol():return json.loads(Path('data/fireball/balanced_protocol.json').read_text())


def positive():
    rows=[dict(kind=k,lane=l,seed=s,success=True,reason='goal_success')for k,l,s in contexts('validation',96)]
    return dict(rows=rows,episodes=96,successes=96,by_kind={k:dict(episodes=32,successes=32)for k in KINDS})


def negative(rate=0.):return dict(episodes=96,first_action_fireball_rate=rate,mean_fireball_probability=.1)


def history():
    rows=[]
    for epoch in range(1,81):
        r=dict(epoch=epoch,percent=epoch*1.25,loss=1.)
        if epoch%10==0:r.update(validation=positive(),negative_validation=negative(),rehearsal_action_agreement=.97,
            eligible=True,minimum_skill_rate=1.,skill_score=1.)
        rows.append(r)
    return rows


def game_report(wins,mode):
    rows=[]
    for i,(seed,side)in enumerate((s,c)for s in protocol()['validation_match_seeds']for c in (1,-1)):
        winner=side if i<wins else -side
        rows.append(dict(seed=seed,candidate_side=side,candidate_mode=JOINT_MODE,baseline_mode=mode,
            outcome='win'if i<wins else'loss',result='bottom'if winner==1 else'top',quality_flags=[]))
    return dict(matches=rows,summary=summarize(rows))


def test_skill_shortlist_is_earliest_three_when_all_scores_tie():
    eligible,short=skill_shortlist(history(),negative(.25),protocol())
    assert len(eligible)==8 and [r['epoch']for r in short]==[10,20,30]


def test_macro_success_cannot_hide_collapsed_tower_finishing():
    rows=history()
    for row in rows:
        if 'validation'not in row:continue
        p=row['validation'];p['successes']=64;p['by_kind']['tower_finish']['successes']=0
        for item in p['rows']:
            if item['kind']=='tower_finish':item.update(success=False,reason='goal_window_missed')
        row.update(eligible=False,minimum_skill_rate=0.,skill_score=2/3)
    assert skill_shortlist(rows,negative(.25),protocol())==([],[])


def test_validation_game_winner_is_selected_only_after_both_guards():
    p=protocol();shortlist=[dict(epoch=e,path=f'epoch_{e}.pt',sha256=str(e),minimum_skill_rate=1.,skill_score=1.)for e in (10,20)]
    games=[]
    for item,parent,v25 in zip(shortlist,(10,9),(18,20)):
        games.append(dict(item,parent=game_report(parent,JOINT_MODE),v25=game_report(v25,'physical_live'),
            match_guards_passed=True,game_score=(parent/20+v25/20)/2))
    chosen=choose_games(games,shortlist,game_report(18,'physical_live'),p,'physical_live')
    assert chosen['epoch']==20


def test_bad_parent_match_score_rejects_candidate_even_with_perfect_v25():
    item=dict(epoch=10,path='epoch_10.pt',sha256='sha',minimum_skill_rate=1.,skill_score=1.)
    game=dict(item,parent=game_report(5,JOINT_MODE),v25=game_report(20,'physical_live'),
        match_guards_passed=False,game_score=.625)
    assert choose_games([game],[item],game_report(18,'physical_live'),protocol(),'physical_live')is None


@pytest.mark.parametrize('corruption',['duplicate','test_seed','mode','outcome','aggregate'])
def test_bad_validation_game_evidence_rejected(corruption):
    r=game_report(15,'physical_live')
    if corruption=='duplicate':r['matches'][0]=deepcopy(r['matches'][1])
    elif corruption=='test_seed':r['matches'][0]['seed']=710001
    elif corruption=='mode':r['matches'][0]['candidate_mode']='physical_live'
    elif corruption=='outcome':r['matches'][0]['result']='draw'
    else:r['summary']['wins']=20
    with pytest.raises(ValueError):verify_games(r,protocol(),'physical_live')


def test_misreported_shortlist_guards_are_rejected():
    rows=history();rows[9]['rehearsal_action_agreement']=.8
    with pytest.raises(ValueError):skill_shortlist(rows,negative(.25),protocol())
