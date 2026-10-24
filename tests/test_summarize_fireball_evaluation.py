from copy import deepcopy

import pytest

from scripts.attack.summarize_fireball_evaluation import fireball_result,duel_result
from scripts.attack.train_fireball_refinement import contexts,KINDS
from sim.eval_defense_duel import summarize,JOINT_MODE


def fireball_fixture():
    rows=[dict(kind=k,lane=l,seed=s,success=True,reason='goal_success')for k,l,s in contexts('test',6)]
    value=dict(rows=rows,episodes=6,successes=6,by_kind={k:dict(episodes=2,successes=2)for k in KINDS})
    return dict(before=deepcopy(value),after=deepcopy(value))


def test_complete_paired_fireball_contexts():
    result=fireball_result(fireball_fixture(),6)
    assert result['after']['tower_unit']['rate']==1


@pytest.mark.parametrize('corruption',['duplicate','off_split','wrong_count','false_success','nonterminal'])
def test_fireball_evidence_corruption_is_rejected(corruption):
    report=fireball_fixture();value=report['after']
    if corruption=='duplicate':value['rows'][0]=deepcopy(value['rows'][1])
    elif corruption=='off_split':value['rows'][0]['seed']=820000
    elif corruption=='wrong_count':value['successes']=5
    elif corruption=='false_success':value['rows'][0]['reason']='goal_window_missed'
    else:value['rows'][0]['reason']=None
    with pytest.raises(ValueError):fireball_result(report,6)


def duel_fixture():
    baselines={'v25':dict(sha256='v25-sha',mode='physical_live')}
    rows=[dict(baseline_id='v25',baseline_sha256='v25-sha',baseline_mode='physical_live',candidate_mode=JOINT_MODE,
               seed=s,candidate_side=side,result='bottom'if side==1 else'top',outcome='win',quality_flags=[])
          for s in range(710001,710011)for side in (1,-1)]
    return dict(matches=rows,baselines=[dict(id='v25',sha256='v25-sha',matches=deepcopy(rows),summary=summarize(rows))],
                historical_summary=summarize(rows),parent_summary=summarize([])),baselines


def test_paired_game_aggregate_verified():
    report,baselines=duel_fixture()
    assert duel_result(report,baselines)['historical']['wins']==20


@pytest.mark.parametrize('corruption',['duplicate','outcome','mode','aggregate'])
def test_paired_game_corruption_is_rejected(corruption):
    report,baselines=duel_fixture()
    if corruption=='duplicate':report['matches'][0]=deepcopy(report['matches'][1])
    elif corruption=='outcome':report['matches'][0]['result']='draw'
    elif corruption=='mode':report['matches'][0]['candidate_mode']='legacy_training'
    else:report['historical_summary']['wins']=19
    with pytest.raises(ValueError):duel_result(report,baselines)
