from copy import deepcopy

import pytest

from scripts.attack.train_fireball_refinement import contexts,KINDS
from scripts.attack.verify_fireball_selectivity import select_epoch,negative_rate,positive_score


def positive():
    rows=[dict(kind=k,lane=l,seed=s,success=True,reason='goal_success')for k,l,s in contexts('validation',96)]
    return dict(rows=rows,episodes=96,successes=96,by_kind={k:dict(episodes=32,successes=32)for k in KINDS})


def negative(rate=.25):
    return dict(episodes=96,first_action_fireball_rate=rate,mean_fireball_probability=.3)


def history():
    rows=[]
    for epoch in range(1,41):
        r=dict(epoch=epoch,percent=2.5*epoch,loss=1.)
        if epoch%5==0:r.update(validation=positive(),negative_validation=negative(),
            rehearsal_action_agreement=.95,selection_score=.9375,eligible=True)
        rows.append(r)
    return rows


def test_selection_retains_earliest_tied_eligible_epoch():
    assert select_epoch(history(),negative(.5))['epoch']==5


def test_lower_negative_cast_rate_can_improve_score():
    rows=history();rows[9]['negative_validation']=negative(.125);rows[9]['selection_score']=.96875
    assert select_epoch(rows,negative(.5))['epoch']==10


def test_no_candidate_is_valid_when_all_epochs_fail_retention_guard():
    rows=history()
    for r in rows:
        if 'validation'in r:r.update(rehearsal_action_agreement=.8,eligible=False)
    assert select_epoch(rows,negative(.5))is None


@pytest.mark.parametrize('mutation',['score','nan','guard','progress','missing_epoch','context','aggregate'])
def test_rejects_misreported_selection(mutation):
    rows=history()
    if mutation=='score':rows[4]['selection_score']=1.
    elif mutation=='nan':rows[4]['selection_score']=float('nan')
    elif mutation=='guard':rows[4]['rehearsal_action_agreement']=.8
    elif mutation=='progress':rows[0]['percent']=50.
    elif mutation=='missing_epoch':rows.pop()
    elif mutation=='context':rows[4]['validation']['rows'][0]['seed']=1020000
    else:rows[4]['validation']['successes']=95
    with pytest.raises(ValueError):select_epoch(rows,negative(.5))


@pytest.mark.parametrize('rate',[-.01,1.01,float('nan'),.123])
def test_negative_counts_must_be_finite_possible_counts(rate):
    with pytest.raises(ValueError):negative_rate(negative(rate))


def test_nonterminal_positive_rows_cannot_enter_selection():
    report=deepcopy(positive());report['rows'][0].update(success=False,reason=None);report['successes']-=1
    with pytest.raises(ValueError):positive_score(report)
