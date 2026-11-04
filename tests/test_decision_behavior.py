from copy import deepcopy
import pytest
from scripts.attack.decision_behavior import economy,card_usage


def record():
    return dict(elapsed_ms=60000,accepted_card_counts={'cannon':2},decision_economy=dict(
        decisions=100,forced_wait=90,voluntary_wait=5,plays=5,invalid=0,available_below4=99,
        cards={'cannon':dict(in_hand=100,legal=10,selected=5),'musketeer':dict(in_hand=100,legal=0,selected=0)}))


def test_forced_wait_is_not_a_policy_choice_and_unavailable_card_is_unknown():
    r=economy([record()])
    assert r['forced_wait_fraction']==.9
    assert r['voluntary_wait_given_choice']==.5
    assert r['valid_choice_decisions']==10
    assert r['cards']['musketeer']['selected_given_legal'] is None
    assert r['cards']['cannon']['selected_given_legal']==.5
    assert r['available_below4_fraction_all_decisions']==.99


def test_missing_economy_is_unknown_not_zero_or_a_partial_cohort_average():
    assert economy([dict()])==dict(known_games=0,total_games=1,complete=False)
    result=economy([record(),dict()])
    assert result==dict(known_games=1,total_games=2,complete=False)


def test_card_rates_use_actual_duration_and_keep_acceptance_separate_from_requests():
    r=record();long=deepcopy(r);long['elapsed_ms']=180000;long['accepted_card_counts']={'cannon':6}
    usage=card_usage([r,long]);assert usage['cannon']==dict(per_game=4,per_minute=2)
    assert usage['musketeer']==dict(per_game=0,per_minute=0)
    assert economy([r,long])['cards']['cannon']['selected']==10


@pytest.mark.parametrize('change',['partition','overflow','negative','opportunities'])
def test_malformed_decision_evidence_is_rejected(change):
    row=record();stats=row['decision_economy']
    if change=='partition':stats['plays']+=1
    elif change=='overflow':stats['available_below4']=101
    elif change=='negative':stats['forced_wait']=-1
    else:stats['cards']['cannon']['selected']=11
    with pytest.raises(ValueError):economy([row])
