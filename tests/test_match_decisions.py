import pytest
from scripts.attack.summarize_match_decisions import aggregate


def row(duration,count,legal,total):
    return dict(elapsed_ms=duration,outcome='loss',accepted_card_counts={'cannon':count},
        accepted_card_lane_counts={'left':{'cannon':count}},
        decision_economy=dict(decisions=total,forced_wait=total-legal,voluntary_wait=legal-count,
            plays=count,invalid=0,available_below4=total-legal,available_elixir_sum=4*legal,
            cards={'cannon':dict(in_hand=total,legal=legal,selected=count)}),
        resolved_spell_counts={},empty_spell_counts={},empty_spell_elixir=0,elixir_overflow=0,
        own_tower_damage=100,nonlethal_crown_only_elixir=0)


def test_opportunities_and_time_are_pooled_without_treating_frames_as_matches():
    data=aggregate([row(10000,1,1,4),row(50000,3,7,16)])
    cannon=data['cards']['cannon']
    assert data['games']==2 and cannon['per_game']==2
    assert cannon['per_minute']==4
    assert cannon['selected_given_legal']==.5
    assert data['plays_given_choice']==.5
    assert data['forced_wait_fraction']==.6


def test_no_legal_choice_is_missing_rate_not_zero_percent_skill():
    data=aggregate([row(10000,0,0,4)])
    assert data['plays_given_choice']is None
    assert data['cards']['cannon']['selected_given_legal']is None


def test_impossible_opportunity_counts_are_rejected():
    data=row(10000,1,1,4)
    data['decision_economy']['cards']['cannon']['legal']=0
    with pytest.raises(ValueError,match='opportunity'):aggregate([data])
