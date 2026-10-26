import json
from scripts.kaggle.stream_training import record_evaluation_event


def test_completed_parts_form_honest_lower_bound_without_duplicate_pair():
    p={}
    model=json.dumps(dict(event='POOL_TEST_COMPLETE',model='parent',games=120))
    assert record_evaluation_event(p,model,10)
    assert not record_evaluation_event(p,model,11)
    pair=json.dumps(dict(event='POOL_PAIR_COMPLETE',pair=['parent','v25'],games=20))
    assert record_evaluation_event(p,pair,12)
    reverse=json.dumps(dict(event='POOL_PAIR_COMPLETE',pair=['v25','parent'],games=20))
    assert not record_evaluation_event(p,reverse,13)
    assert p['evaluation']['games_lower_bound']['completed']==140
    assert not p['evaluation']['games_lower_bound']['quality_verified']


def test_unknown_model_and_wrong_game_count_are_rejected():
    for model,games in [('missing',120),('parent',119)]:
        assert not record_evaluation_event({},json.dumps(dict(event='POOL_TEST_COMPLETE',model=model,games=games)),1)
