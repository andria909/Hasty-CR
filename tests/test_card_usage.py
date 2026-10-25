import pytest

from scripts.attack.evaluate_card_usage import card_summary,counted_match


def test_summary_includes_absent_cards_and_uses_elapsed_time():
    rows=[dict(elapsed_ms=60000,candidate_cards={'hog_rider':2}),
          dict(elapsed_ms=120000,candidate_cards={'hog_rider':4})]
    summary=card_summary(rows)
    assert summary['hog_rider']['mean_per_match']==3
    assert summary['hog_rider']['std_population']==1
    assert summary['hog_rider']['plays_per_game_minute']==2
    assert summary['musketeer']['total']==0
    assert summary['musketeer']['mean_per_match']==0
    with pytest.raises(ValueError):card_summary([dict(elapsed_ms=1,candidate_cards={'hog_rider':-1})])


def test_skeleton_card_is_one_cast_even_when_it_spawns_three_units(monkeypatch):
    from sim import arena
    from sim.defense_assets import load_defense_assets
    from sim.runner import resolve_deck
    from sim.env import DECK_26
    cards,spells,_,_=load_defense_assets();cards.update(resolve_deck(cards,DECK_26))
    def fake_run(*args,match_factory,**kwargs):
        match=match_factory(cards=cards,decks=(list(DECK_26),list(DECK_26)),seed=1,spells=spells)
        player=match.players[1];player.elixir=10000
        player.hand=['skeletons','hog_rider','cannon','zap']
        player.queue=[c for c in DECK_26 if c not in player.hand]
        assert match.play_card(1,'skeletons',arena.Point(3500,22000))
        assert len([e for e in match.battle.entities.values()if e.side==1 and e.name=='skeleton'])==3
        return dict(plays={'1':1,'-1':0},elapsed_ms=1000)
    monkeypatch.setattr('scripts.attack.evaluate_card_usage.run_match',fake_run)
    row=counted_match(None,None,'cpu','physical_live',1,1,cards,spells,
        dict(geometry='native_anchors_candidate_v1',action_delay_ms=150,max_ms=400000))
    assert row['candidate_cards']=={'skeletons':1}
    assert row['opponent_cards']=={}
    assert len(row['accepted_card_events'])==1


def test_verifier_recomputes_casts_and_rejects_duplicate_paired_games():
    from copy import deepcopy
    from scripts.attack.summarize_card_usage import verify_model
    from sim.eval_defense_duel import summarize,outcome
    model=dict(id='example',mode='physical_live',sha256='pinned',file='policy.pt')
    protocol=dict(seeds=[1],seats=[1,-1],opponent=model)
    rows=[]
    for seat in (1,-1):
        rows.append(dict(seed=1,candidate_side=seat,candidate_mode='physical_live',baseline_mode='physical_live',
            result='bottom',outcome=outcome('bottom',seat,True),elapsed_ms=60000,quality_flags=[],
            plays={'1':1,'-1':1},candidate_cards={'skeletons':1},opponent_cards={'hog_rider':1},
            accepted_card_events=[dict(card='skeletons',side=seat,at_ms=100),dict(card='hog_rider',side=-seat,at_ms=200)]))
    report=dict(model=model,opponent=model,records=rows,matches=2,summary=summarize(rows),
        cards=card_summary(rows),opponent_cards=card_summary(rows,'opponent_cards'))
    assert verify_model(report,model,protocol)['cards']['skeletons']['mean_per_match']==1
    changed=deepcopy(report);changed['records'][0]['candidate_cards']['skeletons']=3
    with pytest.raises(ValueError,match='accepted events'):verify_model(changed,model,protocol)
    changed=deepcopy(report);changed['records'][1]=changed['records'][0]
    with pytest.raises(ValueError,match='schedule'):verify_model(changed,model,protocol)
