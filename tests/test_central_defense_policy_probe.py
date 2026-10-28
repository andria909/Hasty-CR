from collections import Counter
from scripts.attack.probe_central_defense_policies import contexts,opening,summarize
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26
from sim.runner import resolve_deck


def test_suite_is_balanced_and_public_opening_pays_actual_enemy_cost():
    rows=contexts()
    assert len(rows)==64 and len({tuple(r.values())for r in rows})==64
    assert set(Counter((r['own_mask'],r['threat'],r['lane'])for r in rows).values())=={4}
    cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    for threat in ('hog_rider','balloon'):
        r=next(r for r in rows if r['threat']==threat and r['own_mask']==0)
        m,hp=opening(r,cards,spells)
        assert m.players[-1].elixir_spent==cards[threat].cost
        assert m.players[1].elixir_spent==0 and m.elapsed_ms==1500
        assert sorted(m.players[1].hand+m.players[1].queue)==sorted(DECK_26)
        assert hp['left']==hp['right']==0 and hp['king']>0


def test_summary_counts_paid_actions_not_requested_oracle_plans():
    rows=[dict(actions=[dict(card='cannon',cell=[8,22])],clear_without_damage=False,
               total_damage=100,elixir_spent=3,rejections=1),
          dict(actions=[],clear_without_damage=True,total_damage=0,elixir_spent=0,rejections=0)]
    s=summarize(rows)
    assert s['mean_card_plays']['cannon']==.5 and s['mean_paid_elixir']==1.5
    assert s['clear_without_damage']==1 and s['mean_damage']==50
    assert s['rejected_commands']==1 and s['cannon_outside_center']==0
