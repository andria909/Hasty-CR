from copy import deepcopy

from scripts.gamestate.summarize_live_rewards import tower_loss


def board():
    return {'units': [dict(entity_id=side*3+i, side=side,
                           name='KingTower' if i==0 else 'PrincessTower',
                           hp=1000, max_hp=1000)
                      for side in (0, 1) for i in range(3)]}


def test_disappearing_tower_is_not_damage_without_terminal_evidence():
    initial=board(); final=deepcopy(initial); final['units'].pop()
    assert tower_loss(initial,final,1,missing_princess_confirmed=False) is None
    assert tower_loss(initial,final,1,missing_princess_confirmed=True)==1


def test_missing_king_remains_unknown_even_with_crown_result():
    initial=board(); final=deepcopy(initial); final['units'].pop(3)
    assert tower_loss(initial,final,1,missing_princess_confirmed=True) is None


def test_midmatch_and_changed_maximum_are_not_full_match_rewards():
    initial=board(); final=deepcopy(initial); initial['units'][3]['hp']=900
    assert tower_loss(initial,final,1,missing_princess_confirmed=True) is None
    initial=board(); final['units'][3]['max_hp']=1100
    assert tower_loss(initial,final,1,missing_princess_confirmed=True) is None


def test_hp_fraction_ignores_different_tower_maxima_and_enemy_damage():
    initial=board(); initial['units'][3].update(hp=2000,max_hp=2000)
    final=deepcopy(initial); final['units'][3]['hp']=1000
    final['units'][0]['hp']=0
    assert tower_loss(initial,final,1,missing_princess_confirmed=True)==.5
