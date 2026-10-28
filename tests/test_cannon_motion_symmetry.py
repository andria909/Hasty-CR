import pytest
from scripts.attack.audit_cannon_tower_states import run_case
from scripts.attack.audit_cannon_stress import CELLS
from sim.defense_assets import load_defense_assets
from sim.runner import resolve_deck
from sim.env import DECK_26
from sim.engine import _signed_trunc_div


@pytest.mark.parametrize('value',[0,1,84,85,999,1001,1234567890123456789])
def test_vector_rounding_is_sign_symmetric(value):
    assert _signed_trunc_div(value,7)==value//7
    assert _signed_trunc_div(-value,7)==-(value//7)


@pytest.mark.parametrize('route,decision,choice',[
    ('pocket',0,'central_near'),('bridge',5500,'king_front')])
@pytest.mark.parametrize('delay',[1350,2025])
def test_paid_hog_cannon_targets_and_every_position_are_mirrored(route,decision,choice,delay):
    cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    traces={};results={}
    for lane in ['left','right']:
        traces[lane]=[]
        def capture(match):
            traces[lane].append([(e.uid,e.pos.x,e.pos.y,e.hitpoints,e.target_uid)
                for e in match.battle.entities.values()if not e.is_tower])
        results[lane]=run_case(cards,spells,0,3,lane,'hog_rider',delay,choice,
            route=route,decision_ms=decision,candidate_cells=CELLS,trace=capture)
    assert results['left']['total_damage']==results['right']['total_damage']
    assert results['left']['pulled']==results['right']['pulled']
    assert len(traces['left'])==len(traces['right'])
    for left,right in zip(traces['left'],traces['right']):
        mirrored_tower_uid={1:2,2:1,4:5,5:4}
        assert [(uid,18000-x,y,hp,mirrored_tower_uid.get(target,target))for uid,x,y,hp,target in left]==right
