from types import SimpleNamespace as NS
import numpy as np
import pytest

from sim.board_potential import public_board_potential,shaping
from sim.continuous_push import ContinuousPushEnv
from sim.entities import make_unit
from sim.arena import Point


@pytest.mark.parametrize('gamma',[.9,.999,1.])
def test_discounted_shaping_cannot_pay_for_cycling_or_change_terminal_return(gamma):
    for path in ([5.,8.,3.,0.],[5.,10.,5.,10.,5.,0.]):
        reward=sum(gamma**i*shaping(a,b,gamma,terminal=i==len(path)-2)
            for i,(a,b)in enumerate(zip(path,path[1:])))
        assert reward==pytest.approx(-5.)
    assert shaping(5,7,gamma,terminal=False)==pytest.approx(gamma*7-5)
    assert shaping(5,7,gamma,terminal=True)==-5


def test_potential_uses_only_own_bank_and_visible_living_units():
    class PrivateBank:
        @property
        def elixir(self):raise AssertionError('Enemy bank must not be read')
    units={1:NS(alive=True,is_tower=False,name='knight',side=1,hitpoints=50,max_hitpoints=100),
           2:NS(alive=True,is_tower=False,name='musketeer',side=-1,hitpoints=100,max_hitpoints=100),
           3:NS(alive=True,is_tower=True,name='king',side=-1,hitpoints=10000,max_hitpoints=10000)}
    match=NS(players={1:NS(elixir=5000),-1:PrivateBank()},battle=NS(entities=units))
    assert public_board_potential(match,{'knight':3.,'musketeer':4.})==2.5
    units[2].hitpoints=75
    assert public_board_potential(match,{'knight':3.,'musketeer':4.})==3.5


@pytest.mark.parametrize('card',['knight','musketeer','cannon','hog_rider','skeletons','electro_spirit'])
def test_bank_to_full_health_units_does_not_create_placement_bonus(card):
    env=ContinuousPushEnv(continuous_profile={'board_potential_weight':.15});env.reset(seed=3)
    before=public_board_potential(env.match,env._board_values)
    spec=env.cards[card];env.match.players[1].elixir-=spec.cost*1000
    for _ in range(max(1,spec.summon_number)):
        env.match.battle.add(make_unit(0,spec.unit,1,Point(3500,24000),0))
    assert public_board_potential(env.match,env._board_values)==pytest.approx(before)


def test_auxiliary_reward_does_not_change_physics_inputs_or_action_availability():
    baseline=ContinuousPushEnv();shaped=ContinuousPushEnv(continuous_profile={'board_potential_weight':.15})
    baseline.reset(seed=7);shaped.reset(seed=7);initial=shaped._metrics()['board_potential'];difference=0.;steps=0
    for index in range(32):
        a=0 if index%2 else int(np.flatnonzero(baseline._mask())[1])if baseline._mask()[1:].any()else 0
        left=baseline.step(a);right=shaped.step(a);steps+=1
        np.testing.assert_array_equal(left[0]['planes'],right[0]['planes'])
        np.testing.assert_array_equal(left[0]['scalars'],right[0]['scalars'])
        np.testing.assert_array_equal(left[4]['action_mask'],right[4]['action_mask'])
        difference+=.999**index*(right[1]-left[1])
        if left[2]or left[3]:break
    expected=.15*(-initial+.999**steps*shaped._metrics()['board_potential'])
    assert difference==pytest.approx(expected,abs=1e-9)
    assert shaped._metrics()['board_shaping_discounted_return']==pytest.approx(expected,abs=1e-9)


def test_trainer_rejects_a_discount_mismatch_before_collecting_rollouts(tmp_path):
    from sim.train_attack_ppo import parser,train
    p=tmp_path/'profile.json';p.write_text('{"board_potential_weight":0.15,"shaping_gamma":0.9}')
    args=parser().parse_args(['--output',str(tmp_path/'run'),'--device','cpu','--curriculum','continuous_v1',
        '--continuous-profile',str(p),'--gamma','0.999'])
    with pytest.raises(ValueError,match='must equal PPO gamma'):train(args)
