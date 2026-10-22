from types import SimpleNamespace as NS

import numpy as np
import pytest

from sim.defense_audit_policy import defender_action


def fake(hand, lane='right', actions=()):
    mask=np.zeros(2321,dtype=bool)
    mask[0]=True
    mask[list(actions)]=True
    return NS(match=NS(players={1:NS(hand=hand)}), scenario=NS(lane=lane)), {'action_mask':mask}


def action(slot,x,y):
    return 1+slot*576+y*18+x


def test_hold_default_and_unavailable_targets_do_not_evict_unknowns():
    env,info=fake(['unknown','cannon','zap','knight'])
    assert defender_action(env,info)==0
    assert defender_action(env,info,'cannon_cycle')==0
    info['action_mask'][action(0,14,16)]=True
    assert defender_action(env,info,'cannon_cycle')==0


@pytest.mark.parametrize('lane,x', [('right',14),('left',3)])
def test_cannon_priority_over_hand_order_and_mirrored_lane(lane,x):
    env,info=fake(['knight','cannon','skeletons','zap'],lane,
                  [action(0,x,16),action(1,x,16)])
    assert defender_action(env,info,'cannon_cycle')==action(1,x,16)
    info['action_mask'][action(1,x,16)]=False
    assert defender_action(env,info,'cannon_cycle')==action(0,x,16)


@pytest.mark.parametrize('card,x,y', [('musketeer',14,20),('hog_rider',3,16),
                                      ('zap',3,6),('fireball',3,6)])
@pytest.mark.parametrize('lane', ['left','right'])
def test_card_targets(card,x,y,lane):
    if lane=='left':x=17-x
    env,info=fake([card],lane,[action(0,x,y)])
    assert defender_action(env,info,'cannon_cycle')==action(0,x,y)


def test_unknown_modes_and_missing_lane_fail_closed():
    env,info=fake(['cannon'],lane='unknown')
    with pytest.raises(ValueError,match='Unknown audit'):
        defender_action(env,info,'typo')
    with pytest.raises(ValueError,match='explicit scenario lane'):
        defender_action(env,info,'cannon_cycle')


def test_real_firecracker_escort_regression_policy_actually_plays():
    from sim.defense_env import DefenseEnv
    env=DefenseEnv(split='train',
        scenario_id='pair:firecracker:hog_rider:skeletons/back/x1/right',max_ms=60000)
    _,info=env.reset(seed=42)
    plays=0
    for _ in range(120):
        selected=defender_action(env,info,'cannon_cycle')
        assert info['action_mask'][selected]
        plays+=selected!=0
        _,_,terminated,truncated,info=env.step(selected)
        assert not str(info['terminal_reason']).startswith('invalid_script')
        if env._script_index==len(env.scenario.plays) or terminated or truncated:
            break
    assert plays>0
    assert env._script_index==3
    assert [e['card']for e in env.match.play_log if e['side']==-1]==[
        'firecracker','hog_rider','skeletons']
    reasons=info['metrics']['script_trigger_reasons']
    assert any(r['reason']=='escort_engaged' and r['escort']=='firecracker' for r in reasons)
