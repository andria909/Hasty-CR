"""PPO u1164 Princess escort failure: real legal simulator defense, no network."""
import pytest

from sim.defense_env import DefenseEnv

SCENARIO='pair:goblin_barrel:knight:princess:skeletons/back/x1/right'


def cannon_cycle(env,info):
    player=env.match.players[1]
    for card in ('cannon','skeletons','electro_spirit','knight','musketeer','hog_rider','zap','fireball'):
        if card not in player.hand:
            continue
        x,y=(14,20)if card=='musketeer'else(3,16)if card=='hog_rider'else(3,6)if card in ('zap','fireball')else(14,16)
        action=1+player.hand.index(card)*576+y*18+x
        if info['action_mask'][action]:
            return action
    return 0


@pytest.mark.parametrize('seed',[42,0,1,900000])
def test_princess_combat_releases_full_knight_barrel_skeletons_schedule(seed):
    env=DefenseEnv(split='train',scenario_id=SCENARIO,max_ms=60000)
    _,info=env.reset(seed=seed)
    # Original stage2 source deterministically failed this exact seed42 policy at45050ms.
    for _ in range(120):
        _,_,terminal,truncated,info=env.step(cannon_cycle(env,info))
        assert not str(info['terminal_reason']).startswith('invalid_script')
        if terminal or truncated:
            break
    assert env._script_index==len(env.scenario.plays)==4
    assert [row['card']for row in env.match.play_log if row['side']==-1]==[
        'princess','knight','goblin_barrel','skeletons']
    reasons=info['metrics']['script_trigger_reasons']
    assert len(reasons)==1
    assert reasons[0]['escort']=='princess'and reasons[0]['card']=='knight'
    assert reasons[0]['reason']=='escort_engaged'and reasons[0]['at_ms']<45000
