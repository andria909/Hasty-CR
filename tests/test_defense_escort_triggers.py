"""Policy-dependent escort trigger regressions using simulator state only."""
from sim.defense_env import DefenseEnv
from sim.defense_scenarios import Scenario,ScheduledPlay
from sim.entities import IDLE,WINDUP

FAILING_ID='pair:firecracker:hog_rider:skeletons/back/x1/right'


def defender_action(env,info):
    """Legal deterministic Cannon/cycle defense; no network or gradients."""
    player=env.match.players[1]
    for card in ('cannon','skeletons','electro_spirit','knight','musketeer','hog_rider','zap','fireball'):
        if card not in player.hand:
            continue
        x,y=14,16
        if card=='musketeer':x,y=14,20
        if card=='hog_rider':x,y=3,16
        if card in ('zap','fireball'):x,y=3,6
        action=1+player.hand.index(card)*576+y*18+x
        if info['action_mask'][action]:
            return action
    return 0


def test_real_cannon_cycle_does_not_deadlock_alive_firecracker_before_bridge():
    env=DefenseEnv(split='train',scenario_id=FAILING_ID,max_ms=60000)
    _,info=env.reset(seed=42)
    for _ in range(120):
        _,_,terminated,truncated,info=env.step(defender_action(env,info))
        assert not str(info['terminal_reason']).startswith('invalid_script')
        if env._script_index==len(env.scenario.plays)or terminated or truncated:
            break
    assert env._script_index==3
    assert [event['card']for event in env.match.play_log if event['side']==-1]==['firecracker','hog_rider','skeletons']
    reasons=info['metrics']['script_trigger_reasons']
    assert len(reasons)==1 and reasons[0]['reason']=='escort_engaged'
    assert reasons[0]['escort']=='firecracker' and reasons[0]['card']=='hog_rider'
    assert reasons[0]['at_ms']<45000


def escort_env():
    scenario=Scenario('escort-trigger-test','escort-trigger-test','medium','train','right',(
        ScheduledPlay('firecracker',position='back'),
        ScheduledPlay('hog_rider',near_bridge_after='firecracker')))
    env=DefenseEnv(scenario=scenario,max_ms=60000)
    env.reset(seed=42)
    escort=next(u for u in env.match.battle.entities.values()if u.side==-1 and not u.is_tower)
    return env,escort


def test_alive_slow_escort_with_acquired_target_does_not_trigger_early():
    env,escort=escort_env()
    escort.speed_mt_per_sec=0
    escort.state=IDLE
    escort.target_uid=env.match.towers[1]['right'].uid
    assert escort.hitpoints==escort.max_hitpoints and escort.pos.y<11500
    env.match.battle.now_ms=1000
    env._process_enemy()
    assert env._script_index==1 and not env._script_engaged_units
    assert env._script_trigger_reasons=={}


def test_never_seen_escort_remains_invalid_instead_of_dropping_attack():
    scenario=Scenario('invalid-unseen','invalid-unseen','medium','train','left',(
        ScheduledPlay('hog_rider',near_bridge_after='firecracker',max_wait_ms=1000),))
    env=DefenseEnv(scenario=scenario,max_ms=5000)
    env.reset(seed=42)
    for _ in range(3):
        _,_,terminated,truncated,info=env.step(0)
        if terminated or truncated:
            break
    assert truncated and not terminated
    assert info['terminal_reason']=='invalid_script_trigger_timeout'
    assert env._script_index==0 and env.match.players[-1].elixir_spent==0


def test_combat_trigger_is_latched_and_diagnostic_not_repeated_while_waiting_elixir():
    env,escort=escort_env()
    env.match.players[-1].elixir=0
    escort.state=WINDUP
    env._process_enemy()
    assert env._script_index==1
    first=list(env._script_trigger_reasons.values())
    escort.state=IDLE
    env.match.battle.now_ms=1000
    env._process_enemy()
    assert list(env._script_trigger_reasons.values())==first
    env.match.players[-1].elixir=10000
    env._process_enemy()
    assert env._script_index==2
    assert list(env._script_trigger_reasons.values())==first


def test_older_snapshot_without_trigger_diagnostics_restores_as_empty_state():
    env,_=escort_env()
    snapshot=env.snapshot()
    snapshot.pop('_script_engaged_units')
    snapshot.pop('_script_trigger_reasons')
    env.restore(snapshot)
    assert env._script_engaged_units==set()
    assert env._metrics()['script_trigger_reasons']==[]
