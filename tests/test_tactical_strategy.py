from collections import defaultdict,Counter

import pytest

from sim.tactical_strategy import TacticalStrategyEnv,catalog,CONTRACT


def exercise(goal):
    env=TacticalStrategyEnv(split='train',geometry='native_anchors_candidate_v1')
    case=next(s for s in env.scenarios if getattr(s.scenario,'tactical_goal',None)==goal and
        s.scenario.initial_elixir==10 and s.scenario.lane=='left'and '/y12500/'in s.scenario_id)
    obs,info=env.reset(seed=42,scenario_id=case.scenario_id)
    return env,obs,info


def test_task_feature_is_public_presence_not_privileged_exercise_label():
    env,obs,info=exercise('supported_hog')
    assert len(obs['scalars'])==309
    assert obs['scalars'][-3:].tolist()==[0.,0.,0.]
    assert info['metrics']['task']=='attack'
    assert info['metrics']['observation_contract']==CONTRACT
    assert not any(e.side==1 and e.name=='musketeer'for e in env.match.battle.entities.values())


def test_related_lanes_hands_and_positions_never_cross_splits():
    cases=catalog('native_anchors_candidate_v1');families=defaultdict(set)
    for case in cases:families[case.family_id].add(case.split)
    assert all(len(splits)==1 for splits in families.values())
    for split,count in (('train',72),('validation',18),('test',18)):
        assert set(Counter(c.tactical_goal for c in cases if c.split==split).values())=={count}


@pytest.mark.parametrize('goal',['supported_hog','air_defense_counterpush','hog_defense_counterpush'])
def test_real_reference_actions_can_complete_each_new_goal(goal):
    env,obs,info=exercise(goal);first=False;second=False
    for _ in range(180):
        player=env.match.players[1];card=None
        if goal=='hog_defense_counterpush':
            threat_alive=any(env.match.battle.entities.get(uid)and env.match.battle.entities[uid].alive
                for uid in env._active._tactical_initial_enemy)
            card='cannon'if not first else 'hog_rider'if not second and not threat_alive else None
            x,y=(7,20)if card=='cannon'else (3,18)
        else:
            card='musketeer'if not first else 'hog_rider'if not second else None
            x,y=(3,22)if card=='musketeer'else (3,18)
        action=0
        if card and card in player.hand:
            candidate=1+player.hand.index(card)*576+y*18+x
            if info['action_mask'][candidate]:action=candidate
        if action:
            if first:second=True
            else:first=True
        obs,reward,done,truncated,info=env.step(action)
        if done or truncated:break
    assert info['terminal_reason']=='goal_success'
    assert info['metrics']['hog_target_tower_damage']>0
    if goal!='hog_defense_counterpush':
        assert info['metrics']['musketeer_accepted_plays']==1
        assert info['metrics']['musketeer_unit_damage']>0
