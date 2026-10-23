from contextlib import nullcontext
import pickle
from types import SimpleNamespace as NS

import pytest
from sim.attack_env import AttackEnv
from sim.evaluation_terminal_damage import observe_terminal_damage


def setup(**changes):
    fields=dict(scenario_id='terminal-evaluation-test',family_id='terminal-evaluation-test',split='train',goal_kind='spell_finish',
        lane='left',target_lane='king',initial_elixir=10,hand_seed=42,initial_hand=('fireball','zap','hog_rider','knight'),
        enemy_initial_elixir=10,tower_hp=(NS(side=-1,lane='king',hp=1),),initial_units=(),enemy_plays=(),
        deadline_ms_range=(8000,14000),deadline_visibility='hidden',geometry='legacy_v1',opportunity_start_ms=0)
    fields.update(changes);return NS(**fields)


def play(card,instrumented,**changes):
    env=AttackEnv(scenario=setup(**changes),action_delay_ms=0);env.reset(seed=42)
    x,y=(3,7) if changes.get('target_lane')=='left' else (9,3);first=0 if card is None else 1+env.match.players[1].hand.index(card)*576+y*18+x
    trace=[];context=observe_terminal_damage(env) if instrumented else nullcontext(None)
    with context as observer:
        for i in range(100):
            obs,reward,terminal,truncated,info=env.step(first if i==0 else 0)
            trace.append((obs['planes'].tobytes(),obs['scalars'].tobytes(),reward,terminal,truncated,info['action_mask'].tobytes()))
            if terminal or truncated:break
        else:raise AssertionError('Expected bounded terminal episode')
        enriched=observer.enrich(info['metrics']) if observer else info['metrics']
    return trace,pickle.dumps(env.snapshot()),info['metrics'],enriched


@pytest.mark.parametrize('card',['fireball','zap'])
def test_actual_king_finisher_attribution_preserves_entire_trajectory_and_snapshot(card):
    plain=play(card,False);observed=play(card,True)
    assert plain[:3]==observed[:3]
    metrics=observed[3]
    assert metrics['terminal_reason']=='enemy_king_destroyed' and metrics['successful_spell_finish'] is None
    assert metrics['successful_spell_finish_v2'] is True
    assert len(metrics['terminal_damage_events_v2'])==1
    event=metrics['terminal_damage_events_v2'][0]
    assert event['source_kind']=='spell' and event['source_name']==card and event['damage']==1 and event['source_side']==1


def test_nonterminal_princess_finish_is_not_double_counted():
    plain=play('zap',False,target_lane='left',tower_hp=(NS(side=-1,lane='left',hp=1),))
    observed=play('zap',True,target_lane='left',tower_hp=(NS(side=-1,lane='left',hp=1),))
    assert plain[:3]==observed[:3]
    assert observed[3]['successful_spell_finish_v2'] is True
    assert observed[3]['terminal_damage_events_v2']==[]


def test_context_restores_functions_after_error_and_rejects_overlap():
    from sim import spells
    from sim.engine import Battle
    original_spell=spells.apply_spell;original_land=Battle._land
    env=AttackEnv(scenario=setup())
    with pytest.raises(ValueError):
        with observe_terminal_damage(env):
            with pytest.raises(RuntimeError,match='overlap'):
                with observe_terminal_damage(env):pass
            raise ValueError('evaluation stopped')
    assert spells.apply_spell is original_spell and Battle._land is original_land


def test_missing_terminal_source_remains_unknown():
    from sim.evaluation_terminal_damage import TerminalDamageObserver
    env=AttackEnv(scenario=setup());env.reset(seed=42)
    observer=TerminalDamageObserver(env)
    result=observer.enrich(dict(task='attack',terminal_reason='enemy_king_destroyed',successful_spell_finish=None,
                                hog_target_tower_damage=0,hog_damage_ledger_complete=False))
    assert result['successful_spell_finish_v2'] is None
    assert result['hog_damage_ledger_complete_v2'] is False


@pytest.mark.parametrize('unit_card',['hog_rider','knight'])
def test_terminal_unit_source_attributes_hog_only_and_preserves_mechanics(unit_card):
    probe=AttackEnv(scenario=setup());probe.reset(seed=42);point=probe.match.towers[-1]['king'].pos
    unit=NS(card=unit_card,side=1,x=point.x,y=point.y+1000,deployed_ms_ago=1000)
    plain=play(None,False,goal_kind='hog_tower_damage',initial_units=(unit,))
    observed=play(None,True,goal_kind='hog_tower_damage',initial_units=(unit,))
    assert plain[:3]==observed[:3]
    metrics=observed[3]
    assert metrics['terminal_reason']=='enemy_king_destroyed'
    assert metrics['hog_target_tower_damage']==0
    assert metrics['hog_target_tower_damage_v2']==(1 if unit_card=='hog_rider' else 0)
    assert metrics['hog_damage_ledger_complete_v2'] is True
    assert metrics['successful_spell_finish_v2'] is False


def test_unknown_attribution_is_not_counted_as_failed_or_successful_finish():
    from sim.eval_strategy import summarize_terminal_attribution
    rows=[dict(goal_kind='spell_finish',successful_spell_finish_v2=v) for v in (True,False,None)]
    rows+=[dict(goal_kind='hog_tower_damage',hog_target_tower_damage_v2=0,hog_damage_ledger_complete_v2=False)]
    summary=summarize_terminal_attribution(rows)
    assert summary['spell_finish']==dict(episodes=3,known=2,unknown=1,success=1,rate=.5)
    assert summary['hog_hit']==dict(episodes=1,known=0,unknown=1,success=0,rate=None)
