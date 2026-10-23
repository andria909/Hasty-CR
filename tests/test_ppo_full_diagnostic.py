import json
import runpy
import sys
from types import SimpleNamespace as NS

from scripts.kaggle.run_ppo_full_diagnostic import TRACE_WRAPPER


def test_diagnostic_observer_retains_original_result_and_unmodified_invalid_target(monkeypatch,capsys):
    from sim.defense_env import DefenseEnv
    from sim import arena
    sentinel=object()
    def original(env):
        env.terminal_reason='invalid_script_play_timeout'
        return sentinel
    monkeypatch.setattr(DefenseEnv,'_process_enemy',original)
    calls=[];monkeypatch.setattr(runpy,'run_module',lambda *args,**kwargs:calls.append((args,kwargs)))
    monkeypatch.setattr(sys,'path',list(sys.path))
    namespace={};exec(compile(TRACE_WRAPPER,'trace_run.py','exec'),namespace)
    point=arena.Point(-200,24000);player=NS(hand=['zap'],elixir=10000)
    match=NS(elapsed_ms=45050,players={-1:player},spells={'zap':NS(can_place_on_water=True)},cards={'zap':NS(cost=2)},
             can_play=lambda *args:True,play_log=[],battle=NS(entities={}))
    env=NS(match=match,scenario=NS(scenario_id='diagnostic-fixture',plays=[NS(card='zap')]),seed=49,_script_index=0,
           terminal_reason=None,_target=lambda event:point)
    assert namespace['observed'](env)is sentinel
    record=json.loads(capsys.readouterr().out)
    assert record['event']=='SCRIPT_PLAY_DIAGNOSTIC' and record['target']==[-200,24000]
    assert record['in_arena'] is False and env._target(None)==point
    assert calls==[(('sim.train_defense_ppo',),{'run_name':'__main__'})]
