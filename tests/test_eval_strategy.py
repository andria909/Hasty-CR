import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from sim import eval_strategy as evaluator
from sim.strategy_env import StrategyEnv, OBSERVATION_CONTRACT


def test_frozen_schedule_covers_all_scenarios_and_all_shuffle_seeds():
    protocol=json.loads(Path('data/strategy/evaluation_protocol.json').read_text())
    for split,count in [('validation',1056),('test',804)]:
        env=StrategyEnv(split=split)
        rows=evaluator.context_schedule(protocol,split,env.scenarios)
        assert len(rows)==count
        assert len({(r['scenario_id'],r['seed'])for r in rows})==count
    with pytest.raises(ValueError):evaluator.context_schedule(protocol,'train',env.scenarios)
    protocol['scenario_ids']['test']=protocol['scenario_ids']['test'][:-1]
    with pytest.raises(ValueError,match='exact heldout catalog'):
        evaluator.context_schedule(protocol,'test',env.scenarios)


def test_real_environment_and_network_evaluation_in_both_tasks():
    # Tiny validation integration probe; no trained model selection or test evaluation.
    from sim.train_attack_ppo import initialize_attack_policy
    torch.set_num_threads(2)
    def factory(**options):
        env=StrategyEnv(**options)
        env.scenarios=[next(s for s in env.scenarios if s.task==task)for task in ('attack','defense')]
        return env
    options=dict(max_ms=1000)
    env=factory(split='validation',**options)
    protocol=dict(schema='joint-strategy-evaluation-v1',environment=options,
        scenario_ids=dict(validation=[s.scenario_id for s in env.scenarios]),seeds=dict(validation=[900001]))
    network,_,_=initialize_attack_policy(num_scalars=309)
    report=evaluator.evaluate(network,'cpu',protocol,'validation',env_factory=factory)
    assert report['episodes']==2
    assert {r['task']for r in report['records']}=={'attack','defense'}
    assert report['attack']['overall']['episodes']==1
    assert report['defense']['overall']['episodes']==1
    assert not report['quality_accepted'] and not report['full_match_evaluated']
    assert not report['checkpoint_selection_performed']


def test_source_and_environment_contract_are_checked_before_evaluation(tmp_path):
    from sim.train_attack_ppo import initialize_attack_policy, code_fingerprint
    from sim.arena import resolve_geometry
    network,architecture,_=initialize_attack_policy(num_scalars=309)
    options=dict(action_delay_ms=150,max_ms=240000,geometry='legacy_v1',
        coordinate_mode='physical_live_cells_v2',initial_hand_mode='shuffle',deadline_scale=3.,visible_fraction=.5)
    config=dict(options,task_mode='joint',num_scalars=309,observation_contract=OBSERVATION_CONTRACT,
        geometry_sha256=resolve_geometry('legacy_v1').fingerprint,source_code_sha256=code_fingerprint('joint'))
    blob=dict(schema='attack-ppo-boundary-v1',state_dict=network.state_dict(),architecture=architecture,config=config)
    path=tmp_path/'selected.pt';torch.save(blob,path)
    assert evaluator.load_selected(path,'cpu',options).state_dict()['scalar.0.weight'].shape[1]==309
    with pytest.raises(ValueError,match='differs from training'):
        evaluator.load_selected(path,'cpu',dict(options,deadline_scale=1.))
    blob['config']['source_code_sha256']='different';torch.save(blob,path)
    with pytest.raises(ValueError,match='source semantics differ'):evaluator.load_selected(path,'cpu',options)
