"""Bounded CPU optimizer smoke authorized for attack trainer preparation."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from sim.env import NUM_PLANES,NUM_SCALARS,ACTIONS
from sim.train_ppo import build_network,COMPACT_ARCHITECTURE
from sim.train_attack_ppo import (parser,train,load_attack_state,build_attack_network,
    initialize_attack_policy,make_attack_env,summarize_attack,validate_resume)


def test_zero_extended_timer_warmstart_preserves_legacy_logits():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    old=build_network(NUM_PLANES,NUM_SCALARS,ACTIONS)
    new=build_attack_network()
    metadata=load_attack_state(new,{'state_dict':old.state_dict()})
    assert metadata['appended_timer_columns_zero_initialized']
    assert torch.count_nonzero(new.scalar[0].weight[:,-2:])==0
    planes=torch.rand(1,NUM_PLANES,32,18)
    old_scalars=torch.rand(1,NUM_SCALARS)
    new_scalars=torch.cat((old_scalars,torch.tensor([[.03,1.]])),dim=1)
    with torch.no_grad():
        old_outputs=old(planes,old_scalars);new_outputs=new(planes,new_scalars)
    for a,b in zip(old_outputs,new_outputs):torch.testing.assert_close(a,b,rtol=0,atol=0)
    with pytest.raises(ValueError,match='input shape'):
        load_attack_state(new,{'state_dict':old.state_dict()},exact=True)


def test_make_env_scales_both_splits_without_test_leakage():
    for split in ('train','validation'):
        original=make_attack_env(split=split)
        scaled=make_attack_env(split=split,deadline_scale=3,initial_hand_mode='shuffle')
        assert [s.scenario_id for s in original.scenarios]==[s.scenario_id for s in scaled.scenarios]
        assert scaled.scenarios[0].deadline_ms_range==tuple(x*3 for x in original.scenarios[0].deadline_ms_range)
    with pytest.raises(ValueError,match='test split'):make_attack_env(split='test')


def smoke_args(output,updates=1):
    return parser().parse_args(['--output',str(output),'--task-mode','attack','--device','cpu','--updates',str(updates),
        '--envs','1','--rollout','2','--epochs','1','--minibatch-size','2','--eval-every','0',
        '--max-ms','500','--target-environment-steps','4','--checkpoint-every','50'])


def test_real_cpu_optimizer_update_exact_resume_and_timer_checkpoint(tmp_path):
    torch.set_num_threads(1)
    one=train(smoke_args(tmp_path/'one'))
    checkpoint=one/'update_000001.pt'
    first=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert first['schema']=='attack-ppo-boundary-v1'
    assert first['step']==2 and first['update']==1
    assert first['state_dict']['scalar.0.weight'].shape[1]==NUM_SCALARS+2
    assert first['config']['initial_hand_mode']=='shuffle'
    assert first['sampler']['initial_hand_mode']=='shuffle'
    assert first['environment_states'][0]['_deadline_ms']>0
    metrics=json.loads((one/'metrics.jsonl').read_text())
    assert np.isfinite(metrics['gradient_norm']) and np.isfinite(metrics['loss'])
    assert first['optimiser']['state']
    continuous=train(smoke_args(tmp_path/'continuous',updates=2))
    args=smoke_args(tmp_path/'resume',updates=2);args.resume=str(checkpoint)
    resumed=train(args)
    a=torch.load(continuous/'update_000002.pt',map_location='cpu',weights_only=False)
    b=torch.load(resumed/'update_000002.pt',map_location='cpu',weights_only=False)
    for key in a['state_dict']:
        torch.testing.assert_close(a['state_dict'][key],b['state_dict'][key],rtol=0,atol=0)
    assert a['sampler']==b['sampler']
    assert a['episode_returns']==b['episode_returns']
    assert a['infos'][0]['metrics']==b['infos'][0]['metrics']
    bad=deepcopy(first['config']);bad['initial_hand_mode']='scenario'
    with pytest.raises(ValueError,match='initial_hand_mode'):validate_resume(first,bad)


def test_empty_partial_summary_and_visibility_breakdown():
    assert summarize_attack([])['episode_return']['mean']is None
    result=summarize_attack([dict(timer_visible=False,goal_kind='hog',setup_kind='bridge',
        attack_success=True,episode_return=1),dict(timer_visible=True,attack_success=False)])
    assert result['attack_success']['rate']==.5
    assert result['episode_return']['known']==1
    assert set(result['by']['timer_visible'])=={'False','True'}


def test_parser_upfront_target_and_explicit_cuda_default():
    args=parser().parse_args(['--output','unused'])
    assert args.init is None and args.resume is None
    assert args.target_environment_steps==1000000
    assert args.device=='cuda' and args.initial_hand_mode=='shuffle'
    assert args.eval_episodes==32 and args.checkpoint_every==50


def test_joint_warmstart_309_preserves_logits_and_explicit_version():
    torch.set_num_threads(1)
    old=build_network(NUM_PLANES,NUM_SCALARS,ACTIONS)
    network,_,metadata=initialize_attack_policy({'state_dict':old.state_dict()},
        num_scalars=NUM_SCALARS+3,observation_contract='joint-test-v1')
    assert metadata['target_scalar_count']==NUM_SCALARS+3
    assert metadata['appended_scalar_columns']==3
    assert metadata['observation_contract']=='joint-test-v1'
    assert torch.count_nonzero(network.scalar[0].weight[:,-3:])==0
    planes=torch.rand(1,NUM_PLANES,32,18);scalars=torch.rand(1,NUM_SCALARS)
    with torch.no_grad():
        a=old(planes,scalars)
        b=network(planes,torch.cat((scalars,torch.tensor([[.01,1.,1.]])),dim=1))
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
    with pytest.raises(ValueError,match='contract'):
        load_attack_state(network,{'state_dict':network.state_dict(),'config':{
            'observation_contract':'wrong'}},exact=True,observation_contract='joint-test-v1')


def test_hard_window_termination_and_technical_timeout_bootstrap_differ():
    from sim.train_attack_ppo import generalized_advantage
    rewards=torch.tensor([[1.,1.]])
    values=torch.tensor([[2.,2.]])
    next_values=torch.tensor([[10.,10.]])
    terminal=torch.tensor([[True,False]])
    truncated=torch.tensor([[False,True]])
    _,returns=generalized_advantage(rewards,values,next_values,terminal,truncated,gamma=.9)
    torch.testing.assert_close(returns,torch.tensor([[1.,10.]]))


def test_real_joint_309_optimizer_and_exact_boundary_resume(tmp_path):
    from sim.strategy_env import StrategyEnv,OBSERVATION_CONTRACT as JOINT_CONTRACT
    from sim.strategy_sampler import StrategyScenarioCycle
    torch.set_num_threads(1)
    sample_env=StrategyEnv(max_ms=500)
    for seed in range(20):
        sampler=StrategyScenarioCycle(sample_env.scenarios,seed=seed)
        first=[sampler.next()['scenario_id'].split('::')[0]for _ in range(2)]
        if set(first)=={'attack','defense'}:break
    else:raise AssertionError('No mixed two-seat seeded batch found')
    def arguments(output,updates):
        return parser().parse_args(['--output',str(output),'--device','cpu','--task-mode','joint',
            '--updates',str(updates),'--seed',str(seed),'--envs','2','--rollout','1','--epochs','1',
            '--minibatch-size','2','--eval-every','0','--max-ms','500',
            '--target-environment-steps','4','--deadline-scale','3'])
    one=train(arguments(tmp_path/'joint-one',1))
    first=torch.load(one/'update_000001.pt',map_location='cpu',weights_only=False)
    assert first['state_dict']['scalar.0.weight'].shape[1]==NUM_SCALARS+3
    assert first['config']['observation_contract']==JOINT_CONTRACT
    assert first['sampler']['schema']=='joint-balanced-scenario-cycle-v1'
    metrics=json.loads((one/'metrics.jsonl').read_text())
    assert {episode['task']for episode in metrics['episodes']}=={'attack','defense'}
    assert set(metrics['by_task'])=={'attack','defense'}
    assert first['optimiser']['state']
    continuous=train(arguments(tmp_path/'joint-continuous',2))
    args=arguments(tmp_path/'joint-resume',2);args.resume=str(one/'update_000001.pt')
    resumed=train(args)
    a=torch.load(continuous/'update_000002.pt',map_location='cpu',weights_only=False)
    b=torch.load(resumed/'update_000002.pt',map_location='cpu',weights_only=False)
    for key in a['state_dict']:torch.testing.assert_close(a['state_dict'][key],b['state_dict'][key],rtol=0,atol=0)
    assert a['sampler']==b['sampler']
    assert a['infos'][0]['metrics']==b['infos'][0]['metrics']
    assert a['episode_returns']==b['episode_returns']
    result=json.loads((resumed/'training_result.json').read_text())
    assert result['target_met'] and not result['quality_accepted']
