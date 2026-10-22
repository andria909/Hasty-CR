from copy import deepcopy
import json

import pytest

from sim.defense_log_summary import compact_stdout_lines,summarize_episodes,summarize_update


def episode(difficulty='simple',loss=0,dead=False):
    return dict(difficulty=difficulty,no_tower_damage=loss==0,
                own_tower_hp_loss=dict(left=loss,right=0,king=0),
                own_tower_fraction_loss=dict(left=loss/1000,right=0,king=0),
                own_towers_alive=dict(left=not dead,right=True,king=True),
                terminal_reason='cleared')


def test_complete_outcomes_have_counts_percent_losses_and_difficulty_breakdown():
    records=[episode(),episode('medium',loss=100,dead=True)]
    summary=summarize_episodes(records)
    assert summary['n']==2 and summary['no_damage']==1
    assert summary['no_damage_pct']==50 and summary['no_damage_known']==2
    assert summary['mean_hp_loss']==50 and summary['mean_total_tower_fraction_loss']==pytest.approx(.05)
    assert summary['princess_deaths']==1 and summary['princess_known']==4
    assert summary['king_deaths']==0 and summary['king_survival_pct']==100
    assert summary['timeouts']==summary['invalid']==0
    assert summary['mean_return'] is None # No episode reward was recorded.
    assert summary['by_difficulty']['simple']['no_damage_pct']==100
    assert summary['by_difficulty']['medium']['no_damage_pct']==0


def test_partial_metrics_use_known_denominators_and_do_not_call_unknown_zero_damage():
    partial=dict(own_tower_hp_loss={'left':10},own_towers_alive={'left':False},terminal_reason='invalid_script_play_timeout')
    summary=summarize_episodes([{},partial,episode()])
    assert summary['n']==3 and summary['no_damage_known']==2
    assert summary['no_damage']==1 and summary['no_damage_pct']==50
    assert summary['hp_loss_known']==1 and summary['mean_hp_loss']==0
    assert summary['princess_deaths']==1 and summary['princess_known']==3
    assert summary['king_known']==1
    assert summary['terminal_known']==2 and summary['invalid']==1
    assert summarize_episodes([{'own_tower_hp_loss':{'left':0}}])['no_damage'] is None


def test_empty_summary_has_zero_episode_count_and_null_unobserved_metrics():
    summary=summarize_episodes([])
    assert summary['n']==0 and summary['by_difficulty']=={}
    for name in ['mean_hp_loss','no_damage','no_damage_pct','mean_return','return_std',
                 'princess_deaths','king_deaths','timeouts','invalid']:
        assert summary[name] is None


def test_ppo_training_return_remains_unknown_but_validation_uses_real_recorded_rewards():
    record=dict(update=1,step=16,episodes=[episode()],validation=dict(
        episodes=[dict(metrics=episode(),reward=3),dict(metrics=episode(),reward=7)],mean_reward=5))
    summary=summarize_update(record,'PPO')
    assert summary['train']['mean_return'] is None
    assert summary['validation']['mean_return']==5
    assert summary['validation']['return_std']==2
    assert summarize_episodes([dict(reward=999,difficulty='simple')])['mean_return'] is None
    lines=[json.loads(line)for line in compact_stdout_lines(record,'PPO')]
    assert [line['event']for line in lines]==['train','validation']
    assert lines[0]['episodes']['n']==1 and lines[1]['episodes']['n']==2


def test_grpo_raw_returns_remain_visible_and_validation_mean_does_not_invent_std():
    record=dict(update=2,environment_steps=64,metrics=[episode(),episode('medium')],
                returns=[-10,0],loss=.25,kl=.01,degenerate=False,
                validation=dict(metrics=[episode()],mean_return=3))
    result=summarize_update(record,'GRPO')
    assert result['train']['mean_return']==-5 and result['train']['return_std']==5
    assert result['validation']['mean_return']==3 and result['validation']['return_std'] is None
    missing=summarize_episodes([],returns=[1,3])
    assert missing['n']==0 and missing['mean_return']==2
    assert not missing['return_alignment_complete']


def test_huge_sampler_and_validation_cannot_oversize_stdout_or_mutate_jsonl_record():
    record=dict(update=22,environment_steps=128,metrics=[episode()for _ in range(8)],
                returns=[1,2,3,4,5,6,7,8],loss=.000123,kl=.0123456789,degenerate=False,
                sampler=dict(scenario_ids=[f'full-large-curriculum-scenario-{i}'for i in range(1344)]),
                validation=dict(metrics=[episode()for _ in range(100)],mean_return=1))
    original=deepcopy(record)
    lines=compact_stdout_lines(record,'GRPO')
    assert len(lines)==2 and all(len(line)<1000 for line in lines)
    assert 'sampler' not in lines[0] and 'scenario_ids' not in lines[0]
    assert 'mean_return' in lines[0][:150] and 'loss' in lines[0][:200]
    assert record==original


def test_nonfinite_values_and_incomplete_episode_metadata_do_not_emit_nan_json():
    lines=compact_stdout_lines(dict(update=1,step=8,loss=float('nan'),episodes=[]),'PPO')
    line=json.loads(lines[0])
    assert line['loss'] is None and line['episodes']['n']==0
    assert line['episodes']['no_damage_pct'] is None
