import json

import pytest

from scripts.defense.summarize_training import summarize_file


def test_train_windows_validation_and_missing_rewards_stay_separate(tmp_path):
    source = tmp_path/'metrics.jsonl'
    clean = dict(no_tower_damage=True, difficulty='simple',
                 own_tower_hp_loss=dict(left=0, right=0, king=0))
    hit = dict(no_tower_damage=False, difficulty='complex',
               own_tower_hp_loss=dict(left=100, right=0, king=0))
    records = [dict(update=1, episodes=[clean], validation=dict(
        mean_reward=-3, episodes=[dict(reward=-3, metrics=hit)])),
        dict(update=101, episodes=[hit])]
    source.write_text(''.join(json.dumps(r)+'\n' for r in records))
    result = summarize_file(source, 'PPO')
    assert result['train']['n'] == 2
    assert result['train']['no_damage_pct'] == 50
    assert result['train']['mean_hp_loss'] == 50
    assert result['train']['mean_return'] is None
    assert len(result['windows']) == 2
    assert result['validation'][0]['summary']['mean_return'] == -3
    assert result['validation'][0]['summary']['no_damage'] == 0
    assert result['train']['by_difficulty']['simple']['no_damage_pct'] == 100


def test_grpo_real_returns_and_duplicate_updates_fail(tmp_path):
    source = tmp_path/'updates.jsonl'
    record = dict(update=1, metrics=[{}, {}], returns=[-4, 2])
    source.write_text(json.dumps(record)+'\n')
    assert summarize_file(source, 'GRPO')['train']['mean_return'] == -1
    source.write_text((json.dumps(record)+'\n')*2)
    with pytest.raises(ValueError, match='Non-increasing'):
        summarize_file(source, 'GRPO')
    record['returns'] = [1]
    source.write_text(json.dumps(record)+'\n')
    with pytest.raises(ValueError, match='Return/episode mismatch'):
        summarize_file(source, 'GRPO')
