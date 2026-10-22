import pytest

from scripts.kaggle.collect_defense_stage2 import selected_files


def test_only_reported_checkpoints_are_collected_and_resume_hash_does_not_erase_verified_hash():
    result = {'ppo': dict(checkpoint='ppo/update_002000.pt', sha256='a'*64,
                          selected_checkpoint='ppo/best_validation_001000.pt', selected_sha256='b'*64,
                          resume_checkpoint='ppo/update_002000.pt')}
    assert selected_files(result) == {'ppo/update_002000.pt': 'a'*64,
                                      'ppo/best_validation_001000.pt': 'b'*64}


@pytest.mark.parametrize('path', ['../secret.pt', '/ppo/a.pt', 'grpo/a.pt',
                                 'ppo/../a.pt', 'ppo/a.json', 'ppo//a.pt'])
def test_untrusted_result_paths_cannot_download_other_files(path):
    with pytest.raises(ValueError, match='Unsafe'):
        selected_files({'ppo': dict(checkpoint=path, sha256='a'*64)})


def test_reported_weight_hash_is_required_and_conflicting_aliases_fail():
    with pytest.raises(ValueError, match='Missing'):
        selected_files({'ppo': dict(checkpoint='ppo/a.pt')})
    with pytest.raises(ValueError, match='Conflicting'):
        selected_files({'ppo': dict(checkpoint='ppo/a.pt', sha256='a'*64,
                                   selected_checkpoint='ppo/a.pt', selected_sha256='b'*64)})
