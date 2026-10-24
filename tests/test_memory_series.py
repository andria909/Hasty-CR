from scripts.gamestate.memory_series import terminal_result


def king(side, hp):
    return dict(name='KingTower',side=side,hp=hp,entity_id=10+side)


def tower(identity, side, hp=100):
    return dict(name='PrincessTower',side=side,hp=hp,entity_id=identity)


def test_king_result_uses_explicit_own_side():
    frame={'units':[king(0,0),king(1,100)]}
    assert terminal_result(frame,0,{},'king_destroyed')['result']=='loss'
    assert terminal_result(frame,1,{},'king_destroyed')['result']=='win'


def test_frozen_crowns_count_removed_tower_bodies():
    initial={1:0,2:0,3:1,4:1}
    frame={'units':[king(0,100),king(1,100),tower(1,0),tower(2,0),tower(3,1)]}
    reason='tick_frozen_hand_unavailable'
    assert terminal_result(frame,1,initial,reason)['result']=='loss'
    assert terminal_result(frame,0,initial,reason)['result']=='win'
    assert terminal_result(frame,0,initial,'world_unloaded')['result']=='unknown'


def test_equal_crowns_and_double_king_zero_never_guess_winner():
    initial={1:0,2:0,3:1,4:1}
    frame={'units':[king(0,100),king(1,100),tower(1,0),tower(3,1)]}
    assert terminal_result(frame,0,initial,'tick_frozen_hand_unavailable')['result']=='unknown'
    assert terminal_result({'units':[king(0,0),king(1,0)]},0,initial,'king_destroyed')['result']=='unknown'


def test_queue_retry_requires_matching_verified_lobby_button():
    from io import BytesIO
    from PIL import Image
    from scripts.gamestate.memory_series import lobby_button_matches
    def png(color):
        image=Image.new('RGB',(1440,2560),color)
        stream=BytesIO(); image.save(stream,format='PNG'); return stream.getvalue()
    lobby=png((255,190,10))
    assert lobby_button_matches(lobby,lobby)
    assert not lobby_button_matches(lobby,png((40,90,180)))


def test_remaining_runner_preserves_selected_checkpoint(tmp_path):
    from scripts.gamestate.memory_series import remaining_runner_command
    checkpoint = tmp_path / 'ppo300.pt'
    checkpoint.write_bytes(b'trusted-test-fixture')
    for remaining in range(1, 6):
        command = remaining_runner_command('python', adb='adb', serial='device',
            checkpoint=checkpoint, remaining=remaining, output=tmp_path/'run')
        assert command[command.index('--checkpoint')+1] == str(checkpoint.resolve())
        assert command[command.index('--max-matches')+1] == str(remaining)
    import pytest
    with pytest.raises(ValueError):
        remaining_runner_command('python', adb='adb', serial='device',
            checkpoint=tmp_path/'missing.pt', remaining=5, output=tmp_path/'run')
