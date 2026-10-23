from dataclasses import replace

import pytest

from scripts.gamestate.memory_play import VERIFIED_DECK, own_context, tap_points
from scripts.gamestate.model_bridge import MissingModelInput, NativeModelAdapter
from tests.test_memory_model_bridge import fixture


def test_verified_deck_order_is_distinct_from_seen_card_history():
    private = dict(status='ok', tick=4196, own_side=0, elixir_raw=57104,
                   hand_indices=[0,4,6,2], cycle_indices=[1,3,7,5])
    own = own_context(private)
    assert own.hand == ('hog_rider','electro_spirit','skeletons','zap')
    assert own.next_card == 'cannon'
    assert own.game_seconds == 209.8
    assert own.elixir.value == 5.7104
    assert own.elixir.source == 'arm64_memory'
    assert not own.public_history_complete


def test_private_hand_transition_is_never_filled_in():
    with pytest.raises(MissingModelInput):
        own_context(dict(status='transition_or_ambiguous'))
    with pytest.raises(MissingModelInput):
        own_context(dict(status='ok', hand_indices=[-1,1,2,3], cycle_indices=[4,5,6,7]))


def test_incomplete_public_history_requires_explicit_trial_mode():
    _, snapshot, own = fixture()
    own = replace(own, public_history_complete=False,
                  elixir=replace(own.elixir, source='arm64_memory'))
    default = NativeModelAdapter(own_side=0).encode(snapshot, own)
    assert not default.metadata['ready_for_inference']
    trial = NativeModelAdapter(own_side=0, allow_incomplete_history=True).encode(snapshot, own)
    assert trial.metadata['ready_for_inference']
    assert trial.metadata['warnings'] == ['incomplete_opponent_play_history']
    assert trial.observation['scalars'][0] == .5


def test_calibrated_tap_points_are_scaled_to_actual_android_resolution():
    select, target = tap_points(0,9,21,1440,2560)
    assert select == (445,2281)
    assert target == (750,1388)
    with pytest.raises(ValueError):
        tap_points(4,9,21,1440,2560)


def test_trial_unknown_profile_retains_measured_hp_and_marks_missing_traits():
    _, snapshot, own = fixture()
    snapshot['units'][-1]['name'] = 'GoblinHut_Rework'
    view = NativeModelAdapter(own_side=0, allow_unknown_profiles=True).encode(snapshot, own)
    assert 'unknown_tactical_profile:GoblinHut_Rework' in view.metadata['warnings']
    assert view.observation['planes'][1].sum() == 2  # unknown body is retained
    assert view.observation['planes'][3].sum() > 0


def test_real_incoming_push_is_not_masked_as_enemy_back_line():
    import json
    from pathlib import Path
    snapshot=json.loads(Path('data/validation/memory_v25_idle_20261022.json').read_text())
    context=own_context(snapshot['own_state'],destroyed=((1,'right'),))
    legacy=NativeModelAdapter(own_side=1,allow_incomplete_history=True,
                              allow_unknown_profiles=True).encode(snapshot,context)
    physical=NativeModelAdapter(own_side=1,allow_incomplete_history=True,
                                allow_unknown_profiles=True,
                                coordinate_mode='physical_live').encode(snapshot,context)
    # Cannon, Musketeer and Knight were all wrongly disabled in the real stall.
    for slot in (0,1,3):
        assert not legacy.action_mask[1+slot*576:1+(slot+1)*576].any()
        assert physical.action_mask[1+slot*576:1+(slot+1)*576].any()
    assert physical.observation['planes'][1,16:,:].sum() > 0
    assert physical.metadata['coordinate_mode']=='physical_live'


def test_diagnostic_screenshot_brackets_engine_time_without_using_cv(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts.gamestate.memory_play import diagnostic_screenshot
    frames=iter([{'frame':{'battle_tick':100}},{'frame':{'battle_tick':103}}])
    reader=SimpleNamespace(query=lambda:next(frames))
    monkeypatch.setattr('scripts.gamestate.memory_play.subprocess.check_output',lambda *a,**k:b'PNG')
    result=diagnostic_screenshot(reader,['adb'],tmp_path,'sample',phase='after_ack',
                                  input_host_ns=1,source_tick=80,ack_host_ns=2)
    assert result['engine_tick_before_capture']==100
    assert result['engine_tick_after_capture']==103
    assert (tmp_path/'sample.png').read_bytes()==b'PNG'
    assert result['capture_ms']>=0
