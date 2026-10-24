"""Transport failures must never become stale or fabricated coordinates."""
import json
from types import SimpleNamespace

import pytest

from scripts.gamestate.arm64_entities import CaptureError
from scripts.gamestate.query_arm64 import CurrentStateReader


def reader_with_response(monkeypatch, payload, code=0):
    reader = CurrentStateReader.__new__(CurrentStateReader)
    reader.command = ['test-reader']
    monkeypatch.setattr('scripts.gamestate.query_arm64.subprocess.run',
                        lambda *a, **kw: SimpleNamespace(
                            stdout=json.dumps(payload).encode(), stderr=b'', returncode=code))
    return reader


def test_adb_exec_out_can_mask_native_no_battle_exit(monkeypatch):
    reader = reader_with_response(monkeypatch, {'status': 'no_battle'})
    assert reader.query()['status'] == 'no_battle'
    assert reader.query()['units'] == []


def test_torn_snapshot_is_rejected_not_retried_or_accepted(monkeypatch):
    reader = reader_with_response(monkeypatch, {'sequence': 0, 'rejected': 14})
    row = reader.query()
    assert row['status'] == 'rejected'
    assert row['units'] == []


def test_units_filter_preserves_engine_side_and_coordinates(monkeypatch):
    reader = reader_with_response(monkeypatch, {'entity_count': 2, 'entities': [
        {'entity_id': 5000007, 'side': 1, 'x': 4559, 'y': 22363, 'hp': 178},
        {'entity_id': 4000007, 'side': 0, 'x': 9000, 'y': 3000, 'hp': -1}]})
    row = reader.query()
    assert len(row['units']) == 1
    assert row['units'][0]['side'] == 1
    assert row['units'][0]['x_tiles'] == 4.559
    assert row['units'][0]['y_tiles'] == 22.363
    assert len(row['frame']['entities']) == 2


def test_count_mismatch_fails_closed(monkeypatch):
    reader = reader_with_response(monkeypatch, {'entity_count': 1, 'entities': []})
    with pytest.raises(CaptureError, match='count mismatch'):
        reader.query()


def test_memory_snapshot_includes_separate_mathematical_elixir(monkeypatch):
    from scripts.gamestate.elixir import ElixirTracker
    reader = reader_with_response(monkeypatch, {'entity_count': 0, 'entities': []})
    tracker = ElixirTracker.from_match_start()
    tracker.confirm_play(event_id='hog-1', cost=4, game_seconds=0)
    row = reader.query_with_elixir(tracker, game_seconds=2.8)
    assert row['own_elixir']['value'] == 2
    assert row['own_elixir']['source'] == 'mathematical_standard_mode'
    assert 'own_elixir' not in row['frame']


def test_rejected_snapshot_has_no_usable_elixir(monkeypatch):
    from scripts.gamestate.elixir import ElixirTracker
    reader = reader_with_response(monkeypatch, {'rejected': 14})
    tracker = ElixirTracker.from_match_start()
    assert reader.query_with_elixir(tracker, game_seconds=2.8)['own_elixir'] is None
    assert tracker.estimate().game_seconds == 0


def test_query_exports_pending_budget_and_delayed_confirmations(monkeypatch):
    from scripts.gamestate.elixir import ElixirTracker
    reader = reader_with_response(monkeypatch, {'entity_count': 0, 'entities': []})
    tracker = ElixirTracker.from_match_start()
    tracker.reserve_play(action_id='tap-1', cost=4, game_seconds=0)
    row = reader.query_with_elixir(tracker, game_seconds=2.8)
    assert row['own_elixir']['value'] == 6
    assert row['own_elixir']['available_minimum'] == 2
    assert row['own_elixir']['pending_actions'] == 1
    tracker.confirm_play(event_id='accepted-1', action_id='tap-1',
                         cost=4, game_seconds=0)
    tracker.confirm_gain(event_id='credit-1', amount=1, game_seconds=2.8)
    row = reader.query_with_elixir(tracker, game_seconds=2.8)
    assert row['own_elixir']['value'] == 3
    assert row['own_elixir']['reserved'] == 0
    assert row['own_elixir']['available_minimum'] == 3


def test_own_state_mode_selects_new_binary_without_changing_default(monkeypatch):
    from scripts.gamestate.query_arm64 import BINARY, STATE_BINARY
    monkeypatch.setattr('scripts.gamestate.query_arm64.Memory', lambda adb, serial:
                        SimpleNamespace(adb=[adb,'-s',serial],pid=1,libg_base=0x100000))
    normal = CurrentStateReader('adb','serial')
    private = CurrentStateReader('adb','serial',own_state=True)
    assert BINARY in normal.command
    assert normal.command[-1] == '--auto-once'
    assert STATE_BINARY in private.command
    assert private.command[-1] == '--auto-state-once'


def test_private_state_is_forwarded_without_inventing_missing_values(monkeypatch):
    state = {'status':'ok','world':'0x10000','tick':50,'own_side':1,
             'elixir_raw':60000,'hand_indices':[0,1,2,3],'cycle_indices':[4,5,6,7]}
    reader = reader_with_response(monkeypatch, {'entity_count':0,'entities':[], 'own_state':state,'battle_tick':50})
    assert reader.query()['own_state'] == state
