"""Conservative semantics for the experimental reader, without a live device."""
import subprocess

import pytest

from scripts.gamestate.arm64_entities import (
    CaptureError, FirstObservations, Memory, pointer,
)


def frame(*entities):
    return {"sample_start_ns": 100, "sample_end_ns": 200, "entities": list(entities)}


def entity(number=1, address="0x10000", name="Knight"):
    return {"entity_id": number, "address": address, "name": name,
            "side": 1, "x": 9500, "y": 22500}


def test_attaching_mid_match_is_not_a_placement():
    tracker = FirstObservations()
    assert tracker.update(frame(entity())) == []
    assert tracker.update(frame(entity())) == []
    changed = entity()
    changed["x"] += 100
    assert tracker.update(frame(changed)) == []


def test_reused_address_with_new_identity_is_observed_again():
    tracker = FirstObservations()
    tracker.update(frame(entity()))
    events = tracker.update(frame(entity(number=2)))
    assert len(events) == 1
    assert events[0]["entity_id"] == 2
    assert events[0]["placement_confirmed"] is False
    assert events[0]["event"] == "entity_first_observed"


def test_two_spawned_bodies_are_not_collapsed_to_one_card_command():
    tracker = FirstObservations()
    tracker.update(frame())
    events = tracker.update(frame(entity(1, "0x10000", "Archer"),
                                  entity(2, "0x20000", "Archer")))
    assert len(events) == 2
    assert all(not event["placement_confirmed"] for event in events)


def test_tagged_android_pointers_preserve_the_address():
    assert pointer(0xB400007D30449C00) == 0x7D30449C00


def test_short_batch_fails_instead_of_shifting_following_fields(monkeypatch):
    memory = object.__new__(Memory)
    memory.adb, memory.pid = ["adb"], 1
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: b"12345678")
    with pytest.raises(CaptureError, match="Short batch"):
        memory.batch([(0x10000, 8), (0x20000, 8)])


def test_names_fail_closed_for_non_names():
    with pytest.raises(CaptureError, match="Invalid internal"):
        Memory.decode_name(b"Knight\x00reused")


@pytest.mark.parametrize('version', ['160402020', '160635001'])
def test_each_supported_build_keeps_its_exact_hash_guard(monkeypatch, version):
    from scripts.gamestate.arm64_entities import BUILD_PROFILES
    digest = BUILD_PROFILES[version]['sha256']
    def shell(self, command):
        if command == 'id -u': return '0'
        if command.startswith('getprop'): return 'arm64-v8a'
        if command.startswith('dumpsys'): return 'versionCode=' + version
        if command.startswith('pidof'): return '123'
        if command.startswith('cat /proc/'): return '100000-200000 r-xp 00000000 00:00 1 /app/lib/arm64/libg.so'
        if command.startswith('sha256sum'): return digest + ' libg.so'
        raise AssertionError(command)
    monkeypatch.setattr(Memory, 'shell', shell)
    assert Memory('adb', 'serial').version == version
    digest = '0' * 64
    with pytest.raises(CaptureError, match='hash differs'):
        Memory('adb', 'serial')
