"""Changing privilege transport must not change the state contract or guards."""
import json
import shlex
from types import SimpleNamespace

import pytest

from scripts.gamestate.arm64_entities import CaptureError, LIBG_SHA256, Memory
from scripts.gamestate.query_arm64 import CurrentStateReader, STATE_BINARY
from scripts.gamestate.query_arm64_v2 import main


SU = "/system/xbin/su"
LIB = "/data/app/com.supercell.clashroyale/lib/arm64/libg.so"


def attach(monkeypatch, *, su=SU, uid="0", version="160402020", digest=LIBG_SHA256):
    seen = []

    def output(argv, **kwargs):
        seen.append(argv)
        command = argv[-1]
        if su:
            parts = shlex.split(command)
            assert parts[:2] == [su, "-c"]
            assert len(parts) == 3
            command = parts[2]
        answers = {
            "id -u": uid + "\n",
            "getprop ro.product.cpu.abi": "arm64-v8a\n",
            "dumpsys package com.supercell.clashroyale": f"versionCode={version} minSdk=32",
            "pidof com.supercell.clashroyale": "42\n",
            "cat /proc/42/maps": f"100000-200000 r-xp 00000000 00:00 1 {LIB}\n",
            "sha256sum " + LIB: digest + "  " + LIB,
        }
        return answers[command]

    monkeypatch.setattr("scripts.gamestate.arm64_entities.subprocess.check_output", output)
    return Memory("adb", "device", su=su), seen


def test_su_applies_to_identity_maps_and_hash_checks(monkeypatch):
    memory, commands = attach(monkeypatch)
    assert memory.pid == 42 and memory.libg_base == 0x100000
    assert len(commands) == 6
    assert all(command[:4] == ["adb", "-s", "device", "shell"] for command in commands)


@pytest.mark.parametrize("bad", ["su", "/system/xbin/su;id", "/system/xbin/su -c", ""])
def test_su_must_be_a_path_not_an_injected_command(bad):
    with pytest.raises(CaptureError, match="absolute executable path"):
        Memory("adb", "device", su=bad)


@pytest.mark.parametrize("kwargs,match", [
    ({"uid": "2000"}, "Root access is required"),
    ({"version": "160402021"}, "version differs"),
    ({"digest": "0" * 64}, "hash differs"),
])
def test_su_preserves_fail_closed_guards(monkeypatch, kwargs, match):
    with pytest.raises(CaptureError, match=match):
        attach(monkeypatch, **kwargs)


def test_native_reader_and_binary_batch_reads_use_su(monkeypatch):
    memory, _ = attach(monkeypatch)
    monkeypatch.setattr("scripts.gamestate.query_arm64.Memory", lambda *a, **k: memory)
    reader = CurrentStateReader("adb", "device", own_state=True, su=SU)
    native = shlex.split(reader.command[-1])
    assert reader.command[:4] == ["adb", "-s", "device", "exec-out"]
    assert native[:2] == [SU, "-c"]
    assert shlex.split(native[2]) == [STATE_BINARY, "42", "0x100000", "0x1a36450",
                                    "1", "20", "--auto-state-once"]
    commands = []

    def output(argv, **kwargs):
        commands.append(argv)
        return b"abcdefgh"

    monkeypatch.setattr("scripts.gamestate.arm64_entities.subprocess.check_output", output)
    assert memory.batch([(0x10000, 4), (0x20000, 4)]) == [b"abcd", b"efgh"]
    assert shlex.split(commands[0][-1])[:2] == [SU, "-c"]
    assert "dd if=/proc/42/mem" in shlex.split(commands[0][-1])[2]


@pytest.mark.parametrize("payload,code,status", [
    ({"status": "no_battle"}, 4, "no_battle"),
    ({"rejected": 14}, 3, "rejected"),
    ({"entity_count": 2, "entities": [
        {"entity_id": 5000007, "name": "HogRider", "side": 1, "x": 4559,
         "y": 22363, "hp": 178, "max_hp": 1000, "card_id": 26000021,
         "level_index": 10, "address": "0x10000"},
        {"entity_id": 4000007, "name": "Fireball", "side": 0, "x": 9000,
         "y": 3000, "hp": -1, "max_hp": -1, "card_id": 28000000,
         "level_index": None, "address": "0x20000"}],
      "battle_tick": 50, "own_state": {"status": "ok", "tick": 50,
        "own_side": 1, "hand_indices": [0, 1, 2, 3], "cycle_indices": [4, 5, 6, 7],
        "elixir_raw": 60000, "world": "0x30000"}}, 0, "ok"),
])
def test_json_semantics_identical_across_transports(monkeypatch, payload, code, status):
    monkeypatch.setattr("scripts.gamestate.query_arm64.subprocess.run", lambda *a, **k:
                        SimpleNamespace(stdout=json.dumps(payload).encode(), stderr=b"",
                                        returncode=code))
    readers = [CurrentStateReader.__new__(CurrentStateReader) for _ in range(2)]
    readers[0].command = ["adb", "exec-out", STATE_BINARY]
    readers[1].command = ["adb", "exec-out", shlex.join([SU, "-c", STATE_BINARY])]
    rows = [r.query() for r in readers]
    dynamic = {"request_host_ns", "received_host_ns", "round_trip_ms"}
    assert {k: v for k, v in rows[0].items() if k not in dynamic} == {
        k: v for k, v in rows[1].items() if k not in dynamic}
    assert rows[0]["status"] == status
    assert rows[0]["frame"] == payload
    if status == "ok":
        assert rows[0]["units"][0]["x_tiles"] == 4.559
        assert len(rows[0]["units"]) == 1
        assert rows[0]["own_state"] == payload["own_state"]


def test_cli_forwards_su_without_changing_output(monkeypatch, capsys):
    seen = {}

    def create(adb, serial, binary, own_state, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(query=lambda: {"status": "no_battle", "units": [],
                                              "frame": {"status": "no_battle"}})

    monkeypatch.setattr("scripts.gamestate.query_arm64_v2.create_reader", create)
    assert main(["--adb", "adb", "--serial", "device", "--su", SU, "--own-state"]) == 0
    assert seen == {"su": SU}
    assert json.loads(capsys.readouterr().out)["status"] == "no_battle"
