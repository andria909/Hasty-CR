import subprocess

import pytest

from scripts.gamestate.arm64_entities import CaptureError
from scripts.gamestate.query_arm64_v2 import create_reader, resolve_adb


def test_v2_reports_missing_game_process_without_traceback(monkeypatch):
    def no_process(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1, ["adb", "-s", "emulator-5556", "shell", "pidof com.supercell.clashroyale"])

    monkeypatch.setattr("scripts.gamestate.query_arm64_v2.CurrentStateReader", no_process)
    with pytest.raises(CaptureError, match="Clash Royale is not running"):
        create_reader("adb", "emulator-5556", "/data/local/tmp/reader", False)


def test_v2_reports_root_or_adb_failure_cleanly(monkeypatch):
    def root_failure(*args, **kwargs):
        raise CaptureError("ADB must already be root; this tool changes no device settings")

    monkeypatch.setattr("scripts.gamestate.query_arm64_v2.CurrentStateReader", root_failure)
    with pytest.raises(CaptureError, match="ADB must already be root"):
        create_reader("adb", "emulator-5554", "/data/local/tmp/reader", False)


def test_v2_prefers_explicit_adb_path():
    assert resolve_adb("/custom/platform-tools/adb") == "/custom/platform-tools/adb"


def test_v2_discovers_adb_from_android_home(tmp_path, monkeypatch):
    adb = tmp_path / "platform-tools" / "adb"
    adb.parent.mkdir()
    adb.touch()
    monkeypatch.setenv("ANDROID_HOME", str(tmp_path))
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    monkeypatch.delenv("ANDROID_ADB", raising=False)
    assert resolve_adb() == str(adb)
