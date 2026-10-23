"""Experimental, read-only entity capture for the pinned macOS ARM64 client.

An entity's first observed position is NOT a confirmed card placement command.
Pass a collection or a known live entity address from this process. Addresses
are session-specific; this tool does not bypass the older x86 reader's guards.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import time
from pathlib import Path

PACKAGE = "com.supercell.clashroyale"
VERSION = "160402020"
LIBG_SHA256 = "347afc57c8d168bb67d0defa488a89f76297025cac1e8544b5182bb7e70ff433"
PTR_MASK = 0x00FFFFFFFFFFFFFF


def pointer(value: int) -> int:
    return value & PTR_MASK


def u64(blob: bytes, offset: int = 0) -> int:
    return struct.unpack_from("<Q", blob, offset)[0]


class CaptureError(RuntimeError):
    pass


class Memory:
    def __init__(self, adb: str, serial: str):
        self.adb = [adb, "-s", serial]
        if self.shell("id -u").strip() != "0":
            raise CaptureError("ADB must already be root; this tool changes no device settings")
        if self.shell("getprop ro.product.cpu.abi").strip() != "arm64-v8a":
            raise CaptureError("Only arm64-v8a is supported")
        version = self.shell(f"dumpsys package {PACKAGE}")
        if not re.search(rf"versionCode={VERSION}\b", version):
            raise CaptureError("Client version differs from the pinned layout")
        self.pid = int(self.shell(f"pidof {PACKAGE}").strip())
        self.maps = self.shell(f"cat /proc/{self.pid}/maps")
        libmaps = [line.split() for line in self.maps.splitlines()
                   if line.endswith("/lib/arm64/libg.so")]
        if not libmaps:
            raise CaptureError("libg.so is not mapped")
        self.libg_base = min(int(row[0].split("-")[0], 16) - int(row[2], 16)
                             for row in libmaps)
        libpath = libmaps[0][-1]
        digest = subprocess.check_output(self.adb + ["shell", "sha256sum", libpath],
                                         text=True, timeout=15).split()[0]
        if digest != LIBG_SHA256:
            raise CaptureError("libg.so hash differs from the pinned layout")
        self.name_cache: dict[int, str] = {}

    def shell(self, command: str) -> str:
        return subprocess.check_output(self.adb + ["shell", command],
                                       text=True, timeout=15)

    def batch(self, requests: list[tuple[int, int]]) -> list[bytes]:
        if not requests:
            return []
        if any(pointer(a) < 0x10000 or not 0 < n <= 65536 for a, n in requests):
            raise CaptureError("Invalid memory read request")
        command = "; ".join(
            f"dd if=/proc/{self.pid}/mem bs=4096 iflag=skip_bytes,count_bytes "
            f"skip={pointer(a)} count={n} 2>/dev/null" for a, n in requests)
        blob = subprocess.check_output(self.adb + ["exec-out", command], timeout=10)
        if len(blob) != sum(n for _, n in requests):
            raise CaptureError("Short batch read; frame rejected rather than zero-filled")
        result, at = [], 0
        for _, size in requests:
            result.append(blob[at:at + size])
            at += size
        return result

    def read(self, address: int, size: int) -> bytes:
        return self.batch([(address, size)])[0]

    def names(self, addresses: list[int]) -> dict[int, str]:
        missing = list(dict.fromkeys(pointer(a) for a in addresses
                                     if pointer(a) not in self.name_cache))
        headers = self.batch([(a, 64) for a in missing])
        external: list[tuple[int, int]] = []
        external_keys = []
        for address, blob in zip(missing, headers):
            size = struct.unpack_from("<I", blob, 0x2C)[0]
            if not 0 < size < 128:
                raise CaptureError("Invalid entity data-name length")
            if size < 8:
                raw = blob[0x30:0x30 + size]
                self.name_cache[address] = self.decode_name(raw)
            else:
                external.append((u64(blob, 0x30), size))
                external_keys.append(address)
        for key, raw in zip(external_keys, self.batch(external)):
            self.name_cache[key] = self.decode_name(raw)
        return self.name_cache

    @staticmethod
    def decode_name(raw: bytes) -> str:
        if not re.fullmatch(rb"[A-Za-z0-9_]+", raw):
            raise CaptureError("Invalid internal entity name")
        return raw.decode("ascii")


class EntityReader:
    def __init__(self, memory: Memory, collection: int):
        self.memory = memory
        self.collection = pointer(collection)

    def frame(self) -> dict:
        start = time.monotonic_ns()
        head = self.memory.read(self.collection, 24)
        array = u64(head, 8)
        capacity, count = struct.unpack_from("<ii", head, 0x10)
        if not 0 <= count <= capacity <= 1024:
            raise CaptureError("Invalid collection size; collection may have expired")
        pointers = []
        if count:
            raw = self.memory.read(array, count * 8)
            pointers = [pointer(u64(raw, i * 8)) for i in range(count)]
        if len(set(pointers)) != len(pointers):
            raise CaptureError("Duplicate active entity pointers")
        blobs = self.memory.batch([(a, 0x124) for a in pointers])
        data_addresses, component_requests, component_indexes = [], [], []
        for i, blob in enumerate(blobs):
            if pointer(u64(blob, 0x10)) != self.collection:
                raise CaptureError("Entity does not belong to the collection")
            data_addresses.append(pointer(u64(blob, 0x48)))
            if pointer(u64(blob, 0x18)):
                component_requests.append((u64(blob, 0x18), 24))
                component_indexes.append(i)
        names = self.memory.names(data_addresses)
        hp_requests, hp_indexes = [], []
        for index, blob in zip(component_indexes, self.memory.batch(component_requests)):
            if pointer(u64(blob, 0x10)):
                hp_requests.append((pointer(u64(blob, 0x10)) + 0x10, 8))
                hp_indexes.append(index)
        hp = dict(zip(hp_indexes, (struct.unpack("<ii", b)
                                  for b in self.memory.batch(hp_requests))))
        entities = []
        for index, (address, blob) in enumerate(zip(pointers, blobs)):
            side, x, y = struct.unpack_from("<iii", blob, 0x78)
            if side not in (0, 1) or not -4000 <= x <= 22000 or not -4000 <= y <= 36000:
                raise CaptureError("Entity coordinates or side violate layout bounds")
            health, maximum = hp.get(index, (None, None))
            entities.append({"address": hex(address),
                             "entity_id": struct.unpack_from("<I", blob, 8)[0],
                             "name": names[data_addresses[index]], "side": side,
                             "x": x, "y": y,
                             "card_id": struct.unpack_from("<i", blob, 0xAC)[0],
                             "level_index": (struct.unpack_from("<i", blob, 0x120)[0]
                                             if struct.unpack_from("<I", blob, 8)[0]
                                             // 1_000_000 == 5 else None),
                             "hp": health, "max_hp": maximum})
        # Reallocation or entity insertion/removal during the read invalidates it.
        if self.memory.read(self.collection, 24)[8:24] != head[8:24]:
            raise CaptureError("Collection changed while sampling; frame rejected")
        # Recheck identity and ownership, including when count stays the same.
        if count:
            current = self.memory.read(array, count * 8)
            if current != raw:
                raise CaptureError("Collection members changed while sampling")
            identities = self.memory.batch([(a, 24) for a in pointers])
            for before, after in zip(blobs, identities):
                if before[8:12] != after[8:12] or before[16:24] != after[16:24]:
                    raise CaptureError("Entity identity changed while sampling")
        end = time.monotonic_ns()
        return {"sample_start_ns": start, "sample_end_ns": end,
                "read_ms": (end - start) / 1e6, "collection": hex(self.collection),
                "entity_count": count, "entities": entities,
                "atomic": False, "game_tick": None}


class FirstObservations:
    """Report appearances, never mislabel them as confirmed card placements."""
    def __init__(self):
        self.previous: set[tuple] | None = None

    def update(self, frame: dict) -> list[dict]:
        current = {(e["entity_id"], e["address"], e["name"]) for e in frame["entities"]}
        if self.previous is None:
            self.previous = current
            return []  # Attaching mid-battle is a baseline, not new placements.
        events = []
        for entity in frame["entities"]:
            identity = (entity["entity_id"], entity["address"], entity["name"])
            if identity not in self.previous:
                events.append({"event": "entity_first_observed", **entity,
                               "sample_start_ns": frame["sample_start_ns"],
                               "sample_end_ns": frame["sample_end_ns"],
                               "placement_confirmed": False})
        self.previous = current
        return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", default="/tmp/Hasty-android-tools/platform-tools/adb")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--collection", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 600:
        parser.error("--seconds must be between 0 and 600")
    memory = Memory(args.adb, args.serial)
    reader = EntityReader(memory, args.collection)
    tracker = FirstObservations()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "identity.json").write_text(json.dumps(
        {"pid": memory.pid, "version": VERSION, "libg_sha256": LIBG_SHA256,
         "serial": args.serial, "collection": hex(reader.collection)}, indent=2))
    stop = time.monotonic() + args.seconds
    frames, failures, appearances, latencies = 0, 0, 0, []
    with (args.output / "frames.jsonl").open("x") as stream, \
            (args.output / "events.jsonl").open("x") as events, \
            (args.output / "errors.jsonl").open("x") as errors:
        while time.monotonic() < stop:
            try:
                frame = reader.frame()
                frame["host_wall_ns"] = time.time_ns()
                stream.write(json.dumps(frame) + "\n")
                stream.flush()
                for event in tracker.update(frame):
                    events.write(json.dumps(event) + "\n")
                    events.flush()
                    appearances += 1
                    print(event["name"], event["side"], event["x"], event["y"], flush=True)
                frames += 1
                latencies.append(frame["read_ms"])
            except (CaptureError, subprocess.SubprocessError, struct.error) as error:
                failures += 1
                errors.write(json.dumps({"host_wall_ns": time.time_ns(),
                                         "error": str(error)}) + "\n")
                errors.flush()
                time.sleep(0.05)
    summary = {"frames": frames, "rejected_frames": failures,
               "appearances": appearances, "placements_confirmed": 0,
               "mean_read_ms": sum(latencies) / len(latencies) if latencies else None,
               "max_read_ms": max(latencies) if latencies else None,
               "note": "First appearances only; no completeness or exact placement claim."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
