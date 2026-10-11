#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$ROOT/.venv-live/bin/python"
CHECKPOINT="$ROOT/checkpoints/portable_v11/portable_v11_buildings.pt"
SERIAL="${HASTYCR_SERIAL:-}"
MATCHES=1
MODE=shadow

usage() {
  echo "Usage: ./run-macos.sh [--serial ID] [--matches N] [--checkpoint FILE]"
  echo "                      [--shadow|--tap-friendly|--ladder]"
  echo
  echo "  --shadow        observe and log; never tap (default)"
  echo "  --tap-friendly  tap during a match, but never press Battle"
  echo "  --ladder        allow the bot to press Battle (account-ban risk)"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serial) SERIAL="$2"; shift 2 ;;
    --matches) MATCHES="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --shadow) MODE=shadow; shift ;;
    --tap-friendly) MODE=friendly; shift ;;
    --ladder) MODE=ladder; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

ADB_BIN="${ADB:-$(command -v adb || true)}"
if [[ -z "$ADB_BIN" ]]; then
  echo "adb is missing. Install it with:" >&2
  echo "  brew install --cask android-platform-tools" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Live environment is missing. Create it with:" >&2
  echo "  python3.12 -m venv .venv-live" >&2
  echo "  .venv-live/bin/pip install -e '.[live,dev]'" >&2
  exit 1
fi
if [[ ! -d "$ROOT/vendor/ClashRoyaleBuildABot/clashroyalebuildabot" ]]; then
  echo "Live perception vendor is missing. Follow the macOS setup in README.md." >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

if [[ -z "$SERIAL" ]]; then
  mapfile_output="$($ADB_BIN devices | awk 'NR > 1 && $2 == "device" {print $1}')"
  device_count="$(printf '%s\n' "$mapfile_output" | sed '/^$/d' | wc -l | tr -d ' ')"
  if [[ "$device_count" != "1" ]]; then
    echo "Expected exactly one connected Android device; found $device_count." >&2
    echo "Run 'adb devices', then pass --serial DEVICE_ID." >&2
    exit 1
  fi
  SERIAL="$mapfile_output"
fi

if ! "$ADB_BIN" -s "$SERIAL" shell pm path com.supercell.clashroyale \
    2>/dev/null | grep -q '^package:'; then
  echo "Clash Royale is not installed/reachable on $SERIAL." >&2
  exit 1
fi

ARGS=(
  "$ROOT/scripts/cr_bot.py"
  --adb "$ADB_BIN"
  --serial "$SERIAL"
  --max-matches "$MATCHES"
  --vision yolo
  --rl "$CHECKPOINT"
)

case "$MODE" in
  shadow) ARGS+=(--no-queue --dry-run) ;;
  friendly) ARGS+=(--no-queue) ;;
  ladder)
    echo "WARNING: ladder automation can get the account banned." >&2
    ;;
esac

echo "HastyCR: mode=$MODE device=$SERIAL checkpoint=$(basename "$CHECKPOINT")"
exec "$PYTHON" "${ARGS[@]}"
