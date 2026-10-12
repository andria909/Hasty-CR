# Hasty-CR on a new Mac

This is the recovery checklist for moving the simulator and live bot to a new
macOS machine. It intentionally contains no email address, Supercell login
code, VPN private key, cookies, or other account secrets.

## 1. Clone and install

Install Git, Python 3.12, ADB, and MuMuPlayer. Authenticate `gh` to the private
GitHub repository, then run:

```bash
gh repo clone andria909/Hasty-CR
cd Hasty-CR
python3.12 -m venv .venv-live
.venv-live/bin/pip install -e '.[live,dev]'
git clone https://github.com/Pbatch/ClashRoyaleBuildABot.git vendor/ClashRoyaleBuildABot
git -C vendor/ClashRoyaleBuildABot checkout f89ecbf309e698cb9fccdfb64a6ff2a930e6541e
.venv-live/bin/pip install -e 'vendor/ClashRoyaleBuildABot[cpu]'
```

On Homebrew systems, ADB can be installed with:

```bash
brew install --cask android-platform-tools
```

The current live candidate is:

```text
checkpoints/portable_v20/portable_v20_delay500.pt
```

Its SHA-256 and evaluation results are recorded in
`checkpoints/portable_v20/manifest.json`. This policy was trained with a
500 ms observation-to-action delay to match the measured MuMu capture latency.

## 2. Emulator, VPN, and account

1. Start MuMuPlayer and its existing Android device. Do not create a new
   device if the old exported device can be imported.
2. Install Clash Royale through the Android store used by that MuMu image.
3. If the game is region-blocked, install the Android VPN client **inside the
   emulator** and import a newly generated Android/WireGuard configuration.
   Do not enable the VPN on macOS. This keeps only emulator traffic in the
   tunnel.
4. Never commit the WireGuard configuration: it contains a private key. Revoke
   the old device configuration in the VPN provider when moving machines.
5. Open Clash Royale and complete Supercell ID login manually. Email codes,
   CAPTCHA, consent dialogs, and account recovery must be completed by the
   owner. Never put the email or one-time code in this repository.
6. Select deck slot 1 and verify the live deck is exactly:
   Cannon, Fireball, Hog Rider, Knight, Electro Spirit, Musketeer, Skeletons,
   Zap. Card order in the slot does not matter; card recognition does.

Exporting/importing the whole MuMu device is preferable to repeating the game
and VPN setup, but log in again if Supercell invalidates the device session.

## 3. Connect ADB

MuMu configurations used by this project have exposed ADB at either
`127.0.0.1:16480` or `127.0.0.1:7555`. Try both and keep the one that appears as
`device`:

```bash
adb start-server
adb connect 127.0.0.1:16480
adb connect 127.0.0.1:7555
adb devices
```

Confirm that the game is installed:

```bash
adb -s 127.0.0.1:16480 shell pm path com.supercell.clashroyale
```

Replace the serial in later commands if the working endpoint is `7555`.
`connection refused` means the Android device is not running yet, not that the
bot is broken. If multiple devices are listed, always pass `--serial`.

## 4. Offline simulation and evaluation

Offline simulation does not require MuMu, a VPN, or a Clash Royale account.

Run the focused regression suite:

```bash
.venv-live/bin/python -m pytest -q \
  tests/test_hog_strategy.py tests/test_rl_policy.py \
  tests/test_value_warmup.py tests/test_reward_events.py \
  tests/test_reward_aggregation.py tests/test_sim_env.py \
  tests/test_public_state.py
```

Evaluate v20 against both held-out opponent families:

```bash
.venv-live/bin/python -m scripts.evaluate_pilot \
  --ckpt checkpoints/portable_v20/portable_v20_delay500.pt \
  --episodes 120 --opponents brain meta --seed 985000 \
  --out reports/eval_portable_v20_repeat.json
```

Run bulk headless matches with the scripted policy:

```bash
.venv-live/bin/python -m sim.batch --matches 200 --workers 8 --opponent brain
```

## 5. Live safety ladder

Always test in this order after moving machines.

Shadow mode observes one manually started battle and never taps:

```bash
./run-macos.sh --serial 127.0.0.1:16480 --shadow --matches 1 \
  --checkpoint checkpoints/portable_v20/portable_v20_delay500.pt
```

Friendly mode taps during battles but never presses the Battle button. Start
each friendly battle yourself:

```bash
./run-macos.sh --serial 127.0.0.1:16480 --tap-friendly --matches 3 \
  --checkpoint checkpoints/portable_v20/portable_v20_delay500.pt
```

Ladder mode can press Battle automatically. It can lose trophies and automated
play may violate the game's terms or cause an account ban:

```bash
./run-macos.sh --serial 127.0.0.1:16480 --ladder --matches 5 \
  --checkpoint checkpoints/portable_v20/portable_v20_delay500.pt
```

Do not start a second bot while one is already running. Stop the foreground
runner with `Ctrl-C`; this preserves completed match logs.

## 6. Logs and diagnosis

The important runtime locations are:

- `tmp/live/cr_bot.log` — decisions, capture/policy latency, guards, and errors;
- `tmp/live/matches/` — completed live match traces;
- `tmp/rl/` — training checkpoints and training logs (ignored by Git);
- `reports/` — compact evaluations intended for Git;
- `checkpoints/portable_v20/` — the promoted live candidate and manifest.

Common failures:

- No device in `adb devices`: start the Android device, then reconnect its port.
- `unauthorized`: approve the ADB dialog inside Android and reconnect.
- Package missing: install Clash Royale in this exact MuMu device.
- Bot sees menus but does not queue: `--shadow` and `--tap-friendly` deliberately
  prohibit queue taps.
- Wrong cards or placements: verify deck slot 1 before blaming the policy.
- Slow decisions: inspect `capture_ui_ms`, `board_guard_ms`, and `policy_ms` in
  the log. Policy inference is normally under 2 ms; screen capture is usually
  the bottleneck.
- Game disconnects or refuses the location: check the VPN app inside Android;
  do not route all macOS traffic through it.

## 7. Preserve work

Before changing machines:

```bash
git status --short
git add <intentional source/report files>
git commit -m "Describe the change"
git push origin main
```

Raw replay shards, `.venv-live`, temporary training outputs, account data, and
VPN configurations are intentionally ignored. Promote a checkpoint into a
versioned `checkpoints/portable_vNN/` directory and record its hash and paired
evaluation in a manifest before force-adding the `.pt` file.
