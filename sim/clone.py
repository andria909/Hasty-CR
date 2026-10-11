"""Teach the network to imitate the hand-written brain, before asking it to improve.

PPO from a random start does not work on this problem, and the failure is
instructive rather than mysterious. Over 1.25M steps the agent reached
`crowns_for = 0` in *every* evaluation while winning a third of its matches: it
had found the local optimum of spending cheap cards on defence and winning
time-out tiebreaks on tower health. Scoring a crown needs a coordinated sequence
- get elixir, put a tank at the bridge, put the Hog behind it, defend the
counter-push - and the chance of stumbling onto that by sampling from ~2,300
masked actions is negligible. So the reward is dense enough to learn defence and
far too sparse to learn offence.

Behaviour cloning removes the exploration problem instead of tuning around it.
The hand-written policy already knows how to attack; supervised learning on its
decisions gets the network to the same place in minutes, and PPO can then start
from a policy that at least sends a Hog.

Two details that matter for correctness:

**The teacher plays through the environment, not beside it.** Its chosen action
is the action the environment executes, so the recorded states are the states
that actually follow - on-policy for the teacher. Recording a teacher's opinion
of states produced by someone else's actions is a different and much weaker
dataset.

**"Hold" is most of the data.** The brain decides on roughly one step in ten, and
a classifier trained on that imbalance learns to output nothing. Hold examples
are therefore subsampled to a target share.

    python -m sim.clone --episodes 400
    python -m sim.train_ppo --resume tmp/rl/clone.pt --name ppo_from_clone
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

OUT = ROOT / "tmp" / "rl"
LOG = ROOT / "tmp" / "live" / "rl_train.log"


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} clone: {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def collect(episodes: int, hold_share: float = 0.25,
            seed: int = 500_000, opponent: str = "brain") -> Tuple[dict, np.ndarray]:
    """Play `episodes` matches with the brain in the agent's seat, recording it."""
    from sim.adapter import build_state
    from sim.env import ClashEnv
    from sim.runner import BrainPolicy

    env = ClashEnv(seed=seed, opponent=opponent)
    teacher = BrainPolicy(env._cards, side=1)

    planes: List[np.ndarray] = []
    scalars: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    actions: List[int] = []
    holds = decisions = rejected = 0
    rng = random.Random(0)

    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        teacher.reset()
        while True:
            state = build_state(env.match, 1, env._cards)
            now = env.match.elapsed_ms / 1000.0
            decision = teacher.brain.decide(state, now, now)

            action = 0
            if decision is not None:
                hand = env.match.players[1].hand
                if decision.card in hand:
                    slot = hand.index(decision.card)
                    candidate = ClashEnv.encode(slot, int(decision.x), int(decision.y))
                    if 0 <= candidate < len(info["action_mask"]) \
                            and info["action_mask"][candidate]:
                        action = candidate
                    else:
                        # The brain sometimes wants a tile the engine will not
                        # accept. Recording it would teach an illegal habit.
                        rejected += 1

            keep = action != 0 or rng.random() < hold_share
            if keep:
                planes.append(obs["planes"])
                scalars.append(obs["scalars"])
                masks.append(info["action_mask"])
                actions.append(action)
                holds += action == 0
                decisions += action != 0

            obs, _, terminated, truncated, info = env.step(action)
            if action != 0 and decision is not None:
                teacher.brain.confirm(decision, now)
            if terminated or truncated:
                break
        if (episode + 1) % 50 == 0:
            log(f"collected {episode + 1}/{episodes} episodes, "
                f"{decisions} plays, {holds} holds, {rejected} illegal wishes")

    env.close()
    data = {
        "planes": np.stack(planes),
        "scalars": np.stack(scalars),
        "masks": np.stack(masks),
    }
    return data, np.array(actions, dtype=np.int64)


def spatial_cross_entropy(logits, masks, target, radius: int = 1):
    """Per-row imitation loss with soft credit for nearby legal placements.

    The teacher's tile is not uniquely correct: a Cannon one cell beside its
    recorded position can be strategically equivalent, while ordinary class
    cross-entropy treats it like a Cannon at the opposite corner.  Card slot
    and hold/ability actions remain exact; only legal cells for the same card
    slot inside ``radius`` receive distance-decayed target mass.
    """
    import torch
    from sim.env import CARD_ACTIONS, GRID_H, GRID_W, TILES

    radius = max(0, int(radius))
    masked = logits.masked_fill(~masks, float("-inf"))
    log_probs = torch.log_softmax(masked, dim=1)
    # 0 * -inf is NaN, even though illegal actions have zero target mass.
    log_probs = torch.where(masks, log_probs, torch.zeros_like(log_probs))
    weights = torch.zeros_like(logits)
    for row, raw_action in enumerate(target.detach().cpu().tolist()):
        action = int(raw_action)
        if radius == 0 or action <= 0 or action >= CARD_ACTIONS:
            weights[row, action] = 1.0
            continue
        offset = action - 1
        slot, cell = divmod(offset, TILES)
        y, x = divmod(cell, GRID_W)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < GRID_H and 0 <= nx < GRID_W):
                    continue
                candidate = 1 + slot * TILES + ny * GRID_W + nx
                if bool(masks[row, candidate]):
                    weights[row, candidate] = 1.0 / (1.0 + abs(dx) + abs(dy))
        weights[row] /= weights[row].sum().clamp_min(1.0)
    return -(weights * log_probs).sum(dim=1)


def action_gate_loss(logits, masks, target):
    """Binary loss matching the greedy serve-time hold/play comparison."""
    import torch
    import torch.nn.functional as functional
    from sim.env import CARD_ACTIONS

    play_logits = logits[:, 1:CARD_ACTIONS].masked_fill(
        ~masks[:, 1:CARD_ACTIONS], float("-inf"))
    best_play = play_logits.max(dim=1).values
    # A row with no playable card can only target hold. Keep the margin finite
    # so BCE does not produce inf even in a diagnostic/synthetic batch.
    best_play = torch.where(torch.isfinite(best_play), best_play,
                            logits[:, 0] - 20.0)
    should_play = (target != 0).to(logits.dtype)
    return functional.binary_cross_entropy_with_logits(
        best_play - logits[:, 0], should_play, reduction="none")


def near_play_correct(predicted, target, radius: int = 1):
    """Correct card slot and a placement within ``radius`` cells."""
    from sim.env import CARD_ACTIONS, GRID_W, TILES

    out = []
    for raw_predicted, raw_target in zip(
            predicted.detach().cpu().tolist(), target.detach().cpu().tolist()):
        pred, truth = int(raw_predicted), int(raw_target)
        if truth <= 0 or truth >= CARD_ACTIONS:
            out.append(pred == truth)
            continue
        if pred <= 0 or pred >= CARD_ACTIONS:
            out.append(False)
            continue
        pred_slot, pred_cell = divmod(pred - 1, TILES)
        true_slot, true_cell = divmod(truth - 1, TILES)
        pred_y, pred_x = divmod(pred_cell, GRID_W)
        true_y, true_x = divmod(true_cell, GRID_W)
        out.append(pred_slot == true_slot
                   and abs(pred_x - true_x) <= radius
                   and abs(pred_y - true_y) <= radius)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Behaviour-clone the brain")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hold-share", type=float, default=0.25)
    parser.add_argument("--hold-loss-weight", type=float, default=0.15,
                        help="loss weight for hold rows; prevents the majority "
                             "class from erasing all placement learning")
    parser.add_argument("--placement-radius", type=int, default=1,
                        help="nearby legal tiles given soft imitation credit")
    parser.add_argument("--act-loss-weight", type=float, default=1.0,
                        help="auxiliary weight for deciding play versus hold")
    parser.add_argument("--gate-hold-weight", type=float, default=0.15,
                        help="hold-row weight inside the binary play/hold gate; "
                             "without class balance a 97%% hold corpus collapses")
    parser.add_argument("--play-repeat", type=int, default=1,
                        help="repeat action rows in TRAIN only; validation keeps "
                             "the natural hold/play distribution")
    parser.add_argument("--opponent", default="brain",
                        choices=("meta", "brain", "simple", "mirror"),
                        help="opponent used to generate teacher states")
    parser.add_argument("--data-cache", type=Path,
                        help="reuse/save collected teacher rows as an npz file")
    parser.add_argument("--init", type=Path,
                        help="continue fitting from an existing policy checkpoint")
    parser.add_argument("--name", default="clone")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                        default="auto",
                        help="PyTorch backend; auto prefers CUDA, then Apple MPS")
    parser.add_argument("--architecture", choices=("compact", "wide"),
                        default="compact",
                        help="policy capacity; wide uses the residual v4 network")
    parser.add_argument("--replay-rows", type=Path,
                        help="trusted IL_Replay shards from tools/build_rows.py; "
                             "classic Hog actions are adapted to the live deck")
    parser.add_argument("--replay-repeat", type=int, default=1,
                        help="repeat replay TRAIN indices without leaking copies "
                             "into validation")
    parser.add_argument("--replay-all-results", action="store_true",
                        help="include losing Hog sides; default learns only wins")
    args = parser.parse_args()

    import torch
    from sim.env import ACTIONS, NUM_PLANES, NUM_SCALARS
    from sim.device import torch_device
    from sim.train_ppo import build_network, load_policy_state, masked_distribution

    started = time.time()
    if args.data_cache and args.data_cache.is_file():
        cached = np.load(args.data_cache)
        data = {key: cached[key] for key in ("planes", "scalars", "masks")}
        actions = cached["actions"]
        log(f"loaded dataset cache {args.data_cache}")
    else:
        data, actions = collect(args.episodes, args.hold_share,
                                opponent=args.opponent)
        if args.data_cache:
            args.data_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.data_cache, actions=actions, **data)
            log(f"saved dataset cache {args.data_cache}")
    teacher_count = len(actions)
    replay_splits = np.asarray([], dtype="U5")
    if args.replay_rows:
        from sim.replay_bc import load_replay_rows
        replay_data, replay_actions, replay_splits = load_replay_rows(
            args.replay_rows, winners_only=not args.replay_all_results)
        for key in data:
            data[key] = np.concatenate((data[key], replay_data[key]), axis=0)
        actions = np.concatenate((actions, replay_actions), axis=0)
        replay_plays = int((replay_actions != 0).sum())
        log(f"appended {len(replay_actions)} trusted winning Hog replay rows "
            f"({replay_plays} plays, {len(replay_actions) - replay_plays} holds)")

    played = int((actions != 0).sum())
    log(f"dataset {len(actions)} samples ({played} plays, "
        f"{len(actions) - played} holds) in {time.time() - started:.0f}s")

    device = torch_device(args.device)
    network = build_network(
        NUM_PLANES, NUM_SCALARS, ACTIONS, args.architecture).to(device)
    if args.init:
        if not args.init.is_file():
            raise SystemExit(f"--init {args.init} does not exist")
        blob = torch.load(args.init, map_location=device, weights_only=False)
        load_policy_state(network, blob["state_dict"])
        log(f"initialised from {args.init} (weights only)")
    optimiser = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=1e-4)

    split = int(teacher_count * 0.9)
    order = np.random.default_rng(0).permutation(teacher_count)
    train_idx, val_idx = order[:split], order[split:]
    if len(replay_splits):
        replay_idx = np.arange(teacher_count, len(actions))
        replay_train = replay_idx[replay_splits == "train"]
        replay_val = replay_idx[replay_splits == "val"]
        repeat = max(1, args.replay_repeat)
        train_idx = np.concatenate((train_idx, np.tile(replay_train, repeat)))
        val_idx = np.concatenate((val_idx, replay_val))
        log(f"fixed replay split: {len(replay_train)} train x{repeat}, "
            f"{len(replay_val)} val, "
            f"{int((replay_splits == 'test').sum())} held-out test")
    play_train = train_idx[actions[train_idx] != 0]
    play_repeat = max(1, args.play_repeat)
    if play_repeat > 1 and len(play_train):
        train_idx = np.concatenate(
            (train_idx, np.tile(play_train, play_repeat - 1)))
        log(f"oversampled {len(play_train)} train play rows x{play_repeat}; "
            f"validation remains natural")

    def batch_of(index: np.ndarray):
        return (torch.from_numpy(data["planes"][index]).to(device),
                torch.from_numpy(data["scalars"][index]).to(device),
                torch.from_numpy(data["masks"][index]).to(device),
                torch.from_numpy(actions[index]).to(device))

    best = 0.0
    for epoch in range(args.epochs):
        network.train()
        np.random.shuffle(train_idx)
        total = 0.0
        for start in range(0, len(train_idx), args.batch):
            planes, scalars, masks, target = batch_of(train_idx[start:start + args.batch])
            logits, _value = network(planes, scalars)
            per_row = spatial_cross_entropy(
                logits, masks, target, args.placement_radius)
            weights = torch.where(
                target == 0,
                torch.full_like(per_row, args.hold_loss_weight),
                torch.ones_like(per_row))
            imitation = (per_row * weights).sum() / weights.sum().clamp_min(1.0)
            gate_rows = action_gate_loss(logits, masks, target)
            gate_weights = torch.where(
                target == 0,
                torch.full_like(gate_rows, args.gate_hold_weight),
                torch.ones_like(gate_rows))
            gate = ((gate_rows * gate_weights).sum()
                    / gate_weights.sum().clamp_min(1.0))
            loss = imitation + args.act_loss_weight * gate
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimiser.step()
            total += float(loss.detach()) * len(target)

        network.eval()
        correct = play_correct = near_correct = play_total = act_correct = 0
        with torch.no_grad():
            for start in range(0, len(val_idx), 1024):
                planes, scalars, masks, target = batch_of(val_idx[start:start + 1024])
                logits, _ = network(planes, scalars)
                predicted = masked_distribution(logits, masks).probs.argmax(1)
                correct += int((predicted == target).sum())
                act_correct += int(((predicted != 0) == (target != 0)).sum())
                is_play = target != 0
                play_total += int(is_play.sum())
                play_correct += int((predicted[is_play] == target[is_play]).sum())
                near_correct += sum(near_play_correct(
                    predicted[is_play], target[is_play], args.placement_radius))
        accuracy = correct / max(1, len(val_idx))
        act_accuracy = act_correct / max(1, len(val_idx))
        play_accuracy = play_correct / max(1, play_total)
        near_accuracy = near_correct / max(1, play_total)
        log(f"epoch {epoch + 1}/{args.epochs}  loss {total / len(train_idx):.4f}  "
            f"val {accuracy:.3f}  act {act_accuracy:.3f}  "
            f"plays exact {play_accuracy:.3f} near {near_accuracy:.3f}")
        # A model that always holds has high aggregate accuracy and is useless.
        # Select checkpoints equally on deciding *whether* to act and on the
        # exact action when the teacher acts.
        selection = act_accuracy + near_accuracy
        if selection > best:
            best = selection
            OUT.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": network.state_dict(),
                        "optimiser": optimiser.state_dict(),
                        "step": 0, "val_accuracy": accuracy,
                        "val_play_accuracy": play_accuracy,
                        "val_act_accuracy": act_accuracy,
                        "val_near_play_accuracy": near_accuracy,
                        "architecture": network.architecture},
                       OUT / f"{args.name}.pt")

    log(f"best selection score {best:.3f} -> {OUT / (args.name + '.pt')}")

    from sim.train_ppo import evaluate
    blob = torch.load(OUT / f"{args.name}.pt", map_location=device, weights_only=False)
    load_policy_state(network, blob["state_dict"])
    network.eval()
    result = evaluate(network, device, episodes=16, opponent=args.opponent)
    log(f"EVAL(clone)  W{result['wins']} L{result['losses']} D{result['draws']}  "
        f"crowns {result['crowns_for']}-{result['crowns_against']}  "
        f"hog {result['hog_share']:.0%}  plays/match {result['plays_per_match']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
