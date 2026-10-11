"""Measure how often each shaped reward actually fires for a checkpoint.

This is diagnostic, not another evaluator: it reports raw event totals and
their weighted contribution so a rare spell prediction cannot look important
just because its coefficient is large.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

RAW_FIELDS = (
    "tower_fraction_dealt", "tower_fraction_taken", "elixir_trade",
    "spell_quality", "spell_prediction", "spell_casts", "spell_misses", "kite_quality",
    "reward_total",
)


def summarise(rows: list[dict]) -> dict:
    count = max(1, len(rows))
    return {
        field: {
            "mean_per_episode": sum(float(row[field]) for row in rows) / count,
            "nonzero_episodes": sum(abs(float(row[field])) > 1e-12 for row in rows),
            "total": sum(float(row[field]) for row in rows),
        }
        for field in RAW_FIELDS
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="audit raw reward event frequency")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--opponent", choices=("brain", "meta", "simple", "mirror"),
                        default="meta")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=990_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                        default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")

    import torch
    from sim.device import torch_device
    from sim.env import (ACTIONS, NUM_PLANES, NUM_SCALARS, ClashEnv,
                         RewardWeights)
    from sim.train_ppo import build_network, load_policy_state, masked_distribution

    device = torch_device(args.device)
    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    network = build_network(NUM_PLANES, NUM_SCALARS, ACTIONS).to(device)
    load_policy_state(network, blob["state_dict"])
    network.eval()

    # Non-zero values make every event path execute. EpisodeStats stores the
    # raw quantities before these coefficients are applied.
    weights = RewardWeights(elixir_traded=0.05, spell_quality=0.15,
                            spell_prediction=0.25, spell_miss=-0.05,
                            kite_quality=0.15)
    env = ClashEnv(seed=args.seed, opponent=args.opponent, rewards=weights)
    rows: list[dict] = []
    outcomes = {"wins": 0, "losses": 0, "draws": 0}
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            while True:
                with torch.no_grad():
                    planes = torch.from_numpy(obs["planes"]).unsqueeze(0).to(device)
                    scalars = torch.from_numpy(obs["scalars"]).unsqueeze(0).to(device)
                    mask = torch.from_numpy(info["action_mask"]).unsqueeze(0).to(device)
                    logits, _ = network(planes, scalars)
                    action = int(masked_distribution(logits, mask).probs.argmax(1).item())
                obs, _reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            result = info["result"]
            outcomes["wins" if result == "bottom" else
                     "losses" if result == "top" else "draws"] += 1
            rows.append(asdict(info["stats"]))
    finally:
        env.close()

    summary = summarise(rows)
    coefficients = {
        "tower_fraction_dealt": weights.tower_damage_dealt,
        "tower_fraction_taken": weights.tower_damage_taken,
        "elixir_trade": weights.elixir_traded,
        "spell_quality": weights.spell_quality,
        "spell_prediction": weights.spell_prediction,
        "spell_misses": weights.spell_miss,
        "kite_quality": weights.kite_quality,
    }
    for field, coefficient in coefficients.items():
        summary[field]["weighted_mean_per_episode"] = (
            summary[field]["mean_per_episode"] * coefficient)

    report = {
        "checkpoint": str(args.checkpoint), "opponent": args.opponent,
        "episodes": args.episodes, "seed": args.seed, **outcomes,
        "reward_components": summary,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
