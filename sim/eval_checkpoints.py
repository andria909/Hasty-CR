"""Paired, reproducible comparison of two policy checkpoints.

Both policies play the same opponent/deck on the same episode seeds.  The
paired exact sign test then answers the useful question: on seeds where their
outcomes differ, how often is the candidate better than the baseline?

Example::

    python -m sim.eval_checkpoints \
      --baseline tmp/rl/portable_bc_balanced_v2.pt \
      --candidate tmp/rl/portable_ppo_v2_best.pt \
      --opponents brain,meta --episodes 60 \
      --output reports/portable_v2_paired_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson_interval(successes: int, trials: int,
                    z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided 95% Wilson score interval, including 0/n and n/n."""
    if trials <= 0:
        return 0.0, 1.0
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = proportion + z * z / (2.0 * trials)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z * z / (4.0 * trials * trials))
    return ((centre - radius) / denominator,
            (centre + radius) / denominator)


def paired_sign_test(better: int, worse: int) -> dict:
    """Exact binomial sign test over non-tied paired outcomes."""
    discordant = better + worse
    if discordant == 0:
        return {"better": better, "worse": worse, "tied": None,
                "discordant": 0, "p_one_sided": 1.0, "p_two_sided": 1.0}
    upper = sum(math.comb(discordant, k) for k in range(better, discordant + 1))
    one_sided = upper / (2 ** discordant)
    extreme = min(better, worse)
    lower_tail = sum(math.comb(discordant, k) for k in range(extreme + 1))
    two_sided = min(1.0, 2.0 * lower_tail / (2 ** discordant))
    return {"better": better, "worse": worse, "tied": None,
            "discordant": discordant, "p_one_sided": one_sided,
            "p_two_sided": two_sided}


def compare_games(baseline: list[dict], candidate: list[dict]) -> dict:
    """Compare paired win/draw/loss outcomes and crown differentials."""
    baseline_by_seed = {game["seed"]: game for game in baseline}
    candidate_by_seed = {game["seed"]: game for game in candidate}
    if baseline_by_seed.keys() != candidate_by_seed.keys():
        raise ValueError("baseline and candidate seeds differ")

    rank = {"top": -1, "draw": 0, None: 0, "bottom": 1}
    better = worse = tied = 0
    crown_delta_gain = 0
    for seed in sorted(baseline_by_seed):
        old = baseline_by_seed[seed]
        new = candidate_by_seed[seed]
        difference = rank.get(new["result"], 0) - rank.get(old["result"], 0)
        better += difference > 0
        worse += difference < 0
        tied += difference == 0
        old_crowns = old["crowns_for"] - old["crowns_against"]
        new_crowns = new["crowns_for"] - new["crowns_against"]
        crown_delta_gain += new_crowns - old_crowns

    result = paired_sign_test(better, worse)
    result["tied"] = tied
    result["mean_crown_differential_gain"] = (
        crown_delta_gain / max(1, len(baseline_by_seed)))
    return result


def _load_network(path: Path, device):
    import torch
    from sim.env import ACTIONS, NUM_PLANES, NUM_SCALARS
    from sim.train_ppo import build_network, load_policy_state

    blob = torch.load(path, map_location=device, weights_only=False)
    network = build_network(NUM_PLANES, NUM_SCALARS, ACTIONS).to(device)
    load_policy_state(network, blob["state_dict"])
    network.eval()
    return network


def main() -> int:
    parser = argparse.ArgumentParser(description="paired checkpoint evaluation")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--opponents", default="brain,meta")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=920_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                        default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from sim.device import torch_device
    from sim.train_ppo import evaluate

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    for path in (args.baseline, args.candidate):
        if not path.is_file():
            parser.error(f"checkpoint does not exist: {path}")

    device = torch_device(args.device)
    models = {
        "baseline": _load_network(args.baseline, device),
        "candidate": _load_network(args.candidate, device),
    }
    report = {
        "episodes": args.episodes,
        "seed": args.seed,
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "opponents": {},
    }
    for opponent in (name.strip() for name in args.opponents.split(",")):
        if not opponent:
            continue
        rows = {
            name: evaluate(network, device, episodes=args.episodes,
                           seed=args.seed, opponent=opponent,
                           include_games=True)
            for name, network in models.items()
        }
        comparison = compare_games(rows["baseline"]["games"],
                                   rows["candidate"]["games"])
        for result in rows.values():
            low, high = wilson_interval(result["wins"], args.episodes)
            result["win_rate_wilson_95"] = [low, high]
        report["opponents"][opponent] = {
            **rows,
            "paired_outcome_test": comparison,
            "candidate_significantly_better": (
                comparison["better"] > comparison["worse"]
                and comparison["p_one_sided"] < 0.05),
        }
        print(
            f"{opponent}: baseline {rows['baseline']['wins']}/{args.episodes}, "
            f"candidate {rows['candidate']['wins']}/{args.episodes}; "
            f"paired better/worse/tied "
            f"{comparison['better']}/{comparison['worse']}/{comparison['tied']}, "
            f"p={comparison['p_one_sided']:.6g}", flush=True)

    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
