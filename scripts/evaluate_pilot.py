"""Fixed evaluation matrix for the RL pilot.

Evaluates checkpoints vs held-out opponents on fresh seeds.
Primary metrics: win rate, crown differential, tower HP differential.
Includes Wilson 95% CI, match duration, deterministic greedy policy, no grad.

Fixed seed discipline:
  COMPARE_SEED = 8000  — all checkpoints compared on identical seeds (directly comparable)
  FINAL_SEED   = 9000  — fresh held-out set for final candidate (unseen during any comparison)
Training seeds are drawn from rollout RNG; eval seeds are disjoint by design.

    python -m scripts.evaluate_pilot --ckpt tmp/rl/pilot_best.pt --episodes 60
    python -m scripts.evaluate_pilot --ckpt tmp/rl/pilot_best.pt --all
    python -m scripts.evaluate_pilot --ckpt checkpoints/live_candidate/pilot_best_20260824.pt --episodes 200 --seed 8000
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

COMPARE_SEED = 8000
FINAL_SEED = 9000


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion (95% by default)."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def evaluate_one(ckpt: Path, opponent: str, episodes: int, seed: int, device: str = "cpu") -> dict:
    import torch
    from sim.env import ACTIONS, NUM_PLANES, NUM_SCALARS, ClashEnv
    from sim.train_ppo import (build_network, checkpoint_architecture,
                               load_policy_state, masked_distribution)

    blob = torch.load(str(ckpt), map_location=device, weights_only=False)
    net = build_network(NUM_PLANES, NUM_SCALARS, ACTIONS,
                        checkpoint_architecture(blob)).to(device)
    load_policy_state(net, blob["state_dict"])
    net.eval()

    env = ClashEnv(seed=seed, opponent=opponent)
    wins = losses = draws = 0
    cf = ca = 0
    hog = plays = 0
    cards = Counter()
    spell_casts = spell_misses = hog_spirit_combos = 0
    spell_quality = spell_prediction = kite_quality = 0.0
    illegal_total = 0
    tower_for = 0.0
    tower_against = 0.0
    duration_sum = 0.0
    results = []

    import torch as T
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep * 1000)
        while True:
            with T.no_grad():
                planes = T.from_numpy(obs["planes"]).unsqueeze(0).to(device)
                scalars = T.from_numpy(obs["scalars"]).unsqueeze(0).to(device)
                mask = T.from_numpy(info["action_mask"]).unsqueeze(0).to(device)
                logits, _ = net(planes, scalars)
                action = int(masked_distribution(logits, mask).probs.argmax())
                # verify masking: illegal actions must have zero probability
                probs = masked_distribution(logits, mask).probs[0].cpu().numpy()
                assert bool(info["action_mask"][action]), f"greedy chose illegal action {action}"
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        mine, theirs = info["crowns"]
        cf += mine
        ca += theirs
        wins += info["result"] == "bottom"
        losses += info["result"] == "top"
        draws += info["result"] not in ("bottom", "top")
        hog += info["stats"].cards.get("hog_rider", 0)
        plays += info["stats"].plays
        cards.update(info["stats"].cards)
        spell_casts += info["stats"].spell_casts
        spell_misses += info["stats"].spell_misses
        spell_quality += info["stats"].spell_quality
        spell_prediction += info["stats"].spell_prediction
        kite_quality += info["stats"].kite_quality
        hog_spirit_combos += info["stats"].hog_spirit_combos
        illegal_total += info["stats"].illegal
        if env.match:
            ours = sum(env.match.tower_fractions(1).values())
            theirs2 = sum(env.match.tower_fractions(-1).values())
            tower_for += ours
            tower_against += theirs2
            duration_sum += env.match.elapsed_ms
        results.append((mine, theirs, info["result"]))
    env.close()
    n = episodes
    lo, hi = wilson_ci(wins, n)
    card_counts = {card: cards[card] for card in sorted(cards)}
    card_share = {card: count / max(1, plays)
                  for card, count in card_counts.items()}
    card_per_match = {card: count / max(1, n)
                      for card, count in card_counts.items()}
    defensive_core = sum(cards[card] for card in
                         ("cannon", "musketeer", "knight", "skeletons"))
    return {
        "opponent": opponent,
        "episodes": n,
        "seed": seed,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "crowns_for": cf,
        "crowns_against": ca,
        "crown_diff": (cf - ca) / n,
        "tower_for": tower_for / n,
        "tower_against": tower_against / n,
        "tower_diff": (tower_for - tower_against) / n,
        "hog_share": hog / max(1, plays),
        "card_counts": card_counts,
        "card_share": card_share,
        "card_per_match": card_per_match,
        "defensive_core_share": defensive_core / max(1, plays),
        "defensive_core_per_match": defensive_core / max(1, n),
        "spell_casts_per_match": spell_casts / max(1, n),
        "spell_miss_rate": spell_misses / max(1, spell_casts),
        "spell_quality_per_cast": spell_quality / max(1, spell_casts),
        "spell_prediction_per_cast": spell_prediction / max(1, spell_casts),
        "kite_quality_per_match": kite_quality / max(1, n),
        "hog_spirit_combo_rate": hog_spirit_combos / max(1, hog),
        "plays_per_match": plays / max(1, n),
        "illegal_total": illegal_total,
        "illegal_rate": illegal_total / max(1, plays),
        "duration_s": duration_sum / max(1, n) / 1000.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate pilot checkpoints")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--opponents", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=8000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--all", action="store_true", help="evaluate all tmp/rl/*_best.pt")
    args = parser.parse_args()

    opponents = args.opponents or ["brain", "meta", "simple"]

    if args.all:
        ckpts = sorted(Path("tmp/rl").glob("*_best.pt"))
    else:
        ckpts = [args.ckpt]

    all_results = {}
    for ckpt in ckpts:
        if not ckpt.exists():
            print(f"skip {ckpt} (missing)", file=sys.stderr)
            continue
        print(f"\n=== {ckpt} ===")
        print(f"    mode=eval no_grad greedy argmax | seed={args.seed} episodes={args.episodes}")
        rows = []
        for opp in opponents:
            r = evaluate_one(ckpt, opp, args.episodes, args.seed)
            rows.append(r)
            print(f"  vs {opp:8s}  W{r['wins']:3d} L{r['losses']:3d} D{r['draws']:3d}  "
                  f"win={r['win_rate']:.1%} [{r['wilson_lo']:.1%}-{r['wilson_hi']:.1%}]  "
                  f"crowns {r['crowns_for']:3d}-{r['crowns_against']:3d} "
                  f"diff={r['crown_diff']:+.2f}  tower_diff={r['tower_diff']:+.2f}  "
                  f"hog={r['hog_share']:.0%}  def={r['defensive_core_share']:.0%}  "
                  f"spell_miss={r['spell_miss_rate']:.0%}  "
                  f"illegal={r['illegal_rate']:.1%}  dur={r['duration_s']:.0f}s")
            card_line = " ".join(
                f"{card}={per_match:.1f}/g"
                for card, per_match in sorted(r["card_per_match"].items()))
            print(f"    cards: {card_line}")
        all_results[str(ckpt)] = rows

    blob = {"seed": args.seed, "episodes": args.episodes,
            "compare_seed": COMPARE_SEED, "final_seed": FINAL_SEED,
            "eval_mode": "eval no_grad greedy argmax deterministic",
            "results": all_results}
    out = args.out or Path(f"reports/eval_{ckpts[0].stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
