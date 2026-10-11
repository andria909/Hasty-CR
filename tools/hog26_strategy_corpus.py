"""Extract an auditable Hog 2.6 action corpus from IL_Replay.

The output is deliberately strategy-level rather than pretending that a raw
action stream contains a trustworthy board after simulator divergence.  Exact
state/action pairs come from ``tools/build_rows.py``; this corpus supplies
placement distributions, timing and successful action sequences across every
exact Hog side, including matches our engine cannot fully redrive.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.replay_corpus import OPPONENT, TEAM, iter_corpus  # noqa: E402

HOG26 = frozenset({"cannon", "fireball", "hog_rider", "ice_golemite",
                   "ice_spirits", "musketeer", "skeletons", "log"})
ADAPT = {"ice_golemite": "knight", "ice_spirits": "electro_spirit", "log": "zap"}
ROLE = {
    "hog_rider": "win_condition", "cannon": "central_pull",
    "musketeer": "ranged_core", "ice_golemite": "kite_tank",
    "ice_spirits": "stun_cycle", "skeletons": "distraction_cycle",
    "fireball": "medium_spell", "log": "small_spell",
}
BUILDING_TARGETERS = {
    "hog_rider", "giant", "goblin_giant", "golem", "electro_giant",
    "royal_giant", "battle_ram", "ram_rider", "royal_hogs", "balloon",
    "lava_hound", "wall_breakers",
}


def winner(rec) -> str:
    value = rec.result.lower()
    if value in {"victory", "win"}:
        return TEAM
    if value in {"defeat", "loss"}:
        return OPPONENT
    return TEAM if rec.crowns[TEAM] > rec.crowns[OPPONENT] else OPPONENT


def matchup(deck: frozenset[str]) -> str:
    if deck & {"golem", "giant", "goblin_giant", "electro_giant", "lava_hound"}:
        return "beatdown"
    if deck & {"balloon"}:
        return "balloon"
    if deck & {"graveyard"}:
        return "graveyard"
    if deck & {"xbow", "mortar"}:
        return "siege"
    if deck & {"goblin_barrel", "princess"}:
        return "bait"
    if deck & {"battle_ram", "ram_rider"}:
        return "bridge_spam"
    if deck & {"hog_rider", "royal_hogs"}:
        return "hog_cycle"
    return "control_other"


def own_frame(play, side: str) -> tuple[float, float]:
    x, y = play.x / 1000.0, play.y / 1000.0
    return (x, y) if side == TEAM else (18.0 - x, 32.0 - y)


def quantiles(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {}
    pick = lambda fraction: ordered[round((len(ordered) - 1) * fraction)]
    return {"p25": round(pick(0.25), 2), "median": round(statistics.median(ordered), 2),
            "p75": round(pick(0.75), 2)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "data" / "strategy" / "hog26_expert_v1.jsonl")
    parser.add_argument("--summary", type=Path,
                        default=ROOT / "data" / "strategy" / "hog26_expert_v1_summary.json")
    args = parser.parse_args()
    for path in (args.out, args.summary):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing corpus: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    positions = defaultdict(lambda: {"x": [], "y": []})
    transitions = Counter()
    hog_followups = Counter()
    cannon_after_targeter = 0
    sides = wins = 0
    matchups = Counter()
    for rec in iter_corpus(limit_parts=args.parts):
        for side in (TEAM, OPPONENT):
            if rec.base_deck(side) != HOG26:
                continue
            sides += 1
            did_win = winner(rec) == side
            wins += did_win
            other = OPPONENT if side == TEAM else TEAM
            family = matchup(rec.base_deck(other))
            matchups[family] += 1
            own = [play for play in rec.plays
                   if play.side == side and play.kind == "play_card"
                   and play.engine in HOG26]
            theirs = [play for play in rec.plays
                      if play.side == other and play.kind == "play_card"]
            for previous, current in zip(own, own[1:]):
                if current.tick - previous.tick <= 80:
                    transitions[(ADAPT.get(previous.engine, previous.engine),
                                 ADAPT.get(current.engine, current.engine))] += 1
            for index, play in enumerate(own):
                live = ADAPT.get(play.engine, play.engine)
                x, y = own_frame(play, side)
                positions[live]["x"].append(x)
                positions[live]["y"].append(y)
                if play.engine == "hog_rider":
                    for later in own[index + 1:]:
                        gap = later.tick - play.tick
                        if gap > 80:
                            break
                        hog_followups[ADAPT.get(later.engine, later.engine)] += 1
                if play.engine == "cannon":
                    recent = [event for event in theirs
                              if 0 <= play.tick - event.tick <= 160]
                    cannon_after_targeter += any(
                        event.engine in BUILDING_TARGETERS for event in recent)
                rows.append({
                    "match": rec.tag, "side": side, "won": did_win,
                    "mode": rec.game_mode, "matchup": family,
                    "tick": play.tick, "time_s": round(play.tick / 20.0, 2),
                    "classic_card": play.engine, "live_card": live,
                    "role": ROLE[play.engine], "x": round(x, 2), "y": round(y, 2),
                })

    with args.out.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema": "hastycr.hog26-expert-strategy.v1",
        "source": "VanguardX101/IL_Replay",
        "parts": args.parts or "all", "hog_sides": sides, "winning_sides": wins,
        "actions": len(rows), "matchups": dict(matchups),
        "live_adaptation": ADAPT,
        "placement": {card: {"count": len(value["x"]),
                              "x": quantiles(value["x"]),
                              "y": quantiles(value["y"])}
                      for card, value in sorted(positions.items())},
        "top_transitions_under_4s": [
            {"first": pair[0], "second": pair[1], "count": count}
            for pair, count in transitions.most_common(30)],
        "hog_followups_under_4s": dict(hog_followups.most_common()),
        "cannon_with_recent_building_targeter": cannon_after_targeter,
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{sides} exact Hog sides, {wins} wins, {len(rows)} actions")
    print(f"wrote {args.out} and {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
