"""Run the deployed CV stack on saved real frames, without ADB or game actions."""

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cr_bot as live


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--overlays", type=Path)
    parser.add_argument("--original-units", action="store_true",
                        help="reproduce the original board detector without project corrections")
    args = parser.parse_args()
    live.logger.remove()
    detector = live.Detector(live.DECK)
    if not args.original_units:
        live.configure_unit_detection(detector, None)
    classifier = live.Brain(learn=False, use_advisor=False).classifier
    pairs = json.loads(args.pairs.read_text())["rows"]
    selected = {r["path"] for r in json.loads(args.selection.read_text())} if args.selection else set()
    if args.overlays:
        args.overlays.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print("classifier_ready", classifier.ready, flush=True)
    with args.output.open("w") as output:
        for index, pair in enumerate(pairs):
            for which in ("before_frame", "after_frame"):
                path = pair[which]
                full = Image.open(path).convert("RGB")
                small = full.resize((368, 652))
                state = detector.run(small)
                if state is None:
                    output.write(json.dumps({"path": path, "no_state": True}) + "\n")
                    continue
                scored = classifier.classify_hand_scored(full) if classifier.ready else []
                winners = {}
                for slot, (name, score) in enumerate(scored):
                    if name is not None and (name not in winners or score > scored[winners[name]][1]):
                        winners[name] = slot
                dedup = [name if name is not None and winners.get(name) == slot else None
                         for slot, (name, _) in enumerate(scored)]
                hp_names = ("left_ally_princess_hp", "right_ally_princess_hp",
                            "left_enemy_princess_hp", "right_enemy_princess_hp")
                raw_hp = {key: getattr(state.numbers, key).number for key in hp_names}
                raw_cards = [c.name for c in state.cards]
                state, ally_hp, enemy_hp = live.patch_tower_numbers(state, full)
                units = []
                for side, detections in (("ally", state.allies), ("enemy", state.enemies)):
                    for item in detections:
                        units.append({"name": str(item.unit.name), "side": side,
                                      "confidence": float(item.position.conf),
                                      "bbox_small": list(item.position.bbox),
                                      "tile": [item.position.tile_x, item.position.tile_y]})
                record = {"path": path, "pair_index": index, "which": which,
                          "screen": state.screen.name, "raw_cards_next_then_hand": raw_cards,
                          "ncc_scored": scored, "ncc_after_dedup": dedup,
                          "ready_slots": list(state.ready), "elixir": state.numbers.elixir.number,
                          "hp_raw": raw_hp, "hp_calibrated_ally": ally_hp,
                          "hp_calibrated_enemy": enemy_hp, "units": units,
                          "battle_guard_passed": live.guarded.battle_guard(state)}
                output.write(json.dumps(record, default=lambda value: value.item()) + "\n")
                if path in selected and args.overlays:
                    annotated = full.copy()
                    draw = ImageDraw.Draw(annotated)
                    for unit in units:
                        b = unit["bbox_small"]
                        box = (b[0]*full.width/368,b[1]*full.height/652,
                               b[2]*full.width/368,b[3]*full.height/652)
                        color = "cyan" if unit["side"] == "ally" else "red"
                        draw.rectangle(box, outline=color, width=3)
                        draw.text((box[0],max(0,box[1]-14)),
                                  f'{unit["name"]} {unit["confidence"]:.2f}', fill=color,
                                  stroke_width=1, stroke_fill="black")
                    annotated.save(args.overlays / (Path(path).stem + ".jpg"), quality=95)
            if index % 10 == 0:
                print("pairs",index+1,flush=True)


if __name__ == "__main__":
    main()
