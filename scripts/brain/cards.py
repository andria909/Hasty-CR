"""Independent card recognition with an explicit uncertainty threshold.

The current eight-card deck uses menu artwork captured before the validation
battles. Compare only the upper illustration: borders, elixir badges and level
labels differ between the menu, hand and preview. Normalised greyscale
correlation handles linear dimming/desaturation; uncertain readings abstain.
Legacy full-card references remain available for other decks and custom assets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

VENDOR_CARDS = (
    Path(__file__).resolve().parents[2]
    / "vendor" / "ClashRoyaleBuildABot" / "clashroyalebuildabot" / "images" / "cards"
)

# CARD_CONFIG in upstream constants.py, expressed on the 368x652 screenshot.
_CARD_Y, _CARD_H = 543, 73
_CARD_X0, _CARD_W, _CARD_DX = 84, 61, 69
_SRC_W, _SRC_H = 368, 652

PATCH = 32
MIN_MARGIN = 0.04     # required gap between best and second-best correlation
MIN_SCORE = 0.30      # below this nothing is recognised at all
CURRENT_CARDS = Path(__file__).resolve().parents[2] / "data/card_references/current_20261021"


def next_card_box(width: int, height: int) -> Tuple[int, int, int, int]:
    return tuple(round(v * scale) for v, scale in zip(
        (21, 609, 47, 642), (width / 368, height / 652) * 2))


def _art_signature(image: Image.Image) -> np.ndarray:
    # Exclude frame, cost badge, and deck-menu level label. The upper artwork
    # is shared by menu, affordable/grey hand cards, and the small preview.
    w, h = image.size
    return _signature(image.crop((w * .08, h * .05, w * .92, h * .73)))


def hand_boxes(width: int, height: int) -> List[Tuple[int, int, int, int]]:
    """The four hand-card boxes, scaled to an arbitrary frame size."""
    sx, sy = width / _SRC_W, height / _SRC_H
    boxes = []
    for slot in range(4):
        x0 = _CARD_X0 + slot * _CARD_DX
        boxes.append((
            int(round(x0 * sx)), int(round(_CARD_Y * sy)),
            int(round((x0 + _CARD_W) * sx)), int(round((_CARD_Y + _CARD_H) * sy)),
        ))
    return boxes


def _signature(image: Image.Image) -> np.ndarray:
    """Zero-mean, unit-norm greyscale patch: the NCC-ready form."""
    patch = np.asarray(
        image.convert("L").resize((PATCH, PATCH), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ).ravel()
    patch -= patch.mean()
    norm = np.linalg.norm(patch)
    return patch / norm if norm > 1e-6 else patch


class CardClassifier:
    def __init__(self, deck: List[str], cards_dir: Path | None = None):
        self.deck = list(deck)
        directory = Path(cards_dir or VENDOR_CARDS)
        self.references: Dict[str, np.ndarray] = {}
        self.art_references: Dict[str, np.ndarray] = {}
        for name in self.deck:
            path = directory / f"{name}.jpg"
            if path.exists():
                self.references[name] = _signature(Image.open(path))
            # A custom reference directory must remain self-contained.
            current = CURRENT_CARDS / f"{name}.png"
            if cards_dir is None and current.exists():
                with Image.open(current) as image:
                    self.art_references[name] = _art_signature(image)

    @property
    def ready(self) -> bool:
        return bool(self.deck) and (len(self.references) == len(self.deck)
                                   or len(self.art_references) == len(self.deck))

    def classify_patch(self, image: Image.Image) -> Tuple[Optional[str], float]:
        """Best matching card for one crop, or None when it is ambiguous."""
        return self._match(_signature(image), self.references)

    @staticmethod
    def _match(signature, references) -> Tuple[Optional[str], float]:
        if not references:
            return None, 0.0
        scored = sorted(
            ((float(np.dot(signature, ref)), name)
             for name, ref in references.items()),
            reverse=True,
        )
        best_score, best_name = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else -1.0
        if best_score < MIN_SCORE or (best_score - runner_up) < MIN_MARGIN:
            return None, best_score
        return best_name, best_score

    def classify_hand_scored(
        self, frame: Image.Image
    ) -> List[Tuple[Optional[str], float]]:
        """Four (card, correlation) slot readings from a full-resolution frame.

        The score is what lets a caller settle a duplicate: a Clash Royale hand
        holds four *distinct* cards, so when two slots name the same card the
        weaker correlation is the wrong one.
        """
        boxes = hand_boxes(frame.width, frame.height)
        if len(self.art_references) == len(self.deck):
            return [self._match(_art_signature(frame.crop(box)), self.art_references)
                    for box in boxes]
        return [self.classify_patch(frame.crop(box)) for box in boxes]

    def classify_next(self, frame: Image.Image) -> Tuple[Optional[str], float]:
        """Read the preview independently; abstain if current art is unavailable."""
        if len(self.art_references) != len(self.deck):
            return None, 0.0
        return self._match(_art_signature(frame.crop(next_card_box(*frame.size))),
                           self.art_references)

    def classify_hand(self, frame: Image.Image) -> List[Optional[str]]:
        """Four slot readings from a full-resolution frame."""
        return [name for name, _score in self.classify_hand_scored(frame)]
