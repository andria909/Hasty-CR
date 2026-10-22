"""Versioned observation quantization; never changes world positions or actions.

``physical_live`` is an existing external proxy transform, not an alias for v2.
Selecting v2 changes policy inputs and requires a separately evaluated policy.
"""
from dataclasses import dataclass
from math import floor


@dataclass(frozen=True)
class PolicyCoordinates:
    """Immutable, explicit encoder contract for the fixed 18 by 32 policy grid."""

    name: str

    def __post_init__(self):
        if self.name not in ("legacy_training", "physical_live_cells_v2"):
            raise ValueError(f"Unknown policy coordinate contract: {self.name!r}")

    def cell(self, position, side: int = 1) -> tuple[int, int]:
        # Keep floating division and round(31-y) in this precise order for
        # checkpoint compatibility (including Python's ties-to-even).
        x, y = position.x / 1000.0, position.y / 1000.0
        if self.name == "legacy_training":
            gx, gy = round(x), round(31 - y)
        else:
            gx, gy = floor(x), floor(y)
        gx, gy = int(min(17, max(0, gx))), int(min(31, max(0, gy)))
        if side < 0:
            gx, gy = 17 - gx, 31 - gy
        return gx, gy


LEGACY_TRAINING = PolicyCoordinates("legacy_training")
PHYSICAL_LIVE_CELLS_V2 = PolicyCoordinates("physical_live_cells_v2")


def policy_coordinates(name: str = "legacy_training") -> PolicyCoordinates:
    """Resolve only explicit contract IDs; ambiguous live aliases are rejected."""
    if name == LEGACY_TRAINING.name:
        return LEGACY_TRAINING
    if name == PHYSICAL_LIVE_CELLS_V2.name:
        return PHYSICAL_LIVE_CELLS_V2
    raise ValueError(f"Unknown policy coordinate contract: {name!r}")
