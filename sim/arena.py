"""Arena geometry for the simulator, in the game's own units.

Units and why
-------------
Everything here is **integer millitiles and milliseconds**, because that is what
the client's own data files use: `Range = 800` is 0.8 tiles, `HitSpeed = 1600`
is 1.6 seconds. Converting to floats would mean converting twice - once on the
way in, once on the way out - and inviting exactly the drift that Clash Royale
itself avoids by running fixed-point arithmetic. Integers in, integers out.

Coordinates match `scripts/brain/arena.py` so a policy written against the live
bot runs unchanged in here:

    18 tiles wide, 32 tall, y increasing downward.
    y < 16 is the enemy half, y >= 16 is ours.
    Our princess towers sit at (3, 24) and (14, 24); theirs at (3, 7), (14, 7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple
from math import isqrt as _isqrt

MT = 1000                      # millitiles per tile
TILES_W, TILES_H = 18, 32
WIDTH = TILES_W * MT
HEIGHT = TILES_H * MT

RIVER_Y = 16 * MT              # boundary between the halves
# The river straddles the boundary rather than sitting on one side of it. It
# used to run from 15.0 to 16.0 with the boundary at 16.0, so the whole band
# belonged to the top half: measured from each tower to its own near lip, the
# bottom player walked 8.5 tiles and the top player 7.5. That is a free tile
# on every push, and with a mirrored policy on both seats it showed up as the
# bottom seat winning about two thirds of the time - which quietly invalidates
# any A/B comparison run on this board.
RIVER_TOP = 15 * MT + MT // 2      # 15.5
RIVER_BOTTOM = 16 * MT + MT // 2   # 16.5, so the band is centred on RIVER_Y

# Ground units may only cross at the bridges.
BRIDGE_X = (3 * MT + MT // 2, 14 * MT + MT // 2)

# Bridge width is not constant across the game. It is two tiles in Arenas 1-6
# and 8-9, and three tiles in Arena 7 and Arenas 10-23 - so every arena this
# project cares about, Path of Legends included, has the wider bridge. The
# simulator modelled the narrow one, which funnels a push harder than the real
# board does and matters most for exactly the bridge-spam decks it is meant to
# train against.
#
#   https://clashroyale.fandom.com/wiki/Arenas
#
# Expressed as a half-width in millitiles: 1500 spans tiles 2, 3 and 4 around a
# centre of 3.5.
BRIDGE_TILES = 3
BRIDGE_HALF_WIDTH = BRIDGE_TILES * MT // 2

TICK_MS = 50                   # 20 Hz; the client runs faster, this is enough
                               # to resolve a 1.6s hit speed and is 3x cheaper.


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)


def tile(x: float, y: float) -> Point:
    """Tile coordinates (as used by the policy) -> millitiles at tile centre."""
    return Point(int(x * MT + MT // 2), int(y * MT + MT // 2))


def to_tiles(p: Point) -> Tuple[float, float]:
    return p.x / MT, p.y / MT


def distance(a: Point, b: Point) -> int:
    """Euclidean distance in millitiles.

    Integer sqrt keeps the whole simulation in integers; the error is under one
    millitile, which is a thousandth of a tile and far below anything the game
    resolves.
    """
    # `_isqrt` directly rather than through `isqrt`: a sum of two squares is
    # never negative, so the guard that wrapper exists for cannot fire here,
    # and this is called about two and a half million times a match.
    dx, dy = a.x - b.x, a.y - b.y
    return _isqrt(dx * dx + dy * dy)


def isqrt(value: int) -> int:
    """Exact integer square root.

    This was a float estimate with correction loops, which is the right answer
    written out by hand: `math.isqrt` computes the same floor(sqrt(n)) in C.
    Results are bit-identical, so no simulated match changes - it is the single
    hottest call in the engine at over a million invocations a second, and
    target acquisition is about half of all runtime.
    """
    if value <= 0:
        return 0
    return _isqrt(value)


def is_our_half(p: Point) -> bool:
    return p.y >= RIVER_Y


def crosses_river(a: Point, b: Point) -> bool:
    return (a.y < RIVER_Y) != (b.y < RIVER_Y)


def nearest_bridge_x(x: int) -> int:
    return min(BRIDGE_X, key=lambda bx: abs(bx - x))


def on_bridge(p: Point) -> bool:
    return any(abs(p.x - bx) <= BRIDGE_HALF_WIDTH for bx in BRIDGE_X)


# The centre line, and the band around it the king tower stands on. Princess
# towers sit five and a half tiles either side of centre, the king half a tile
# off it, so this band separates them cleanly without needing to identify a
# tower by name - which matters because towers are built directly rather than
# from a card, and not every construction site passes `king=True`.
CENTRE_X = WIDTH // 2
CENTRE_BAND_MT = 3 * MT // 2       # 1.5 tiles either side


def lane_of(p: Point) -> str:
    """Which half of the arena a point stands in."""
    return "left" if p.x < CENTRE_X else "right"


def is_centre_lane(p: Point) -> bool:
    """True on the centre column, where the king tower stands.

    The king belongs to both lanes: a push down either one ends at it.
    """
    return abs(p.x - CENTRE_X) <= CENTRE_BAND_MT


def same_lane(unit: Point, tower: Point) -> bool:
    """Whether a unit at `unit` is in the lane that `tower` defends.

    Ground troops in Clash Royale are lane-committed. A unit on the right half
    walks at the right princess tower, and when that tower falls it walks at
    the king - it does not cross the arena to the surviving princess tower on
    the far side.

    Straight-line distance says the opposite, and says it in the most common
    situation in the game. From the back of the right lane with the right
    tower already down, the *left* princess tower is about 20.7 tiles away and
    the king about 22.1, so nearest-target sends a Hog diagonally across the
    whole board. That walk does not exist in a real match.
    """
    return is_centre_lane(tower) or lane_of(unit) == lane_of(tower)


def clamp_to_arena(p: Point) -> Point:
    return Point(max(0, min(WIDTH - 1, p.x)), max(0, min(HEIGHT - 1, p.y)))


def in_arena(p: Point) -> bool:
    """Inside the playable board. Spells are limited by this and nothing else."""
    return 0 <= p.x < WIDTH and 0 <= p.y < HEIGHT


# Tower anchors, in tile coordinates, matching the live policy's convention.
# The left princess tower sat on column 4 while the right sat on 14. With the
# arena 18 wide the mirror line is 9, so that put the left tower 4.5 tiles from
# centre and the right one 5.5 - the board was not mirrored left to right, and
# the left tower did not even line up with its own bridge at 3.5. The live
# bot's arena, calibrated against the real game, has them on columns 3 and 14
# of an 0..17 grid, which is 3.5 and 14.5 here. A top-versus-bottom symmetry
# check cannot see this one, because it is the same on both halves.
ALLY_PRINCESS = {"left": tile(3, 24), "right": tile(14, 24)}
ENEMY_PRINCESS = {"left": tile(3, 7), "right": tile(14, 7)}
ALLY_KING = tile(9, 28)
ENEMY_KING = tile(9, 3)


# The "pocket": taking a princess tower opens ground across the river in front
# of where it stood, and the same happens to you when you lose one.
#
# **These two numbers are an approximation and are not sourced.** The rule is
# real and the client enforces it, but the shipped data does not define it -
# `globals.csv` has no deploy-area key and the arena tile map is not in the
# CSVs - and no published reference states it tile by tile. So the shape here
# is a rectangle around the fallen tower, seven tiles wide and reaching from
# one tile behind it to the river.
#
# What would settle it, stated precisely so a later pass cannot quietly swap a
# guess for a guess: in a real match after taking one princess tower, walk a
# placement along the enemy half and record the outermost accepted x on both
# the arena-edge side and the centre side, and the furthest accepted y behind
# the tower. Three numbers, one match.
#
# Getting the width wrong does not break the rule, it mis-sizes the reward for
# a crown - so it is worth measuring, and it was worth far less than the bug
# it replaced, which was the pocket being unreachable from the action space
# entirely.
POCKET_HALF_WIDTH_MT = 3 * MT     # unverified; see above
POCKET_BEHIND_TOWER_MT = MT       # unverified; see above


@dataclass(frozen=True)
class ArenaGeometry:
    """Versioned crown anchors; terrain and pocket shape stay legacy/unverified.

    Coordinates are entity positions, not a claim about rendered/hitbox centres.
    Profiles are immutable and pickle/deepcopy-safe; each Match owns its choice.
    """
    profile_id: str
    bottom: tuple[Point, Point, Point]  # left princess, right princess, king
    top: tuple[Point, Point, Point]
    terrain_id: str = 'legacy_terrain_v1'
    limitations: tuple[str, ...] = ('deployment_pocket_shape_unverified',)

    def __post_init__(self):
        if self.terrain_id != 'legacy_terrain_v1':
            raise ValueError('Configurable terrain is not implemented; refusing unsupported geometry terrain')
        for anchors in (self.bottom,self.top):
            if not isinstance(anchors,tuple) or len(anchors)!=3 or any(not isinstance(p,Point) for p in anchors):
                raise ValueError('Geometry anchors must be immutable triples of Points')
        if not isinstance(self.limitations,tuple):
            raise ValueError('Geometry limitations must be immutable')

    def princess(self, side: int):
        points=self._side(side)
        return dict(left=points[0],right=points[1])

    def king(self, side: int):
        return self._side(side)[2]

    def _side(self, side):
        if side not in (-1,1):
            raise ValueError('Geometry side must be -1 or +1')
        return self.bottom if side==1 else self.top

    @property
    def fingerprint(self):
        import hashlib
        import json
        payload=dict(profile_id=self.profile_id,terrain_id=self.terrain_id,
            bottom=[(p.x,p.y) for p in self.bottom],top=[(p.x,p.y) for p in self.top],
            limitations=self.limitations)
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


LEGACY_GEOMETRY = ArenaGeometry('legacy_v1',
    (ALLY_PRINCESS['left'],ALLY_PRINCESS['right'],ALLY_KING),
    (ENEMY_PRINCESS['left'],ENEMY_PRINCESS['right'],ENEMY_KING))
NATIVE_ANCHORS_CANDIDATE_GEOMETRY = ArenaGeometry('native_anchors_candidate_v1',
    (Point(3500,25500),Point(14500,25500),Point(9000,29000)),
    (Point(3500,6500),Point(14500,6500),Point(9000,3000)),
    limitations=('deployment_pocket_shape_unverified','native_entity_anchor_hitbox_offset_unverified',
                 'terrain_inherited_from_legacy_not_native_validated'))


def resolve_geometry(value=None):
    profiles=(LEGACY_GEOMETRY,NATIVE_ANCHORS_CANDIDATE_GEOMETRY)
    if value is None:
        return LEGACY_GEOMETRY
    for profile in profiles:
        if value==profile.profile_id or value==profile:
            return profile
    raise ValueError('Unknown/unsupported geometry profile')


def deploy_area_ok(p: Point, side: int, enemy_princess_down: Iterable[str] = (), *, geometry=None) -> bool:
    """Whether `side` may deploy a troop at `p`.

    side is +1 for the player at the bottom (ours) and -1 for the top.
    `enemy_princess_down` names the fallen towers of `side`'s opponent, so the
    call is symmetric: passing our own downed towers with side=-1 opens their
    pocket in our half, which is what a lost tower costs you.
    """
    profile=resolve_geometry(geometry)
    if side not in (-1,1):
        raise ValueError('Deployment side must be -1 or +1')
    if not (0 <= p.x < WIDTH and 0 <= p.y < HEIGHT):
        return False
    if side > 0:
        if p.y >= RIVER_Y:
            return True
        for lane in enemy_princess_down:
            centre = profile.princess(-side)[lane]
            if (abs(p.x - centre.x) <= POCKET_HALF_WIDTH_MT
                    and p.y >= centre.y - POCKET_BEHIND_TOWER_MT):
                return True
        return False
    if p.y < RIVER_Y:
        return True
    for lane in enemy_princess_down:
        centre = profile.princess(-side)[lane]
        if (abs(p.x - centre.x) <= POCKET_HALF_WIDTH_MT
                and p.y <= centre.y + POCKET_BEHIND_TOWER_MT):
            return True
    return False
