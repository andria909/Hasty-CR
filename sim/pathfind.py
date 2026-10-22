"""Grid pathfinding for the arena.

The engine walked units straight at their target, detouring to a bridge when
the river was in the way. That is fine in an empty lane and wrong everywhere
else: a unit meeting a building simply pressed into it, and the steering patch
that followed only handled one obstacle directly ahead.

This is a breadth-first flow field over the 18x32 tile grid. Cost is one BFS
per distinct goal and expanded building footprint, cached: 576 tiles.
Units with the same body radius and destination share a field. Buildings
exclude unreachable waypoints; local steering still handles moving bodies.

The river is impassable except on the two bridges, which falls out of the grid
rather than being special-cased in the movement code - so a unit that has to
cross finds a bridge because that is the only way through, not because it was
told to go there.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

from .arena import (BRIDGE_HALF_WIDTH, BRIDGE_X, MT, RIVER_BOTTOM, RIVER_TOP,
                    TILES_H, TILES_W, Point)

# Eight-way movement: diagonals matter, or units walk in staircases around
# corners. The cost of each is carried with it, because counting steps instead
# is what caused units to walk the wrong way for three seconds.
#
# With every step costing 1, the field is a Chebyshev metric: from (7,23) to a
# goal at (3,16) the distance is max(4,7) = 7, and stepping *sideways* to
# (8,23) also gives max(5,7) = 7. The move is free. Whole regions of the board
# are ties whenever |dy| > |dx|, which is most of a lane, and `next_step` broke
# them by list order - where the first entry was (1, 0), right.
#
# A Knight placed two tiles left of centre therefore walked right, visibly, for
# three seconds before turning around and heading for the left bridge. It still
# reached the correct tower, so every endpoint comparison against the reference
# engine agreed and none of them could see it. Measured against that engine on
# the same placement: it goes 7.0 -> 6.4 -> 5.6 -> 4.7 and ours went
# 7.0 -> 7.1 -> 7.4 -> 7.5 before correcting.
#
# 10 for a straight step and 14 for a diagonal is the usual integer stand-in
# for 1 and sqrt(2), and is what the reference engine uses. Sideways is no
# longer free, so the ties that caused this do not exist.
_ORTHOGONAL, _DIAGONAL = 10, 14
_STEPS = ((1, 0, _ORTHOGONAL), (-1, 0, _ORTHOGONAL),
          (0, 1, _ORTHOGONAL), (0, -1, _ORTHOGONAL),
          (1, 1, _DIAGONAL), (1, -1, _DIAGONAL),
          (-1, 1, _DIAGONAL), (-1, -1, _DIAGONAL))

_FIELD_CACHE: Dict[tuple, List[List[int]]] = {}


def walkable(tx: int, ty: int) -> bool:
    """Can a ground unit stand on this tile?

    Everything except the river, and the river is crossable on a bridge.
    """
    if not (0 <= tx < TILES_W and 0 <= ty < TILES_H):
        return False
    centre_mt = ty * MT + MT // 2
    if RIVER_TOP <= centre_mt <= RIVER_BOTTOM:
        x_mt = tx * MT + MT // 2
        return any(abs(x_mt - bx) <= BRIDGE_HALF_WIDTH for bx in BRIDGE_X)
    return True


def distance_field(goal_tx: int, goal_ty: int, blocking_circles=()) -> List[List[int]]:
    """Cost from every tile to the goal, or -1 where there is no route.

    Dijkstra rather than a breadth-first walk, because the steps no longer all
    cost the same - see `_STEPS`.
    """
    blocking_circles = tuple(sorted(blocking_circles))
    key = (goal_tx, goal_ty, blocking_circles)
    cached = _FIELD_CACHE.get(key)
    if cached is not None:
        return cached

    # Circles already include the moving body's radius. A waypoint inside a
    # solid tower cannot be reached, even if its terrain tile is walkable.
    blocked = {(tx, ty) for ty in range(TILES_H) for tx in range(TILES_W)
               if any((tx*MT+MT//2-x)**2 + (ty*MT+MT//2-y)**2 < radius**2
                      for x, y, radius in blocking_circles)}
    field = [[-1] * TILES_W for _ in range(TILES_H)]
    if not (0 <= goal_tx < TILES_W and 0 <= goal_ty < TILES_H):
        _FIELD_CACHE[key] = field
        return field

    # The goal itself may be a tower tile, which is not walkable for standing
    # on but is certainly somewhere units path towards, so it seeds the search
    # regardless.
    field[goal_ty][goal_tx] = 0
    heap = [(0, goal_tx, goal_ty)]
    while heap:
        here, tx, ty = heapq.heappop(heap)
        if here > field[ty][tx]:
            continue                     # a better route already settled this
        for dx, dy, step in _STEPS:
            nx, ny = tx + dx, ty + dy
            if not walkable(nx, ny) or (nx, ny) in blocked:
                continue
            # Prevent diagonal routes through the corner of a solid footprint.
            if dx and dy and ((tx+dx, ty) in blocked or (tx, ty+dy) in blocked):
                continue
            cost = here + step
            if field[ny][nx] == -1 or cost < field[ny][nx]:
                field[ny][nx] = cost
                heapq.heappush(heap, (cost, nx, ny))

    if len(_FIELD_CACHE) >= 512:
        _FIELD_CACHE.pop(next(iter(_FIELD_CACHE)))
    _FIELD_CACHE[key] = field
    return field


def next_step(pos: Point, goal: Point, blocking_circles=()) -> Optional[Point]:
    """The next tile centre to walk towards, following the flow downhill.

    Returns None when the goal is close enough to head for directly, or when
    no route exists - the caller then falls back to walking straight at it,
    which is the right behaviour for a target already in the same open space.
    """
    goal_tx, goal_ty = goal.x // MT, goal.y // MT
    tx, ty = pos.x // MT, pos.y // MT
    if (tx, ty) == (goal_tx, goal_ty):
        return None

    field = distance_field(goal_tx, goal_ty, blocking_circles)
    if not (0 <= tx < TILES_W and 0 <= ty < TILES_H):
        return None
    here = field[ty][tx]
    if 0 <= here <= _DIAGONAL:
        return None                      # adjacent: just walk at it

    # Ties are rare now, but a remaining one must not be settled by the order
    # of `_STEPS`: that is the whole of the bug this cost model replaced. Break
    # them on the straight-line distance to the goal instead, which never
    # prefers a compass direction.
    best = None
    best_key = (here if here >= 0 else float("inf"), 0.0, 0.0)
    for dx, dy, _step in _STEPS:
        nx, ny = tx + dx, ty + dy
        if not (0 <= nx < TILES_W and 0 <= ny < TILES_H):
            continue
        cost = field[ny][nx]
        if cost == -1:
            continue
        if blocking_circles and dx and dy and (field[ty][nx] < 0 or field[ny][tx] < 0):
            continue
        towards = ((nx * MT + MT // 2 - goal.x) ** 2
                   + (ny * MT + MT // 2 - goal.y) ** 2)
        # Equal left/right detours prefer the arena centre, preserving lane
        # reflection instead of the first compass direction in _STEPS.
        centrality = abs(nx*MT+MT//2 - TILES_W*MT//2) if blocking_circles else 0
        if (cost, towards, centrality) >= best_key:
            continue
        best_key = (cost, towards, centrality)
        best = (nx, ny)
    if best is None:
        return None
    return Point(best[0] * MT + MT // 2, best[1] * MT + MT // 2)
