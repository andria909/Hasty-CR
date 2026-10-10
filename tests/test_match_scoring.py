"""Scoring rules that do not require extracted game data."""

from types import SimpleNamespace

from sim.match import Match


def _tower(*, alive: bool = True, hp: float = 1.0):
    return SimpleNamespace(alive=alive, hp_fraction=hp)


def _match(enemy, ours=None):
    if ours is None:
        ours = {name: _tower() for name in ("left", "right", "king")}
    return SimpleNamespace(towers={1: ours, -1: enemy})


def test_direct_king_takedown_is_three_crowns():
    enemy = {"left": _tower(), "right": _tower(), "king": _tower(alive=False)}
    assert Match.crowns_for(_match(enemy), 1) == 3


def test_princess_towers_score_one_each_while_king_lives():
    enemy = {
        "left": _tower(alive=False),
        "right": _tower(alive=False),
        "king": _tower(),
    }
    assert Match.crowns_for(_match(enemy), 1) == 2


def test_tiebreak_includes_damaged_king_tower():
    ours = {
        "left": _tower(hp=0.8), "right": _tower(hp=0.9), "king": _tower(hp=0.2),
    }
    enemy = {
        "left": _tower(hp=0.5), "right": _tower(hp=0.7), "king": _tower(hp=1.0),
    }
    assert Match._tiebreak(_match(enemy, ours)) == "top"
