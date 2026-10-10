from pathlib import Path

from sim.env import DECK_26
from sim.gamedata import load_characters, load_gamedata
from sim.match import Match
from sim.spells import load_spells


def test_a_clean_clone_can_build_the_complete_hog_deck(tmp_path: Path):
    """An absent client dump must select the versioned public fallback."""
    cards = load_gamedata(level=11, root=tmp_path / "missing-client-data")
    assert set(DECK_26) <= cards.keys()
    assert [cards[name].cost for name in DECK_26] == [3, 4, 4, 2, 1, 4, 1, 2]
    assert cards["skeletons"].summon_number == 3
    assert cards["fireball"].unit is None
    assert cards["the_log"].unit is None


def test_public_stats_scale_with_level_and_keep_aliases(tmp_path: Path):
    missing = tmp_path / "missing-client-data"
    level_11 = load_gamedata(level=11, root=missing)
    level_12 = load_gamedata(level=12, root=missing)
    assert level_12["hog_rider"].unit.hitpoints > level_11["hog_rider"].unit.hitpoints
    assert level_12["hog_rider"].unit.damage > level_11["hog_rider"].unit.damage
    assert level_11["ice_spirit"].unit is not None
    assert level_11["ice_golem"].unit is not None


def test_public_fallback_can_finish_a_deterministic_match(tmp_path: Path):
    cards = load_gamedata(level=11, root=tmp_path / "missing-client-data")
    spells = load_spells(level=11, root=tmp_path / "missing-client-data")
    match = Match(cards=cards, decks=(DECK_26, DECK_26), seed=71,
                  spells=spells)
    while not match.finished:
        match.step()
    assert match.result in {"bottom", "top", "draw"}
    assert match.elapsed_ms <= 310_000
    assert len(load_characters(level=11)) >= len(cards)
