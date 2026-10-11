import numpy as np

from sim.public_state import (_elixir_interval, card_vocab, public_feature_count,
                              vector_from_events)


def _play(at, card, cost):
    return {"at_ms": at, "card": card, "cost": cost}


def test_public_vector_reveals_cards_without_hidden_hand_or_exact_elixir():
    events = [_play(1000, "cannon", 3), _play(4000, "fireball", 4)]
    vector = vector_from_events(events, 5000)
    vocab = card_vocab()
    assert vector[vocab.index("cannon")] == 1
    assert vector[vocab.index("fireball")] == 1
    assert vector.shape == (public_feature_count(),)
    assert np.isfinite(vector).all()


def test_recently_played_counter_is_definitely_unavailable():
    vector = vector_from_events([_play(1000, "cannon", 3)], 1500)
    width = len(card_vocab()) + 1
    assert vector[width + card_vocab().index("cannon")] == 1


def test_card_returns_after_four_other_public_plays():
    events = [_play(1000, "cannon", 3)] + [
        _play(2000 + i * 1000, card, 1)
        for i, card in enumerate(("skeletons", "ice_spirits", "log", "knight"))
    ]
    vector = vector_from_events(events, 6500)
    width = len(card_vocab()) + 1
    assert vector[width + card_vocab().index("cannon")] == 0


def test_building_counter_summary_tracks_seen_and_cycle_availability():
    width = len(card_vocab()) + 1
    base = 2 * width
    recent = vector_from_events([_play(1000, "cannon", 3)], 1500)
    assert recent[base + 7] == 1 / 8
    assert recent[base + 8] == 0  # definitely out of their hand

    cycled = vector_from_events([_play(1000, "cannon", 3)] + [
        _play(2000 + i * 1000, card, 1)
        for i, card in enumerate(("skeletons", "ice_spirits", "log", "knight"))
    ], 6500)
    assert cycled[base + 7] == 1 / 8
    assert cycled[base + 8] == 1


def test_no_revealed_building_is_unknown_not_claimed_ready():
    width = len(card_vocab()) + 1
    base = 2 * width
    vector = vector_from_events([_play(1000, "fireball", 4)], 1500)
    assert vector[base + 7] == 0
    assert vector[base + 8] == 0


def test_unknown_cards_use_a_bucket_instead_of_changing_shape():
    vector = vector_from_events([_play(1000, "future_card", 5)], 1200)
    unknown = len(card_vocab())
    assert vector[unknown] == 1


def test_elixir_regeneration_crosses_the_triple_phase_exactly():
    low, high, _recent = _elixir_interval(
        [_play(239_000, "three_musketeers", 9)], 241_000)
    expected = 1.0 + 1000 / 1400 + 1000 / 933
    assert np.isclose(low, expected)
    assert np.isclose(high, expected)
