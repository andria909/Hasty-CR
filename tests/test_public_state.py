import numpy as np

from sim.public_state import (_elixir_interval, card_vocab, public_feature_count,
                              vector_from_events, vector_from_live)


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


def test_client_and_detector_names_use_the_real_card_vocabulary():
    vocab = card_vocab()
    assert len(vocab) == 123, "changing this shifts saved checkpoint inputs"
    events = [_play(1000, "hog_rider", 4),
              _play(2000, "barbarian", 5),
              _play(3000, "ice_spirits", 1)]
    vector = vector_from_events(events, 3500)
    for card in ("hog-rider", "barbarians", "ice-spirit"):
        assert vector[vocab.index(card)] == 1
    assert vector[len(vocab)] == 0, "known cards must not land in unknown"


def test_live_opponent_cards_and_estimated_elixir_reach_the_model():
    from brain.economy import EnemyEconomy
    from brain.opponent import OpponentModel
    opponent = OpponentModel()
    opponent.observe(["barbarian", "hog_rider"], 100.0)
    economy = EnemyEconomy()
    economy.reset(100.0)
    economy.elixir = 3.5
    vector = vector_from_live(opponent, economy, 100.0)
    vocab = card_vocab()
    assert vector[vocab.index("barbarians")] == 1
    assert vector[vocab.index("hog-rider")] == 1
    assert vector[2 * (len(vocab) + 1)] == 0.35


def test_building_and_hog_counter_use_canonical_card_names():
    vocab = card_vocab()
    base = 2 * (len(vocab) + 1)
    events = [_play(1000, "inferno_tower", 5),
              _play(2000, "skeleton_army", 3)]
    simulated = vector_from_events(events, 2500)
    assert simulated[vocab.index("inferno-tower")] == 1
    assert simulated[vocab.index("skeleton-army")] == 1
    assert simulated[base + 5] == 2 / 8
    assert simulated[base + 7] == 1 / 8
    from brain.economy import EnemyEconomy
    from brain.opponent import OpponentModel
    opponent = OpponentModel()
    opponent.observe(["inferno_tower", "skeleton_army"], 2.5)
    economy = EnemyEconomy()
    economy.reset(2.5)
    live = vector_from_live(opponent, economy, 2.5)
    assert live[base + 5] == simulated[base + 5]
    assert live[base + 7] == simulated[base + 7]


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
