"""The live bridge must show the network the same board the simulator did.

A policy trained on `sim.env.observe` and served on a differently-oriented view
does not fail loudly. It plays - competently even, for a while - and defends
the wrong lane. This project has already paid for one silent mirror:
`grid_to_point(13, 20, -1)` was reflecting a placement into the opposite lane,
which made the Hog look unstoppable in every test that used it.

So these tests do not check the bridge against a hand-written expectation.
They build a real simulator match, take `sim.env.observe`'s view of it, then
hand the *same board* to the live path as if vision had reported it, and
require the two to agree cell for cell.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from brain import arena as live_arena                              # noqa: E402
from brain.rl_policy import (apply_live_safety_mask, mask_from_live,
                             observation_from_live)  # noqa: E402
from sim import arena as sim_arena                                 # noqa: E402
from sim.env import (ACTIONS, DECK_26, GRID_W, RIVER_ROW, TILES,   # noqa: E402
                     ClashEnv, legal_mask, observe)


# --------------------------------------------------------- the two grids agree

def test_the_live_and_simulator_grids_are_the_same_convention():
    """The bridge applies no coordinate transform, so this must hold."""
    assert (live_arena.GRID_W, live_arena.GRID_H) == (18, 32)
    assert live_arena.RIVER_Y == RIVER_ROW
    for lane in ("left", "right"):
        live_x, live_y = live_arena.ENEMY_PRINCESS[lane]
        sim_point = sim_arena.ENEMY_PRINCESS[lane]
        assert abs(sim_point.x / sim_arena.MT - live_x) <= 0.5, lane
        assert abs(sim_point.y / sim_arena.MT - live_y) <= 0.5, lane
        live_x, live_y = live_arena.ALLY_PRINCESS[lane]
        sim_point = sim_arena.ALLY_PRINCESS[lane]
        assert abs(sim_point.x / sim_arena.MT - live_x) <= 0.5, lane
        assert abs(sim_point.y / sim_arena.MT - live_y) <= 0.5, lane


def test_our_half_means_the_same_thing_on_both_sides():
    assert live_arena.our_half(SimpleNamespace(x=9, y=RIVER_ROW))
    assert not live_arena.our_half(SimpleNamespace(x=9, y=RIVER_ROW - 1))


# ------------------------------------------------- the same board, both routes

def played_out_match(seed: int = 3, steps: int = 260):
    """A match stepped far enough to have units of both sides on the board."""
    env = ClashEnv(seed=seed, opponent="meta")
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        legal = np.flatnonzero(legal_mask(env.match, env._cards, 1))
        _obs, _r, term, trunc, _info = env.step(int(rng.choice(legal)))
        if term or trunc:
            break
    return env


def live_view_of(match):
    """What vision would have reported for this board, in live types."""
    allies, tracks = [], []
    for entity in match.battle.entities.values():
        if not entity.alive or entity.is_tower:
            continue
        x, y = sim_arena.to_tiles(entity.pos)
        # Live perception has already converted bottom-up detector tiles into
        # the top-down policy grid by this point.
        cell = SimpleNamespace(x=x, y=31 - y)
        if entity.side > 0:
            allies.append((cell, entity.name))
        else:
            tracks.append(SimpleNamespace(cell=cell, name=entity.name))
    player = match.players[1]
    ours = match.tower_fractions(1)
    theirs = match.tower_fractions(-1)
    hand = {card: slot for slot, card in enumerate(player.hand[:4])}
    obs = SimpleNamespace(
        allies=[cell for cell, _ in allies], tracks=tracks,
        elixir=player.elixir / 1000.0, elapsed=match.elapsed_ms / 1000.0,
        multiplier=2800.0 / max(1, match.regen_ms()),
        ally_hp=ours, enemy_hp=theirs, hand=hand,
        # The card detector reads this every frame as slot 0:
        # CARD_CONFIG[0] is its own crop, separate from the four
        # evenly spaced hand slots.
        next_card=player.next_card,
        opponent_events=[event for event in match.play_log if event["side"] == -1])
    return obs, allies


def test_unit_positions_land_in_the_same_cells_as_the_simulator():
    env = played_out_match()
    match = env.match
    obs, allies = live_view_of(match)
    if not allies and not obs.tracks:
        pytest.skip("no units on the board in this rollout")

    from_sim = observe(match, 1)["planes"]
    from_live = observation_from_live(obs, allies)["planes"]
    for plane, label in ((0, "ally units"), (1, "enemy units")):
        assert np.array_equal(from_sim[plane], from_live[plane]), (
            f"{label} plane differs between the simulator and the live "
            f"bridge; the two views are not the same board")


def test_air_and_building_planes_agree_too():
    env = played_out_match()
    obs, allies = live_view_of(env.match)
    from_sim = observe(env.match, 1)["planes"]
    from_live = observation_from_live(obs, allies)["planes"]
    for plane in (4, 5, 6, 7):
        assert np.array_equal(from_sim[plane], from_live[plane]), (
            f"plane {plane} (air/building) differs")


def test_recovering_ally_names_is_what_fills_the_ally_planes():
    """Without names, three of the four ally planes are silently zero."""
    env = played_out_match()
    obs, allies = live_view_of(env.match)
    if not allies:
        pytest.skip("no allies on the board")
    named = observation_from_live(obs, allies)["planes"]
    nameless = observation_from_live(obs, None)["planes"]
    assert np.array_equal(named[0], nameless[0]), "positions should not change"
    assert nameless[2].sum() == 0.0
    assert named[2].sum() > 0.0, (
        "ally hit-point plane is empty even with names; the bridge is not "
        "recovering unit identity")


def test_the_scalars_match_the_simulators():
    env = played_out_match()
    obs, allies = live_view_of(env.match)
    from_sim = observe(env.match, 1)["scalars"]
    from_live = observation_from_live(obs, allies)["scalars"]
    # The whole vector. This used to compare everything *except* the
    # next-card one-hot, and that exemption is what let a sixth of the
    # scalars be wrong in every live match while the test stayed green.
    shared = len(from_sim)
    assert np.allclose(from_sim[:shared], from_live[:shared], atol=1e-3), (
        f"scalars differ:\n sim  {from_sim[:shared]}\n live {from_live[:shared]}")


def test_the_next_card_is_supplied_live():
    """It was never a blind spot, and asserting it was made it permanent.

    This test used to assert the tail summed to zero, calling the next
    card something "live vision cannot see". That was false: the card
    detector reads it every frame as `state.cards[0]`, whose crop is
    declared separately from the four hand slots in CARD_CONFIG. The
    bridge simply never copied it across.

    The cost was real. The policy trained with one of these eight always
    set, then played live with none set - a state occurring nowhere in
    training - and for a cycle deck this is the input that says how far
    away the win condition is. Over five matches against a person: hog
    share 3-5% live against 11-13% in the simulator, 266-360 holds.
    """
    env = played_out_match()
    obs, allies = live_view_of(env.match)
    from_live = observation_from_live(obs, allies)["scalars"]
    tail = 3 + 4 + 4 * len(DECK_26)
    next_card = from_live[tail:tail + len(DECK_26)]
    assert next_card.sum() == 1.0, "exactly one next card must be set"
    assert (next_card.argmax()
            == DECK_26.index(env.match.players[1].next_card))


# ------------------------------------------------------------------- the mask

def test_the_live_mask_matches_the_simulators():
    env = played_out_match()
    obs, allies = live_view_of(env.match)
    from_sim = legal_mask(env.match, env._cards, 1)
    from_live = apply_live_safety_mask(mask_from_live(obs), obs, allies)
    # Abilities are not reachable live, so compare the card actions only.
    assert np.array_equal(from_sim[:1 + 4 * TILES], from_live[:1 + 4 * TILES]), (
        "the live mask allows a different set of placements than the "
        "simulator the policy was trained in")


def test_the_live_mask_opens_the_enemy_half_after_a_crown():
    env = played_out_match()
    obs, _allies = live_view_of(env.match)
    obs.elixir = 10.0
    before = mask_from_live(obs)
    alive = next((lane for lane in ("left", "right")
                  if obs.enemy_hp.get(lane, 0.0) > 0.0), None)
    if alive is None:
        pytest.skip("both enemy princess towers are already down")
    obs.enemy_hp = dict(obs.enemy_hp, **{alive: 0.0})
    after = mask_from_live(obs)
    def enemy_half(mask):
        total = 0
        for slot in range(4):
            grid = mask[1 + slot * TILES:1 + (slot + 1) * TILES].reshape(-1, GRID_W)
            total += int(grid[:RIVER_ROW].sum())
        return total
    assert enemy_half(after) > enemy_half(before), (
        "taking a tower opened no ground live, though it does in the sim")


def test_an_empty_hand_leaves_only_hold():
    obs = SimpleNamespace(allies=[], tracks=[], elixir=0.0, elapsed=0.0,
                          multiplier=1.0, ally_hp={"left": 1.0, "right": 1.0},
                          enemy_hp={"left": 1.0, "right": 1.0}, hand={})
    mask = mask_from_live(obs)
    assert mask[0] and mask.sum() == 1
    assert len(mask) == ACTIONS


def _live_obs(hand, tracks=(), threats=(), threat_score=0.0):
    return SimpleNamespace(
        allies=[], tracks=list(tracks), threats=list(threats),
        threat_score=threat_score, elixir=10.0, elapsed=30.0,
        multiplier=1.0, ally_hp={"left": 1.0, "right": 1.0},
        enemy_hp={"left": 1.0, "right": 1.0}, hand=hand,
    )


def test_live_safety_blocks_hog_while_a_push_is_active():
    threat = SimpleNamespace(cell=SimpleNamespace(x=4, y=22), name="knight")
    obs = _live_obs({"hog_rider": 0, "knight": 1}, tracks=[threat],
                    threats=[threat], threat_score=6.0)
    mask = apply_live_safety_mask(mask_from_live(obs), obs)
    hog = mask[1:1 + TILES]
    knight = mask[1 + TILES:1 + 2 * TILES]
    assert not hog.any()
    assert knight.any()


def test_live_safety_keeps_hog_at_the_two_bridges():
    obs = _live_obs({"hog_rider": 0, "electro_spirit": 1})
    grid = apply_live_safety_mask(mask_from_live(obs), obs)[1:1 + TILES].reshape(32, 18)
    assert grid[16, 4] and grid[18, 13]
    assert not grid[25, 4] and not grid[20, 9]


def test_live_safety_reserves_spirit_for_hog_and_defenders_for_defence():
    obs = _live_obs({"hog_rider": 0, "electro_spirit": 1,
                     "knight": 2, "musketeer": 3})
    mask = apply_live_safety_mask(mask_from_live(obs), obs)
    assert mask[1:1 + TILES].any(), "the Hog attack should remain available"
    for slot in (1, 2, 3):
        assert not mask[1 + slot * TILES:1 + (slot + 1) * TILES].any()

    hog = [(SimpleNamespace(x=4, y=17), "hog_rider")]
    supported = apply_live_safety_mask(mask_from_live(obs), obs, hog)
    spirit = supported[1 + TILES:1 + 2 * TILES].reshape(32, 18)
    assert spirit[17, 4]
    assert not spirit[17, 13], "Spirit must follow the Hog's lane"


def test_live_safety_places_cannon_only_in_the_central_pull_pocket():
    giant = SimpleNamespace(cell=SimpleNamespace(x=4, y=22), name="giant")
    obs = _live_obs({"cannon": 0}, tracks=[giant], threats=[giant])
    cannon = apply_live_safety_mask(mask_from_live(obs), obs)[
        1:1 + TILES].reshape(32, 18)
    assert cannon[19:22, 5:8].any()
    assert not cannon[:, :5].any()
    assert not cannon[:, 8:].any()


def test_live_safety_keeps_ground_cards_out_of_a_pure_air_defence():
    balloon = SimpleNamespace(cell=SimpleNamespace(x=13, y=21), name="balloon")
    obs = _live_obs({"musketeer": 0, "cannon": 1,
                     "knight": 2, "skeletons": 3},
                    tracks=[balloon], threats=[balloon])
    mask = apply_live_safety_mask(mask_from_live(obs), obs)
    assert mask[1:1 + TILES].any(), "Musketeer must remain available"
    for slot in (1, 2, 3):
        assert not mask[1 + slot * TILES:1 + (slot + 1) * TILES].any()


def test_hog_without_spirit_waits_until_elixir_is_full():
    obs = _live_obs({"hog_rider": 0})
    obs.elixir = 9.9
    assert not apply_live_safety_mask(mask_from_live(obs), obs)[1:1 + TILES].any()
    obs.elixir = 10.0
    assert apply_live_safety_mask(mask_from_live(obs), obs)[1:1 + TILES].any()


def test_live_safety_does_not_cast_fireball_on_one_moving_unit():
    one = SimpleNamespace(cell=SimpleNamespace(x=4, y=20), name="knight")
    obs = _live_obs({"fireball": 0}, tracks=[one])
    mask = apply_live_safety_mask(mask_from_live(obs), obs)
    assert not mask[1:1 + TILES].any()


def test_live_safety_allows_spell_on_a_real_cluster():
    tracks = [
        SimpleNamespace(cell=SimpleNamespace(x=4, y=20), name="archer"),
        SimpleNamespace(cell=SimpleNamespace(x=5, y=20), name="archer"),
    ]
    obs = _live_obs({"fireball": 0, "zap": 1}, tracks=tracks)
    mask = apply_live_safety_mask(mask_from_live(obs), obs)
    assert mask[1:1 + TILES].any()
    assert mask[1 + TILES:1 + 2 * TILES].any()


# ----------------------------------------------------- the whole live decision

def _checkpoint():
    """A real checkpoint, preferring a vetted one. Skips if none exists.

    This used to name four specific files under `tmp/rl`, which is the scratch
    directory a training run writes into - so it exercised whatever happened to
    be lying there, including `hog26v5_best.pt`, a collapsed policy that played
    its win condition 0% of the time. Vetted checkpoints live under
    `checkpoints/` with a manifest recording what they scored, so those come
    first and scratch is only the fallback.

    A clone has neither, and skipping is right there: the weights are far too
    large to ship in the repository.
    """
    for root in (ROOT / "checkpoints", ROOT / "tmp" / "rl"):
        if not root.is_dir():
            continue
        found = sorted(
            root.glob("**/*.pt"),
            # Scratch may contain behaviour clones and interrupted/last
            # snapshots alongside the evaluated policy.  Prefer the artefact
            # explicitly selected as best; alphabetical order picked a clone
            # that holds on every board and made this live-policy test depend
            # on unrelated leftovers in tmp/rl.
            key=lambda path: (not path.name.endswith("_best.pt"),
                              "ppo" not in path.name.lower(), str(path)))
        if found:
            return found[0]
    pytest.skip("no trained checkpoint on disk (expected in a fresh clone)")


def _network(checkpoint):
    import torch
    from sim.env import NUM_PLANES, NUM_SCALARS
    from sim.train_ppo import build_network, load_policy_state
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net = build_network(NUM_PLANES, NUM_SCALARS, ACTIONS)
    load_policy_state(net, blob["state_dict"])
    net.eval()
    return net


def _greedy(net, view, mask):
    import torch
    with torch.no_grad():
        planes = torch.from_numpy(view["planes"]).unsqueeze(0)
        scalars = torch.from_numpy(view["scalars"]).unsqueeze(0)
        logits, _ = net(planes, scalars)
        logits = logits[0].float()
        logits[~torch.from_numpy(mask)] = float("-inf")
        return int(torch.argmax(logits).item())


def _blind_hp(view):
    """The same view with the hit-point planes removed.

    Live vision reports that a unit is there, not how hurt it is, so planes 2
    and 3 are the one part of the observation the bridge cannot reproduce.
    Zeroing them on both sides isolates everything it *can*: orientation,
    binning, ownership, air, buildings, scalars and the mask.
    """
    scrubbed = {"planes": view["planes"].copy(), "scalars": view["scalars"]}
    scrubbed["planes"][2] = 0.0
    scrubbed["planes"][3] = 0.0
    return scrubbed


def test_the_same_board_gets_the_same_action_given_the_same_information():
    """The invariant the whole bridge exists to hold.

    Not "does it play a card" - a policy that plays 50 cards in a 600-decision
    match holds on most frames, and asserting it must act on an arbitrary frame
    tests nothing. What must be true is that the two routes to the network
    agree: same board and same information in, same choice out.

    Hit points are excluded because live cannot supply them; that gap is
    measured separately below rather than hidden here.
    """
    pytest.importorskip("torch")
    net = _network(_checkpoint())
    agreed = compared = 0
    for seed in range(8):
        env = played_out_match(seed=seed, steps=120 + 40 * seed)
        if env.match.finished:
            continue
        obs, allies = live_view_of(env.match)
        sim_mask = legal_mask(env.match, env._cards, 1)
        live_mask = apply_live_safety_mask(mask_from_live(obs), obs, allies)
        differing = np.flatnonzero(sim_mask != live_mask)
        assert not len(differing), (
            f"masks differ at seed {seed}: {differing[:12].tolist()} "
            f"sim={sim_mask[differing[0]]} live={live_mask[differing[0]]} "
            f"elixir={obs.elixir} hand={env.match.players[1].hand[:4]} allies={[(n, c.x, c.y) for c, n in allies]} "
            f"tracks={[(t.name, t.cell.x, t.cell.y) for t in obs.tracks]}")
        from_sim = _greedy(net, _blind_hp(observe(env.match, 1)), sim_mask)
        from_live = _greedy(net, _blind_hp(observation_from_live(obs, allies)),
                            live_mask)
        compared += 1
        agreed += from_sim == from_live
    assert compared, "no comparable boards"
    assert agreed == compared, (
        f"the live bridge chose a different action than the simulator on "
        f"{compared - agreed} of {compared} identical boards, with hit points "
        f"excluded - so this is a real orientation, binning or mask bug")


def test_everything_except_hit_points_reproduces_exactly():
    """Planes 2 and 3 are the only ones allowed to differ."""
    env = played_out_match()
    obs, allies = live_view_of(env.match)
    from_sim = observe(env.match, 1)["planes"]
    from_live = observation_from_live(obs, allies)["planes"]
    for plane in (0, 1, 4, 5, 6, 7):
        assert np.array_equal(from_sim[plane], from_live[plane]), (
            f"plane {plane} differs, and only 2 and 3 are permitted to")


def test_the_hit_point_gap_is_real_and_is_measured_not_assumed():
    """How often full-HP vision changes the decision, as a number.

    This is the cost of not measuring damage live. It is recorded here so a
    change in it is visible, and so nobody has to guess whether the
    approximation is harmless. It is not asserted to be zero - it is not.
    """
    pytest.importorskip("torch")
    net = _network(_checkpoint())
    changed = compared = 0
    for seed in range(12):
        env = played_out_match(seed=seed, steps=100 + 30 * seed)
        if env.match.finished:
            continue
        obs, allies = live_view_of(env.match)
        mask = legal_mask(env.match, env._cards, 1)
        true_hp = _greedy(net, observe(env.match, 1), mask)
        full_hp = _greedy(net, observation_from_live(obs, allies), mask)
        compared += 1
        changed += true_hp != full_hp
    assert compared >= 6, "not enough boards to say anything"
    share = changed / compared
    assert share <= 0.5, (
        f"treating every unit as undamaged changed the chosen action on "
        f"{changed}/{compared} boards ({share:.0%}). Above half means the "
        f"live policy is effectively playing a different game than the "
        f"trained one, and vision needs to carry health")


def test_every_live_decision_it_does_make_is_legal():
    """Swept across elixir and threats, because most single frames are holds."""
    pytest.importorskip("torch")
    from brain.rl_policy import RLBrain, card_cost
    brain = RLBrain(_checkpoint())
    played = []
    clock = 1000.0
    for elixir in range(1, 11):
        for lane_x in (3, 9, 14):
            clock += 1.0
            state = _live_state(
                hand=["hog_rider", "electro_spirit", "cannon", "zap"],
                elixir=elixir,
                enemies=[("giant", lane_x, 12), ("musketeer", lane_x, 10)],
                allies=[("knight", lane_x, 20)])
            decision = brain.decide(state, elapsed=45.0, now=clock)
            if decision is None:
                continue
            played.append((elixir, decision))
            assert decision.card in {"hog_rider", "electro_spirit", "cannon", "zap"}
            assert 0 <= decision.x < GRID_W and 0 <= decision.y < 32
            assert 0 <= decision.slot < 4
            assert card_cost(decision.card) <= elixir, (
                f"played {decision.card} at {elixir} elixir")
            if decision.card not in ("zap", "fireball"):
                assert decision.y >= RIVER_ROW, (
                    f"{decision.card} placed at y={decision.y}, in the enemy "
                    f"half, with both their towers standing")
    if not played:
        pytest.skip("this checkpoint greedily holds on every swept board")


def _live_state(**kwargs):
    sys.path.insert(0, str(ROOT / "tests"))
    from test_brain import make_state
    return make_state(**kwargs)


# ------------------------------------------------ no language model, ever

def test_the_learned_brain_never_builds_an_advisor():
    """The rl path must not touch Qwen, and must not need Ollama running.

    Four separate things have to hold, and each of them alone would be
    enough for the bot to sit waiting on an LLM that is not there.
    """
    import inspect

    from brain.policy import Brain
    from brain.rl_policy import RLBrain

    # 1. `Brain` only builds an Advisor when explicitly asked, and the
    #    learned brain's internal observer does not ask.
    assert "use_advisor: bool = False" in inspect.getsource(Brain.__init__)
    assert Brain(learn=False).advisor is None
    assert "use_advisor=False" in inspect.getsource(RLBrain._build_observer)

    # 2. The decision itself is a forward pass, with nothing to wait on.
    decide = inspect.getsource(RLBrain.decide).lower()
    for forbidden in ("advisor", "ollama", "qwen", "requests", "urllib",
                      "http", "socket"):
        assert forbidden not in decide, f"{forbidden} reached the rl decision"
    assert "argmax" in decide and "logits" in decide

    # 3. The runner branches before the advisor is ever constructed.
    runner = (ROOT / "scripts" / "cr_bot.py").read_text(encoding="utf-8")
    assert "brain = RLBrain(" in runner
    assert "brain = Brain(use_advisor=args.advisor)" in runner

    # 4. And the launcher refuses to claim otherwise.
    launcher = (ROOT / "run.ps1").read_text(encoding="utf-8")
    assert "$useAdvisor = (-not $NoAdvisor) -and ($Brain -eq 'rules')" in launcher


# ---------------------------------------------- the whole runner interface

def runner_attributes() -> set:
    """Every `brain.<name>` the runner touches, read from its source.

    Derived rather than listed. A hand-written list is a snapshot that goes
    stale the first time someone adds a call, and the failure is not a test
    going red - it is the bot standing in a live match logging the same
    AttributeError sixty times a minute, which is exactly what happened.
    """
    import re
    names = set()
    for line in (ROOT / "scripts" / "cr_bot.py").read_text(
            encoding="utf-8").splitlines():
        # `from brain.policy import Brain` is a module path, not a call on the
        # brain object; only the latter has to exist at runtime.
        if "import" in line:
            continue
        names.update(re.findall(r"\bbrain\.([a-z_]+)", line))
    return names


def test_the_learned_brain_answers_everything_the_runner_asks_of_it():
    from brain.rl_policy import RLBrain

    checkpoint = ROOT / "checkpoints" / "mirror" / "mirror_best.pt"
    if not checkpoint.exists():
        import pytest
        pytest.skip("no mirror checkpoint on disk")
    brain = RLBrain(checkpoint)

    missing = sorted(name for name in runner_attributes()
                     if not hasattr(brain, name))
    assert not missing, (
        f"cr_bot calls brain.{{{', '.join(missing)}}} and RLBrain has none of "
        "them; in a real match this throws every frame and the bot does nothing")


def test_the_match_lifecycle_runs_without_a_simulator_or_an_emulator():
    """reset -> decide -> confirm, the sequence the runner actually drives."""
    from brain.policy import Candidate
    from brain.rl_policy import RLBrain

    checkpoint = ROOT / "checkpoints" / "mirror" / "mirror_best.pt"
    if not checkpoint.exists():
        import pytest
        pytest.skip("no mirror checkpoint on disk")
    brain = RLBrain(checkpoint)

    brain.reset()
    assert brain.plays == 0
    assert brain.committed_elixir == 0.0

    brain.confirm(Candidate(card="hog_rider", slot=0, x=9, y=20,
                            tag="t", weight_key="rl", features={}, score=1.0),
                  obs_now=1.0)
    assert brain.plays == 1, "a confirmed play has to reach the observer"
    assert brain.play_counts.get("hog_rider") == 1
    # `committed_elixir` is the rule engine's defensive budget and only moves
    # for its own `defend*` tags, which a learned policy never emits. It stays
    # zero here by design; the runner only logs it.
    assert brain.committed_elixir == 0.0

    assert brain.book is None                 # learn=False
    assert brain.advisor is None              # and there is no LLM here
    summary = brain.summary()
    assert isinstance(summary, str), "the match record stores this verbatim"
    assert "brain=rl" in summary and "step=" in summary

    brain.reset()
    assert brain.plays == 0, "reset must clear the counters for the next match"
