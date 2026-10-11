"""A behaviour clone arrives with a random critic, and PPO must not act on it.

`sim/clone.py` fits the action head only:

    logits, value = network(planes, scalars)
    loss = criterion(logits, target)

`value` is computed and discarded. So a checkpoint from it holds a policy worth
75% against the meta pool and a value head that is still at initialisation.

PPO's advantage is `returns - values`. Fed a random critic it is noise with a
large magnitude, and the clipped objective happily takes big steps on it. Three
separate runs started from that clone and fell from 75% to roughly 35%, each
time losing the win condition entirely - the failure was read as a reward
problem twice and an opponent problem once before the critic was suspected.

The fix is to fit the critic first with the policy frozen. These tests assert
the freeze is total: not "the policy changes little", but that it is
bit-identical afterwards, because "little" compounds over hundreds of updates.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from sim.env import ACTIONS, NUM_PLANES, NUM_SCALARS      # noqa: E402
from sim.train_ppo import (WIDE_ARCHITECTURE, build_network,
                           load_policy_state)  # noqa: E402


def freeze_for_warmup(network, active: bool) -> None:
    """The same rule `sim.train_ppo.set_warmup` applies."""
    for name, parameter in network.named_parameters():
        parameter.requires_grad_(not active or name.startswith("value"))


def test_the_clone_script_still_does_not_train_the_value_head():
    """If this ever changes, the warmup default should be revisited."""
    source = (ROOT / "sim" / "clone.py").read_text(encoding="utf-8")
    assert "logits, _value = network" in source, (
        "sim/clone.py no longer fits only the action head; if it now trains "
        "the critic too, --value-warmup may no longer be needed")


def test_only_the_value_head_is_trainable_during_warmup():
    network = build_network(NUM_PLANES, NUM_SCALARS, ACTIONS)
    freeze_for_warmup(network, True)
    trainable = {n for n, p in network.named_parameters() if p.requires_grad}
    assert trainable, "nothing is trainable during warmup"
    assert all(n.startswith("value") for n in trainable), sorted(trainable)
    frozen = {n for n, p in network.named_parameters() if not p.requires_grad}
    assert any(n.startswith(("placement", "card_bias", "hold")) for n in frozen)
    assert any(n.startswith("conv") for n in frozen), (
        "the shared trunk is trainable during warmup, so value gradients "
        "still move the features the policy reads")


def test_a_warmup_update_leaves_the_policy_bit_identical():
    """The whole point: identical, not merely close."""
    network = build_network(NUM_PLANES, NUM_SCALARS, ACTIONS)
    optimiser = torch.optim.Adam(network.parameters(), lr=1e-3, eps=1e-5)
    planes = torch.randn(8, NUM_PLANES, 32, 18)
    scalars = torch.randn(8, NUM_SCALARS)
    returns = torch.randn(8)

    before = {n: p.detach().clone() for n, p in network.named_parameters()}
    freeze_for_warmup(network, True)
    for _ in range(5):
        _logits, value = network(planes, scalars)
        loss = ((value - returns) ** 2).mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

    for name, parameter in network.named_parameters():
        if name.startswith("value"):
            assert not torch.equal(parameter, before[name]), (
                f"{name} did not move; the critic is not learning")
        else:
            assert torch.equal(parameter, before[name]), (
                f"{name} changed during value warmup - the policy is drifting "
                f"while it was supposed to be frozen")


def test_the_action_distribution_is_unchanged_by_warmup():
    """Same board in, same logits out, before and after fitting the critic."""
    network = build_network(NUM_PLANES, NUM_SCALARS, ACTIONS)
    optimiser = torch.optim.Adam(network.parameters(), lr=1e-3, eps=1e-5)
    planes = torch.randn(4, NUM_PLANES, 32, 18)
    scalars = torch.randn(4, NUM_SCALARS)

    network.eval()
    with torch.no_grad():
        before, _ = network(planes, scalars)

    freeze_for_warmup(network, True)
    for _ in range(10):
        _logits, value = network(planes, scalars)
        loss = ((value - torch.randn(4)) ** 2).mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        after, _ = network(planes, scalars)
    assert torch.equal(before, after), (
        "the policy's logits moved during value warmup")


def test_unfreezing_restores_full_training():
    network = build_network(NUM_PLANES, NUM_SCALARS, ACTIONS)
    freeze_for_warmup(network, True)
    freeze_for_warmup(network, False)
    assert all(p.requires_grad for p in network.parameters()), (
        "some parameters stayed frozen after warmup ended")


def test_the_critic_actually_improves_during_warmup():
    """A freeze that also froze the critic would be silently useless."""
    torch.manual_seed(0)
    network = build_network(NUM_PLANES, NUM_SCALARS, ACTIONS)
    optimiser = torch.optim.Adam(network.parameters(), lr=1e-3, eps=1e-5)
    planes = torch.randn(16, NUM_PLANES, 32, 18)
    scalars = torch.randn(16, NUM_SCALARS)
    returns = torch.full((16,), 5.0)

    freeze_for_warmup(network, True)
    first = last = None
    # 200 steps, not 40: with the trunk frozen the critic is a single linear
    # layer over fixed features, and it converges steadily rather than fast.
    # That slowness is why warmup runs at its own larger learning rate.
    for step in range(200):
        _logits, value = network(planes, scalars)
        loss = ((value - returns) ** 2).mean()
        if step == 0:
            first = float(loss.detach())
        last = float(loss.detach())
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    assert last < first * 0.6, (
        f"value loss went {first:.3f} -> {last:.3f}; the critic is barely "
        f"fitting, so the warmup would not buy anything")


def test_v2_checkpoint_migration_preserves_logits_exactly():
    torch.manual_seed(4)
    old = build_network(NUM_PLANES, NUM_SCALARS, ACTIONS)
    planes = torch.randn(3, NUM_PLANES, 32, 18)
    scalars = torch.randn(3, NUM_SCALARS)
    with torch.no_grad():
        expected, _ = old(planes, scalars)
    old_state = {
        name: value for name, value in old.state_dict().items()
        if not name.startswith(("slot_query.", "coordinate."))}

    torch.manual_seed(999)
    migrated = build_network(NUM_PLANES, NUM_SCALARS, ACTIONS)
    assert load_policy_state(migrated, old_state)
    with torch.no_grad():
        actual, _ = migrated(planes, scalars)
    assert torch.equal(actual, expected)
def test_current_checkpoint_needs_no_migration():
    network = build_network(NUM_PLANES, NUM_SCALARS, ACTIONS)
    clone = build_network(NUM_PLANES, NUM_SCALARS, ACTIONS)
    assert not load_policy_state(clone, network.state_dict())


def test_appended_public_features_preserve_old_checkpoint_logits():
    torch.manual_seed(41)
    network = build_network(NUM_PLANES, NUM_SCALARS, ACTIONS)
    old_state = dict(network.state_dict())
    old_state["scalar.0.weight"] = old_state["scalar.0.weight"][:, :-2]
    migrated = build_network(NUM_PLANES, NUM_SCALARS, ACTIONS)
    assert load_policy_state(migrated, old_state)

    planes = torch.randn(3, NUM_PLANES, 32, 18)
    old_scalars = torch.randn(3, NUM_SCALARS - 2)
    new_scalars = torch.cat((old_scalars, torch.zeros(3, 2)), dim=1)
    with torch.no_grad():
        expected, _ = network(planes, new_scalars)
        actual, _ = migrated(planes, new_scalars)
    assert torch.equal(actual, expected)


def test_appended_tactical_planes_preserve_old_checkpoint_logits():
    """Eight-plane checkpoints must remain valid after observation v2."""
    torch.manual_seed(42)
    old = build_network(8, NUM_SCALARS, ACTIONS)
    old_planes = torch.randn(3, 8, 32, 18)
    scalars = torch.randn(3, NUM_SCALARS)
    with torch.no_grad():
        expected, _ = old(old_planes, scalars)

    migrated = build_network(NUM_PLANES, NUM_SCALARS, ACTIONS)
    assert load_policy_state(migrated, old.state_dict())
    new_planes = torch.cat(
        (old_planes, torch.randn(3, NUM_PLANES - 8, 32, 18)), dim=1)
    with torch.no_grad():
        actual, _ = migrated(new_planes, scalars)
    assert torch.equal(actual, expected)


def test_wide_policy_has_more_capacity_and_the_same_contract():
    compact = build_network(NUM_PLANES, NUM_SCALARS, ACTIONS)
    wide = build_network(
        NUM_PLANES, NUM_SCALARS, ACTIONS, WIDE_ARCHITECTURE)
    compact_params = sum(value.numel() for value in compact.parameters())
    wide_params = sum(value.numel() for value in wide.parameters())
    assert wide_params > compact_params * 4

    planes = torch.randn(2, NUM_PLANES, 32, 18)
    scalars = torch.randn(2, NUM_SCALARS)
    logits, values = wide(planes, scalars)
    assert logits.shape == (2, ACTIONS)
    assert values.shape == (2,)
    assert wide.architecture == WIDE_ARCHITECTURE
