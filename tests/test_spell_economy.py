from copy import deepcopy

from sim.adapter import grid_to_point
from sim.arena import Point
from sim.entities import make_unit
from sim.spell_economy import SpellEconomyTracker
from tests.test_spell_waste import match_with
import pytest


def resolve(match):
    for _ in range(160):
        match.step(50)


def test_crown_only_fireball_is_paid_and_does_not_change_combat():
    match = match_with('fireball'); control = deepcopy(match)
    tracker = SpellEconomyTracker(match).attach()
    at = grid_to_point(9, 1, 1)
    assert match.play_card(1, 'fireball', at)
    assert control.play_card(1, 'fireball', at)
    assert tracker.metrics()['dormant_at_request_king_only_elixir'] == 0
    resolve(match); resolve(control)
    metrics = tracker.metrics()
    assert metrics['empty_spell_elixir'] == 0
    assert metrics['nonlethal_crown_only_elixir'] == 4
    assert metrics['dormant_at_request_king_only_elixir'] == 4
    assert match.towers[-1]['king'].hitpoints == control.towers[-1]['king'].hitpoints
    assert match.players[1].elixir == control.players[1].elixir
    assert deepcopy(tracker).metrics() == metrics == tracker.metrics()


def test_lethal_crown_spell_is_not_charged_as_nonlethal_chip():
    match = match_with('zap'); tracker = SpellEconomyTracker(match).attach()
    target = match.towers[-1]['left']; target.hitpoints = 1
    assert match.play_card(1, 'zap', target.pos)
    metrics = tracker.metrics()
    assert metrics['nonlethal_crown_only_elixir'] == 0
    assert metrics['crown_finishing_spell_counts'] == {'zap': 1}


def test_troop_and_tower_contact_retains_possible_spell_utility():
    match = match_with('zap'); tracker = SpellEconomyTracker(match).attach()
    tower = match.towers[-1]['left']
    target = make_unit(0, match.cards['knight'].unit, -1,
                       Point(tower.pos.x, tower.pos.y+500), 0)
    match.battle.add(target)
    assert match.play_card(1, 'zap', tower.pos)
    metrics = tracker.metrics()
    assert metrics['nonlethal_crown_only_elixir'] == 0
    assert metrics['troop_contact_spell_counts'] == {'zap': 1}


def test_princess_chip_and_already_active_king_are_separate_from_dormant_king():
    for kind in ('left', 'king'):
        match = match_with('zap'); tracker = SpellEconomyTracker(match).attach()
        if kind == 'king':
            match.towers[-1]['king'].target_only_buildings = False
        assert match.play_card(1, 'zap', match.towers[-1][kind].pos)
        metrics = tracker.metrics()
        assert metrics['nonlethal_crown_only_elixir'] == 2
        assert metrics['dormant_at_request_king_only_elixir'] == 0


def test_optional_environment_penalizes_once_and_resumes_exactly():
    from sim.continuous_spell_economy import SpellEconomyContinuousPushEnv
    control = SpellEconomyContinuousPushEnv(action_delay_ms=0)
    penalized = SpellEconomyContinuousPushEnv(action_delay_ms=0, dormant_king_weight=.5,
                                             crown_only_weight=.25, spell_waste_weight=.1)
    for seed in range(100):
        control.reset(seed=seed); penalized.reset(seed=seed)
        if 'zap' in control.match.players[1].hand:
            break
    slot = control.match.players[1].hand.index('zap')
    action = 1+slot*576+3*18+9
    state = penalized.snapshot()
    a = control.step(action); b = penalized.step(action)
    assert b[4]['spell_economy_step_penalty'] == 1
    assert b[4]['empty_spell_step_penalty'] == 0
    assert b[1] == pytest.approx(a[1]-1)
    assert penalized.step(0)[4]['spell_economy_step_penalty'] == 0
    penalized.restore(state)
    again = penalized.step(action)
    assert again[1:4] == b[1:4]
    assert again[4]['metrics'] == b[4]['metrics']


def test_training_factory_requires_explicit_supported_economy_configuration():
    from sim.train_attack_ppo import make_training_env
    kwargs = dict(curriculum='continuous_v1', tower_support_weight=.15,
                  spell_waste_weight=.1, crown_only_weight=0., dormant_king_weight=.4)
    env = make_training_env(**kwargs)
    _, info = env.reset(seed=2)
    assert info['metrics']['dormant_king_weight'] == .4
    for patch in (dict(crown_only_weight=None), dict(pending_inputs=True),
                  dict(tactical_start_fraction=.25), dict(curriculum='classic'),
                  dict(dormant_king_weight=float('nan'))):
        with pytest.raises(ValueError):
            make_training_env(**dict(kwargs, **patch))


def test_actual_economy_ppo_optimizer_and_exact_resume(tmp_path):
    import torch
    from sim.train_attack_ppo import parser, train, validate_resume
    from scripts.kaggle.preflight_global_full_pair import equal_state
    torch.set_num_threads(1)
    def args(name, updates):
        return parser().parse_args(['--output', str(tmp_path/name), '--device', 'cpu',
            '--curriculum', 'continuous_v1', '--task-mode', 'joint',
            '--tower-support-weight', '.15', '--spell-waste-weight', '.1',
            '--crown-only-weight', '.1', '--dormant-king-weight', '.4',
            '--action-delay-ms', '1350', '--updates', str(updates), '--envs', '1',
            '--rollout', '2', '--epochs', '1', '--minibatch-size', '2',
            '--eval-every', '0', '--max-ms', '180000', '--target-environment-steps', '4'])
    first = train(args('one', 1))/'update_000001.pt'
    whole = train(args('whole', 2))/'update_000002.pt'
    resumed_args = args('resumed', 2); resumed_args.resume = str(first)
    resumed = train(resumed_args)/'update_000002.pt'
    a = torch.load(whole, map_location='cpu', weights_only=False)
    b = torch.load(resumed, map_location='cpu', weights_only=False)
    for key in ('state_dict', 'optimiser', 'rng', 'infos', 'environment_states'):
        if key == 'environment_states':
            assert a[key][0]['dormant_king_weight'] == b[key][0]['dormant_king_weight'] == .4
        else:
            assert equal_state(a[key], b[key])
    bad = dict(a['config'], dormant_king_weight=.8)
    with pytest.raises(ValueError, match='dormant_king_weight'):
        validate_resume(a, bad)
