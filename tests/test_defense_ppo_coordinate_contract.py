"""Contract selection only: no optimization, rollouts, devices, or weights."""
from types import SimpleNamespace

import pytest

from sim.arena import resolve_geometry
from sim.eval_defense import environment_contract, parser as eval_parser
from sim.train_defense_ppo import parser as train_parser


def test_train_defaults_and_historical_namespace():
    args = train_parser().parse_args(['--init', 'unused', '--output', 'unused'])
    assert args.geometry == 'legacy_v1'
    assert args.coordinate_mode == 'physical_live'
    old = SimpleNamespace()
    assert environment_contract(
        geometry=getattr(old, 'geometry', 'legacy_v1'),
        coordinate_mode=getattr(old, 'coordinate_mode', 'physical_live')) == environment_contract({})


@pytest.mark.parametrize('geometry', ['legacy_v1', 'native_anchors_candidate_v1'])
@pytest.mark.parametrize('mode', ['physical_live', 'legacy_training', 'physical_live_cells_v2'])
def test_flags_and_new_checkpoint_roundtrip(geometry, mode):
    flags = ['--init', 'unused', '--output', 'unused', '--geometry', geometry,
             '--coordinate-mode', mode]
    train = train_parser().parse_args(flags)
    contract = environment_contract(geometry=train.geometry, coordinate_mode=train.coordinate_mode)
    assert contract['geometry_sha256'] == resolve_geometry(geometry).fingerprint
    evaluation = eval_parser().parse_args(flags + ['--split', 'validation'])
    assert environment_contract(contract, geometry=evaluation.geometry,
                                coordinate_mode=evaluation.coordinate_mode) == contract
    assert environment_contract(contract) == contract


def test_eval_absent_flags_infer_and_plain_v25_defaults():
    args = eval_parser().parse_args(['--init', 'unused', '--output', 'unused', '--split', 'test'])
    assert args.geometry is None and args.coordinate_mode is None
    plain = environment_contract({}, geometry=args.geometry, coordinate_mode=args.coordinate_mode)
    assert plain['geometry'] == 'legacy_v1'
    assert plain['coordinate_mode'] == 'physical_live'
    new = environment_contract(geometry='native_anchors_candidate_v1',
                               coordinate_mode='physical_live_cells_v2')
    assert environment_contract(new, geometry=args.geometry, coordinate_mode=args.coordinate_mode) == new


@pytest.mark.parametrize('config', [{}, {'architecture': 'v25'}, {'geometry': 'legacy_v1'}])
def test_missing_metadata_only_permits_historical_contract(config):
    assert environment_contract(config)['coordinate_mode'] == 'physical_live'
    for overrides in ({'geometry': 'native_anchors_candidate_v1'},
                      {'coordinate_mode': 'legacy_training'},
                      {'coordinate_mode': 'physical_live_cells_v2'}):
        with pytest.raises(ValueError):
            environment_contract(config, **overrides)


def test_new_checkpoint_mismatches_and_incomplete_metadata_rejected():
    contract = environment_contract(geometry='native_anchors_candidate_v1',
                                    coordinate_mode='physical_live_cells_v2')
    with pytest.raises(ValueError, match='mismatch'):
        environment_contract(contract, coordinate_mode='physical_live')
    with pytest.raises(ValueError, match='mismatch'):
        environment_contract(contract, geometry='legacy_v1')
    with pytest.raises(ValueError, match='SHA mismatch'):
        environment_contract(dict(contract, geometry_sha256='wrong'))
    with pytest.raises(ValueError, match='complete'):
        environment_contract({k: v for k, v in contract.items() if k != 'geometry_sha256'})


@pytest.mark.parametrize('parser', [train_parser, eval_parser])
def test_help_documents_both_flags(parser, capsys):
    with pytest.raises(SystemExit) as stopped:
        parser().parse_args(['--help'])
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert '--geometry' in help_text and '--coordinate-mode' in help_text
