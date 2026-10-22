"""Coordinate resume compatibility without running rollout or gradient updates."""
import pytest

from sim import arena
from sim.train_defense_grpo import coordinate_contract_metadata,resume_config_differences


def config(geometry='legacy_v1',coordinates='physical_live'):
    return dict(environment_code_sha256='same-code',seed=42,
                **coordinate_contract_metadata(geometry,coordinates))


def test_training_defaults_preserve_current_geometry_and_proxy_encoder():
    values=coordinate_contract_metadata()
    assert values==dict(geometry='legacy_v1',geometry_sha256=arena.LEGACY_GEOMETRY.fingerprint,
                        coordinate_mode='physical_live')
    assert resume_config_differences(config(),config())==[]


@pytest.mark.parametrize('coordinates',['physical_live','legacy_training','physical_live_cells_v2'])
def test_native_geometry_metadata_is_exact_and_not_accepted_by_default_resume(coordinates):
    native=config('native_anchors_candidate_v1',coordinates)
    assert native['geometry_sha256']==arena.NATIVE_ANCHORS_CANDIDATE_GEOMETRY.fingerprint
    assert resume_config_differences(native,native)==[]
    assert 'geometry_sha256'in resume_config_differences(native,config())


def test_legacy_missing_metadata_only_accepts_exact_default_and_same_code_hash():
    previous=dict(environment_code_sha256='same-code',seed=42)
    assert resume_config_differences(previous,config())==[]
    for current in (config(coordinates='legacy_training'),config(coordinates='physical_live_cells_v2'),
                    config('native_anchors_candidate_v1')):
        differences=resume_config_differences(previous,current)
        assert 'missing_geometry_coordinate_metadata_for_nondefault_contract'in differences
    current=config();current['environment_code_sha256']='new-source'
    assert 'environment_code_sha256'in resume_config_differences(previous,current)
    current=config();current['geometry_sha256']='changed-anchor-fingerprint'
    assert 'geometry_sha256'in resume_config_differences(config(),current)


def test_invalid_geometry_and_coordinates_fail_closed():
    with pytest.raises(ValueError):coordinate_contract_metadata('guessed-geometry')
    with pytest.raises(ValueError):coordinate_contract_metadata(coordinate_mode='implicit-new')
