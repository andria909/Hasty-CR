from dataclasses import replace
import itertools
import numpy as np
import pytest
from sim.cannon_placement import CENTRAL_MODE,CENTRAL_CELLS,restrict_cannon
from sim.continuous_push import ContinuousPushEnv
from sim.env import ACTIONS


def test_intersection_never_unlocks_occupied_cells_or_other_cards():
    mask=np.ones(ACTIONS,dtype=bool);mask[1+21*18+8]=False
    result=restrict_cannon(mask,['cannon','fireball','hog_rider','zap'],CENTRAL_MODE)
    assert result[0] and not result[1+21*18+8]
    assert result[1:577].sum()==len(CENTRAL_CELLS)-1
    np.testing.assert_array_equal(result[577:],mask[577:])
    assert mask[1:577].sum()==575 # input was not mutated
    none=mask.copy();none[1:577]=False
    assert not restrict_cannon(none,['cannon','fireball','hog_rider','zap'],CENTRAL_MODE)[1:577].any()


@pytest.mark.parametrize('lost',list(itertools.product((False,True),repeat=4)))
@pytest.mark.parametrize('reflect',[0,1])
def test_all_princess_tower_states_stay_central_and_mirror_symmetric(lost,reflect):
    env=ContinuousPushEnv(continuous_profile={'lane_reflection_fraction':reflect})
    env.reset(seed=4)
    towers=sorted([e for e in env.match.battle.entities.values()if e.is_tower and e.name!='king_tower'],key=lambda e:e.uid)
    assert len(towers)==4
    for dead,tower in zip(lost,towers):
        if dead:tower.hitpoints=0
    env.match.players[1].hand=['cannon','fireball','hog_rider','zap'];env.match.players[1].elixir=10000
    mask=env._mask();positions={(int(x),int(y))for y,x in np.argwhere(mask[1:577].reshape(32,18))}
    assert positions and positions<=set(CENTRAL_CELLS)
    assert mask[0] and mask[577:].any()
    # Mask includes no unlocked enemy pocket even after enemy tower death.
    assert all(18<=y<=23 and 6<=x<=11 for x,y in positions)


@pytest.mark.parametrize('rotated',[False,True])
def test_native_adapter_uses_identical_shared_constraint(rotated):
    from tests.test_memory_model_bridge import fixture
    from scripts.gamestate.model_bridge import NativeModelAdapter
    _,row,own=fixture(rotated=rotated)
    own=replace(own,hand=('cannon','fireball','knight','skeletons'),next_card='zap')
    args=dict(own_side=0,coordinate_mode='physical_live_cells_v2')
    old=NativeModelAdapter(**args).encode(row,own)
    new=NativeModelAdapter(**args,cannon_placement_mode=CENTRAL_MODE).encode(row,own)
    np.testing.assert_array_equal(new.action_mask,restrict_cannon(old.action_mask,own.hand,CENTRAL_MODE))
    assert new.action_mask[1:577].any()
    assert new.metadata['cannon_placement_mode']==CENTRAL_MODE
