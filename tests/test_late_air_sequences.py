import json
from pathlib import Path
import numpy as np
import pytest
from copy import deepcopy
from scripts.attack.build_late_air_sequences import replay
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26
from sim.runner import resolve_deck


def source():
    data=json.loads(Path('reports/late_air_native_search_20261027.json').read_text())
    cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    return data,cards,spells


def test_pending_first_card_does_not_hide_second_free_slot():
    data,cards,spells=source()
    context=next(c for c in data['contexts'] if c['opening']['seed']==3800057)
    rows=replay(context,cards,spells)
    assert len(rows)==2
    assert rows[1]['pending_count']==1 and rows[1]['reserved_elixir']==4
    assert rows[1]['scalars'][0]<rows[0]['scalars'][0]-.3
    slot=context['best']['commands'][0]['slot']
    assert not rows[1]['masks'][1+slot*576:1+(slot+1)*576].any()
    assert rows[1]['masks'][rows[1]['action']]
    assert rows[1]['action']//576 != rows[0]['action']//576


def test_cycled_zap_waits_for_real_slot_and_reproduces_clear():
    data,cards,spells=source()
    context=next(c for c in data['contexts'] if c['opening']['seed']==3800085)
    rows=replay(context,cards,spells)
    assert len(rows)==21 and sum(r['action']==0 for r in rows)==19
    assert rows[-1]['decision_ms']-rows[0]['decision_ms']==5000
    assert all(np.all(~r['success_masks'] | r['masks']) for r in rows)
    broken=deepcopy(context);broken['best']['own_tower_damage']['left']=1
    with pytest.raises(ValueError,match='outcome did not reproduce'):
        replay(broken,cards,spells)
