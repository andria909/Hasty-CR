import ast
import runpy
from types import SimpleNamespace as NS

import pytest

from scripts.kaggle.run_selected_ppo_train_evaluation import schedule
from scripts.kaggle.prepare_selected_ppo_train_evaluation import standalone


def test_train_contexts_are_fixed_and_disjoint_between_workers():
    scenarios=[NS(split='train',scenario_id=n)for n in ('defense::a','attack::b')]
    rows=schedule(scenarios)
    assert len(rows)==6 and {r['seed']for r in rows}=={42,43,44}
    assert {tuple(sorted(r.items()))for r in rows[::2]}.isdisjoint({tuple(sorted(r.items()))for r in rows[1::2]})
    with pytest.raises(ValueError):schedule([NS(split='test',scenario_id='test')])
    with pytest.raises(ValueError):schedule([scenarios[0],scenarios[0]])


def test_train_controller_is_standalone_and_keeps_fixed_policy_pins(tmp_path):
    config={'policy_sha256':'fixed-policy'};source=standalone(config)
    assert not [n for n in ast.walk(ast.parse(source))if isinstance(n,ast.ImportFrom)and(n.module or '').startswith('scripts.')]
    path=tmp_path/'train.py';path.write_text(source);values=runpy.run_path(str(path),run_name='prepared_train_evaluation')
    assert values['LAUNCH_CONFIG']==config
    assert len(values['schedule']([NS(split='train',scenario_id='train')]))==3
