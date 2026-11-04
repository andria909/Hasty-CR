"""Actual full-budget command rehearsal for the choice-aware PPO variant."""
import argparse
import importlib.util
from pathlib import Path


def child(path):
    source=Path(__file__).with_name('preflight_policy_choice_pilot.py')
    spec=importlib.util.spec_from_file_location('choice_full_shared',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.child(path,full=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child',required=True)
    child(p.parse_args().child)
