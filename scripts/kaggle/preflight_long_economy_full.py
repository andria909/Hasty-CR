"""Exercise full-budget commands for long pilots; retain paid reward audits."""
import argparse
import importlib.util
import json
from pathlib import Path


def child(path):
    source=Path(__file__).with_name('preflight_spell_interaction_pilot.py')
    spec=importlib.util.spec_from_file_location('long_full_shared',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.child(path,full=True)
    root=Path(json.loads(Path(path).read_text())['root'])
    target=root/'child_preflight.json';report=json.loads(target.read_text())
    if report['full_steps']!=4096000 or report['original_pilot_command'] is not None:
        raise ValueError('Full command required; pilot preflight cannot qualify')
    report['long_economy_full_verified']=True
    target.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--child',required=True)
    child(parser.parse_args().child)
