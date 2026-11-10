"""Matched entity64 decoding/geometry pilot, fresh seeds, unchanged paid physics.

All candidates use grouped greedy evaluation and the original sampled PPO
distribution. Control versus relative isolates local attention geometry;
control versus sharp isolates spatial entropy weight. No checkpoint promotion.
"""
import argparse
import json
from pathlib import Path

from scripts.kaggle import entity_experiment as base
from scripts.kaggle import submit_entity_experiment as transport

ARCHS = dict(decode64='entity_transformer_d64_l2_v1',
             relative64='entity_transformer_d64_l2_relative_v1',
             sharp64='entity_transformer_d64_l2_v1')
ACCOUNTS = dict(decode64=(0, 'andriashadrin'), relative64=(1, 'vertel0'),
                sharp64=(2, 'andreiperelaiko'))
SEEDS = (522, 523)
CASE_OPTIONS = {case: ['--evaluation-action-mode', 'grouped_argmax',
                       '--spatial-entropy-weight', '.1' if case == 'sharp64' else '1']
                for case in ARCHS}


def configure():
    # Shared transport/budget/preflight implementation, explicit study constants.
    for module in (base, transport):
        module.ARCHS, module.ACCOUNTS, module.SEEDS = ARCHS, ACCOUNTS, SEEDS
    base.CASE_OPTIONS = CASE_OPTIONS


def prepare(root):
    configure()
    root = base.prepare(root)
    path = root / 'dataset/protocol.json'
    protocol = json.loads(path.read_text())
    protocol.update(schema='entity-decoder-pilot-v1',
        hypotheses={'decode64': 'grouped deterministic execution baseline',
                    'relative64': 'same encoder and parameter count, explicit cell/body distance attention bias',
                    'sharp64': 'same network, reduce spatial entropy pressure from 1 to .1'},
        limits='Grouped argmax is not joint MAP. No automatic live deployment. Same 72 validation contexts, two new seeds; no test tuning.',
        diagnostic='tmp/entity_decoding_20261110_v3/summary.json')
    path.write_text(json.dumps(protocol, indent=2) + '\n')
    prep_path = root / 'preparation.json'
    prep = json.loads(prep_path.read_text())
    prep['pins']['protocol.json'] = base.sha(path)
    prep_path.write_text(json.dumps(prep, indent=2) + '\n')


def remote(root, case):
    configure()
    base.remote(root, case)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'preflight', 'submit', 'watch'))
    p.add_argument('--root', required=True)
    p.add_argument('--case', choices=tuple(ARCHS))
    a = p.parse_args()
    configure()
    if a.mode == 'prepare':
        prepare(a.root)
    elif a.mode == 'preflight':
        base.preflight(a.root)
    elif a.mode == 'submit':
        transport.submit(a.root, a.case, slug_suffix='1110',
                         remote_module='scripts.kaggle.entity_decoder_experiment')
    else:
        transport.watch(a.root, a.case)
