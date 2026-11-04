"""Fixed GAE x auxiliary card-balance factorial; rewards and physics unchanged."""
import argparse
import json

from scripts.kaggle.prepare_air_exposure_pilot import validate_shared_trial, STEPS
from scripts.kaggle.prepare_spell_interaction_pilot import prepare as prepare_factorial

VARIANT = 'credit_assignment_v1'
SEEDS = [472, 473]
CASES = (('control', .95, .03), ('longer', .98, .03),
         ('no_balance', .95, 0.), ('longer_no_balance', .98, 0.))
CONTROLS = {'longer': 'control', 'no_balance': 'control', 'longer_no_balance': 'no_balance'}


def validate_protocol(p):
    if (p.get('study_variant') != VARIANT or p.get('pilot_seeds') != SEEDS or
            p.get('pilot_steps_each') != STEPS or p.get('total_pilot_steps') != 8 * STEPS or
            p.get('paired_controls') != CONTROLS or len(p['cases']) != 4):
        raise ValueError('Fixed credit-assignment factorial required')
    common = None
    for row, (name, lam, balance) in zip(p['cases'], CASES):
        if (row['id'], row['gae_lambda'], row['card_balance_weight'], row['initial_id'],
                row['focus'], row['spell_waste_weight']) != (name, lam, balance, 'parent', 'uniform', .3):
            raise ValueError('Credit grid or common initializer differs')
        fixed = {k: v for k, v in row.items() if k not in ('id', 'gae_lambda', 'card_balance_weight')}
        if common is not None and fixed != common:
            raise ValueError('Only GAE lambda and auxiliary card balance may vary')
        common = fixed
        validate_shared_trial(row, card_balance_weight=balance)


def prepare(source, output, design):
    return prepare_factorial(source, output, design, variant=VARIANT, validator=validate_protocol,
                             experiment='credit_assignment', slug='hastycr-credit1104')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source', 'output', 'design'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.design)))
