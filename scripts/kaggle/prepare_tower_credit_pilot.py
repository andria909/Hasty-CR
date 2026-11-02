"""Fixed positive tower-credit x empty-spell-cost experiment."""
import argparse
import json

from scripts.kaggle.prepare_air_exposure_pilot import validate_shared_trial,STEPS
from scripts.kaggle.prepare_spell_interaction_pilot import prepare as prepare_factorial

VARIANT='tower_credit_v1'
SEEDS=[462,463]
CASES=(('control01',.15,.1),('control03',.15,.3),('tower05_01',.05,.1),('tower05_03',.05,.3))
CONTROLS={'control03':'control01','tower05_01':'control01','tower05_03':'control03'}


def validate_protocol(p):
    if (p.get('study_variant')!=VARIANT or p.get('pilot_seeds')!=SEEDS or
            p.get('pilot_steps_each')!=STEPS or p.get('total_pilot_steps')!=8*STEPS or
            p.get('paired_controls')!=CONTROLS or len(p['cases'])!=4):
        raise ValueError('Fixed tower-credit factorial required')
    common=None
    for row,(name,tower,cost)in zip(p['cases'],CASES):
        if (row['id'],row['tower_support_weight'],row['spell_waste_weight'],row['initial_id'],row['focus'])!=(name,tower,cost,'parent','uniform'):
            raise ValueError('Positive tower grid or common initializer differs')
        fixed={k:v for k,v in row.items()if k not in ('id','tower_support_weight','spell_waste_weight')}
        if common is not None and fixed!=common:raise ValueError('Only tower and empty-spell costs may vary')
        common=fixed;validate_shared_trial(row,tower_support_weight=tower)


def prepare(source,output,design):
    return prepare_factorial(source,output,design,variant=VARIANT,validator=validate_protocol,
        experiment='tower_credit',slug='hastycr-tower-credit1102')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','output','design'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output,a.design)))
