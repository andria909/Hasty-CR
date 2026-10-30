"""Bank teacher pilots require actual new-physics/reward and BC data evidence."""
import json
from pathlib import Path
from scripts.attack.verify_mana_campaign import verify_mana
from scripts.kaggle.prepare_bank_teacher_pilot import VARIANT


def validate_protocol(p):
    if p['study_variant']!=VARIANT or p['pilot_seeds']!=[362,363] or p['total_pilot_steps']!=819200:
        raise ValueError('Wrong teacher study/seeds/budget')
    expected=[('control',0,0.),('low',200,0.),('medium',800,0.),('high',800,.1)]
    if [(r['id'],r['warmstart_updates'],r['supervised_replay_weight'])for r in p['cases']]!=expected:
        raise ValueError('Changed teacher comparisons')
    common=None
    for row in p['cases']:
        identity={k:v for k,v in row.items()if k not in ('id','warmstart_updates','supervised_replay_weight')}
        if common is None:common=identity
        elif common!=identity:raise ValueError('Confounded teacher comparison')
        for k,v in dict(hog_sight_clipping=1,action_delay_ms=1350,validation_action_delay_ms=1350,
                        cannon_center_only=1,tower_support_weight=.15,spell_waste_weight=.1,
                        sequence_wait_fraction=.5,card_balance_weight=.03).items():
            if row.get(k)!=v:raise ValueError('Teacher execution differs: '+k)


def verify(collected,prepared):
    result=verify_mana(collected,prepared,protocol_validator=validate_protocol,expected_variant=VARIANT)
    collected,prepared=Path(collected),Path(prepared)
    p=json.loads((prepared/'dataset/protocol.json').read_text())
    m=json.loads((prepared/'dataset/air_manifest.json').read_text())
    if p['teacher_data_sha256']!=m['data_sha256']:raise ValueError('Teacher data differs')
    for row in p['cases']:
        if not row['warmstart_updates']:continue
        for seed in p['pilot_seeds']:
            r=json.loads((collected/f"pilot_{row['id']}_s{seed}_warmstart/result.json").read_text())
            if r.get('sequence_wait_fraction')!=.5:raise ValueError('BC sampler differs')
    result['bank_teacher_audit']=dict(verified=True,data_sha256=m['data_sha256'],groups=p['teacher_groups'])
    result['promotion_blocked']=not m.get('cannon_first_audited',False)
    if result['promotion_blocked']:
        result['promotion_block_reason']='Teacher omitted legal Cannon-first alternatives; reproduced lower-damage counterexample'
    return result
