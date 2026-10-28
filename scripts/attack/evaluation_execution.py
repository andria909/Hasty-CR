"""Explicit game execution contract shared by heldout tests and duels.

Missing contracts mean the historical225ms/2100ms/unrestricted protocol only.
New blueprints must bind an explicit contract before checkpoint evaluation.
"""
from copy import deepcopy
import math

LEGACY = dict(schema='evaluation-execution-v1', action_delay_ms=225,
              decision_interval_ms=250, action_cooldown_ms=2100,
              cannon_placement_mode='unrestricted', geometry='native_anchors_candidate_v1',
              coordinate_mode='physical_live_cells_v2', physics_tick_ms=50, max_ms=310000)
CURRENT = dict(LEGACY, action_delay_ms=1350, action_cooldown_ms=0,
               cannon_placement_mode='central_defense_v1')
CORRECTED = dict(CURRENT, schema='evaluation-execution-v2',
                 normal_range_mode='edge_to_edge_v1', target_lock_mode='legacy_periodic_v1')


def validate_execution(value):
    if not isinstance(value,dict):
        raise ValueError('Complete explicit evaluation execution contract required')
    template=CORRECTED if value.get('schema')=='evaluation-execution-v2' else LEGACY
    if set(value)!=set(template):
        raise ValueError('Complete explicit evaluation execution contract required')
    for key in ('schema','geometry','coordinate_mode','physics_tick_ms','max_ms'):
        if type(value[key]) is not type(template[key]) or value[key]!=template[key]:
            raise ValueError('Unsupported evaluation execution '+key)
    for key in ('action_delay_ms','decision_interval_ms','action_cooldown_ms'):
        if type(value[key]) is not int or not 0<=value[key]<=10000:
            raise ValueError('Invalid execution timing '+key)
    if value['decision_interval_ms']<50 or value['decision_interval_ms']%50:
        raise ValueError('Decision interval must be a positive whole physics tick')
    if value['cannon_placement_mode'] not in ('unrestricted','central_defense_v1'):
        raise ValueError('Unknown evaluation Cannon contract')
    if template is CORRECTED:
        if value['normal_range_mode'] not in ('edge_to_edge_v1','legacy_target_edge_v1'):
            raise ValueError('Unknown normal range mode')
        if value['target_lock_mode'] not in ('in_range_v1','legacy_periodic_v1'):
            raise ValueError('Unknown target retention mode')
    return deepcopy(value)


def execution_for(protocol):
    value=protocol.get('execution')
    if value is None:
        if protocol.get('schema','').endswith(('-v2','-v3')):
            raise ValueError('Version2/3 requires explicit execution')
        return deepcopy(LEGACY)
    result=validate_execution(value)
    if protocol.get('schema','').endswith('-v3') and result['schema']!='evaluation-execution-v2':
        raise ValueError('Version3 requires explicit combat geometry')
    runtime=protocol.get('evaluation_runtime',{})
    for key in set(runtime)&set(result):
        if runtime[key]!=result[key]:raise ValueError('Evaluation runtime contradicts execution '+key)
    return result


def combat_modes(execution):
    e=validate_execution(execution)
    return dict(normal_range_mode=e.get('normal_range_mode','legacy_target_edge_v1'),
                target_lock_mode=e.get('target_lock_mode','legacy_periodic_v1'))


def configure_battle(battle,execution):
    for key,value in combat_modes(execution).items():setattr(battle,key,value)
    battle._validate_combat_geometry()


def training_execution(config):
    """Describe an explicitly configured training run; old runs stay untagged."""
    profile=config.get('continuous_profile',{})
    if not isinstance(profile,dict) or not {'source_radius_reach','target_lock_in_range'}<=profile.keys():
        return None
    for key in ('source_radius_reach','target_lock_in_range','cannon_center_only'):
        if isinstance(profile.get(key),bool) or profile.get(key) not in (0,1):
            raise ValueError('Invalid training combat flag '+key)
    result=dict(CORRECTED,
        normal_range_mode='edge_to_edge_v1' if profile['source_radius_reach'] else 'legacy_target_edge_v1',
        target_lock_mode='in_range_v1' if profile['target_lock_in_range'] else 'legacy_periodic_v1',
        cannon_placement_mode='central_defense_v1' if profile['cannon_center_only'] else 'unrestricted',
        geometry=config['geometry'],coordinate_mode=config['coordinate_mode'])
    for key in ('action_delay_ms','max_ms','action_cooldown_ms','decision_interval_ms'):
        value=profile[key] if key in ('action_cooldown_ms','decision_interval_ms') else config[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or int(value)!=value:
            raise ValueError('Training timing is not an integer '+key)
        result[key]=int(value)
    return validate_execution(result)


def require_training_execution(model,execution):
    if model.get('training_execution')!=execution:
        raise ValueError('Candidate training physics differs from corrected evaluation')


def environment_kwargs(execution):
    e=validate_execution(execution)
    return dict(action_delay_ms=e['action_delay_ms'],max_ms=e['max_ms'],geometry=e['geometry'],
        coordinate_mode=e['coordinate_mode'],continuous_profile=dict(
            action_cooldown_ms=e['action_cooldown_ms'],decision_interval_ms=e['decision_interval_ms'],
            cannon_center_only=int(e['cannon_placement_mode']=='central_defense_v1'),tower_damage_all=0,
            source_radius_reach=int(combat_modes(e)['normal_range_mode']=='edge_to_edge_v1'),
            target_lock_in_range=int(combat_modes(e)['target_lock_mode']=='in_range_v1')))


def restrict_placements(mask,hand,execution):
    if execution['cannon_placement_mode']=='unrestricted':
        return mask
    from sim.cannon_placement import restrict_cannon
    return restrict_cannon(mask,hand,execution['cannon_placement_mode'])


def check_row_execution(row,execution):
    if row.get('execution')!=execution:
        raise ValueError('Per-game execution differs from bound protocol')
    if execution['schema']=='evaluation-execution-v2':
        for key,value in combat_modes(execution).items():
            if row.get(key)!=value:raise ValueError('Actual battle geometry differs: '+key)
    if 'cannon_outside_center' not in row or type(row['cannon_outside_center']) is not int:
        raise ValueError('Missing actual Cannon placement audit')
    if row['cannon_outside_center']<0 or (execution['cannon_placement_mode']=='central_defense_v1' and row['cannon_outside_center']):
        raise ValueError('Cannon played outside required central cells')


def check_duel_execution(row,execution):
    check_row_execution(row,execution)
    count=row.get('executed_commands')
    if type(count) is not int or count<0:
        raise ValueError('Invalid executed command count')
    if count!=sum(sum(v.values())for v in row['accepted_cards'].values())+sum(row['rejected_commands'].values()):
        raise ValueError('Executed command accounting differs')
    delay=((execution['action_delay_ms']+49)//50)*50
    expected=delay if count else None
    if row.get('actual_command_delay_min_ms')!=expected or row.get('actual_command_delay_max_ms')!=expected:
        raise ValueError('Actual command delay differs from bound execution')
