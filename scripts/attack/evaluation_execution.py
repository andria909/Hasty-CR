"""Explicit game execution contract shared by heldout tests and duels.

Missing contracts mean the historical225ms/2100ms/unrestricted protocol only.
New blueprints must bind an explicit contract before checkpoint evaluation.
"""
from copy import deepcopy

LEGACY = dict(schema='evaluation-execution-v1', action_delay_ms=225,
              decision_interval_ms=250, action_cooldown_ms=2100,
              cannon_placement_mode='unrestricted', geometry='native_anchors_candidate_v1',
              coordinate_mode='physical_live_cells_v2', physics_tick_ms=50, max_ms=310000)
CURRENT = dict(LEGACY, action_delay_ms=1350, action_cooldown_ms=0,
               cannon_placement_mode='central_defense_v1')


def validate_execution(value):
    if not isinstance(value,dict) or set(value)!=set(LEGACY):
        raise ValueError('Complete explicit evaluation execution contract required')
    for key in ('schema','geometry','coordinate_mode','physics_tick_ms','max_ms'):
        if type(value[key]) is not type(LEGACY[key]) or value[key]!=LEGACY[key]:
            raise ValueError('Unsupported evaluation execution '+key)
    for key in ('action_delay_ms','decision_interval_ms','action_cooldown_ms'):
        if type(value[key]) is not int or not 0<=value[key]<=10000:
            raise ValueError('Invalid execution timing '+key)
    if value['decision_interval_ms']<50 or value['decision_interval_ms']%50:
        raise ValueError('Decision interval must be a positive whole physics tick')
    if value['cannon_placement_mode'] not in ('unrestricted','central_defense_v1'):
        raise ValueError('Unknown evaluation Cannon contract')
    return deepcopy(value)


def execution_for(protocol):
    value=protocol.get('execution')
    if value is None:
        if protocol.get('schema','').endswith('-v2'):
            raise ValueError('Version2 requires explicit execution')
        return deepcopy(LEGACY)
    result=validate_execution(value)
    runtime=protocol.get('evaluation_runtime',{})
    for key in set(runtime)&set(result):
        if runtime[key]!=result[key]:raise ValueError('Evaluation runtime contradicts execution '+key)
    return result


def environment_kwargs(execution):
    e=validate_execution(execution)
    return dict(action_delay_ms=e['action_delay_ms'],max_ms=e['max_ms'],geometry=e['geometry'],
        coordinate_mode=e['coordinate_mode'],continuous_profile=dict(
            action_cooldown_ms=e['action_cooldown_ms'],decision_interval_ms=e['decision_interval_ms'],
            cannon_center_only=int(e['cannon_placement_mode']=='central_defense_v1'),tower_damage_all=0))


def restrict_placements(mask,hand,execution):
    if execution['cannon_placement_mode']=='unrestricted':
        return mask
    from sim.cannon_placement import restrict_cannon
    return restrict_cannon(mask,hand,execution['cannon_placement_mode'])


def check_row_execution(row,execution):
    if row.get('execution')!=execution:
        raise ValueError('Per-game execution differs from bound protocol')
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
