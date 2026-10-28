"""Current paid sequence contract; changing it invalidates demonstration pins."""
EXECUTION=dict(command_delay_ms=1350,action_cooldown_ms=0,decision_interval_ms=250,
               cannon_placement_mode='central_defense_v1',physics_tick_ms=50)


def require_execution(protocol):
    if protocol.get('execution')!=EXECUTION:
        raise ValueError('Sequence execution differs; regenerate demonstration search')
