"""Deterministic legal stress actions, not a learned or quality-scored defender."""

DEFENDERS = ('hold_only', 'cannon_cycle')
CARD_PRIORITY = ('cannon', 'skeletons', 'electro_spirit', 'knight',
                 'musketeer', 'hog_rider', 'zap', 'fireball')


def defender_action(env, info, defender='hold_only'):
    """Use only the supplied legal mask; mirror right-lane fixture for left.

    Cannon and cheap troops go to the threatened bridge, Musketeer behind it;
    Hog and spells target the opposite lane. Unavailable targets remain hold.
    This stresses script scheduling, not strategic playing strength.
    """
    if defender not in DEFENDERS:
        raise ValueError(f'Unknown audit defender: {defender}')
    mask = info['action_mask']
    if len(mask) < 2305 or not mask[0]:
        raise ValueError('Audit requires a full legal action mask including hold')
    if defender == 'hold_only':
        return 0
    hand = env.match.players[1].hand
    lane = getattr(env.scenario, 'lane', None)
    if lane not in ('left', 'right'):
        raise ValueError(f'Cannon cycle requires an explicit scenario lane: {lane!r}')
    for card in CARD_PRIORITY:
        if card not in hand[:4]:
            continue
        x, y = 14, 16
        if card == 'musketeer':
            x, y = 14, 20
        elif card == 'hog_rider':
            x, y = 3, 16
        elif card in ('zap', 'fireball'):
            x, y = 3, 6
        if lane == 'left':
            x = 17-x
        action = 1 + hand.index(card)*576 + y*18 + x
        if mask[action]:
            return action
    return 0
