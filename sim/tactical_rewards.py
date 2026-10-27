"""Small auxiliary signals based on resolved outcomes, never cast frequency."""


def fireball_outcomes(events, side=1):
    value = 0.
    casts = clusters = unit_hits = 0
    for event in events:
        if event.get('caster_side') != side or event.get('side') != side or event.get('spell') != 'fireball' or not event.get('resolved'):
            continue
        casts += 1
        hits = [h for h in event.get('hits', []) if h['target_side'] == -side and h['damage'] > 0]
        units = [h for h in hits if not h['is_tower']]
        unit_hits += len(units)
        clusters += len(units) >= 3
        # Fraction of paid unit value actually removed, not raw damage or count.
        removed = sum(max(0, h['elixir_milli']) / 1000 * min(1., h['damage'] / max(1, h['max_hp'])) for h in units)
        if units:
            # A heuristic exchange threshold of half a Fireball's cost; capped
            # so auxiliary shaping cannot outweigh winning a full match.
            value += max(-2., min(6., removed - 2.))
            if removed >= 2 and any(h['is_tower'] for h in hits):
                value += .25
        # Tower-only damage is already rewarded by the main tower objective;
        # empty casts are handled by the separate resolved-miss penalty.
    return dict(fireball_exchange_value=value, resolved_fireballs=casts,
                fireball_unit_hits=unit_hits, cluster_fireballs=clusters)


def reserve_potential(match):
    """Own public bank only, capped at one four-elixir defensive answer."""
    return min(4., max(0., match.players[1].elixir / 1000))
