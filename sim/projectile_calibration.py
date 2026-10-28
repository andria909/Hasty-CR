"""Only XY projectile speeds supported by saved native 20Hz traces.

Evidence: reports/defense_plan/native_training_projectiles_20261026.json and
reports/defense_plan/recorded_projectile_clips_20261028.json (Cannon/Archer).
Unmeasured projectile kinds retain their historical conversion.
"""
MEASURED_RAW={'TowerPrincessProjectile':600,'KingProjectile':1000,
    'MusketeerProjectile':1000,'FireSpiritsProjectile':400,'ArrowsSpell':1100,
    'FireballSpell':600,'chr_wizardProjectile':600,'RocketSpell':350,
    'SpearGoblinProjectile':500,'BombSkeletonProjectile':400,
    'TowerCannonball':1000,'ArcherArrow':600}


def projectile_speed(name,raw):
    return raw*20 if MEASURED_RAW.get(name)==raw else raw*1000//60
