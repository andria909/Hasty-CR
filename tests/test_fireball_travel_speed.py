from types import SimpleNamespace as NS

from sim.arena import Point
from sim.engine import Battle
from sim.public_gamedata import load_public_spells
from sim.spells import load_spells,cast_spell


def test_public_fireball_matches_native_tick_motion():
    spec=load_public_spells(11)['fireball']
    assert spec.projectile_speed_mt_per_sec==12000
    assert spec.travels_from_king


def test_loaded_fireball_reaches_twelve_tile_aim_in_one_second(tmp_path):
    # Missing CSV directory exercises the same portable public path as Kaggle.
    spec=load_spells(11,root=tmp_path/'missing')['fireball']
    battle=Battle()
    cast_spell(battle,spec,Point(9000,17000),1,origin=Point(9000,29000))
    assert battle.spell_impacts[0][0]==1000
    assert battle.spell_cast_log[0]['due_ms']==1000


def test_client_section_and_public_fireball_speed_agree(tmp_path,monkeypatch):
    import sim.gamedata
    monkeypatch.setattr(sim.gamedata,'load_gamedata',lambda **kw:
        {'fireball':NS(unit=None,form='Normal',rarity='Rare')})
    characters=tmp_path/'characters';characters.mkdir()
    (characters/'fireballspell.toml').write_text(
        '[PROJECTILE.FireballSpell]\nDamage = 100\nRadius = 2500\nSpeed = 600\n')
    assert load_spells(11,root=tmp_path,scale=lambda damage,*_:damage)['fireball'].projectile_speed_mt_per_sec==12000
