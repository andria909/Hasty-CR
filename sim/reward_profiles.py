"""Explicit finite reward profiles for paired PPO coefficient experiments."""
from dataclasses import asdict,fields
import hashlib
import json
import math
from pathlib import Path
import re

from sim.defense_env import DefenseRewards
from sim.attack_env import AttackRewards

SCHEMA='ppo-reward-profile-v1'


def load_reward_profile(path=None):
    data=json.loads(Path(path).read_text())if path else dict(schema=SCHEMA,id='baseline',defense={},attack={})
    if (not isinstance(data,dict)or set(data)-{'schema','id','defense','attack'}or data.get('schema')!=SCHEMA
            or not isinstance(data.get('id'),str)or not re.fullmatch('[a-z0-9_-]{1,64}',data['id'])):
        raise ValueError('Invalid explicit reward profile schema/id')
    normalized=dict(schema=SCHEMA,id=data['id'])
    for name,cls,negative in [('defense',DefenseRewards,{'own_hp_fraction','own_princess_destroyed','own_king_destroyed','elixir_spent','spell_miss','illegal'}),
                              ('attack',AttackRewards,{'elixir','missed_window','own_king_death','illegal'})]:
        supplied=data.get(name,{})
        if not isinstance(supplied,dict)or set(supplied)-{f.name for f in fields(cls)}:
            raise ValueError('Unknown reward coefficient: '+name)
        values=dict(asdict(cls()),**supplied)
        for key,value in values.items():
            if isinstance(value,bool)or not isinstance(value,(int,float))or not math.isfinite(value):
                raise ValueError('Reward coefficients must be finite numbers')
            if (key in negative and value>0)or(key not in negative and value<0):
                raise ValueError('Reward coefficient has the wrong sign: '+key)
        normalized[name]={key:float(value)for key,value in values.items()}
    digest=hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return normalized,digest


def reward_objects(profile):
    return DefenseRewards(**profile['defense']),AttackRewards(**profile['attack'])
