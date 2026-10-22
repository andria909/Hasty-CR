"""Deterministic attack curriculum preparation; no training or acceptance claim.

Deadlines are provisional task parameters, not calibrated physics. Scenario IDs,
private deadline draws and visibility variants must never be policy inputs.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
import hashlib
import json
import math
import random

from sim import arena

SCHEMA='attack-v1'
SPLITS=('train','validation','test')
GOALS=('hog_tower_damage','spell_finish')
DEFAULT_DEADLINES={'bridge':(12000,24000),'back':(18000,36000),'pocket':(10000,20000),
                   'support':(18000,36000),'building':(20000,40000),'finisher':(8000,14000)}


def _side(side):
    if not isinstance(side,int)or isinstance(side,bool)or side not in (-1,1):
        raise ValueError('Side must be -1 or +1')


def _point(x,y):
    if (not isinstance(x,int)or not isinstance(y,int)or isinstance(x,bool)or isinstance(y,bool)
            or not 0<=x<arena.WIDTH or not 0<=y<arena.HEIGHT):
        raise ValueError('Setup positions must be integer physical millitiles inside the arena')


@dataclass(frozen=True)
class TowerHPSetup:
    side: int
    lane: str
    hp: int|None=None  # None preserves the actual tower profile maximum.

    def __post_init__(self):
        _side(self.side)
        if self.lane not in ('left','right','king'):
            raise ValueError('Unknown tower lane')
        if self.hp is not None and (not isinstance(self.hp,int)or isinstance(self.hp,bool)or self.hp<0):
            raise ValueError('Tower HP must be nonnegative absolute HP or None')


@dataclass(frozen=True)
class UnitSetup:
    card: str
    side: int
    x: int
    y: int
    deployed_ms_ago: int=0

    def __post_init__(self):
        _side(self.side);_point(self.x,self.y)
        if (not isinstance(self.card,str)or not self.card or
                not isinstance(self.deployed_ms_ago,int)or isinstance(self.deployed_ms_ago,bool)or self.deployed_ms_ago<0):
            raise ValueError('Invalid initial unit setup')


@dataclass(frozen=True)
class AttackScheduledPlay:
    at_ms: int
    card: str
    x: int
    y: int
    position_kind: str='fixed_guess'
    trigger_card: str|None=None
    visibility_delay_ms: int=0

    def __post_init__(self):
        _point(self.x,self.y)
        if (any(not isinstance(t,int)or isinstance(t,bool)or t<0 for t in (self.at_ms,self.visibility_delay_ms))
                or not isinstance(self.card,str)or not self.card):
            raise ValueError('Invalid enemy play schedule')
        if self.position_kind not in ('fixed_guess','reactive_visible','counterspell'):
            raise ValueError('Unknown enemy defense trigger')
        if self.position_kind=='reactive_visible'and not self.trigger_card:
            raise ValueError('Reactive defense requires an explicit visible trigger card')


@dataclass(frozen=True)
class AttackScenario:
    scenario_id: str
    family_id: str
    split: str
    goal_kind: str
    lane: str
    target_lane: str='auto'
    initial_elixir: float=10
    enemy_initial_elixir: float=10
    hand_seed: int|None=None
    initial_hand: tuple[str,...]|None=None
    tower_hp: tuple[TowerHPSetup,...]=()
    initial_units: tuple[UnitSetup,...]=()
    enemy_plays: tuple[AttackScheduledPlay,...]=()
    deadline_ms_range: tuple[int,int]=(12000,24000)
    deadline_visibility: str='mixed'
    opportunity_start_ms: int=0
    geometry: str='legacy_v1'
    setup_kind: str='bridge'
    finisher_spell: str|None=None
    finisher_tower_damage: int|None=None
    finisher_hp_relation: str|None=None
    single_cast_sufficient: bool|None=None
    spell_profile_sha256: str|None=None
    paired_case_id: str=''
    limitations: tuple[str,...]=('attack_deadline_physics_calibration_required',)

    def __post_init__(self):
        if not self.scenario_id or not self.family_id or self.split not in SPLITS or self.goal_kind not in GOALS:
            raise ValueError('Invalid attack scenario identity/split/goal')
        if self.lane not in ('left','right')or self.target_lane not in ('auto','left','right','king'):
            raise ValueError('Invalid attack lane/target lane')
        for amount in (self.initial_elixir,self.enemy_initial_elixir):
            if not math.isfinite(amount)or not 0<=amount<=10:
                raise ValueError('Initial elixir must be finite and in [0,10]')
        if (not isinstance(self.deadline_ms_range,tuple)or len(self.deadline_ms_range)!=2 or
                any(not isinstance(n,int)or isinstance(n,bool)for n in self.deadline_ms_range)or
                not 0<self.deadline_ms_range[0]<self.deadline_ms_range[1]):
            raise ValueError('Deadline requires a strictly positive nondegenerate millisecond range')
        if self.deadline_visibility not in ('hidden','visible','mixed')or self.opportunity_start_ms<0:
            raise ValueError('Invalid deadline visibility/opportunity start')
        arena.resolve_geometry(self.geometry)
        if self.initial_hand is not None and (not isinstance(self.initial_hand,tuple)or len(self.initial_hand)!=4 or
                len(set(self.initial_hand))!=4 or any(not isinstance(c,str)or not c for c in self.initial_hand)):
            raise ValueError('Initial hand must be four unique card names')
        for value,kind in ((self.tower_hp,TowerHPSetup),(self.initial_units,UnitSetup),(self.enemy_plays,AttackScheduledPlay)):
            if not isinstance(value,tuple)or any(not isinstance(item,kind)for item in value):
                raise ValueError('Scenario setups must be typed immutable tuples')
        if len({(t.side,t.lane)for t in self.tower_hp})!=len(self.tower_hp):
            raise ValueError('Conflicting tower HP setup')
        if self.goal_kind=='spell_finish':
            if (self.finisher_spell not in ('fireball','zap')or self.target_lane=='auto' or
                    not isinstance(self.finisher_tower_damage,int)or self.finisher_tower_damage<=0 or
                    self.finisher_hp_relation not in ('below','equal','above')or
                    not isinstance(self.single_cast_sufficient,bool)):
                raise ValueError('Finisher requires explicit tower target and exact crown damage boundary labels')
            hp=next((t.hp for t in self.tower_hp if t.side==-1 and t.lane==self.target_lane),None)
            if hp is None or hp<=0:
                raise ValueError('Finisher needs positive explicit enemy tower HP')
            relation='below'if hp<self.finisher_tower_damage else 'equal'if hp==self.finisher_tower_damage else 'above'
            if relation!=self.finisher_hp_relation or self.single_cast_sufficient!=(hp<=self.finisher_tower_damage):
                raise ValueError('Finisher HP/crown damage labels disagree')

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls,data):
        data=dict(data)
        for field,kind in (('tower_hp',TowerHPSetup),('initial_units',UnitSetup),('enemy_plays',AttackScheduledPlay)):
            data[field]=tuple(kind(**row)for row in data.get(field,()))
        for field in ('deadline_ms_range','limitations','initial_hand'):
            if data.get(field)is not None:data[field]=tuple(data[field])
        return cls(**data)


def family_split(family):
    value=int(hashlib.sha256((SCHEMA+':'+family).encode()).hexdigest()[:8],16)%100
    return 'train'if value<70 else 'validation'if value<85 else 'test'


def sample_episode_context(scenario,seed,*,visible_probability=.5):
    """PRIVATE task draw: never expose this dict as observation or action mask."""
    if not 0<=visible_probability<=1:raise ValueError('Invalid visible timer probability')
    case=scenario.paired_case_id or scenario.scenario_id
    rng=random.Random(f'{SCHEMA}:deadline:{seed}:{case}')
    deadline=rng.randint(*scenario.deadline_ms_range)
    visible=(scenario.deadline_visibility=='visible'if scenario.deadline_visibility!='mixed'else
             random.Random(f'{SCHEMA}:visibility:{seed}:{case}').random()<visible_probability)
    return dict(deadline_ms=deadline,deadline_visible=visible,
                hand_seed=scenario.hand_seed if scenario.hand_seed is not None else int(seed))


def build_catalog(*,spells=None,geometry='legacy_v1',deadline_ranges=None,
                  visibility_modes=('hidden','visible'),include_counterspell=True):
    profile=arena.resolve_geometry(geometry)
    if spells is None:
        from sim.defense_assets import load_defense_assets
        _,spells,_,_=load_defense_assets()
    from types import SimpleNamespace
    crown={name:int(spells[name].damage_to(SimpleNamespace(is_tower=True)))for name in ('fireball','zap')}
    if any(value<=1 for value in crown.values()):raise ValueError('Exact positive level11 crown damage profiles are required')
    ranges=dict(DEFAULT_DEADLINES)
    if deadline_ranges:ranges.update(deadline_ranges)
    scenarios=[]
    hog_hand=('hog_rider','fireball','zap','skeletons')
    spell_hand=('fireball','zap','electro_spirit','skeletons')

    def append(family,case,goal,lane,*,setup='bridge',timer_kind='bridge',**options):
        for visible in visibility_modes:
            case_id=f'{family}/{case}/{lane}'
            scenarios.append(AttackScenario(scenario_id=f'{case_id}/timer_{visible}',family_id=family,
                split=family_split(family),goal_kind=goal,lane=lane,geometry=profile.profile_id,
                deadline_ms_range=tuple(ranges[timer_kind]),deadline_visibility=visible,
                paired_case_id=case_id,setup_kind=setup,**options))

    for lane in ('left','right'):
        x=3500 if lane=='left'else 14500
        other='right'if lane=='left'else 'left'
        # Numeric fractions use actual default crown max at runtime, not invented HP.
        from sim.towers import princess_tower
        maximum=princess_tower(11).hitpoints
        low=max(1,maximum//3);high=max(low+1,maximum*2//3)
        for setup in ('bridge','back','pocket'):
            for hp_case in ('lower_lane','tie','single_princess_dead'):
                if setup=='pocket'and hp_case!='single_princess_dead':continue
                hp=(TowerHPSetup(-1,lane,0 if hp_case=='single_princess_dead'else low),
                    TowerHPSetup(-1,other,high if hp_case!='tie'else low))
                append('hog:solo:unopposed',setup+':'+hp_case,'hog_tower_damage',lane,
                    setup=setup,timer_kind=setup,tower_hp=hp,initial_hand=hog_hand)
        for support,kind in [('musketeer','musketeer_ahead'),('knight','tank_ahead'),('giant','tank_ahead'),
                             ('skeletons','cheap_support'),('electro_spirit','cheap_support')]:
            for y in (18500,21500):
                append('hog:support:'+support,f'{kind}:y{y}','hog_tower_damage',lane,
                    setup=kind,timer_kind='support',initial_hand=hog_hand,
                    tower_hp=(TowerHPSetup(-1,lane,low),TowerHPSetup(-1,other,high)),
                    initial_units=(UnitSetup(support,1,x,y,1000),))
        for building in ('cannon','tesla','bomb_tower'):
            for timing in ('fixed_guess','reactive_visible'):
                # Fixed guess is chosen before seeing the new building. Reactive
                # defense is scheduled only after the actor's Hog becomes visible.
                bx=6500 if lane=='left'else 11500
                play=AttackScheduledPlay(1200,building,bx,9000,timing,
                    trigger_card='hog_rider'if timing=='reactive_visible'else None,
                    visibility_delay_ms=300 if timing=='reactive_visible'else 0)
                append('hog:building:'+building,timing,'hog_tower_damage',lane,
                    setup='building_prediction',timer_kind='building',initial_hand=hog_hand,
                    tower_hp=(TowerHPSetup(-1,lane,low),TowerHPSetup(-1,other,high)),enemy_plays=(play,))
        for spell in ('fireball','zap'):
            relevant=spells[spell]
            profile_data={key:getattr(relevant,key)for key in ('name','damage','crown_tower_percent','tower_damage_override')}
            profile_sha=hashlib.sha256(json.dumps(profile_data,sort_keys=True).encode()).hexdigest()
            for relation,delta in (('below',-1),('equal',0),('above',1)):
                for mana in (6,10):
                    for finish_kind in (('unopposed','king','counterspell')if include_counterspell else ('unopposed','king')):
                        hp=crown[spell]+delta
                        enemy=()
                        target='king'if finish_kind=='king'else lane
                        towers=(TowerHPSetup(-1,target,hp),)
                        if finish_kind=='king':
                            towers+=(TowerHPSetup(-1,lane,0),)
                        if finish_kind=='counterspell':
                            position=profile.princess(1)[lane]
                            enemy=(AttackScheduledPlay(2500,'fireball',position.x,position.y,'counterspell'),)
                            towers+=(TowerHPSetup(1,lane,max(1,crown['fireball']-1)),)
                        family=f'finisher:{spell}:'+finish_kind
                        append(family,f'{relation}:mana{mana}','spell_finish',lane,
                            setup='spell_finisher',timer_kind='finisher',target_lane=target,initial_elixir=mana,
                            initial_hand=spell_hand,tower_hp=towers,enemy_plays=enemy,
                            finisher_spell=spell,finisher_tower_damage=crown[spell],finisher_hp_relation=relation,
                            single_cast_sufficient=delta<=0,spell_profile_sha256=profile_sha)
    ids=[scenario.scenario_id for scenario in scenarios]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate attack scenario IDs')
    report=dict(schema=SCHEMA,count=len(scenarios),families=len({s.family_id for s in scenarios}),
        by_split={split:sum(s.split==split for s in scenarios)for split in SPLITS},
        crown_damage_level11=crown,geometry=profile.profile_id,geometry_sha256=profile.fingerprint,
        calibrated=False,trained=False,quality_accepted=False,
        limitations=['deadline ranges require reference-action physics calibration before training',
            'tower damage labels follow the loaded simulator profiles, not an independent current-game verification',
            'fixed-guess and reactive-building variants are different timing conditions, not measured prediction accuracy',
            'support bodies are synthetic initial setups, not replayed real defenses',
            'bridge/back/pocket are setup labels; actual Hog placement is unconstrained and must be measured separately',
            'solo Hog setup and HP variants share one family and therefore one split; held-out solo coverage is incomplete',
            'initial hands are fixed per goal; shuffled hand/cycle curriculum coverage is not implemented']+
            ([]if include_counterspell else ['counterspell races explicitly excluded from this catalog']),
        goal_split_coverage={split:{goal:sum(s.split==split and s.goal_kind==goal for s in scenarios)for goal in GOALS}for split in SPLITS})
    report['missing_goal_split_coverage']=[f'{split}:{goal}'for split in SPLITS for goal in GOALS
        if not report['goal_split_coverage'][split][goal]]
    report['full_goal_split_coverage']=not report['missing_goal_split_coverage']
    return sorted(scenarios,key=lambda s:s.scenario_id),report


def select_scenarios(split='train',scenario_id=None,*,geometry='legacy_v1'):
    if split not in SPLITS:raise ValueError('Unknown attack split')
    selected=[s for s in build_catalog(geometry=geometry)[0]if s.split==split and
              (scenario_id is None or s.scenario_id==scenario_id)]
    if not selected:raise ValueError('No attack scenarios match this split/ID')
    return selected


def scenarios_for_split(split):return select_scenarios(split)


if __name__=='__main__':
    scenarios,report=build_catalog()
    print(json.dumps(dict(report=report,scenarios=[s.to_dict()for s in scenarios]),indent=2))
