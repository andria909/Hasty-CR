"""Fit Fireball XY velocity against recorded native battle ticks, without live IO."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path

from scripts.gamestate.audit_recorded_tick_speeds import quantiles


def fit(samples):
    if len(samples)<3 or samples[-1][0]-samples[0][0]<5:return None
    mt=sum(s[0]for s in samples)/len(samples)
    mx=sum(s[1]for s in samples)/len(samples);my=sum(s[2]for s in samples)/len(samples)
    denominator=sum((s[0]-mt)**2 for s in samples)
    if not denominator:return None
    vx=sum((s[0]-mt)*(s[1]-mx)for s in samples)/denominator
    vy=sum((s[0]-mt)*(s[2]-my)for s in samples)/denominator
    residual=max(math.hypot(x-(mx+vx*(t-mt)),y-(my+vy*(t-mt)))for t,x,y in samples)
    if residual>10:return None
    return dict(samples=len(samples),span_ticks=samples[-1][0]-samples[0][0],
                speed_millitiles_per_tick=math.hypot(vx,vy),max_fit_residual_millitiles=residual)


def audit(paths,clock_report):
    tracks=defaultdict(dict);sources=[];counts=Counter();segment=0
    for source,path in enumerate(map(Path,paths)):
        h=hashlib.sha256();last=None;last_tick=None
        with path.open('rb')as stream:
            for raw in stream:
                h.update(raw);d=json.loads(raw);frame=d.get('frame',{});tick=frame.get('battle_tick')
                if d.get('status')!='ok'or type(tick)is not int:last=None;continue
                identity=tuple(sorted(e['address']for e in frame.get('entities',[])if e.get('name')=='KingTower'))
                if len(identity)!=2:continue
                if identity!=last or (last_tick is not None and tick<last_tick):segment+=1
                last=identity;last_tick=tick
                for e in frame.get('entities',[]):
                    if e.get('name')!='FireballSpell':continue
                    counts['fireball_observations']+=1
                    key=(source,segment,e['entity_id'],e['address'])
                    sample=(tick,e['x'],e['y'])
                    if tick in tracks[key]and tracks[key][tick]!=sample:
                        tracks[key][tick]=None;counts['conflicting_same_tick_samples']+=1
                    elif tick not in tracks[key]:tracks[key][tick]=sample
        sources.append(dict(path=str(path),sha256=h.hexdigest(),bytes=path.stat().st_size))
    fitted=[]
    for by_tick in tracks.values():
        samples=[s for _,s in sorted(by_tick.items())if s is not None]
        result=fit(samples)
        if result:fitted.append(result)
    from sim.spells import load_spells
    simulation=load_spells(11)['fireball'].projectile_speed_mt_per_sec
    clocks=json.loads(Path(clock_report).read_text())
    if {s['sha256']for s in clocks['sources']}!={s['sha256']for s in sources}:
        raise ValueError('Tick-rate evidence uses different captures')
    tick_speeds=[r['speed_millitiles_per_tick']for r in fitted]
    result=dict(schema='recorded-fireball-native-velocity-v1',sources=sources,
        clock_evidence_sha256=hashlib.sha256(Path(clock_report).read_bytes()).hexdigest(),
        measured_tick_rate_hz=clocks['device_clock_tick_rate_hz'],declared_tick_rate_hz=20,
        counts=dict(counts,tracks=len(tracks),linearly_fitted_tracks=len(fitted)),
        speed_millitiles_per_tick=quantiles(tick_speeds),
        speed_millitiles_per_second_at_20hz=quantiles([v*20 for v in tick_speeds]),
        fit_residual_millitiles=quantiles([r['max_fit_residual_millitiles']for r in fitted]),
        original_simulator_speed_millitiles_per_second=simulation,
        simulator_spells_source_sha256=hashlib.sha256(Path('sim/spells.py').read_bytes()).hexdigest(),
        simulator_changed=False,live_games_started=False,physics_accepted=False,
        limitations=['Tick is read after coordinates; fits with >10mt residual are rejected',
            'Fits prove XY travel speed, not exact birth/impact/render or AOE/hitbox semantics',
            'Flight tracks are correlated within matches; no independent-match confidence claim',
            'No change to other projectile speeds is justified by Fireball-only data'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('paths',nargs='+',type=Path)
    p.add_argument('--clock-report',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();r=audit(a.paths,a.clock_report);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k]for k in ('counts','speed_millitiles_per_tick','speed_millitiles_per_second_at_20hz','original_simulator_speed_millitiles_per_second')}))
