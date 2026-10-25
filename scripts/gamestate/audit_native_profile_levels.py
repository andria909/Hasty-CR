"""Compare measured body max HP with versioned per-level rows, without guessing.

No live access, balance patch, level assignment or engine changes. A missing
exact match stays unresolved; native level_index+1 is only a named hypothesis.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def audit(path):
    native=json.loads(Path(path).read_text());base=Path('data/royaleapi')
    tables=[];sources={}
    for name in ('cards_stats_characters.json','cards_stats_building.json'):
        raw=(base/name).read_bytes();sources[name]=hashlib.sha256(raw).hexdigest();tables.extend(json.loads(raw))
    normalize=lambda value:re.sub('[^a-z0-9]','',value.lower())if isinstance(value,str)else ''
    starts=dict(Common=1,Rare=3,Epic=6,Legendary=9,Champion=11,Experimental=9)
    rows=[]
    for body in native['rows']:
        candidates=[r for r in tables if normalize(r.get('name',''))==normalize(body['name'])and r.get('hitpoints_per_level')]
        result=dict(body=body,exact_hp_matches=[],index_plus_one_hypothesis=[])
        if len(candidates)!=1:
            result['status']='missing_or_ambiguous_runtime_profile';rows.append(result);continue
        profile=candidates[0];start=starts[profile.get('rarity','Common')];values=profile['hitpoints_per_level']
        for index,hp in enumerate(values):
            if hp in body['native_max_hp']:result['exact_hp_matches'].append(dict(level=start+index,max_hp=hp))
        for native_index in body['native_level_index']:
            if type(native_index)is not int:continue
            level=native_index+1;index=level-start
            if 0<=index<len(values):result['index_plus_one_hypothesis'].append(dict(level=level,expected_public_max_hp=values[index],native_max_hp=body['native_max_hp']))
        result['status']='exact_hp_match'if result['exact_hp_matches']else 'no_exact_hp_match'
        rows.append(result)
    return dict(schema='native-profile-level-comparison-v1',source_audit_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        public_sources=sources,rows=rows,automatic_level_assignment=False,profiles_changed=False,
        limitations=['Exact HP agreement does not validate every damage, speed or ability field.',
            'Level index interpretation, client balance version and max-HP extraction require independent confirmation.',
            'A nearby public HP value is not a reason to silently replace measured native data.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=audit(args.native);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([dict(name=r['body']['name'],side=r['body']['native_side'],status=r['status'],matches=r['exact_hp_matches'],index_hypothesis=r['index_plus_one_hypothesis'])for r in report['rows']if r['body']['native_side']==1]))
