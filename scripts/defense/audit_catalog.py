"""Execute scripted attacks with a hold-only defender; no policy training."""
import argparse
import hashlib
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from sim.defense_env import DefenseEnv
from sim.defense_scenarios import build_catalog

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--difficulty',action='append')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    scenarios,_=build_catalog()
    if args.difficulty:scenarios=[s for s in scenarios if s.difficulty in args.difficulty]
    counts=Counter()
    start=time.monotonic()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    metadata={'seed':42,'defender':'hold_only','catalog_sha256':hashlib.sha256(json.dumps([s.to_dict()for s in scenarios],sort_keys=True).encode()).hexdigest(),
              'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()for name in ('sim/defense_env.py','sim/defense_scenarios.py','sim/public_gamedata.py','sim/public_scenario_spells.py','sim/engine.py','sim/pathfind.py','sim/defense_assets.py','sim/public_area_spells.py','sim/public_stat_scaling.py','sim/entities.py','sim/gamedata.py','sim/spells.py','sim/match.py')},'scenario_count':len(scenarios)}
    args.output.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with args.output.open('x') as stream:
        for index,scenario in enumerate(scenarios):
            env=DefenseEnv(scenario=scenario,split=scenario.split)
            env.reset(seed=42)
            for _ in range(1000):
                _,_,terminal,truncated,info=env.step(0)
                if terminal or truncated:break
            else:raise RuntimeError('Environment exceeded step bound')
            metrics=info['metrics']
            counts[metrics['terminal_reason']]+=1
            stream.write(json.dumps({'scenario_id':scenario.scenario_id,'metrics':metrics})+'\n')
            stream.flush()
            if index%50==0:print(json.dumps({'completed':index+1,'total':len(scenarios),'reasons':counts,'seconds':time.monotonic()-start}),flush=True)
    summary={'completed':len(scenarios),'reasons':dict(counts),'seconds':time.monotonic()-start,**metadata}
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
