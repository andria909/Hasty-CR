"""Explicit train-only scenario subsets; never filters validation or test."""
import hashlib
import json
from pathlib import Path


def curriculum_selection(scenarios,path=None):
    scenarios=list(scenarios)
    if any(s.split!='train' for s in scenarios):
        raise ValueError('Curriculum may only select train scenarios')
    available={s.scenario_id:s for s in scenarios}
    if path is None:
        identifiers=sorted(available)
        content=json.dumps(identifiers,separators=(',',':')).encode()
    else:
        content=Path(path).read_bytes()
        payload=json.loads(content)
        identifiers=payload.get('scenario_ids') if isinstance(payload,dict) else payload
        if not isinstance(identifiers,list) or not identifiers or any(not isinstance(s,str) or not s for s in identifiers):
            raise ValueError('scenario-ids must be a nonempty JSON string list or {scenario_ids:[...]}')
        if len(set(identifiers))!=len(identifiers):
            raise ValueError('Duplicate curriculum scenario IDs')
        unsupported=set(identifiers)-set(available)
        if unsupported:
            raise ValueError(f'Unknown, wrong-split or difficulty-excluded train scenario IDs: {sorted(unsupported)}')
        identifiers=sorted(identifiers)
    if not identifiers:
        raise ValueError('Empty train curriculum')
    return [available[s] for s in identifiers],dict(scenario_ids=identifiers,
        scenario_ids_sha256=hashlib.sha256(content).hexdigest(),
        curriculum_scope='selected train subset; not a claim of complete plan coverage')
