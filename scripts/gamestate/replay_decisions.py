"""Replay exact captured inputs offline; never connects to Android or sends taps."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.gamestate.model_bridge import MemoryPolicy
from sim.env import NUM_PLANES, NUM_SCALARS, ACTIONS


def load_capture(path, *, joint):
    with np.load(path,allow_pickle=False) as data:
        planes=data['planes'];scalars=data['scalars'];mask=data['action_mask']
        metadata=json.loads(str(data['metadata']))
    if (planes.shape!=(NUM_PLANES,32,18) or scalars.shape!=(NUM_SCALARS+(3 if joint else 0),)
            or mask.shape!=(ACTIONS,) or planes.dtype!=np.float32 or scalars.dtype!=np.float32
            or mask.dtype!=np.bool_ or not np.isfinite(planes).all() or not np.isfinite(scalars).all()
            or not mask.any()):
        raise ValueError(f'Invalid exact-input tensor contract: {path}')
    action=metadata['decision']['action']
    if type(action)is not int or not 0<=action<ACTIONS or not mask[action]:
        raise ValueError('Recorded action must be available in the recorded mask')
    return planes,scalars,mask,metadata


def replay(checkpoint,captures,*,compare=False):
    checkpoint=Path(checkpoint);files=sorted(Path(captures).glob('*.npz'))
    if not files:raise ValueError('No exact captured decisions found')
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    policy=MemoryPolicy(checkpoint);torch=policy.torch;torch.set_num_threads(1)
    rows=[]
    for path in files:
        planes,scalars,mask,metadata=load_capture(path,joint=policy.joint)
        source=metadata['checkpoint_sha256']
        if source!=digest and not compare:
            raise ValueError('Checkpoint SHA differs; --compare is required for a different model')
        for key in ('coordinate_mode','geometry'):
            if metadata[key]!=getattr(policy,key):
                raise ValueError('Captured coordinate contract differs: '+key)
        with torch.inference_mode():
            logits,value=policy.network(torch.from_numpy(planes)[None],torch.from_numpy(scalars)[None])
            if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
                raise ValueError('Network returned nonfinite values')
            logits[0,~torch.from_numpy(mask)]=-torch.inf
            action=int(logits.argmax(-1)[0]);predicted=float(value[0])
            indices=torch.topk(logits[0],min(5,int(mask.sum()))).indices.tolist()
        recorded=metadata['decision']
        rows.append(dict(file=str(path),match=metadata.get('match'),tick=metadata.get('tick'),
            source_checkpoint_sha256=source,recorded_action=recorded['action'],action=action,
            same_action=action==recorded['action'],value=predicted,
            value_abs_difference=abs(predicted-recorded['value']),top_actions=indices))
    return dict(schema='exact-policy-decision-replay-v1',checkpoint_sha256=digest,
        mode='candidate_comparison' if compare else 'same_weights_reproduction',
        samples=len(rows),same_actions=sum(row['same_action']for row in rows),
        maximum_value_difference=max(row['value_abs_difference']for row in rows),rows=rows,
        limitation='Decision agreement is not tactical quality or evidence of successful placement.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--captures',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compare',action='store_true')
    args=parser.parse_args();report=replay(args.checkpoint,args.captures,compare=args.compare)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in report.items()if k!='rows'}))
