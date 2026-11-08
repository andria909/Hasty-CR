"""Single-thread CPU comparison on accepted recorded native frames and dense stress."""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch

from sim.entity_policy import ARCHITECTURES
from sim.train_ppo import build_network, COMPACT_ARCHITECTURE
from sim.env import ACTIONS
from scripts.gamestate.model_bridge import NativeModelAdapter, MissingModelInput
from scripts.gamestate.memory_play import own_context


def main():
    p=argparse.ArgumentParser();p.add_argument('--frames',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();torch.set_num_threads(1);torch.manual_seed(8)
    frames=[];last_tick=-100
    with Path(args.frames).open() as stream:
        for line in stream:
            row=json.loads(line);own=row.get('own_state',{})
            if row.get('status')!='ok' or own.get('status')!='ok' or own['tick']<last_tick+50:continue
            if len([u for u in row['units'] if u['name']=='PrincessTower'])!=4:continue
            frames.append(row);last_tick=own['tick']
            if len(frames)>=32:break
    if len(frames)<8:raise ValueError('Need eight or more accepted real frames')
    adapter=NativeModelAdapter(own_side=frames[0]['own_state']['own_side'],allow_incomplete_history=True,
        allow_unknown_profiles=True,coordinate_mode='physical_live_cells_v2',entity_inputs=True)
    views=[];encode_ms=[]
    for row in frames:
        began=time.perf_counter_ns()
        try:
            view=adapter.encode(row,own_context(row['own_state']))
        except MissingModelInput:
            continue  # Same rejection as live inference; never invent missing HP.
        encode_ms.append((time.perf_counter_ns()-began)/1e6)
        scalars=np.concatenate((view.observation['scalars'],np.array([0,0,float(not view.metadata['enemy_bodies_present'])],np.float32)))
        views.append((view,torch.from_numpy(scalars)[None]))
    if len(views)<8:raise ValueError('Fewer than eight accepted live inference inputs')
    results={}
    with torch.inference_mode():
        for arch in (COMPACT_ARCHITECTURE,*ARCHITECTURES):
            net=build_network(20,309,ACTIONS,arch).eval();entity=arch in ARCHITECTURES
            inputs=[(torch.from_numpy(v.observation['entities' if entity else 'planes'])[None],s)for v,s in views]
            for i in range(30):net(*inputs[i%len(inputs)])
            times=[]
            for repeat in range(10):
                for x,s in inputs:
                    began=time.perf_counter_ns();net(x,s);times.append((time.perf_counter_ns()-began)/1e6)
            stress=inputs[0][0].clone()
            if entity:
                stress[:]=stress[:,0:1];stress[:,:,0]=1
            dense=[]
            for i in range(100):
                began=time.perf_counter_ns();net(stress,inputs[0][1]);dense.append((time.perf_counter_ns()-began)/1e6)
            stats=lambda x:dict(median_ms=float(np.median(x)),p95_ms=float(np.percentile(x,95)),max_ms=max(x),n=len(x))
            results[arch]=dict(parameters=sum(p.numel()for p in net.parameters()),real_frames=stats(times),dense_stress=stats(dense))
    result=dict(platform=platform.platform(),torch=torch.__version__,threads=1,batch=1,
        source=args.frames,frames=len(views),rejected_frames=len(frames)-len(views),note='Random weights measure architecture cost, not quality; dense stress is synthetic.',
        models=results,adapter_encode_median_ms=float(np.median(encode_ms)))
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
