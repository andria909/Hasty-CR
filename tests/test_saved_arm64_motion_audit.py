import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.audit_saved_arm64_motion import make_tracks,motion_windows,sample_rows


def test_motion_windows_use_game_ticks_and_are_nonoverlapping_without_nominal_speed_filter():
    track=[]
    for i in range(60):
        body=dict(name='Knight',entity_id=42,address='0x1',side=1,x=3500,y=3000+120*i,hp=100)
        track.append(dict(tick=2*i,device=.1*i,body=body,index=i,bodies=[body]))
    windows,rejected=motion_windows(track,{'Knight':SimpleNamespace(range_mt=0)})
    assert len(windows)>5
    assert all(w['game_tiles_per_second']==pytest.approx(1.2)for w in windows)
    assert all(b['start_tick']>a['end_tick']for a,b in zip(windows,windows[1:]))
    assert not rejected


def test_native_target_fields_are_counted_without_inference_and_repeated_ticks_deduplicate():
    body=dict(name='Knight',entity_id=42,address='0x1',side=1,x=3500,y=3000,hp=100,target_uid=9)
    row=dict(status='ok',units=[body],own_state=dict(status='ok',world='0xaa',tick=10),frame=dict(device_start_ns=1000))
    tracks,fields,coverage=make_tracks([row,row],'one-source')
    assert len(tracks)==1 and len(next(iter(tracks.values())))==1
    assert fields['target_uid']==2
    assert coverage['active_clock_rows']==2


def test_sampling_is_bounded_and_records_only_the_read_prefix_hash(tmp_path):
    path=tmp_path/'frames.jsonl'
    lines=[json.dumps({'row':i}).encode()+b'\n'for i in range(5)]
    path.write_bytes(b''.join(lines))
    rows,metadata=sample_rows(path,max_rows=2,max_bytes=10000)
    assert rows==[{'row':0},{'row':1}]
    assert metadata['sampled_prefix_sha256']==hashlib.sha256(b''.join(lines[:2])).hexdigest()
    assert not metadata['full_file_read']
