import json
from scripts.kaggle.watch_continuous_training import sync_status


def test_local_replayed_stream_does_not_regress_saved_training_steps(tmp_path):
    (tmp_path/'live_stream').mkdir();status=tmp_path/'status.json'
    status.write_text(json.dumps(dict(kernel='owner/job',progress_snapshot={'trials':{
        'pilot':{'target_steps':102400,'steps':81920,'percent':80}}})))
    source=tmp_path/'live_stream/progress.json';source.write_text(json.dumps(dict(kernel='owner/job',
        trials={'pilot':{'target_steps':102400,'steps':2048,'percent':2}})))
    assert sync_status(tmp_path,status)
    assert json.loads(status.read_text())['progress_snapshot']['trials']['pilot']['percent']==80
    source.write_text(json.dumps(dict(kernel='different/job',trials={})))
    assert not sync_status(tmp_path,status)
