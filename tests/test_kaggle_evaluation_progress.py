import json

from scripts.kaggle.stream_training import record_evaluation_event,replay_evaluation_progress


def event(progress,completed,total):
    return record_evaluation_event(progress,json.dumps(dict(event='heldout_game_progress',completed=completed,total=total)),10)


def test_reconnect_cannot_double_count_or_regress_completed_games():
    progress={}
    assert event(progress,20,380)
    assert not event(progress,20,380)
    assert not event(progress,0,380)
    assert not event(progress,40,360)
    assert event(progress,40,380)
    assert progress['evaluation']['games']['percent']==100*40/380
    assert progress['evaluation']['games']['quality_verified']is False


def test_invalid_counters_and_unrelated_cuda_do_not_create_progress():
    progress={}
    for completed,total in ((True,380),(381,380),(-1,380),(20,0),(20.5,380)):
        assert not event(progress,completed,total)
    assert not record_evaluation_event(progress,'{"event":"CUDA_READY"}',0)
    assert progress=={}


def test_saved_log_replay_tracks_parts_and_actual_games_separately(tmp_path):
    rows=[dict(event='CUDA_READY',role='candidate',checkpoint_sha256='pinned'),
          dict(event='heldout_part_complete',part='fresh_seeds'),
          dict(event='heldout_part_complete',part='fresh_seeds'),
          dict(event='heldout_game_progress',completed=20,total=380)]
    path=tmp_path/'events.jsonl'
    path.write_text(''.join(json.dumps(dict(time=i,data=json.dumps(r)))+'\n'for i,r in enumerate(rows)))
    progress=replay_evaluation_progress(path)['evaluation']
    assert progress['role']=='candidate'
    assert progress['completed_parts']==['fresh_seeds']
    assert progress['games']['completed']==20
