import json
import pytest
from scripts.attack.audit_native_series import audit


def series(tmp_path, duplicate_ack=False):
    matches=[]
    for global_number in (1,3):
        run=tmp_path/str(global_number);run.mkdir()
        ack=dict(event='PLAY_ACK',match=1,card='cannon',source_tick=40,source_to_ack_game_ms=1400)
        events=[dict(event='MATCH_START',match=1,checkpoint_sha256='pinned'),
                dict(event='PLAY',match=1,card='cannon',tick=40,grid=[3,20],timing={}),ack,
                dict(event='MATCH_END',match=1)]
        if duplicate_ack:events.append(ack)
        (run/'events.jsonl').write_text('\n'.join(json.dumps(e)for e in events))
        matches.append(dict(number=global_number,run=str(run),local_match=1,result='win'))
    matches.insert(1,dict(number=2,interrupted=True,result='loss'))
    path=tmp_path/'series.json';path.write_text(json.dumps(dict(status='complete',requested=3,matches=matches)))
    return path


def test_restarts_keep_global_matches_and_exclude_disconnected_game(tmp_path):
    result=audit(series(tmp_path),'pinned')
    assert result['total_matches']==3 and result['uninterrupted_matches']==2
    assert result['excluded_interrupted_matches']==[2]
    assert result['wins']==2 and result['cards']['cannon']['confirmed']==2
    assert result['cards']['cannon']['mean_per_match']==1


def test_duplicate_confirmation_is_not_extra_usage(tmp_path):
    with pytest.raises(ValueError,match='Duplicate'):
        audit(series(tmp_path,True),'pinned')


def test_changed_checkpoint_cannot_be_merged_into_metrics(tmp_path):
    with pytest.raises(ValueError,match='changed checkpoint'):
        audit(series(tmp_path),'different')
