from scripts.kaggle.queue_evaluation_recovery import terminal_slot


def test_only_matching_authoritative_terminal_status_frees_slot():
    for status in ('COMPLETE','ERROR','CANCEL_ACK'):
        assert terminal_slot(dict(kernel='owner/job',remote_status=status),'owner/job')
        assert not terminal_slot(dict(kernel='other/job',remote_status=status),'owner/job')
    for status in ('RUNNING','QUEUED',None,'ConnectionError'):
        assert not terminal_slot(dict(kernel='owner/job',remote_status=status),'owner/job')
