import json
import numpy as np
from scripts.gamestate.decision_capture import DecisionCapture


def test_capture_owns_arrays_and_limits_samples(tmp_path):
    output=tmp_path/'capture'
    recorder=DecisionCapture(output,every=2,limit=2)
    planes=np.zeros((20,32,18),dtype=np.float32)
    scalars=np.arange(309,dtype=np.float32)
    mask=np.zeros(2305,dtype=bool);mask[0]=True
    for tick in range(6):
        recorder.capture(planes,scalars,mask,{'action':0},{'tick':tick})
        scalars+=1
    recorder.close()
    files=sorted(output.glob('*.npz'));assert len(files)==2
    with np.load(files[0],allow_pickle=False)as data:
        np.testing.assert_array_equal(data['scalars'],np.arange(309,dtype=np.float32))
        assert json.loads(str(data['metadata']))['tick']==0
        assert data['action_mask'].sum()==1
    summary=json.loads((output/'summary.json').read_text())
    assert summary['seen']==6 and summary['written']==2 and summary['error']is None


def test_writer_failure_is_reported_without_changing_control(tmp_path):
    recorder=DecisionCapture(tmp_path/'capture',every=1)
    (recorder.output/'000001.npz').write_bytes(b'existing')
    recorder.capture(np.zeros(1),np.zeros(1),np.ones(1,dtype=bool),{'action':0},{})
    recorder.close()
    assert recorder.error.startswith('FileExistsError:')
    assert (recorder.output/'000001.npz').read_bytes()==b'existing'
