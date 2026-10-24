import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from eval_katacr_original import match_predictions, summarize


def test_exact_class_matching_is_one_to_one_and_confidence_ordered():
    targets=np.array([[0,0,10,10,2,1],[20,20,30,30,3,0]],dtype=float)
    predictions=np.array([[0,0,10,10,.9,2,.8],
                          [0,0,10,10,.8,2,.8],
                          [20,20,30,30,.7,2,.1],
                          [20,20,30,30,.6,3,.1]])
    correct,matched=match_predictions(predictions,targets)
    assert correct[:,0].tolist()==[True,False,False,True]
    assert matched.tolist()==[0,-1,-1,1]
    assert correct[0].all() and correct[3].all()


def test_empty_and_stricter_iou_thresholds():
    target=np.array([[0,0,10,10,1,0]],dtype=float)
    prediction=np.array([[0,0,8,10,.9,1,.2]],dtype=float)
    correct,_=match_predictions(prediction,target)
    assert correct[0,0] and not correct[0,-1]
    correct,matched=match_predictions(prediction,np.zeros((0,6)))
    assert not correct.any() and matched.tolist()==[-1]


def test_aggregate_accepts_class_sets(tmp_path):
    targets=np.array([[0,0,10,10,1,0]],dtype=float)
    predictions=np.array([[0,0,10,10,.9,1,.1]],dtype=float)
    correct,_=match_predictions(predictions,targets)
    result=summarize([{'tp':correct,'predictions':predictions,'targets':targets}],
                     ['unused','knight'],{1},tmp_path,'example')
    assert result['objects']==1 and result['classes_present']==1
    assert result['confidence_0.4_IoU_0.5']['tp']==1
    assert result['mAP50']>.99
