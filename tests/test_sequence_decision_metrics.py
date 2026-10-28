import json
from pathlib import Path
import torch
from sim.train_attack_ppo import initialize_attack_policy
from sim.pending_inputs import CONTRACT
from scripts.attack.evaluate_air_position_labels import score_checkpoint


def test_sequence_report_separates_wait_and_deploy(tmp_path):
    net,architecture,_=initialize_attack_policy(num_planes=24,num_scalars=313,observation_contract=CONTRACT)
    path=tmp_path/'model.pt';torch.save(dict(state_dict=net.state_dict(),config=dict(architecture=architecture,observation_contract=CONTRACT)),path)
    report=score_checkpoint(path,'data/late_air_pending_motionfix_20261028')
    wait=report['by_decision_type']['wait']['local_heldout'];deploy=report['by_decision_type']['deploy']['local_heldout']
    assert wait['examples']==152 and deploy['examples']==16
    assert wait['successful_choices']+deploy['successful_choices']==report['local_heldout']['successful_choices']
