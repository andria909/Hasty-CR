"""Run actual frozen pending/control PPO commands and explicit matching eval."""
import argparse
import importlib.util
import json
from pathlib import Path


def child(path):
    import torch
    import sim.train_attack_ppo as trainer
    from sim.pending_inputs import CONTRACT
    spec=json.loads(Path(path).read_text());pending=spec['trial']['pending_inputs']
    initialize,evaluate=trainer.initialize_attack_policy,trainer.evaluate
    def configured_initialize(*args,**kwargs):
        if pending:kwargs.update(num_planes=24,num_scalars=313,observation_contract=CONTRACT)
        return initialize(*args,**kwargs)
    def configured_evaluate(*args,**kwargs):
        kwargs['env_options']=dict(kwargs.get('env_options',{}),pending_inputs=pending)
        return evaluate(*args,**kwargs)
    trainer.initialize_attack_policy=configured_initialize;trainer.evaluate=configured_evaluate
    filename=Path(__file__).with_name('preflight_global_full_pair.py')
    module_spec=importlib.util.spec_from_file_location('shared_preflight',filename)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module);module.child(path)
    root=Path(spec['root']);report_path=root/'child_preflight.json';report=json.loads(report_path.read_text())
    if ('--pending-inputs'in report['original_full_command'])!=pending:raise ValueError('Captured pending flag differs')
    checkpoint=torch.load(root/'whole/update_000002.pt',map_location='cpu',weights_only=False)
    config=checkpoint['config'];shape=(config['num_planes'],config['num_scalars'])
    if shape!=((24,313)if pending else(20,309)) or config.get('pending_inputs')!=pending:
        raise ValueError('Actual trained input differs from trial')
    report.update(pending_inputs=pending,actual_input_shape=shape)
    report_path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared');p.add_argument('--child');a=p.parse_args()
    if a.child:child(a.child)
    else:
        from scripts.kaggle.preflight_sequence_pilot import preflight
        preflight(a.prepared,child_script=__file__)
