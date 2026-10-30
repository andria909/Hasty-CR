"""Freeze a CPU search from an already verified source-unpacking controller."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
from scripts.attack import search_late_air_defense as helpers


def prepare(base_controller,output,slug,support=False,timing=False,frontier=False):
    if support and timing:raise ValueError('Timing study uses a separate single-threat protocol')
    if frontier and not timing:raise ValueError('Frontier requires timing study')
    output=Path(output);(output/'job').mkdir(parents=True,exist_ok=False)
    tree=ast.parse(Path(base_controller).read_text())
    config=ast.literal_eval(tree.body[-1].value.args[0])
    module='search_bank_timing' if timing else 'search_bank_cannon'
    probe=Path('scripts/attack/'+module+'.py').read_text()
    imports='from copy import deepcopy\nfrom scripts.attack.sequence_execution import EXECUTION\nfrom sim.adapter import grid_to_point\nfrom sim.cannon_placement import restrict_cannon\nfrom sim.eval_defense_duel import mechanical_mask\n'
    probe_tree=ast.parse(probe)
    replaced=0
    for index,node in enumerate(probe_tree.body):
        if isinstance(node,ast.ImportFrom)and node.module=='scripts.attack.search_late_air_defense':
            if {n.name for n in node.names}!={'rollout','rank'}:
                raise ValueError('Unexpected search helper imports')
            inline=ast.parse(imports+'\n'.join(inspect.getsource(getattr(helpers,n))for n in ('helper_cell','rollout','rank')))
            probe_tree.body[index:index+1]=inline.body
            replaced+=1
            break
    if replaced!=1:raise ValueError('One helper import must be replaced')
    ast.fix_missing_locations(probe_tree);probe=ast.unparse(probe_tree)+'\n'
    writes=runs=0
    for node in ast.walk(tree):
        if timing and isinstance(node,ast.Constant)and isinstance(node.value,str):
            node.value=node.value.replace('scripts/attack/search_bank_cannon.py','scripts/attack/search_bank_timing.py').replace('scripts.attack.search_bank_cannon','scripts.attack.search_bank_timing')
        if isinstance(node,ast.Call)and isinstance(node.func,ast.Attribute):
            if node.func.attr=='write_text'and node.args and isinstance(node.args[0],ast.Constant):
                node.args[0]=ast.Constant(probe);writes+=1
            if node.func.attr=='run'and node.args and isinstance(node.args[0],ast.List):
                args=node.args[0].elts
                if any(isinstance(v,ast.Constant)and v.value in ('scripts.attack.search_bank_cannon','scripts.attack.search_bank_timing')for v in args):
                    args[:]=[v for v in args if not(isinstance(v,ast.Constant)and v.value=='--support')]
                    if support:args.append(ast.Constant('--support'))
                    if frontier:args.append(ast.Constant('--frontier'))
                    runs+=1
    if writes!=1 or runs!=1:raise ValueError('Unexpected source-controller layout')
    ast.fix_missing_locations(tree);code=ast.unparse(tree)+'\n';compile(code,'train.py','exec')
    (output/'job/train.py').write_text(code);(output/'expanded_probe.py').write_text(probe)
    kernel=config['owner']+'/'+slug
    metadata=dict(id=kernel,title=slug,code_file='train.py',language='python',kernel_type='script',
        is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=False,
        dataset_sources=[config['owner']+'/'+config['slug']],competition_sources=[],kernel_sources=[])
    (output/'job/kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    prep=dict(kernel=kernel,expected_version=1,cpu_only=True,support=support,timing=timing,frontier=frontier,contexts=36,
        bootstrap_sha256=hashlib.sha256(code.encode()).hexdigest(),probe_sha256=hashlib.sha256(probe.encode()).hexdigest(),
        source_manifest_sha256=config['source_manifest_sha256'],dataset=metadata['dataset_sources'][0],teacher_labels_used=False)
    (output/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n');return prep


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-controller',required=True);p.add_argument('--output',required=True)
    p.add_argument('--slug',required=True);p.add_argument('--support',action='store_true');a=p.parse_args()
    print(json.dumps(prepare(a.base_controller,a.output,a.slug,a.support)))
