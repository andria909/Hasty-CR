"""Isolated arithmetic ablation using a supplied historical engine source.

No simulator files or running GPU jobs are edited. The baseline must differ
from current engine only by the documented vector-rounding fix.
"""
import argparse
import ast
import inspect
import json
from pathlib import Path
import textwrap
from scripts.attack.audit_cannon_stress import audit
from sim.engine import Battle


def symmetric_div(numerator,denominator):
    if denominator<=0:raise ValueError('Positive vector norm required')
    return numerator//denominator if numerator>=0 else -((-numerator)//denominator)


def run(output,*,include_separation=False,baseline_engine=None):
    all_names=('_move','_avoid_buildings','_separate')
    originals={name:getattr(Battle,name)for name in all_names}
    codes={name:textwrap.dedent(inspect.getsource(method))for name,method in originals.items()}
    if baseline_engine:
        baseline=Path(baseline_engine).read_text();tree=ast.parse(baseline)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Battle')
        codes={n.name:textwrap.dedent(''.join(baseline.splitlines(keepends=True)[n.lineno-1:n.end_lineno]))
               for n in cls.body if isinstance(n,ast.FunctionDef)and n.name in all_names}
    names=('_move','_avoid_buildings')+(('_separate',)if include_separation else ())
    try:
        for name in all_names:
            changed=codes[name]
            if name in names:
                terms=(('push','gap'),('a_share','norm'),('b_share','norm'))if name=='_separate' else (('step','gap'),)
                for amount,norm in terms:
                    for axis in ('dx','dy'):
                        before=f'{axis} * {amount} // {norm}';after=f'_symmetric_div({axis} * {amount}, {norm})'
                        if before not in changed:raise ValueError('Supply the historical baseline engine before the rounding fix')
                        changed=changed.replace(before,after)
            scope=dict(originals[name].__globals__,_symmetric_div=symmetric_div)
            exec(compile(changed,'<symmetric-movement-probe>','exec'),scope)
            setattr(Battle,name,scope[name])
        result=audit(output)
        provenance=dict(methods=list(names),rounding='signed truncation towards zero',source_modified=False,
                        baseline_engine=str(baseline_engine))
        Path(str(output)+'.override.json').write_text(json.dumps(provenance,indent=2)+'\n')
        return result
    finally:
        for name,original in originals.items():setattr(Battle,name,original)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--include-separation',action='store_true');p.add_argument('--baseline-engine',type=Path)
    a=p.parse_args();run(a.output,include_separation=a.include_separation,baseline_engine=a.baseline_engine)
