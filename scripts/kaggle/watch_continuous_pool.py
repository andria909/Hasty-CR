"""Bounded local verification after the600-game collector finishes."""
import argparse
import json
from pathlib import Path
import time

from scripts.attack.verify_continuous_pool import verify


def watch(root,output):
    root=Path(root);deadline=time.monotonic()+50000
    while time.monotonic()<deadline:
        for receipt in sorted(root.glob('collected*/collection_receipt.json')):
            try:complete=json.loads(receipt.read_text()).get('complete')
            except(ValueError,OSError):continue
            if not complete:continue
            report=verify(receipt.parent,root);Path(output).write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(dict(event='POOL600_VERIFIED',standings=report['standings'])),flush=True)
            return
        time.sleep(30)
    raise TimeoutError('Local verification deadline; remote job unchanged')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();watch(a.root,a.output)
