"""Start a named observer process using external account credentials.

The token is inherited through the environment, never placed in process args.
exec preserves the PID and writes tracebacks to the caller's durable log.
"""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--token-index',type=int,choices=(0,1,2),required=True)
    p.add_argument('--mode',choices=('stream','collect'),required=True)
    args,forwarded=p.parse_known_args()
    if forwarded and forwarded[0]=='--':forwarded=forwarded[1:]
    if '--api-user'not in forwarded:p.error('Explicit observer owner required')
    item=json.loads((Path.home()/'.kaggle/access_tokens.json').read_text())['tokens'][args.token_index]
    token=item if isinstance(item,str)else item['token']
    os.environ['KAGGLE_API_TOKEN']=token.strip()
    module={'stream':'scripts.kaggle.stream_training','collect':'scripts.kaggle.collect_final_evaluation'}[args.mode]
    os.execv(sys.executable,[sys.executable,'-u','-m',module,*forwarded])


if __name__=='__main__':main()
