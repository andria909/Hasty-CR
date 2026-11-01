"""Wait locally for verified PPO artifacts, then write paired behavior reports."""
import argparse
import json
from pathlib import Path
import time

from scripts.attack.compare_ppo_pilot_behavior import compare_study, render


def main(config_path):
    config = json.loads(Path(config_path).read_text())
    pending = list(config['studies'])
    deadline = time.monotonic()+86400
    while pending and time.monotonic() < deadline:
        for study in list(pending):
            if not Path(study['proof']).exists():
                continue
            result = compare_study(study['root'], study['proof'])
            Path(study['output']).write_text(json.dumps(result, indent=2)+'\n')
            Path(study['markdown']).write_text(render(result))
            pending.remove(study)
            print(json.dumps(dict(kernel=result['kernel'], paired_report=study['output'])), flush=True)
        if pending:
            time.sleep(30)
    if pending:
        raise TimeoutError('Paired report observer expired; remote jobs unchanged')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    main(parser.parse_args().config)
