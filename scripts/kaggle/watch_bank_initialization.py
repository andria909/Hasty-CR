"""Local receipts/progress only; the collector owns 20-minute remote checks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from scripts.attack.verify_bank_initialization import verify


def watch(root,status,verified,markdown):
    root=Path(root);status=Path(status);next_sync=0;deadline=time.monotonic()+86400
    while time.monotonic()<deadline:
        if time.monotonic()>=next_sync:
            state=json.loads(status.read_text());source=root/'live_stream/progress.json'
            if source.exists():
                p=json.loads(source.read_text())
                if p.get('kernel')==state['kernel']:state['progress_snapshot']=p
            state['last_local_sync_ns']=time.time_ns();status.write_text(json.dumps(state,indent=2)+'\n')
            subprocess.run([sys.executable,'-m','scripts.defense.update_done'],check=True)
            next_sync=time.monotonic()+1200
        receipt=root/'collected/collection_receipt.json'
        if receipt.exists()and json.loads(receipt.read_text()).get('complete'):
            proof=verify(root);Path(verified).write_text(json.dumps(proof,indent=2)+'\n')
            lines=['# Подбор инициализации на смешанных навыках','',
                'Проверено6400CUDA-обновлений BC; шагов среды0. Это локальная подгонка к демонстрациям, не игровой винрейт.','',
                '| Вариант | Худший навык на обоих seed | Среднее по навыкам | Худший WAIT/постановка | Локальный порог |',
                '|---|---:|---:|---:|---|']
            for row in proof['ranking']:
                lines.append(f"| {row['case']['id']} | {row['worst_skill']:.1%} | {row['mean_skill']:.1%} | {row['worst_wait_or_deploy']:.1%} | {'да'if row['local_fit_gate']else'нет'} |")
            lines+=['','Отложена целая группа354. Данные состоят из коррелированных локальных примеров. '
                    'Следующий шаг — полные матчи в симуляции/PPO-пилот; автоматического большого обучения и реальных игр нет.','']
            Path(markdown).write_text('\n'.join(lines))
            state=json.loads(status.read_text());state.update(status='complete_updates_verified',verified_report=str(verified))
            status.write_text(json.dumps(state,indent=2)+'\n');subprocess.run([sys.executable,'-m','scripts.defense.update_done'],check=True)
            return
        time.sleep(30)
    raise TimeoutError('Local initialization observer expired; remote task unchanged')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','status','verified','markdown'):p.add_argument('--'+key,required=True)
    a=p.parse_args();watch(a.root,a.status,a.verified,a.markdown)
