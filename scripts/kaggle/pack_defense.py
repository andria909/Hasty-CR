"""Build an explicit private Kaggle input allowlist; never upload credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--owner',required=True)
    parser.add_argument('--slug',default='hastycr-defense-smoke-inputs')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True,exist_ok=False)
    files=set()
    for folder in ('sim','scripts/brain','src'):
        files.update(p for p in (root/folder).rglob('*.py')if p.is_file()and not p.is_symlink())
    files.update((root/'data/royaleapi').glob('*.json'))
    files.update((root/'tests').glob('test_defense*.py'))
    for pattern in ('test_public_gamedata.py','test_public_miner.py','test_public_scenario_spells.py','test_public_heal_spirit.py','test_public_graveyard.py','test_public_shooter_links.py','test_public_moving_projectile_area.py','test_clone_shields.py','test_flow*.py','test_tower_pathfinding.py'):
        files.update((root/'tests').glob(pattern))
    for rel in ('data/meta_decks.json','data/latency/mumu_m5_20261021.json','data/validation/graveyard_public_coverage_20261022.json','scripts/gamestate/audit_sim_alignment.py','LICENSE','README.md'):
        if (root/rel).is_file():files.add(root/rel)
    files.update((root/'data/defense').glob('*.json'))
    manifest={}
    with tarfile.open(args.output/'source.tar.gz','w:gz')as archive:
        for path in sorted(files):
            blob=path.read_bytes()
            if re.search(rb'KGAT_[A-Za-z0-9]{12,}',blob):
                raise ValueError('Credential-like content found in upload allowlist')
            rel=str(path.relative_to(root))
            manifest[rel]=hashlib.sha256(blob).hexdigest()
            archive.add(path,arcname=rel,recursive=False)
    checkpoint=root/'checkpoints/portable_v25/portable_v25_cannon_economy_best.pt'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest!='d65ab55893592c0e4655a1882db459c4a7fb1ebd9e10afaf2daa4d2b369fc972':
        raise ValueError('Baseline v25 hash changed')
    shutil.copy2(checkpoint,args.output/'v25.pt')
    (args.output/'manifest.json').write_text(json.dumps(dict(files=manifest,checkpoint_sha256=digest,archive_sha256=hashlib.sha256((args.output/'source.tar.gz').read_bytes()).hexdigest()),indent=2)+'\n')
    (args.output/'dataset-metadata.json').write_text(json.dumps(dict(
        id=f'{args.owner}/{args.slug}',title='HastyCR Defense Smoke Inputs',licenses=[{'name':'other'}]),indent=2)+'\n')
    print(json.dumps(dict(files=len(files),checkpoint_sha256=digest,output=str(args.output))))


if __name__=='__main__':main()
