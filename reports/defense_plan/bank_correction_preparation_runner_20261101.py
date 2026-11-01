import json,os,subprocess,time
from pathlib import Path
from scripts.attack.filter_bank_correction_frontier import build
from scripts.kaggle.prepare_bank_correction_pair import prepare
from scripts.kaggle.preflight_bank_correction_pair import preflight
from scripts.attack.verify_bank_correction_pair import verify
from scripts.kaggle.local_artifact_transfer import file_sha
state_path=Path('reports/defense_plan/bank_correction_preparation_state_20261101.json')
if state_path.exists():raise RuntimeError('Existing preparation state; inspect before retry')
root=Path('tmp/kaggle/bank_correction_pair_20261101')
frontier=Path('data/mixed_bank_match_frontier_20261101')
def save(phase,**extra):
    state=dict(phase=phase,pid=os.getpid(),root=str(root),frontier=str(frontier),updated_ns=time.time_ns(),submitted=False,**extra)
    temporary=state_path.with_name(state_path.name+'.writing')
    temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(state_path)
    print(json.dumps(state),flush=True)
try:
    save('waiting_for_independent_replays')
    deadline=time.monotonic()+21600
    while time.monotonic()<deadline:
        s=json.loads(Path('reports/defense_plan/bank_match_search_state_20261101.json').read_text())
        if s.get('postprocess_phase')=='complete':break
        if s.get('postprocess_phase')=='error':raise RuntimeError('Replay/assembly failed; inspect source logs')
        process=subprocess.run(['ps','-p',str(s['postprocess_pid']),'-o','command='],capture_output=True,text=True)
        if 'scripts.attack.watch_bank_match_search'not in process.stdout:raise RuntimeError('Replay process absent; no replacement started')
        time.sleep(30)
    else:raise TimeoutError('Preparation wait expired; existing jobs unchanged')
    raw=Path(s['dataset'])
    if file_sha(raw/'examples.npz')!=s['dataset_sha256']:raise ValueError('Assembled data changed')
    save('filtering_verified_labels')
    result=build(raw,'reports/defense_plan/bank_match_search_verified_20261101.json',frontier)
    save('freezing_real_pair',examples=result['examples'])
    prepare('tmp/kaggle/crown_economy_pilot_20261030','data/mixed_bank_skills_margin_safe_spells_20261030',frontier,
        'tmp/kaggle/bank_initialization_20261030_v3/collected/retain001_s392/policy.pt',root,'hastycr-bank-correct1101',owner='vertel0')
    save('running_real_five_skill_preflight')
    checked=preflight(root)
    if checked['technical_identical']or not checked['submission_allowed']:raise ValueError('Real treatment preflight required')
    save('independent_preflight_scoring')
    scored=verify(root,technical_smoke=True)
    (root/'preflight_independent_verification.json').write_text(json.dumps(scored,indent=2)+'\n')
    save('ready_for_submission',kernel='vertel0/hastycr-bank-correct1101',preflight_sha256=file_sha(root/'preflight.json'),
        dataset_sha256=result['data_sha256'],examples=result['examples'],corrections=result['correction_examples'])
except BaseException as exc:
    save('inspection_required',error_type=type(exc).__name__,error=str(exc));raise
