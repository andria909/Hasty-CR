import hashlib, json, shutil, tarfile
from pathlib import Path, PurePosixPath

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare_source(source, output, manifest_sha):
    if sha(source / 'manifest.json') != manifest_sha:
        raise ValueError('Manifest SHA mismatch')
    manifest = json.loads((source / 'manifest.json').read_text())
    files = manifest['files']
    for name in files:
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name or (str(p) != name):
            raise ValueError('Unsafe member')
    output.mkdir(parents=True, exist_ok=False)
    if (source / 'source.tar.gz').exists():
        if sha(source / 'source.tar.gz') != manifest['archive_sha256']:
            raise ValueError('Archive SHA mismatch')
        with tarfile.open(source / 'source.tar.gz') as stream:
            members = stream.getmembers()
            if len(members) != len(files) or {m.name for m in members} != set(files) or any((not m.isfile() for m in members)):
                raise ValueError('Archive members differ')
            stream.extractall(output, filter='data')
    else:
        original = (source / 'source').resolve()
        for name in files:
            p = original / name
            if p.is_symlink() or not p.resolve().is_relative_to(original):
                raise ValueError('Unsafe source')
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
    for name, digest in files.items():
        if sha(output / name) != digest:
            raise ValueError('Source member SHA mismatch')
    return manifest

def execute(config):
    import subprocess, sys
    from concurrent.futures import ThreadPoolExecutor

    def mount(slug):
        root = Path('/kaggle/input')
        paths = [p for p in (root / slug, root / 'datasets' / config['owner'] / slug) if p.is_dir()]
        if len(paths) != 1:
            raise ValueError('Ambiguous dataset mount')
        return paths[0]
    inputs = mount(config['slug'])
    source = mount(config['source_slug'])
    for name, digest in config['pins'].items():
        if name == 'extensions.tar.gz' and (not (inputs / name).exists()):
            continue
        if sha(inputs / name) != digest:
            raise ValueError('Frozen focus input differs')
    project = Path('/kaggle/working/focus_project')
    prepare_source(source, project, config['source_manifest_sha256'])
    for name, digest in config['replaced'].items():
        if sha(project / name) != digest:
            raise ValueError('Unexpected source migration')
    if (inputs / 'extensions.tar.gz').exists():
        with tarfile.open(inputs / 'extensions.tar.gz') as stream:
            members = stream.getmembers()
            if len(members) != len(config['extensions']) or {m.name for m in members} != set(config['extensions']):
                raise ValueError('Unexpected extension set')
            if any((not m.isfile() or PurePosixPath(m.name).is_absolute() or '..' in PurePosixPath(m.name).parts for m in members)):
                raise ValueError('Unsafe extension')
            stream.extractall(project, filter='data')
    else:
        base = (inputs / 'extensions').resolve()
        for name in config['extensions']:
            original = base / name
            target = project / name
            if original.is_symlink() or not original.resolve().is_relative_to(base):
                raise ValueError('Unsafe expanded extension')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
    for name, digest in config['extensions'].items():
        if sha(project / name) != digest:
            raise ValueError('Focus source differs')
    sys.path.insert(0, str(project))
    import torch
    from sim.train_attack_ppo import initialize_attack_policy, evaluate
    from sim.continuous_push import CONTRACT
    from scripts.attack.select_continuous_focus import select_initial, select_profiles
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError('Two CUDA devices required')
    output = Path('/kaggle/working/continuous_focus')
    output.mkdir(exist_ok=False)
    print(json.dumps(dict(event='INPUTS_VERIFIED', protocol_sha256=config['pins']['protocol.json'])), flush=True)
    print(json.dumps(dict(event='CUDA_READY', gpus=[torch.cuda.get_device_name(i) for i in (0, 1)])), flush=True)
    options = dict(curriculum='continuous_v1', validation_selection='worst_family', continuous_focus='uniform', geometry='native_anchors_candidate_v1', coordinate_mode='physical_live_cells_v2', action_delay_ms=config.get('evaluation_action_delay_ms', 225), max_ms=310000)
    options.update(crown_only_weight=0.0, dormant_king_weight=0.0)
    options.update({'continuous_profile': {'overflow_penalty': 0.3, 'missed_fireball_penalty': 0.0, 'fireball_value_reward': 0.25, 'reserve_potential_weight': 0.1, 'own_hp_penalty': 6.0, 'barbarian_train_fraction': 0.35, 'lane_reflection_fraction': 0.5, 'action_cooldown_ms': 0, 'cannon_center_only': 1, 'tower_damage_all': 1, 'source_radius_reach': 1, 'target_lock_in_range': 0, 'hog_sight_clipping': 1, 'shaping_gamma': 0.999, 'decision_interval_ms': 250}, 'tower_support_weight': 0.15, 'spell_waste_weight': 0.1, 'tactical_start_fraction': 0.0})
    options.update({'continuous_profile': {'overflow_penalty': 0.3, 'missed_fireball_penalty': 0.0, 'fireball_value_reward': 0.25, 'reserve_potential_weight': 0.1, 'own_hp_penalty': 6.0, 'barbarian_train_fraction': 0.35, 'lane_reflection_fraction': 0.5, 'action_cooldown_ms': 0, 'cannon_center_only': 1, 'tower_damage_all': 1, 'source_radius_reach': 1, 'target_lock_in_range': 0, 'hog_sight_clipping': 1, 'shaping_gamma': 0.999, 'decision_interval_ms': 250}, 'tower_support_weight': 0.15, 'spell_waste_weight': 0.1, 'tactical_start_fraction': 0.0})

    def reference(model, index):
        blob = torch.load(inputs / model['file'], map_location='cpu', weights_only=False)
        net, _, _ = initialize_attack_policy(blob, device=f'cuda:{index % 2}', num_scalars=309, observation_contract=CONTRACT)
        report = evaluate(net, f'cuda:{index % 2}', episodes=36, seed=900100, env_options=options, task_mode='joint')
        result = dict(id=model['id'], sha256=model['sha256'], validation=report)
        (output / (model['id'] + '_reference.json')).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(event='FOCUS_REFERENCE_COMPLETE', model=model['id'], score=report['selection_score'])), flush=True)
        return result
    references = []

    def ref_group(index):
        return [reference(m, index) for m in config['models'][index::2]]
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(ref_group, i) for i in (0, 1)]
        for f in futures:
            references.extend(f.result())
    initial = select_initial(references)
    (output / 'initial_selection.json').write_text(json.dumps(initial, indent=2) + '\n')
    initial_model = next((m for m in config['models'] if m['id'] == initial['selected']))
    protocol = json.loads((inputs / 'protocol.json').read_text())
    if 'air_examples.npz' in config['pins']:
        air_data = project / 'air_warmstart_data'
        air_data.mkdir()
        shutil.copyfile(inputs / 'air_examples.npz', air_data / 'examples.npz')
        shutil.copyfile(inputs / 'air_manifest.json', air_data / 'manifest.json')

    def run(trial, device, seed, full=False):
        run_id = ('full_' if full else 'pilot_') + trial['id'] + '_s' + str(seed)
        directory = output / run_id
        steps = 4096000 if full else 102400
        run_initial = initial_model
        if trial.get('warmstart_updates', 0):
            warm = output / (run_id + '_warmstart')
            subprocess.run([sys.executable, '-m', 'scripts.attack.train_air_position_warmstart', '--data', str(air_data), '--init', str(inputs / initial_model['file']), '--output', str(warm), '--device', device, '--updates', str(trial['warmstart_updates']), '--kl-weight', str(trial['warmstart_kl_weight']), '--seed', str(seed)], cwd=project, check=True)
            run_initial = dict(id=run_id + '_warmstart', file=str(warm / 'policy.pt'), sha256=sha(warm / 'policy.pt'))
            print(json.dumps(dict(event='WARMSTART_COMPLETE', trial_id=run_id, updates=trial['warmstart_updates'])), flush=True)
        profile = output / (run_id + '_profile.json')
        profile.write_text(json.dumps(dict(overflow_penalty=trial['overflow_penalty'], missed_fireball_penalty=trial.get('missed_fireball_penalty', 0.0), board_potential_weight=trial.get('board_potential_weight', 0.0), shaping_gamma=trial.get('gamma', 0.999), action_cooldown_ms=2100, decision_interval_ms=250)))
        tactical = {k: trial[k] for k in ('fireball_value_reward', 'reserve_potential_weight', 'own_hp_penalty', 'barbarian_train_fraction', 'lane_reflection_fraction') if k in trial}
        if tactical:
            coefficients = json.loads(profile.read_text())
            coefficients.update(tactical)
            profile.write_text(json.dumps(coefficients))
        coefficients = json.loads(profile.read_text())
        coefficients.update({k: trial[k] for k in ('action_cooldown_ms', 'cannon_center_only', 'tower_damage_all')})
        profile.write_text(json.dumps(coefficients))
        coefficients = json.loads(profile.read_text())
        coefficients.update({key: trial[key] for key in ('source_radius_reach', 'target_lock_in_range')})
        profile.write_text(json.dumps(coefficients))
        coefficients = json.loads(profile.read_text())
        coefficients['hog_sight_clipping'] = trial['hog_sight_clipping']
        profile.write_text(json.dumps(coefficients))
        coefficients = json.loads(profile.read_text())
        coefficients['hog_sight_clipping'] = trial['hog_sight_clipping']
        profile.write_text(json.dumps(coefficients))
        args = [sys.executable, '-c', "import torch,runpy;torch.set_num_threads(1);runpy.run_module('sim.train_attack_ppo',run_name='__main__')", '--init', str(inputs / run_initial['file']), '--output', str(directory), '--device', device, '--task-mode', 'joint', '--curriculum', 'continuous_v1', '--geometry', 'native_anchors_candidate_v1', '--coordinate-mode', 'physical_live_cells_v2', '--continuous-profile', str(profile), '--continuous-focus', trial['focus'], '--validation-selection', 'worst_family', '--validation-seed', '900100', '--eval-episodes', '36', '--entropy-coef', str(trial['entropy_coef']), '--history-frames', str(trial.get('history_frames', 1)), '--card-balance-weight', str(trial['card_balance_weight']), '--card-balance-target', 'units', '--policy-temperature', str(trial.get('policy_temperature', 1.0)), '--card-exploration-epsilon', str(trial.get('card_exploration_epsilon', 0.0)), '--action-delay-ms', str(trial.get('action_delay_ms', 225)), '--max-ms', '310000', '--seed', str(seed), '--envs', '8', '--rollout', '128', '--updates', str(steps // 1024), '--target-environment-steps', str(steps), '--epochs', '3', '--minibatch-size', '256', '--lr', '0.00005', '--gamma', str(trial.get('gamma', 0.999)), '--gae-lambda', str(trial.get('gae_lambda', 0.95)), '--value-warmup-steps', '20480', '--target-kl', '.02', '--checkpoint-every', '250' if full else '50', '--eval-every', '250' if full else '50', '--wallclock-seconds', '34000' if full else '7200']
        args += ['--crown-only-weight', str(trial['crown_only_weight']), '--dormant-king-weight', str(trial['dormant_king_weight'])]
        args += ['--tactical-start-fraction', str(trial['tactical_start_fraction'])]
        if trial.get('validation_action_delay_ms') is not None:
            args += ['--validation-action-delay-ms', str(trial['validation_action_delay_ms'])]
        if trial.get('supervised_replay_weight', 0.0):
            args += ['--supervised-replay-data', str(air_data), '--supervised-replay-weight', str(trial['supervised_replay_weight'])]
        if trial['spell_waste_weight'] is not None:
            args += ['--spell-waste-weight', str(trial['spell_waste_weight'])]
        args += ['--tower-support-weight', str(trial['tower_support_weight'])]
        process = subprocess.Popen(args, cwd=project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(json.dumps(dict(event='TRIAL_START', trial_id=run_id, target_steps=steps, device=device)), flush=True)
        with (output / (run_id + '.log')).open('x') as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                try:
                    r = json.loads(line)
                except ValueError:
                    print(line, end='', flush=True)
                    continue
                if isinstance(r, dict) and 'update' in r and ('actual_cuda' in r):
                    print(json.dumps(dict(event='TRIAL_PROGRESS', trial_id=run_id, step=r['step'], update=r['update'], target_steps=steps, percent=r['percent'], actual_cuda=r['actual_cuda'])), flush=True)
        if process.wait():
            raise RuntimeError('Focus training failed: ' + run_id)
        boundary = json.loads((directory / 'training_result.json').read_text())
        if not boundary.get('target_met') or boundary.get('step') != steps:
            raise ValueError('Incomplete focus budget')
        rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
        validations = [r for r in rows if 'validation' in r]
        eligible = [r for r in validations if r['validation']['worst_archetype_points'] >= 0.1]
        chosen = max(eligible or validations, key=lambda r: r['validation']['selection_score'])
        checkpoint = directory / f"update_{chosen['update']:06d}.pt"
        if not checkpoint.exists():
            raise ValueError('Selected boundary was not saved')
        result = dict(id=run_id, trial=trial, seed=seed, step=steps, target_met=True, selected_update=chosen['update'], selected_with_family_guard=bool(eligible), checkpoint=str(checkpoint.relative_to(output)), checkpoint_sha256=sha(checkpoint), initial_model=run_initial, validation=chosen['validation'], quality_accepted=False)
        (output / (run_id + '_summary.json')).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(event='FOCUS_TRIAL_COMPLETE', trial_id=run_id, score=chosen['validation']['selection_score'], family_guard=bool(eligible))), flush=True)
        return result

    def pilots(index):
        return [run(trial, f'cuda:{index}', seed) for trial in protocol['cases'][index::2] for seed in protocol.get('pilot_seeds', (62, 63))]
    runs = []
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(pilots, i) for i in (0, 1)]
        for f in futures:
            runs.extend(f.result())
    selection = select_profiles(runs, seeds=protocol.get('pilot_seeds', (62, 63)))
    (output / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    full = []
    if selection['gate_met'] and config.get('allow_full_training', True):
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(run, r['trial'], f'cuda:{i}', 64, True) for i, r in enumerate(selection['selected'])]
            full = [f.result() for f in futures]
    result = dict(schema='continuous-focus-cohort-result-v1', status='complete' if full else 'pilot_complete' if not config.get('allow_full_training', True) else 'pilot_complete_no_eligible_pair', pilot_steps=819200, full_steps=4096000 * len(full), pilot_budgets_met=True, full_gate_met=selection['gate_met'], full_runs=full, initial_selection=initial, protocol_sha256=config['pins']['protocol.json'], quality_accepted=False, live_checkpoint_changed=False, test_used_for_selection=False)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
execute({'owner': 'vertel0', 'slug': 'hastycr-crown-econ1030-inputs', 'source_slug': 'hastycr-crown-econ1030-inputs', 'allow_full_training': False, 'models': [{'id': 'parent', 'file': 'parent.pt', 'sha256': '40d028fae25b30a23272d97bcc16d04c45f6e297c4f3b244a501d3bbd181aed9'}], 'source_manifest_sha256': '3325d51f7978c8805e851624cd8d2bb7e70d133811545a01ceb6b20f9d7c9f44', 'extensions': {'scripts/attack/select_continuous_focus.py': '1e14ea73a783398c955fbf60427da1d8eb130cda0deb9288832b0fd14978dda4', 'scripts/attack/sequence_execution.py': '863caf63f7b4c37f5537a7077265db44c6e6edc79874f2e78331b1e325fb1552', 'scripts/attack/train_air_position_warmstart.py': '3b64260768fc340a130b0c37778b6b91cbd03d1031bc9d5017deb60f8b362049', 'sim/__init__.py': '4a80c4d95c85aec6d443170c32b30eaa8e33726fc37447e4bc07a214e9d05aa2', 'sim/action_audit.py': '02ab59b6246539cab799c393ae218b9e7b3f71a7a86398baea3b8da6577706bf', 'sim/adapter.py': '3d12212efeb1583179382281729c7e2bd52daeaec94079e29265fde9bbf7f428', 'sim/arena.py': 'eec33eccad91593c43f2c050c3c78f5f9b464150361354a977949ee9357f71e4', 'sim/attack_env.py': '55601f481e1f015d8b359de2b67eb09b67e1518dea5188454a576171b66e02ff', 'sim/attack_sampler.py': 'b712e4cbc39176922534be00e5eeb96f2d1256e605782fb135fa48f6def8256b', 'sim/attack_scenarios.py': '78c19577c993b5e25ef58c5a4092512f3050ae933ef53cc20baab4712eb92ba5', 'sim/batch.py': 'bafd6be3fc3481a3229afa386d6c566f00d07839136fa224757f0141626bac06', 'sim/board_potential.py': 'd1bd576c918ae327a1dffdfed1d3fcf2b8f4f4874b2f7c6c06950b5fe8f565fc', 'sim/campaign.py': 'bf0bb98e9c3217da33e34b6dbb16a5b9696e947d7d9c244b79415a0e8c94639f', 'sim/cannon_placement.py': '441ceaaee92a72e2a63d889b2584501d042cbfbf65d5d5a3a45f1ceec275c39c', 'sim/card_balance.py': '182ebbe811f03b221ce4b10ddf3716f86a2f69597464754c5b22ff199829cf2e', 'sim/card_catalog_audit.py': '853741730fb299e651432493dc0927e9375aa0da158084275d6e55b172e96d4c', 'sim/card_exploration.py': '7f806281021604e2403f9095cb959aef89ed95bdac0bd7e0c2fb713909950aea', 'sim/check.py': 'f6efd4a87952dacb12acc43f2cdd41a1c8c6b98019ce53fa02f503699c2d22a7', 'sim/clone.py': '7e990e09041173346746b638e3f72f9df3fd2ab14a2d11744e1e97082a2ceba3', 'sim/continuous_history.py': '6d43354976b8c86338fe317047a32367e869740fc6c64a03779b02dfa03ca64c', 'sim/continuous_pending.py': 'df0fbff311b3300c5f248f8cbfcbd9a49cd058c71e86cdd1e4dcf0bd10acd0d1', 'sim/continuous_push.py': 'ef70efe6abba7d11338ce58cc7ae8e997e696b20b674114cd654e7d4cf258fb3', 'sim/continuous_sampling.py': '3309dd2bf9950a9ab354d72ea6ee51833a4d979318c4f832396f60c470b50137', 'sim/continuous_spell_economy.py': '3d6a89bb93d58186a1d23c04058bbc34ea13b3921cd429442b06567d94979939', 'sim/continuous_spell_waste.py': 'e3e3da719c880aa43338fd463341e13f6b45148678b070d897f1fbd95ccb3169', 'sim/continuous_tactical_starts.py': 'fc797f9bc6c20b2e1c166ca568570ce6e6850d1a4f359b0084c5a04f7c60ac46', 'sim/continuous_tower_support.py': '63b1f0f61501b2a981e6abd55d8f78fa9dae34f3f977a8de3fd50217592ae4fd', 'sim/continuous_validation.py': 'a4d166e38b3870fc0793e729495a858503753b2563323bf74df7dcad64aef656', 'sim/coverage.py': 'c706864da376e64a04f0d19a99df96e2cc06f0d77232d01887757fe73e3bbd5a', 'sim/decision_economy.py': '4f002c1d9ca2ebc7f462ef5843e2588c7a234d810df8cb871369db211fd72cd1', 'sim/deck_builder.py': '171050648629f4937942218c2db4b8a8ae54540516d2a5abab80afdbb121c78e', 'sim/defense_assets.py': '1e786a8fb1050498e01955df5c0d89cd2a5e004766e2a373913236ce8d9e435a', 'sim/defense_audit_policy.py': '208f3f3b1bae9b3cbb35b9be3cc0e79054f6bbfa61044318a39f763a7d6393c3', 'sim/defense_checkpointing.py': '7daec105f62a94fdc934534e264b849a75d8b968819eb114705161018b1c8fb6', 'sim/defense_curriculum.py': 'a64b7bcbad38bc526d23f947923a6a16e8abe62b9512e81d5b1bd5e5071e2e5a', 'sim/defense_env.py': 'd863667c900bf74a966bf2a9b7c458d66df3fcf1ec45d30cc18903a731e986f3', 'sim/defense_log_summary.py': '291d2c0493745b49dbcd6358a156197af94a3832d97395493b19a91189e7b8f8', 'sim/defense_protocol.py': '8dd9836a09bedd573892b9fa1d7fdd29303b99bb24a35cffd4296308a069a1c8', 'sim/defense_sampler.py': 'cf63dc285ba193be30e8ca9b071138f42776ee0791bae106e658c0658e6542ad', 'sim/defense_scenarios.py': '5e3e7f5301d1e9c1c698a2088ed7e7579fef6f28b71407d4b1121340ec132e31', 'sim/device.py': '9b73680bbd890fb4cf7b7f86a8de365a63df64ac3f2f18e13094854ec99eed05', 'sim/diagnostics.py': '0440d225edbe7a54d4084304c456423057a6a310945645051555a2cbd231f4fd', 'sim/engine.py': '8a15162403bc53816569b5ba95cfe1c200adae39769f23f269ee49a1c72b466f', 'sim/entities.py': '0b40d065cd95fc93d3b4b9df7359c2bdae4178e67bf524ecae3c7fa4f6b0f522', 'sim/env.py': 'cda7ad6b1431132e39480ccb99214ba8eaedea80383a59d4836bebf2e32063f9', 'sim/eval_checkpoints.py': '382c1dd26628379189e092a5444b72e2b3a0383705f19b04ff7b27eb52f7d930', 'sim/eval_defense.py': '41797224f0208246667b0d90213a78f21b715f0276bd74ac9afe17f1881c7967', 'sim/eval_defense_duel.py': '659def3089d3afcdce539b5145371caf7ea407003b11a5c36016b3498a6feb5c', 'sim/eval_strategy.py': '7bfd2c1a22112f287f34d484141b9dba32ba63260a7296a5fe351a46f588cd73', 'sim/evaluation_source_contract.py': '1742c19a4350aa89424af51fcf38fc3b010865263d067db52042ae98b0cddd0c', 'sim/evaluation_terminal_damage.py': '8a6dd78f94f4f92f682ccf2ab06639ec5b3e0ce3b44a4e8da715423b36ce427d', 'sim/evolutions.py': 'fdbb7602ca42783a945b5fde4e9f6ee8a59f6c0fa199c31091945e67fb59fd95', 'sim/fireball_balanced.py': '648d342ce88026b00d46ff3925ab964e1bb54783ba12378f14b4e4acd5e786c0', 'sim/fireball_curriculum.py': 'fd06f88174b94702677d7f0c72f80c458fe65d8d7ed4444ba1219301a09a71d2', 'sim/fireball_decision_pairs.py': '8db516dad1febb649f056be17ad5d8fd4329ae505a19f1a382931b1d09fdffd4', 'sim/gamedata.py': '3f25582079b64a1381c6a350b0420fdfb799b6f3e662040d102ee8ce6d693287', 'sim/hand_slot_augmentation.py': 'ef2d15d1d34acc9d360dac24130548e73cdcebf1a022c12a87005094d3f475ed', 'sim/kite_geometry.py': '854d55606e19bb41baa2b4fc4faba3bc5af4a171ee7e1d8f34572d4c6a085149', 'sim/lane_reflection.py': '49e4b1a2bffbda5eb31cdf713362b9c195de98ecfba270960fe6559050d97eef', 'sim/latency.py': '747f9ef0938932bc6bbb9f13f32e211f1b1b612f2e30a12d7628c0a46b9c636c', 'sim/level_audit.py': '9d94b5f1e7a0634a5d667284877974b3f2d6d8304f9291d4b444d5b613360c96', 'sim/level_stat_audit.py': '655ca6b79e4ebfd929c24a543a7c38d62995cbff55a26fca027946ad084a28bc', 'sim/match.py': '65932ba8fb6aacd022b2adffa7ca411b7728ad172cad8196f55981c46fdd7f5c', 'sim/meta_decks.py': '5bfd479c6f8e2d875814f6f980d47f793131d498d19578f80ba1d065542ba13b', 'sim/model_tokens.py': 'ad668193e4c00993748dcf96bae4fafcb1763469ad57ea5e335e45e8143741c1', 'sim/opponents.py': '8800efa603698fb05c25c97761657c26f13d3e4983827e95cfd825bbd17598e5', 'sim/pathfind.py': 'e06a3aed8f8c4887c637583be021145b1e6b32bf82da52df04bec8b432cfb83a', 'sim/pending_inputs.py': 'a083b0e0b5a47548fa98b58ba50359eae67006dfc64f8a426bfa5ffff64fa1f0', 'sim/pending_slot_augmentation.py': 'c706e9c7fff0c834ec35378124b8873f7f31589957e9cedb1ab2a8163710eb9e', 'sim/policy_coordinates.py': '5bdb504301d43b9e31cd6e6069492d55667c1abb45d1b25cb70eb0d1ff140ffa', 'sim/policy_soup.py': 'f7cfbc566dc1c37f6bf792370af6dafb974f4319ec6094d4641736fc19c1dd30', 'sim/ppo_tuning.py': 'a71d0b70aa9c438cc8ad2d8133eee3cbffb19ec18ae6bd9b071df2875654417d', 'sim/probe_plan.py': '373c402b452bf967bdd370c22d811d21001d01472e6ecf004e7d8f2675235c1a', 'sim/projectile_calibration.py': '88e78fad22aa3e3af440a07150e16d4b617ffc88aeb59282d10e8c35019c44fc', 'sim/public_area_spells.py': '9944347f87ab00ad6dbdbe86d35e957a7c935f2dfdb326ed1939c417c596e61e', 'sim/public_frame_history.py': '59413b1b5ed9788f432dd3cb42a15b3196a92d14793701cba1dc11d21e35bf1c', 'sim/public_gamedata.py': 'b64465c1c99fef36ea9cbafec596e2743df0dfd05a59c635ed65e18fe60d39de', 'sim/public_scenario_spells.py': 'eb496528f76937658f6205582734a32ed888a622a987165d76163dbe131f1fe6', 'sim/public_stat_audit.py': 'dd06a593519d824c1c9431a84e71a5e252aa71b0cc283c6f38c376fe94832792', 'sim/public_stat_scaling.py': '25dfdb5b64da6aac4f0d9ce5fe7bbb2dfaf50b26f55a0a5789140524ddff0909', 'sim/public_state.py': 'bde206b83f5b8ee013e425069221a70af680a5758230d6ef0aacda48170ff404', 'sim/readiness.py': 'db697f362e6f7e294af189360f45df73917a6c5aeaadbb702d1f3c4a08e7e603', 'sim/redrive.py': '043ff968f95b388461e980a846e64cd67d60e1702043e0bf3e3892f79051c572', 'sim/replay_bc.py': '35a43f360786d8563bcdbcdfb6fa51e46d81cc69d6500e62995bd55be88cc9b7', 'sim/replay_corpus.py': '97d3c42d8adb0adb4ea1e54cf85f351d32d00d650d867e0a92a4559f2beb0bcc', 'sim/replay_rows.py': '222e1992836888b4769339e32e2e91e76fdd6f7e510f89c74e42d51c01d41cde', 'sim/reward_events.py': '969944b763b262e7f6e6f733e77918983a919934def0c7180f928a5b52d5e75f', 'sim/reward_profiles.py': 'd47df813842eacbb643fe5b0f6eac7cc26a17bf11ee0c07267bb32d138b625df', 'sim/runner.py': 'bde0abba924e9f693fa1acaa04b1f87e032362f954eb84076034ededa8a27b33', 'sim/search_agent.py': '419d0cc8ab0a995c5b02a9217a1e546a8b4b6bb4ae9f844de8ecc57e7b105323', 'sim/selfplay.py': '71807a93544da449428c759a7539cdb7af0d46a9a48f25b8ff3ecfbb9e6eb787', 'sim/spell_economy.py': '8426cbf91378b4d2484edbf35f727b56764608795999ed12f6b0de5e9ce66b8b', 'sim/spell_waste.py': '5fb8fde2744e497eb9fa7cd40bc2d93f1835b7d4d1bd100f9dec6a0bd707ff95', 'sim/spells.py': 'dd492c5958cbd54360d6a8b327168e70c0a065901983a66771e58616c14cf13d', 'sim/strategy_env.py': 'cca83c612d18d7412c3bc2472041580a5ded5d320a5f5dff67c9826b95f35027', 'sim/strategy_sampler.py': 'e1b7d1145b9ffeccaf6d0e43417d18f628061175e819885a931462aca3b45e7e', 'sim/supervised_replay.py': '51791a64f00aa6af959e3c9d36139e7e91e6594f6e94455e1a77702862cd03f6', 'sim/sweep.py': 'da71dd33b55c1071b492485e1898abedc0a328df0fbc4c4a121454696d5116f3', 'sim/tactical_rewards.py': '757c4cee5137c847277cbc399e96fb13fd35151ec020f49a16480f3a305c6a49', 'sim/tactical_strategy.py': '0b64aa5efce6ecd80cd14149dcf0b24314678234f0cd076b300f86de7a18fc96', 'sim/token_agent.py': '8c45b3ab20712419673c1c5f8c2665e8718d971ce006e11e32c7eb1f618fb653', 'sim/tower_support.py': '7850dd531c978f71e7b1a735fe3a5d39b17a81569e6682e8e6df54547a24ebb9', 'sim/towers.py': '92cb040be4bf14c69e488ffc8d11663685d87f21820e165f7a83e57dd8eee826', 'sim/train_attack_grpo.py': 'e190d87f5f964d5acae6bcf665782164561da32adaf0591bf951bffd405c0e61', 'sim/train_attack_ppo.py': '8043f4f84da5cef28c2361433ebd80d83685796b522b7a3c80a31f03a1561341', 'sim/train_defense_grpo.py': '4b37bd3a949ae191b19fea2857d66e745a986e6924fd9b661b4e84556bf1e551', 'sim/train_defense_ppo.py': '9d3b7ea4fb50f3aa898d1ca6df09e80287b8170c20db466a280d78198d573b31', 'sim/train_ppo.py': '26b4ae4a18d6de7a091378672c28da15043ee0f7e8c64655609b431a2c66c81b', 'sim/train_tokens.py': '543265a3fbee0e323da60e088e7d7d7ca7eff04c29b21f2b069cc276464f9d76', 'sim/validate.py': '5c5cd0dd7dacf49bc93b85aaaff91ac3357d424b7c646a7e968aacca940acf61', 'sim/vecenv.py': '71b4af93f778a3a068a11af27194d3fec69fa9d138c5de5712b6b9840b1c036e', 'sim/watch.py': '3a76d46b204a273e079e6d3f4a6873660af25c964ec43e702fb151b6bd88c9e4'}, 'replaced': {'sim/__init__.py': '4a80c4d95c85aec6d443170c32b30eaa8e33726fc37447e4bc07a214e9d05aa2', 'sim/action_audit.py': '02ab59b6246539cab799c393ae218b9e7b3f71a7a86398baea3b8da6577706bf', 'sim/adapter.py': '3d12212efeb1583179382281729c7e2bd52daeaec94079e29265fde9bbf7f428', 'sim/arena.py': 'eec33eccad91593c43f2c050c3c78f5f9b464150361354a977949ee9357f71e4', 'sim/attack_env.py': '7aa9a3ff66ab8a947282506a19fc9316a75ce0ecb81f11aaa7cfda447bf2f78a', 'sim/attack_sampler.py': 'b712e4cbc39176922534be00e5eeb96f2d1256e605782fb135fa48f6def8256b', 'sim/attack_scenarios.py': '78c19577c993b5e25ef58c5a4092512f3050ae933ef53cc20baab4712eb92ba5', 'sim/batch.py': 'bafd6be3fc3481a3229afa386d6c566f00d07839136fa224757f0141626bac06', 'sim/campaign.py': 'bf0bb98e9c3217da33e34b6dbb16a5b9696e947d7d9c244b79415a0e8c94639f', 'sim/card_catalog_audit.py': '853741730fb299e651432493dc0927e9375aa0da158084275d6e55b172e96d4c', 'sim/check.py': 'f6efd4a87952dacb12acc43f2cdd41a1c8c6b98019ce53fa02f503699c2d22a7', 'sim/clone.py': '7e990e09041173346746b638e3f72f9df3fd2ab14a2d11744e1e97082a2ceba3', 'sim/coverage.py': 'c706864da376e64a04f0d19a99df96e2cc06f0d77232d01887757fe73e3bbd5a', 'sim/deck_builder.py': '171050648629f4937942218c2db4b8a8ae54540516d2a5abab80afdbb121c78e', 'sim/defense_assets.py': '1e786a8fb1050498e01955df5c0d89cd2a5e004766e2a373913236ce8d9e435a', 'sim/defense_audit_policy.py': '208f3f3b1bae9b3cbb35b9be3cc0e79054f6bbfa61044318a39f763a7d6393c3', 'sim/defense_checkpointing.py': '7daec105f62a94fdc934534e264b849a75d8b968819eb114705161018b1c8fb6', 'sim/defense_curriculum.py': 'a64b7bcbad38bc526d23f947923a6a16e8abe62b9512e81d5b1bd5e5071e2e5a', 'sim/defense_env.py': 'd863667c900bf74a966bf2a9b7c458d66df3fcf1ec45d30cc18903a731e986f3', 'sim/defense_log_summary.py': '291d2c0493745b49dbcd6358a156197af94a3832d97395493b19a91189e7b8f8', 'sim/defense_protocol.py': '8dd9836a09bedd573892b9fa1d7fdd29303b99bb24a35cffd4296308a069a1c8', 'sim/defense_sampler.py': 'cf63dc285ba193be30e8ca9b071138f42776ee0791bae106e658c0658e6542ad', 'sim/defense_scenarios.py': '5e3e7f5301d1e9c1c698a2088ed7e7579fef6f28b71407d4b1121340ec132e31', 'sim/device.py': '9b73680bbd890fb4cf7b7f86a8de365a63df64ac3f2f18e13094854ec99eed05', 'sim/diagnostics.py': '0440d225edbe7a54d4084304c456423057a6a310945645051555a2cbd231f4fd', 'sim/engine.py': 'fa6da95053c68839954f9c7bccc8c5a832e3bc742dfce17bbf8e4863f28e205c', 'sim/entities.py': '91bd66c38c4562a47c188eb398357116010f0d9e9f4f7b6ccc639e3a3f4d826f', 'sim/env.py': '537ca677efbd56021538190f2c47d469dfa598292ec98fcf7407f718b45eec0c', 'sim/eval_checkpoints.py': '382c1dd26628379189e092a5444b72e2b3a0383705f19b04ff7b27eb52f7d930', 'sim/eval_defense.py': '41797224f0208246667b0d90213a78f21b715f0276bd74ac9afe17f1881c7967', 'sim/eval_defense_duel.py': '659def3089d3afcdce539b5145371caf7ea407003b11a5c36016b3498a6feb5c', 'sim/eval_strategy.py': '7bfd2c1a22112f287f34d484141b9dba32ba63260a7296a5fe351a46f588cd73', 'sim/evaluation_source_contract.py': '1742c19a4350aa89424af51fcf38fc3b010865263d067db52042ae98b0cddd0c', 'sim/evaluation_terminal_damage.py': '8a6dd78f94f4f92f682ccf2ab06639ec5b3e0ce3b44a4e8da715423b36ce427d', 'sim/evolutions.py': 'fdbb7602ca42783a945b5fde4e9f6ee8a59f6c0fa199c31091945e67fb59fd95', 'sim/fireball_balanced.py': '648d342ce88026b00d46ff3925ab964e1bb54783ba12378f14b4e4acd5e786c0', 'sim/fireball_curriculum.py': 'fd06f88174b94702677d7f0c72f80c458fe65d8d7ed4444ba1219301a09a71d2', 'sim/fireball_decision_pairs.py': '8db516dad1febb649f056be17ad5d8fd4329ae505a19f1a382931b1d09fdffd4', 'sim/gamedata.py': 'c58fe70c2f512042a7ab9dbc783e7152e850a1960a7263f14840ce6030135a65', 'sim/latency.py': '747f9ef0938932bc6bbb9f13f32e211f1b1b612f2e30a12d7628c0a46b9c636c', 'sim/level_audit.py': '9d94b5f1e7a0634a5d667284877974b3f2d6d8304f9291d4b444d5b613360c96', 'sim/level_stat_audit.py': '655ca6b79e4ebfd929c24a543a7c38d62995cbff55a26fca027946ad084a28bc', 'sim/match.py': '88ff55912ed71b4c8044f73f4b158ae7bfb823125dbd047e1f6d46a2868a40f5', 'sim/meta_decks.py': '5bfd479c6f8e2d875814f6f980d47f793131d498d19578f80ba1d065542ba13b', 'sim/model_tokens.py': 'ad668193e4c00993748dcf96bae4fafcb1763469ad57ea5e335e45e8143741c1', 'sim/opponents.py': '8800efa603698fb05c25c97761657c26f13d3e4983827e95cfd825bbd17598e5', 'sim/pathfind.py': '1add0f57e26c75093a9a5cc6576cdac67a94bdcd69bafc8f0e4a955aaeec127a', 'sim/policy_coordinates.py': '5bdb504301d43b9e31cd6e6069492d55667c1abb45d1b25cb70eb0d1ff140ffa', 'sim/policy_soup.py': 'f7cfbc566dc1c37f6bf792370af6dafb974f4319ec6094d4641736fc19c1dd30', 'sim/ppo_tuning.py': 'a71d0b70aa9c438cc8ad2d8133eee3cbffb19ec18ae6bd9b071df2875654417d', 'sim/probe_plan.py': '373c402b452bf967bdd370c22d811d21001d01472e6ecf004e7d8f2675235c1a', 'sim/public_area_spells.py': '9944347f87ab00ad6dbdbe86d35e957a7c935f2dfdb326ed1939c417c596e61e', 'sim/public_gamedata.py': '61763d0502a3a64d6e8c4e43d1ee7025e324ed6c3927f258380e6a5cb80273c3', 'sim/public_scenario_spells.py': 'eb496528f76937658f6205582734a32ed888a622a987165d76163dbe131f1fe6', 'sim/public_stat_audit.py': 'dd06a593519d824c1c9431a84e71a5e252aa71b0cc283c6f38c376fe94832792', 'sim/public_stat_scaling.py': '25dfdb5b64da6aac4f0d9ce5fe7bbb2dfaf50b26f55a0a5789140524ddff0909', 'sim/public_state.py': 'bde206b83f5b8ee013e425069221a70af680a5758230d6ef0aacda48170ff404', 'sim/readiness.py': 'db697f362e6f7e294af189360f45df73917a6c5aeaadbb702d1f3c4a08e7e603', 'sim/redrive.py': '043ff968f95b388461e980a846e64cd67d60e1702043e0bf3e3892f79051c572', 'sim/replay_bc.py': '35a43f360786d8563bcdbcdfb6fa51e46d81cc69d6500e62995bd55be88cc9b7', 'sim/replay_corpus.py': '97d3c42d8adb0adb4ea1e54cf85f351d32d00d650d867e0a92a4559f2beb0bcc', 'sim/replay_rows.py': '222e1992836888b4769339e32e2e91e76fdd6f7e510f89c74e42d51c01d41cde', 'sim/reward_events.py': '969944b763b262e7f6e6f733e77918983a919934def0c7180f928a5b52d5e75f', 'sim/reward_profiles.py': 'd47df813842eacbb643fe5b0f6eac7cc26a17bf11ee0c07267bb32d138b625df', 'sim/runner.py': 'bde0abba924e9f693fa1acaa04b1f87e032362f954eb84076034ededa8a27b33', 'sim/search_agent.py': '419d0cc8ab0a995c5b02a9217a1e546a8b4b6bb4ae9f844de8ecc57e7b105323', 'sim/selfplay.py': '71807a93544da449428c759a7539cdb7af0d46a9a48f25b8ff3ecfbb9e6eb787', 'sim/spells.py': 'f8e641b7cdade4d73ce27867ef3998d6e98429d035ef954b33bd2d8f1c5259cd', 'sim/strategy_env.py': 'cca83c612d18d7412c3bc2472041580a5ded5d320a5f5dff67c9826b95f35027', 'sim/strategy_sampler.py': 'e1b7d1145b9ffeccaf6d0e43417d18f628061175e819885a931462aca3b45e7e', 'sim/sweep.py': 'da71dd33b55c1071b492485e1898abedc0a328df0fbc4c4a121454696d5116f3', 'sim/token_agent.py': '8c45b3ab20712419673c1c5f8c2665e8718d971ce006e11e32c7eb1f618fb653', 'sim/towers.py': '92cb040be4bf14c69e488ffc8d11663685d87f21820e165f7a83e57dd8eee826', 'sim/train_attack_grpo.py': 'e190d87f5f964d5acae6bcf665782164561da32adaf0591bf951bffd405c0e61', 'sim/train_attack_ppo.py': '52ae119f51cfb6abe8f28991cbcb51ca07d7065a7f5a797739e40f2acebda88b', 'sim/train_defense_grpo.py': '4b37bd3a949ae191b19fea2857d66e745a986e6924fd9b661b4e84556bf1e551', 'sim/train_defense_ppo.py': '9d3b7ea4fb50f3aa898d1ca6df09e80287b8170c20db466a280d78198d573b31', 'sim/train_ppo.py': '26b4ae4a18d6de7a091378672c28da15043ee0f7e8c64655609b431a2c66c81b', 'sim/train_tokens.py': '543265a3fbee0e323da60e088e7d7d7ca7eff04c29b21f2b069cc276464f9d76', 'sim/validate.py': '5c5cd0dd7dacf49bc93b85aaaff91ac3357d424b7c646a7e968aacca940acf61', 'sim/vecenv.py': '71b4af93f778a3a068a11af27194d3fec69fa9d138c5de5712b6b9840b1c036e', 'sim/watch.py': '3a76d46b204a273e079e6d3f4a6873660af25c964ec43e702fb151b6bd88c9e4'}, 'pins': {'manifest.json': '3325d51f7978c8805e851624cd8d2bb7e70d133811545a01ceb6b20f9d7c9f44', 'extensions.tar.gz': '129cb8a1bf3eb5fb270f06a2d1a8c3d686122df6a3e1b03b2575eded5bce3607', 'protocol.json': '6ce1aa35bb3de18d7aa4b5bc0ecbf58f55d884672ec24d511e4a07106856daf3', 'parent.pt': '40d028fae25b30a23272d97bcc16d04c45f6e297c4f3b244a501d3bbd181aed9', 'air_examples.npz': '0c5284b46257932758dd0aaa08f82342db8c4de605bf9d81b253b4303f7f27ff', 'air_manifest.json': '1c36798ef31fd9b4ae7e7dd707692f514ef7e4171e8d63d821f555d1886e352f', 'predecessor_verified.json': '8f31f2cef33d13e9526082fff8186e40d7378802b7d3880397b9dc8c05ce2c49'}, 'evaluation_action_delay_ms': 1350})
