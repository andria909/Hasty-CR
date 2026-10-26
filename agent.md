## CURRENT:600-game evaluation VERIFIED; next handoff armed, goal ACTIVE
Observer-only maintenance: corrected-slot streamPID65012 replaces63861, resumes SAME live_stream/events.jsonl/progress.json without truncation,100reconnects. Wrapper waits until prior observation+1200s before reconnecting;GPU untouched. monitoring.json authoritative. stream_training --resume requiresmatchingkernel/existingfiles andpreserves counters/raw events;7streamtests pass. Collector63862/postprocessor64243/focuscoordinator64595 unchanged.
Active objective is iterative improvement: ideas -> short experiments ->2-4full models ->independent metrics ->repeat. Preservev25/secrets, no live withoutnewrequest. Prior UI-paused notes below are historical.

reports/defense_plan/continuous_control_pool_verified_20261026.json and readable reports/continuous_control_pool_20261026.md:600games checked (480random-opponent tests+120pairedduels). Balanced81/120test,41/60tournament,14/20v25,12/20parent; strong94/120,24/60,8/20v25,11/20parent; parent96/120,25/60,8/20v25;v2576/120,30/60. Mskmean balanced.2417,strong1.333,parent.0333,v25.0417;zerooverflow all. Not universal improvement; no promotion. Current corrected-slot models are NOT these old global-block controls. CUDAjob control-eval terminalCOMPLETE,11filescollected andverifierpassed;collector64087 intentionallystoppedafterterminalproof forcompletion-triggeredcollection.

New validation audit reports/defense_plan/continuous_validation_seed_audit_20261026.json: old48validation rows replay24physical seeded states twice (24/24paired endpointmetrics identical aftertaskmetadata removed). Preserve oldnumberswithlimitation. Current runningv2 job remainsfrozen andhasoldvalidation method; no silenthotfix. FUTURE worst_family validation nowseed+task_index withCLI --validation-seed900100; metrics exportepisode_seed/match_seed. Focus selectors reject duplicate72seeds ormissingfamilies.12-game exact-template CPUsmoke confirms12distinctphysical seeds.

NEXT transition coordinatorPID64595isARMED andconfirmedalive, waitingfor reports/defense_plan/continuous_slots_verified_20261026.json. State tmp/kaggle/continuous_focus_handoff_20261026/state.json; immutabletemplate tmp/kaggle/continuous_focus_template_20261026_v2 SHA c802d416e99c6eeb53d95347d2ea2d742b75ed3a151ffbe52bdca1e3d03c9f06. Itwill usefrozenbind.py, notmutableworkingtree, bindtwo verifiedv2weights+original9d6...,self-containedprivatedatasetownervertel0, uploadandpushONEkernel vertel0/hastycr-continuous-focus-20261026. Output tmp/kaggle/continuous_focus_cohort_20261026. Tokens onlyreadexternalregistryindex1,neverGit/logs. Idempotentpersistentrequestflags andownedkernellisting preventduplicateGPUjobsafterambiguouspush.2testsensure noAPIbeforeprerequisites andnoGPUrepushafteruncertainresponse. NoGPUjobsubmittedforfocusyet. reports/defense_plan/continuous_focus_handoff_20261026.json.

FocusGPUcontrollerfirstre-evaluatesparent+2v2weights on72unique balancedvalidation states, select_initial onlyby.5macro+.5worst.4configs uniform/air_golem3x ×entropy.01/.03;8pilots62/63×102400, allshareselectedinit/freshoptimizer. select_profiles requiresbothseedsandminimumfamily.1;fewerthan2eligible ->clean pilot_complete_no_eligible_pair, NOautomaticfull.Otherwise2×4096000seed64. Everyevalboundarysavesregularcheckpoint;selectedeligiblecheckpointcanbeearlierthanunconstrainedglobalbest.Collectorprofilecontinuous-focus downloadsselected+latestonly;verifier checksallactualconsecutiveCUDAcounters,72distinctcontexts,initial/pilotselectionandpins. Scripts prepare_continuous_focus.py,advance_continuous_focus.py,verify_continuous_focus.py. Current selector sourcehash mustmatchfrozen template whenverifying; usefrozen source iflatereditingit.

## Current: next hard-family cohort prepared from TRAIN evidence, not test
reports/defense_plan/continuous_train_family_diagnostics_20261026.json is reproducible via scripts.attack.analyze_continuous_training on the two completed control metrics.jsonl files, updates>=3000. balanced2262completed TRAINgames55.53%W, air15.96%,golem11.11%;strong2451games42.47%W,air8.82%,golem13.97%. Both zero mean manaoverflow; uniformcard regularization alone did not fix weak archetypes. These are stochastic TRAIN outcomes, not current corrected-model tests.

Prepared data/ppo_tuning/continuous_focus/protocol.json:4configs=uniform/air_golem3x sampling ×entropy.01/.03;2seeds62/63×102400, same overflow.15/JSunits.1. CLI now supports --continuous-focus and --validation-selection worst_family. New ContinuousScenarioCycle gives distinct seeds to repeats and exact cursor resume. Worst-family validation stratifies all six archetypes and scores.5macro+.5worst; planned72games(36/task). Eligibilityfloor.1perfamily is ONLY a short-pilot training gate, not finalacceptance. Do not launch blindly before selecting/pinning initial weights from corrected-slot validation; no test-based hyperparameter tuning. This next cohort IS NOT launched. CPUsmoke16steps plus12full validation games completed with proper6family coverage;4sampler/selector tests and combined21passed. Running remote sources remain immutable.

Local corrected-training postprocessor replaced ONLY to add local SSE->status/done sync every1200s; currentPID64243 (see monitoring.json authoritative). No extra remote polling, no GPUrestart. Progress merge rejects replay regressions. New POOL_TEST/PAIR event parsing derives only a lowerbound for completed evaluationgames, deduping reversed pairs;11monitor tests passed. Existing evaluation stream imported earlier code: use replay_evaluation_progress(saved events) to derive lowerbound without an API request. Last replay400/600 from3finished120game modelparts+2finished20game pairparts; no claim that this is the exact current count or final verified report.

## Current follow-through: independent600-game control evaluation submitted
Corrected-slot PPO continues andreiperelaiko/hastycr-continuous-slots-20261026. Prior global-block control training completed and independently verified in reports/defense_plan/continuous_training_verified_20261026.json: balanced bestupdate1250 SHA041d9038... validation28W20L/48, strong bestupdate2500 SHA1da05e8...30W18L/48; both full4096000 budgets verified, neither promoted.

New independent evaluation andriashadrin/hastycr-continuous-control-eval-20261026 VERSION1, root tmp/kaggle/continuous_control_eval_20261026_v2. Tiny initialdatasetv1 preserved; datasetv2 only adds explicit seed in heldout reportrows.600games=4frozenweights×120synthetic testgames +6pairs×20pairedduels. Models: two completed global-block controls, original9d6...,v25. ALL evaluated under corrected independent-slot execution225/2100slot/250ms and same calibrated physics; original contracts306/309 and physical_live/joint modes explicit. This test does NOT choose coefficients/checkpoints or claim live quality. CPUduel300000ms completed with0rejectedcommands.30targeted tests including3 verifier context/counter/draw tests passed. Collector profilecontinuous-pool; verify_continuous_pool rejects missing/duplicate seeds, changedSHA/outcomes/cardmeans/overflow and incomplete600budget. Monitors64086stream/64087collector/64088localverifier, APIchecks20min. Report continuous_control_pool_launch_20261026.json now confirms INPUTS_VERIFIED and2TeslaT4. Afterterminalcollection writes continuous_control_pool_verified_20261026.json. Compare strong/balanced/v25/parent on test and cardusage without retuning on test, then evaluate the corrected-slot models separately after their validation-only selection.

## CURRENT: per-slot execution v2 is actually training
User rejected global ACK waiting. Live memory_play now tracks pending commands PER SLOT; ACK checks only that slot, no extra1s cooldown after replacement, and only requires fresh tick instead of global500ms throttle. PlacementBudget reserves cost immediately, caps at fresh native balance, conservatively regenerates single-income while pending, reconciles once all ACKs arrive. Other slots remain playable; no live games started.

New immutable Kaggle andreiperelaiko/hastycr-continuous-slots-20261026 VERSION1,2TeslaT4 confirmed actual trial updates. Root tmp/kaggle/continuous_slots_20261026.8pilot×102400+2full×4096000=9011200. New contract joint_public_presence_continuous_slots_v2. Sim commands225ms, only used-slot refill2100ms, decision interval250ms, independent pending queue/reserved mana. Controller source dataset andreiperelaiko/hastycr-native-fireball-candidate-20261025-inputs, parent full_joint_economy_soft.pt SHA9d6..., all source pins verified; CPUoptimizer/checkpoint smoke passed. Monitoring stream63861/collector63862/postprocess63863 every20min. Reports continuous_slots_training_20261026.json and later continuous_slots_verified_20261026.json. Old andriashadrin continuous_ppo job is GLOBAL-BLOCK CONTROL ONLY; never call its weights corrected slot behavior. Do not silently modify running/frozen source.

stream_training now recognizes4096000 full budgets (old TRIAL_PROGRESS wrongly used102400); old full progress must be recomputed from steps, not trusted printed percentages from old observer. No GPU restart needed for this observer fix. Model/v25 unchanged.

## CURRENT26Oct: full-game PPO actually running; live stopped
User explicitly authorizes autonomous large training and coefficient studies. No further live games until a separate request. Existing UI goal remains paused: create_goal(new objective) rejected because unfinished goal exists; do not falsely complete old goal. Work continues under explicit user instructions, with durable Kaggle controller.

Kaggle andriashadrin/hastycr-continuous-ppo-20261026 VERSION1, confirmed INPUTS_VERIFIED and2TeslaT4 plus actual TRIAL_PROGRESS. Root tmp/kaggle/continuous_ppo_20261026, immutable protocol/source extension pins.8pilot runs=4configs×2seeds×102400; top2 then4096000 each, total9011200. Fresh original9d6... initialization/optimizer per run, value-head-only warmup20480,targetKL.02. Live/best_model/v25 unchanged. Monitors63255stream/63256collector every1200s; localpostprocessor63306. Collector profilecontinuous-ppo. Report reports/defense_plan/continuous_training_20261026.json, design reports/rl_training_design_20261026.md. Quality NOT accepted; independent full-match tests against saved models still required after completion. Never restart GPU for observer SSL/403.

New sim/continuous_push.py: six legal8-card opponent decks, normal4hand/cycle/mana, random and ScriptedOpponent styles, real full Match terminal.309public inputs, no hidden scenario task. Natural overflow ledger in Match only counts clipped regular income. Trial leak0/.15/.35 and JSunits0/.1/.3; no quota/spam reward.225ms command+2100ms postexecution cooldown from1.5×actual live medians. Frozen original source3be0 plus explicit11filemigration; do not confuse old evaluations with this physics. Nativeclock median19.99Hz; new measured projectile allowlist raw×20, tower princess12k/king20k; unmeasured kinds retain oldscale. Reports native_training_{clock,projectiles,latency}_20261026.json. Kite helper closer; two actual Knight acquisition controls pass, NOT universal live kite proof.

Live defense economy_soft b9b34... series root tmp/live/defense_economy_bs_20261026_live10: initial1connectionlost +recovery5started +final2started =8of10 started; user stopped further play for training. First recovery2matches manually assisted, excluded from autonomous attribution. Subsequent recovery/final suffered queue and parser regressions; do not claim clean10game series. Last screenshot was collection UI, no new Battle. All memory_play/memory_series processes absent. memory_play skips same-world backwards ticks<=20, larger/world changes still failclosed; live bridge changes from prior turns remain uncommitted separately. Preserve rawlogs/screenshots outsideGit.

37targeted trainer/collector/mechanics tests passed before launch; later3new kite/serialization tests added. Exact package CPUpreflight4full games completed(parent0W4L), nounknown. Three unrelated older tower-troop fixture tests fail because clientTOMLs absent and loader usespublic fallback; do not claim whole suite green. Completed prior3-arm study102400each (balance.02) did not improve newskills, never promoted. Full20model400game usage report verified in reports/card_usage_20261025.md.

## Reproducible targeting triage and evaluation percentages
`scripts/gamestate/triage_projectile_targets.py --evidence tmp/native_projectile_target_recovery25_20261025_v2.json --output reports/defense_plan/native_targeting_mismatch_triage.json` replaces the ad hoc triage step. v2 report reproduces27mismatches;10cases retain an assumed target between scans but select a different target on full rescan. This identifies a simulator scan-phase hypothesis, NOT verified game behavior, and does not modify active/frozen combat source.
`scripts/kaggle/stream_training.py` now records heldout_game_progress actual completed/total/percent separately from completed exercise parts and quality verification. Duplicate/replayed/regressing counters and changed totals are rejected;3newprogress tests plus16combined monitor/collector tests passed. `replay_evaluation_progress(path)` derives the same state from saved SSE logs without remote calls. Active stream processes imported the previous version before this change; do not claim their in-memory code changed. Offline replay saved to each run's saved_evaluation_progress.json shows both roles completed fixed384/fresh384/king96; no actual game-count event yet captured, so game percent remains unavailable, not guessed. Existing SSE streams interrupted (candidateConnectionError,parentSSLError) and wait for their configured20min reconnect; this is NOT evidence of terminal GPU jobs and no restart was requested. First collector checks around20min after each launch. Current observations/source/context and saved logs are authoritative; reconcile at scheduled poll.

## Native held-out evaluations actually running on two notebooks
Both version1 notebooks on ownerandreiperelaiko confirmed INPUTS_VERIFIED source3be0 and CUDA_READY2TeslaT4: candidate `andreiperelaiko/hastycr-native-fireball-candidate-20261025` SHA485442..., root `tmp/kaggle/native_fireball_candidate_20261025_v3`; original reference `andreiperelaiko/hastycr-native-fireball-parent-reference-20261025` SHA9d6..., root `tmp/kaggle/native_fireball_parent_20261025_shared`. The parent requested slug lacked `reference`, but Kaggle returned the actual title-derived URL; preparation.json was updated to actual handle, do not launch the old requested handle again. Both jobs use READY dataset `andreiperelaiko/hastycr-native-fireball-candidate-20261025-inputs`: contains parent as full_joint_economy_soft.pt, so no repeated128MB upload required. New prepare_reference_from_candidate preserves all dataset bytes and adds only a separate role/weight-file bootstrap. Verifier respects explicit policy_file and checks its digest against original parent. Both source/weights and CUDA confirmed; candidate fixed384/fresh384/king96 exercises completed in logs, no numeric results collected yet. Candidate380games,parent360 plus804scenario tests each. Observers30461/30462(candidate),30542/30543(parent); collector profile native-fireball-evaluation, firstpoll1200sec, monitor paths in launch_status report. No notebook restarts, no training or live games. Original attemptedparentuploadaccount0 endedHTTP404; originalaccount3HTTP401; those have NO GPU jobs. Known account0 previousdataset READY proves it is not blanket token revocation. Current launch/status dashboard supersedes earlier upload-wait entries below.
Target audit segment bug fixed: previous projectile lookup now includes trace segment to avoid numeric ID reuse across battles. Regression test simulates two battles with identical projectile IDs and host-wrapped frames;4targetaudittests passed. Recomputed recovery25 evidence tmp/native_projectile_target_recovery25_20261025_v2.json:73/100freshagreements unchanged, one mismatch explained by earlier inferred target retention. `reports/defense_plan/native_targeting_mismatch_triage.json` preserves27mismatches:9insidecurrentreach,15withinsource-radius-extension,3unknown-target-radius. These inferred labels do not justify editing production reach/targeting. Raw snapshots remain outside Git.

## Native evaluation collection/verifier and recorded targeting follow-up
`scripts/attack/verify_native_fireball_evaluation.py` checks prepared archive/extension/policy SHA, complete terminal receipt, every exercise seed/task/lane, king Fireball terminal attribution,804 strategy records, all19 opponent identities/modes and paired-seat game rows, recomputed summaries and raw game ledger. Collector profile `native-fireball-evaluation` added. 24 relevant tests plus3 projectile audit tests passed. Native input uploads remain unconfirmed: candidate session11216(ownerandreiperelaiko), parent fallback session81071(ownerandriashadrin), parent originalownerdemjanklimov returned401. Direct dataset status checks returned403 forcandidate/account1parent; this does not prove token revocation or terminal upload failure. No GPU notebook submitted; do not duplicate pending uploads. `reports/defense_plan/fireball_native_evaluation_launch_status.json` records handles and paths; parent fallback root `tmp/kaggle/native_fireball_parent_20261025_account1`. Old completed validation observers17515/17516 stopped only after terminal collection+independent audit.
`scripts/gamestate/audit_projectile_targets.py` now accepts host-wrapped `frame` as well as original raw rows. Read-only audit of remaining+recovery25 saved recordings:43949frames,562inferredisolatedimpacts,103comparable troop/building sources;freshacquisition76agreements/27disagreements, Musketeer45/68,Archer18/20,Cannon13/15. Source-radius exploratory hypothesis improves Musketeer55/68 but is NOT applied. Labels are inferred homing disappearance+HP loss, not verified target IDs; priorlock/scanphase/buffs/deploymentstate missing. This is NOT targeting accuracy and does NOT accept mechanics. Compact report `reports/defense_plan/recorded_native_targeting_exploration.json`; detailed raw event evidence only in tmp/native_projectile_target_{remaining,recovery25}_20261025.json, never stage it.

## Balanced validation audited; native held-out runner prepared
Separate validation report `reports/defense_plan/fireball_balanced_match_validation.json`: epoch50 SHA485442... passed unchanged actual game guards: candidate-v25 16W4L, parent-v25 16W4L, candidate-parent9W11L. Skill validation23/32 tower,32/32 building,31/32 splash, negative0/96. Original phase rejection due95% action proxy is preserved; separate protocol treats agreement as diagnostic. No independent test acceptance or live promotion.
`scripts/attack/evaluate_native_fireball.py` evaluates one pinned policy on native geometry/12000 Fireball: fixed384+fresh384 cases,96 king finish cases with original immediate terminal semantics and observed damage attribution,804 strategy cases, candidate380/parent360 games against19 pinned opponents. `scripts/kaggle/prepare_native_fireball_evaluation.py` packages exact frozen148training files plus only3new evaluation extensions; all19 local opponent hashes verified. Current final prepared roots `tmp/kaggle/native_fireball_{candidate,parent}_20261025_v3` (earlier v1/v2 packages unsubmitted, preserved). Candidate ownerandreiperelaiko, parentownerdemjanklimov. Dataset creation attempted, parent returnedHTTP401; candidate request pending at time of note. No GPU push yet, do not call these running. Recheck existing upload handle/dataset before retrying; do not duplicate remote jobs. CPU frozen-package preflight confirms804 contexts and12000speed,6candidate exercise smoke, two oracle KingFireball terminal kills;30 relevant tests passed. Coretraining files and originalweights unchanged.

## V3 result audit and native held-out protocol are ready
`scripts/attack/verify_fireball_balanced.py` audits the complete source archive, all 80 epochs, each skill floor, the shortlist, paired validation games, selection and finite selected weights. Run with `--collected RUN/collected --prepared tmp/kaggle/fireball_balanced_20261025 --output reports/defense_plan/fireball_balanced_candidate.json`. A completed phase without an eligible candidate is a valid rejection. Ten verifier tests and eighteen combined verifier/collector tests passed. The balanced collector now preserves all eight epoch_10.pt through epoch_80.pt files as well as the selected candidate; the existing observer may have imported its earlier version, so check terminal collection contents.
`data/fireball/native_evaluation_protocol.json` and `native_strategy_protocol.json` freeze the future comparison: exact 3be0 training files, native geometry and 12000 Fireball speed, 804 strategy test cases, 384 fixed Fireball cases plus 384 fresh-seed cases per policy. The opponent pool contains the fifteen approved portable models and all four recent full PPO models, with explicit 306/309 input modes and SHA pins. Test games use seeds 710001 through 710010, distinct from validation 910001 through 910010. Candidate: 380 games. Original parent reference: 360 games, excluding itself. The old 219/300 and 19/20 results used legacy geometry and 10000 speed and cannot serve as a new-physics reference.
Next independent work while v3 runs: prepare and run original-parent reference evaluation on a free account, using the exact frozen 3be0 source plus evaluation extensions and the approved weights. No native evaluation GPU package has been prepared yet. Keep every training file identical; use a declared weights-only native evaluation loader, never forge a legacy checkpoint configuration. Test data must not select training coefficients or epochs. Live games remain stopped; targeting and broader physical parity are unverified.

## V3 balanced phase реально запущена VERSION1 на2T4
andriashadrin/hastycr-fireball-balanced-20261025: INPUTS_VERIFIED/CUDA_READY на2TeslaT4 ирастущие3072demonstrations; gradientepochещёнеувиден. reports/defense_plan/fireball_balanced_status.json, пакет tmp/kaggle/fireball_balanced_20261025; source3be0c6dea..., protocolc8f5b600...; sourceпереходотd357явнозадекларирован4files+3additions. Init/teacherОРИГИНАЛ9d6..., НЕотбракованныеFireballweights. Новаяфазаoptimizerfresh. GPU0training/GPU1validationgames,sourceпинитсядоисполнения;80epochs,parsepercentизреальныхrow. monitoring.json streamPID12366/collector12367,profilefireball-balanced,firstpoll1200сек. Не делатьежесекундныеAPIcalls.
Новая sim.fireball_balanced: pairedpositiveCE+negativeclassmass+.5logoddscontrast(eachbatch) +demonstrationCE+2KL; teacherrehearsal2048actionable publicstates реальных regularMatchсимуляций(nativegeometry), неabstractscenariotasklabel и неforced-WAIT-onlystates. Positive3072 +1024sameHPpairs. FireballEnv task==joint_attack_task, flight12000; mechanical_mask использует match.geometry (новый4йsourcechange).
data/fireball/balanced_protocol.json фиксируетfloors tower>=.70,building>=.95,towerunit>=.95,agreement>=.95,negative_rate<=teacher+.05. Изeligibleepochs validation-only top3, затемкаждый40validationgames (20vsparent+20vsv25),seed910001..910010/seat swap. Teacher тоже20vsV25 наНОВОЙсреде; приниматьтолькосcorevsparent>=.45 иv25score>=newteacherref-.05; средиpassingвыборпоgameScore/minSkill/epoch. Ниtest710001,ниtestсценариивранжированиеНЕвходят. Еслинеeligible/notpassed — candidatefalse безfallbackнеудачныхepochs. ЭтоНЕприёмкавсегоплана80%vsкаждой/physics,targetingнепринят.
ExactpackageCPUsmoke:6/6referenceimpacts,12actualteacherrehearsalstates/input309/Fireball12000;tests15balanced+duels и13balanced+collector. Mergephysics65tests passed. Publicpocketформа всёещёunverified, менятьgeometryнеозначаетполнуюфизическуюприёмку.
InitialdatasetCLI timeout180, ноsamehandle exactownersearch/status подтвердилиcreated/ready; datasetНЕсоздавалсяповторно. FirstSaveKernel404, exactownersearchNotfound/statusdenied, retryсоздалVERSION1(неrestartlivejob). Всеreceiptsсохранены. Actualurlизуспешногореспонса. Нетliveбоёв, исходныйbest_model/v25неподменены. ПослеCOMPLETE profilebalancedcollect→независимыйauditprotocol/80history/floors/shortlist/games/source/SHA; еслиcandidatepassed, отдельныйnewphysicsheldouttest/all15+parentmatches. Старые1a34/10kотчёты исторические, несмешиватьсnative12k.

## Исправления среды перенесены в main после сохранения обеих прежних фаз
Main теперь Fireball12000mt/s в обоихloaders иFireballEnv taskflag==joint_attack_task. Прежние10k/always1 snapshots остаются вtmp/Kaggle packages иGit8608396/d357manifest; ихстарые результаты НЕпересчитаны/НЕпереименованы вновуюфизику. Дляповторнойпроверкистаройselectivity validation надоиспользоватьточныйстарыйsource snapshot, strictreplayнаизменённомmain намереннооткажет. Исходныеweights/v25неизменны. Следующаяфаза —новыйoptimizer сявнымphysics/inputmigration ипарнымиpositive+negative вКАЖДОМbatch; отдельныеvalidationfloors по3навыкам. Не запускатьстарые prepare_* безявногоновогоprotocol/method, не выдаватьстаруюфазузаexactresume.

## Обе Fireball-фазы COMPLETE/выгружены/проверены, обе НЕ приняты
V1 independenttest reports/defense_plan/fireball_refinement_full_metrics.json verified: 384newcases×2weights,804oldtest,320games, frozen1a34 source/10kFireball. NewFireballtest: tower1/128→92/128,building63/128→128/128,tower+unit33/128→128/128. Defense noDamage361/708→360/708. Но all15wins219/300→180/300, v2519/20→14W5L1D/20, противoriginalparent5W15L/20. НЕpromote. Raw tmp/kaggle/fireball_evaluation_20261025_v2/collected_confirmed; старыеobserversостановленыпослеindependentverify.
V2 selectivity COMPLETE: reports/defense_plan/fireball_selectivity_candidate.json verified cSOURCEd357/epoch15/SHAe44365185b8a03c68be57817b0b5f48c85fd88d62d88d0a4e6e12f720a0649bb. 96negativevalidation actualweights replay confirms firstFB80/96→0/96(initialv1→v2), originalparent26/96. BUTvalidationtower0/32,building32/32,towerunit31/32; agreement94.53%. Guard/meanselection корректновоспроизведены, ноmeanobjectiveпозволилколлапсоднойзадачи. Rejectbeforetest;НЕзапускатьдорогойtestдлявыбораweights приочевидномvalfailure. Весотдельно checkpoints/ppo_joint_economy_soft_fireball_selectivity_20261025/policy.pt; v1иoriginal/v25сохранены. Raw tmp/kaggle/fireball_selectivity_20261025/collected_confirmed. Старыйcollector7267истream7266остановленылишьпослеCOMPLETE+audit/replay.
Дополнительныйконкретныйinputbug reports/defense_plan/fireball_deployment_observation_audit.json: trainingFireball taskflagalways1, аdeployedjoint_enemy_presence даёт0 приenemyunit/building. Towerfinishсовпадает1, buildingfinish/towerunitНЕсовпадают. Отдельныйworktree /Users/andriashadrin/projects/Hasty-CR-fireball-speed-20261025, branch fix/fireball-native-speed-20261025: исправлены2Fireballvelocityloaders10k→12k(commit750c5ba) иFireballEnv._observeнаexactjoint_attack_task(commit1f0c29d);46tests passed. Reports corrections наветке. Mainоставленoldphysicsдляпроверкиv2; онаТЕПЕРЬвыполнена, можноexplicitmerge/migrationпослесохранениярезультатов. Archivedoldjobs/source не менять. НикакихGPUjobsвэтих2handlesбольшенет, новыеобученияещёнеподготовлены.
Следующее: main/source переходнаисправленнуюсреду, balancedpairedpositive+negative EACHbatch (v2overweightnegativeиtowercollapsed), validationfloorsдляКАЖДОЙиз3задач плюсguardстарыхнавыков/новыеvalidationmatches. Сначалазафиксироватьпротокол; НЕтюнитьподoldtest/drop39wins. Приобученииновыйoptimizerweights-only, декларироватьphysics/inputmigration; rawtest неиспользоватьдляранжирования. Исходная9d6...остаётсяbest_model.json иliveпо-прежнемузапрещён; goalНЕcomplete.
## Ветка fix/fireball-native-speed-20261025: Fireball скорость исправлена
Это отдельныйworktree с12kmt/s, НЕmain и НЕисточник активныхстарыхKagglejobs. Источник77нативныхполётов median11989.5mt/s. Изменены толькоFireballconversion вsim/spells.py иsim/public_gamedata.py; остальныеprojectilesсохраняютстарыеединицы доихизмерений. Тестстарого1000/60expectation поправлен по независимымтрассам; добавленыpublic/clientsection checks и12tile/1squeuedimpact.38tests passed. reports/defense_plan/fireball_native_speed_correction.json содержитold/newSHA2files иevidenceSHA.
Не менятьmain передselectivityaudit: егоreplay требуетd357source. Сначалапроверитьстарыеterminalартефактынаmain, затемexplicitnewsourcephase/единаяcorrectedphysicsоценка. Не выдаватьстарые10k результатызаизмеренияпри12k и неresumeoptimizerсо сменойфизики.

## Новые реальные tick-замеры: базовые скорости близки, Fireball существенно расходится
Только read-only анализ старойseries30/remaining+recovery25, новыхбоёвнет. reports/defense_plan/recorded_native_tick_speeds.json:45321frames,35759validprivateclock,4389 clockwindows median20.0096Hz. Ровные1swindows/hpconstant: Hog2381.53vs2400mt/s(n50), Knight1184.21vs1200(n345),Musketeer1185.08vs1200(n429),Skeleton1781.91vs1800(n699). Не менятьбазовыескорости: приблизительносогласованы;targets/buffs/charge/hitboxesНЕприняты. NativeC читает tick ПОСЛЕкоординат, неatomicfullframe; reportявноограничен. Старыйfrozen_mechanics_evidence.json верен длясвоихv6inputs, ноv8inputsдобавляюттики.
ВАЖНОFireball: reports/defense_plan/recorded_native_fireball_velocity.json,80flighttracks/720observations,77linearfits сresidual<=10mt,median599.476mt/tick=11989.5mt/s при20Hz. Текущие BOTHCSV sim/spells.py и public fallback sim/public_gamedata.py используютraw600*1000/60=10000mt/s. Предполагаемаякоррекция ИМЕННОFireball rawSpeed*20=12000, другиеprojectilesНЕменятьбезихзамеров. Actualimpactмомент/AOEнепроверены, толькоXYvelocity.9unit-tests двухaudithelpers прошли; input/sourceSHAсохранены, rawcapturesвGitНЕидут.
Нельзя молчапоменятьsource текущегоselectivity-verifier/replay: онпинитd357...иактивныстарыеfrozenjobs. Исправлениеготовитьвотдельномworktree/branch, сохранитьmain дляпроверкистарыхweights; затемexplicitnewtraining/evalphysicsmigration. Не выдаватьold10k-flightweights заalreadytrainedна12k. Обновитьdoneпосленовогоэтапа.

## Верификатор selectivity-фазы готов
scripts/attack/verify_fireball_selectivity.py проверяет terminal receipt/pins/source, неизменный40epoch protocol,1024train+96validation пар,positivevalidation контексты,выборпоscoreиguard,finite309weights. Приотсутствииeligibleepoch сохраняетverified rejection вместоошибочнойприёмки. Поумолчанию зановоисполняет96negativevalidation состояний наcandidate иoriginalparent с2CPUthreads: проверяетреальныйfirst-action cast rate иmean mass; требуетточногосовпадениявсехpackagefiles передreplay. Этоvalidationverification, НЕtest.15тестовselector/guards/context/nonfinitecounts прошли.
После selectivity terminal collection: .venv-live/bin/python -m scripts.attack.verify_fireball_selectivity --collected tmp/kaggle/fireball_selectivity_20261025/collected --prepared tmp/kaggle/fireball_selectivity_20261025 --output reports/defense_plan/fireball_selectivity_candidate.json. Еслиcandidatefalse —невыдумыватьновыйcheckpoint. Еслиtrue —отдельносохранитьSHA-вес, выполнитьновыйнеподбирающийtest/defense/matchregression сявнымmethodv2/source contract. Старыйv1verifier/methodloader намереннонепринимаетv2 безрасширения.

## Коррекция Fireball по train-контролям отправлена VERSION1 на свободный аккаунт0
reports/defense_plan/fireball_train_controls.json:48парtrain ситуаций/2weights,4сек horizon. Пустая полная башня: parent25/48 случаев сFireball, candidate39/48. Это предпочтениеchip, НЕ доказательство стратегическойошибкикаждогоброска. Новые sim/fireball_decision_pairs.py и scripts/attack/train_fireball_selectivity.py: неизменный input309; одинаковыеhand/mana/mask/privateclock и различающийсяHP; положительныйкаст проверяетсяреальнымimpact. Negative — пустаяFULLHPбашня, не HP=Fireball+1 (комбоZap/поддержка могутбытьправильными). Loss штрафуетсуммарнуюмассуFireballкласса, не учитвсегдаWAIT, не меняетинференсмаски.
VERSION1 andriashadrin/hastycr-fireball-select-20261025 source d3578634..., начат наTeslaT4; INPUTS_VERIFIED/CUDA_READY иdemonstrationsрост подтверждены reports/defense_plan/fireball_selectivity_launch.json. Пакет tmp/kaggle/fireball_selectivity_20261025; streamPID7266, collector7267, profilefireball-selectivity,20мин. Исходник9d6...teacher, init042963...firstcandidate, обаSHAпинятся. Новыйoptimizerphase/неresume.40epochs:positive3072impactepisodes,1024negativepairs,1536originalteacherrehearsal; validation96positive+96negative. Фиксированныеweights positive1/negative1/KL2;selectscore positive_success-.25negative_firstcast;guard agreement>=.90 and negative_rate<=originalparent_rate+.05. Protocol/source frozen ДОчисел pendingv1test. Test нечитался/неподбиралкоэфы.
ПервыйdatasetCreate404 подтверждён exactownersearch No datasetsfound, затемcreateуспешен/ready, GPUjob единственныйversion1. Ничего не удалено. Базоваяпроверкаv1 продолжаетandreiperelaiko/hastycr-fireball-eval-20261025: scheduledCLI RUNNING (авторитетно), SSESSLобрыв — не GPUfailure, не перезапускать. Проверятьраз20мин, не угадыватьчислоотсыгранныхматчей. Collector5582ещёжив, причиназадержкилокальногоobserverпоканеподтверждена.
Добавлен scripts.attack.summarize_fireball_evaluation: после terminalfullcollection проверяет frozen1a34package, всеSHA,384×2newtestcontexts,804oldtestexactcontexts/attribution,320pairedgames идвойныеагрегаты, зановоverifiesoriginalparentreport. CLI --collected EVAL/collected --prepared EVAL --evaluation-package tmp/kaggle/ppo_selected_evaluation_20261023_v3 --parent-prepared tmp/kaggle/ppo_selected_final_evaluations_20261023/andreiperelaiko --parent-collected .../andreiperelaiko/collected --output reports/... .11tests наконтексты/исходы;5pairedcontroltests;collectorcombined23passed. Для новойselectivityphase нуженотдельный verifier еёscore/guard/config/пары/finiteweights; старыйv1verifier намеренноотвергаетметодv2. Никакогоpromoteпоtrain/validation, liveзапрещён,goalнеcomplete.

## Fireball candidate COMPLETE; независимая проверка реально идёт на2T4
Кандидат после40epochs проверен scripts.attack.verify_fireball_checkpoint: SHA042963aa..., best_epoch40 выбран по newvalidation и>=90%rehearsalagreement. reports/defense_plan/fireball_refinement_candidate.json: validation башня1/32→23/32, здание9/32→32/32, tower+unit6/32→32/32; agreement94.466%. Это НЕ test. Вес сохранён отдельно checkpoints/ppo_joint_economy_soft_fireball_20261025/policy.pt; исходныйbest_model/v25 неизменныSHA. Fulltraining collected в tmp/kaggle/fireball_refinement_20261025_v2/collected. Training observers остановлены только после terminalCOMPLETE+проверки.
Новый независимый eval VERSION1 andreiperelaiko/hastycr-fireball-eval-20261025 реальноINPUTS_VERIFIED/CUDA_READY на2TeslaT4. reports/defense_plan/fireball_refinement_evaluation_launch.json. Пакет tmp/kaggle/fireball_evaluation_20261025_v2; monitoring.json streamPID5581/collector5582,20мин. Profile fireball-evaluation. Общая замороженная среда1a34..., НИ ОДИН старый файл не заменяется; добавлены3SHA-pinnedмодуля. CPUsmoke на точномfrozenpackage прошёл6validationcasts и15baselineweights admission. Отдельно6тестовverifier,12verifier/collector passed.
Проверка: parent+candidate по384новыхFireballtestситуации (768эпизодов), candidate старый804-case test сterminal-v2,320pairedgames:15старых×20 плюсparent×20,seed710001–710010/сменасторон. Веса ЗАРАНЕЕ зафиксированы. ПослеCOMPLETE скачатьprofilefireball-evaluation, проверитьконтексты/SHA/агрегаты/320schedule, сравнитьрегресс с прежними candidate-parent804 и300matchreports. Не тюнитьпоtest, не менятьисходныйbest/live поvalidation.
Перваяuploadпопытка evaldataset упалаSSL наGetDatasetStatus ДОcreate; ownedlisting доказалотсутствиеdataset, повторcreateуспешен. Всеerrorreceipts сохранены вtmp/...eval_20261025 иv2. ПовторныхGPUзапусков НЕТ. Bootstrap поддерживаетauto-extracted extensions/ иarchive, проверяетвсеmemberSHA. Учебнаяветкаплана сохраняетsupervisedphase, не называтьPPOoptimizerresume.

## Fireball-дообучение выбранной joint economy_soft запущено
Пользователь явно просит доучить именно checkpoint 9d6ac7d1...: добивание башен/обычных зданий и одновременный hit башня+идущий за ней юнит. Новые sim/fireball_curriculum.py и scripts/attack/train_fireball_refinement.py: отдельная supervised BC-фаза с impact-verified demonstrations и KL-повторением исходной политики на старых TRAIN-состояниях, НЕ PPO optimizer resume. Сохранить исходник/v25. Вход joint309 без новых признаков, фактический Fireball-only успех, одна cast-event для двойного попадания.
Kaggle andreiperelaiko/hastycr-fireball-refine-20261025 VERSION1, RUNNING и TeslaT4/cuda0/source SHA доказаны в reports/defense_plan/fireball_refinement_status.json. Пакет tmp/kaggle/fireball_refinement_20261025_v2, private dataset ready. Source78419cfc..., всеallowlist SHA проверяются bootstrap. Новыйtrain3072 ситуаций, frozen rehearsal1536 состояний,40epochs; selection толькоnewvalidation96, min90% rehearsalargmaxagreement — proxy, НЕ приёмка защиты. Архив старого протокола НЕ изменён. Первоначальный stream observer умер при auth; заменён только observer, GPU НЕ повторялся.
monitoring.json: stream PID3438 (live_stream_retry/events.jsonl), collector PID3641 с первым запросом через1200сек и дальнейшими1200сек. collector profilefireball сохраняет terminal result/candidate/metrics/baseline/audit, неproject source. Не polling каждые секунды. Проверки34tests passed, CPUsmoke1epoch выполнен, train oracle96/96 подтверждён. Кандидата ЕЩЁ НЕТ локально, не объявлять наученным. После COMPLETE выгрузить, проверитьSHA/init/конечность, зафиксировать validationselectedepoch и провести новый независимый Fireball test + прежнюю defense/match regression на общих контекстах. Test не использовать для выбора кандидата. Новыеweights не подменяют best_model.json/live до проверки.

## Все четыре описательных TRAIN-замера проверены
Отчёты ppo_defense_spell_precision_train_metrics.json и ppo_joint_economy_soft_train_metrics.json добавлены после независимой проверки terminal COMPLETE, SHA весов/исходников и всех 4452 контекстов на модель. Все четыре модели: train/val/test/300 матчей готовы. done.md пересобран scripts.defense.update_done.
Train noDamage: defense spell_precision 2492/4032 (61.81%), defense economy_soft 2629/4032 (65.20%), joint economy_soft 2174/4032 (53.92%), joint tower_survival 2137/4032 (53.00%). У всех TRAIN spell finish 0 среди известных исходов; неизвестные сохранены отдельно. Новое обучение и live не запускались; прежние веса сохранены.
Следующее: train-only исправление/проверка целевой награды добивания и покрытия сценариев; затем validation подбор без использования test. Полная приёмка всё ещё не пройдена: 80% против каждой прошлой модели и физика реальной игры не доказаны.

## Лучшая модель, сохранённая по запросу пользователя 25 октября 2026
PPO «атака + защита · economy_soft»: `checkpoints/ppo_joint_economy_soft_20261023/policy.pt`.
Точный SHA-256 и метрики — в `best_model.json`. В симуляторе: v25 19/20 (95%),
все 15 соперников 219/300 (73%, общий результат равен defense spell_precision).
Это сохранённый кандидат; live не переключён, соответствие физики реальной игре не принято.

## ВСЕ4 finaleval COMPLETE, осталось2описательныхtrain-прогона
Полные4rawколлекции EVAL/OWNER/collected сохранены и независимыйverifier прошёл.
reports/defense_plan/ppo_selected_four_final_comparison.json:7440val/testэпизодов+1200matches,
одинаковый common1a34/native306/309. Observedv25:SpellPrecision15/20,DefEco13/20,
JointEco19/20,JointTower15/20. All15:219/300,197/300,219/300,196/300.
Ниодна не80%противКАЖДОЙстарой, physicsнепринят, goalНЕcomplete.
Отчёты: ppo_defense_spell_precision_full_metrics.json, ppo_joint_economy_soft_full_metrics.json
плюсдвапрежних. Не использоватьtestдлявыбораweights/коэфов/teacher.
TRAIN DefEco тожеCOMPLETE/4452records validated: ppo_defense_economy_soft_train_metrics.json;
noDamage2629/4032=65.20%,meanHP523.74,4timeouts,0invalidscripts. Таймауты явно вdone.
TRAIN JointTower ужеготов ранее. Новыеtrain-only0/2 version1 реально2T4 иобаworkerрастут:
reports/defense_plan/ppo_last_two_train_evaluations_launch.json. Allfinalevalobserver0/2
остановлены послеполнойвыгрузки, GPUнеперезапускался. Andrei firstSaveKernel404:
послеownedlisting retryсоздал VERSION1(неversion2), новоеобучениеНЕперезапускалось.
СледитьзаTRAIN0/2 по monitoring.json вtmp/kaggle/ppo_selected_train_evaluations_20261023/OWNER;
приCOMPLETE collectorprofiletrain→summarize_selected_ppo_train→добавить reportфайлы.
НОВЫЙ scripts.defense.update_done автоматическипересобираетПОНЯТНЫЙdashboard из
проверенныхJSON, сверяетcheckpointSHAсfour_full_completion, пропускинепревращаетвнули.
Команда .venv-live/bin/python -m scripts.defense.update_done. Не appendисториювdone.
Послеоставшихся2train: все4таблицы; следующийtrain-onlyдиагноз/исправлениеfinisher
целей/наград иcoverageZap, безпереносастарыхtestfamilies вtrain и безсмены4зафиксированныхweights.
Реальныебоипо-прежнемузапрещены; v25неизменна.

## Втораяполнаяоценка готова: defense economy_soft
EVAL/vertel0/collected COMPLETE/completeTrue и независимыйverifier прошёл.
reports/defense_plan/ppo_defense_economy_soft_full_metrics.json.
Observed: v25 13/20=65%,all15 197/300=65.67%; testdefnoDamage357/708=50.42%,
meanHPloss809.69, princess655/708, king700/708. Непромоутить по1победеобщейразницы.
Tower_survival по-прежнемуv25 15/20/all15 196/300.
Defecon TRAIN-only version1 отправлен наосвободившийсяvertel0,2T4/bootstrap
и обаworker растут; reports/defense_plan/ppo_defense_economy_soft_train_launch.json.
Пакет tmp/kaggle/ppo_selected_train_evaluations_20261023/vertel0, monitoring.json текущийPID/log;
collectorprofile train. Finalevalobserververtel0 остановлен послесохранения.
Оставшиесяfinaleval andriashadrin/andreiperelaiko продолжаются потемжеkernels;
не перепушивать. ПослеихCOMPLETE коллекция+верификация+trainjobready.
done.md теперьсравнивает2модели иихtrain/val/test, таблица73строки, безappendистории.

## Текущие3finaleval: skillsчасти завершены, duelsещёидут
reports/defense_plan/ppo_remaining_evaluations_skill_parts_complete.json:все3реально
evaluating на2T4/common1a34, liveparts validation1056/test804 завершены.
Andria startupSSE404 восстановилсяавтоматически; ниодинGPUjob не перезапущен.
Все3 API RUNNING по20минпроверке; уAndreiSDKHTTPError перепроверенCLI
в EVAL/andreiperelaiko/scheduled_status_recheck.json. Ждатьтотжеkernel, не повторять.
Числовыеval/test reportпокане выгружены: finalcollectorнамеренноterminal-only;
не выдумыватьwinrates/процентduels изчислазавершённыхчастей.
ПослеCOMPLETE: collectprofile strategy, verify/summarize_selected_ppo, обновитьdone,
наосвободившемсяаккаунте отправитьУЖЕprepared trainjob4452.
Пути preparedtrainpackages: tmp/kaggle/ppo_selected_train_evaluations_20261023/OWNER/job.

## Train-only диагностика нулевого добивания выполнена
reports/defense_plan/ppo_train_spell_finisher_probe.json:2парныхtrain-ситуации Fireball
unopposed/king belowHP, fixedseedс Fireballвруке, realweights tower_survival безобновления.
Reference accuratecast2/2v2finishTrue, policy0/2. PolicyunopposedпервыйFireball
(9500,1500) мимо princess(3500,7500), timeout; kingcase убиваетHog/войсками.
ОтдельныйenemyKingDeathrewardиunconditionalhogDamage reward вAttackEnv допускают
обход spellgoal; это2примера, нечастотноеобъяснениевсехошибок. Cast/атрибуцияработаютнаreference.
Вpublished TRAIN finishers ВСЕ Fireball:3families counterspell/king/unopposed,72scenarios.
Zapfinisher family вtrainотсутствует. Не переставлятьужеиспользованные testfamilies вtrain,
не тюнитькоэфыпоtest. Текущие3finalevalGPUjobs/frozen1a34не менялись.
Host-only trainpackages остальных3 подготовлены в tmp/kaggle/ppo_selected_train_evaluations_20261023/OWNER,
ещёНЕотправлены: дождаться finalevalCOMPLETE ииспользоватьготовыеprivateinputs.

## 24октября: все4 fullбюджета иперваяtrainоценка ПРОВЕРЕНЫ
reports/defense_plan/ppo_selected_four_full_completion.json:4×2048000/update2000,полные
коллекции/weightsSHA проверены; all4 prepared/selector passed. Recovery checkpoint b9b34f2b...
train-only tower_survival COMPLETE,4452records/1484×3seeds verified;
reports/defense_plan/ppo_joint_tower_survival_train_metrics.json. Новый verifier
scripts/defense/summarize_selected_ppo_train.py проверяетcatalog/contextSHA/nativecontract.11тестов.
Train: noDamage2137/4032=53%,Hog138/204=67.65%,spell0/193known+23unknown.
done.md актуализирован:4/4,train-строки заполнены; НЕappendжурнал.
НОВЫЕversion1 finaleval отправлены наandriashadrin/andreiperelaiko/vertel0,
OWNER/hastycr-ppo-selected-final-eval-20261023. Всеpublishedinputsmetadata проверены,
контроллерпроверяетsource/weightsSHAдооценки. EVAL/OWNER/monitoring.json PID/логи.
Andrei ужеSSE evaluating2T4; Andria SDKstatusRUNNING ноstartupSSE404, этоНЕсбойGPU;
ждатьscheduledreconnect/проверитьтотжеkernel послеинтервала, не перепушивать.
Vertel толькоотправлен, проверитьегоSSE/GPU. Allsource одинаковыйcommon1a34...
Selecteddatasets для0/2 теперьуспешносозданы послеNo datasetsfound-audit;
раннийcreate404 Андрея не создалdataset. Для recoveryтожеprivateinputsготовы.
Старые fullobserver0/1/2 иtrainobserver3 остановлены толькопосле verified COMPLETE/полнойколлекции.
Далее собрать3finaleval, независимыйsummarize_selected_ppo, обновитьdoneтаблицы;
потомtrain-only техже3моделей изихEVAL preparedчерез prepare_selected_ppo_train_evaluation
наосвободившихсяаккаунтах. Ниtestниноваяtrainописательнаяоценкане меняютweights/профили.
80%-цельпротивкаждойисторическоймоделииphysics parity ещёне достигнуты; goalНЕcomplete.

## Текущий этап после запроса пояснить таблицы
done.md переписан с понятными названиями/знаменателями и краткими определениями. Это dashboard,
НЕ appendжурнал. Пользователь просит понятность, ане одниаббревиатуры; писатьwins/total рядомс%.
Обучены3/4: andriashadrin spell_precision иandreiperelaiko joint economy_soft теперьCOMPLETE,
обе коллекции+selector+prepared проверены. Оставшийся recovery defense economy_soft ещёRUNNING.
Andrei selected_dataset upload вернулошибку ПОСЛЕuploadфайлов: проверитьlaststderr иdatasetsfiles
передповтором, не создаватьдубликатыblind. Andria selected_dataset ещёНЕзагружен; оба finaleval надоотправить.
Tower_survival finaleval COMPLETE, rawreports сохранены, независимыйverifiedmetrics JSONготов.
НОВЫЙ train-only evaluator: scripts/kaggle/run_selected_ppo_train_evaluation.py +prepare_selected_ppo_train_evaluation.py.
Он безобновленияweights на2workerprocess/2T4 оценивает1484train сценария×3seeds42,43,44=4452episodes.
Version1 demjanklimov/hastycr-ppo-selected-train-eval-20261023 реальноevaluating на2T4,
обаworker растут: reports/defense_plan/ppo_selected_train_evaluation_first_launch.json.
Пакет tmp/kaggle/ppo_selected_train_evaluations_20261023/demjanklimov, monitoring.json текущийPID/log.
Collector profile train добавлен. Первыйlocalcollector умер приauthenticate ReadTimeout,
проверенmissingPID и заменёнтолькоobserver; GPUjob НЕперезапущен.
12тестов traincontroller/collector/summary прошли. Исходники и GPU-доказательство сохраняются в Git вместе с понятным dashboard.
Для train послеCOMPLETE проверить4452contexts/3seeds/source/policySHA, пересчитатьsummarize_split
по rawtrain.json и дополнитьtrain-строкиdone, не подменятьроднымиtrainingreward.

## Новое требование пользователя к done.md
done.md теперь КОРОТКИЙ DASHBOARD, не журнал: текущийэтап, таблицы train/val/test/винрейтов.
НЕдобавлять длиннуюисторию; подробныеfacts/proofs в reports/defense_plan и agent.md.
Обновлять ячейки для каждой новой модели только изпроверенных rawreports.
Новый запрос: отдельно измеритьконечныеweights наTRAIN; это ещёне запускалось.
В первую fullоценку вошла общая2 tower_survival: report ppo_joint_tower_survival_full_metrics.json.
Observed симуляция196W/104L/300=65.33%,v25 15W/5L/20=75%. Gate outcomes ВСЕunknown изза
coverageflags; НЕпоказывать0% как винрейт и НЕобъявлять80%-приёмку. Verifier теперь разделяет
observed_simulation и acceptance unknown. Test noDamage350/708=49.44%, Hog18/24=75%, spell0/72.
Validation Hog25/48, spell0/126known+18unknown. Тест/valнеиспользоватьдляподбораweights.
Вторая full joint economy_soft API COMPLETE и result2048000/update2000 подтверждены,
collectorfolder ещёвыгружаетtrain/metrics (85MB), не перезапускатьGPU.

## Первая полная модель сохранена, finaleval фактически на2T4
Demjanklimov joint tower_survival: full API COMPLETE,2048000/update2000, SHA5e01303f135fe10888b037ed1997fc43be5b012a421b798f446d5db2297cd7c2.
full/demjanklimov/collected полный; selector finitepolicy/optimizer/8snapshots прошёл.
Prepared EVAL/demjanklimov; selectedprivate dataset загружен. Новый finaleval version1
demjanklimov/hastycr-ppo-selected-final-eval-20261023 достиг evaluating на2TeslaT4,
common1a34..., input309 и explicitclamp migration проверены.
reports/defense_plan/ppo_selected_first_full_evaluation_launch.json.
EVAL=tmp/kaggle/ppo_selected_final_evaluations_20261023; monitoring.json PID/логи/collectorprofile strategy.
Не перепушивать. Новая evaluation ещёне завершена, testметрик пока нет.
Старые fullcollector/stream демьян остановлены после COMPLETE+сохранения, файлы сохранены.
GetDatasetStatus для готового evalinput стал404, но полный datasetsfiles165names/sizes совпал;
GPUcontroller дополнительно проверил SHAвсех164sourcefiles. Не перезаливать/удалять dataset из-за status404.
Stagevalidation128/10v25:score.556801, v25win.7, defenseQ.78125, attackQ.28125;
это НЕfull1056val/804test и не80%-приёмка. Остальные3 ещёне collected, originalандрейstreamlast99.55%;
ждатьего terminalcollector/свежегоSSE, не повторятьGPUjob из-заобрывастрима.

## Независимая сводка finaleval готова
После finaleval collection запускать scripts.defense.summarize_selected_ppo:
--collected EVAL/OWNER/collected --prepared EVAL/OWNER
--evaluation-package tmp/kaggle/ppo_selected_evaluation_20261023_v3 --output NEW_REPORT.json.
Проверяет terminalCOMPLETE, всеSHA, mode/nativecoordinates,1056val/804test exactcontexts,
300pairedgames/all15baselines/modes/SHA, пересчитывает noDamage/HP/survival/elixir
и terminal-v2 Hog/spell known/unknown. Invalidscript/противоречивыеHPметки отвергаются.
Неизвестный outcome запрещает подтверждение80%дляbaseline; rawreward неиспользуется.
Финальный quality_accepted остаётсяFalse: physics parity/medium threshold ещёне приняты.
19тестов verifier/protocol/evalcontroller прошли; настоящиеfinalresults покаотсутствуют.

## Finaleval inputs ready на всех4, local preparation watcher работает
Все4 private OWNER/hastycr-ppo-selected-eval-v3-20261023 API ready, архив одинаковый1a34b396...
reports/defense_plan/ppo_selected_evaluation_inputs_ready.json. Upload/status receipts в
tmp/kaggle/ppo_selected_evaluation_inputs_20261023/OWNER/.
Кампания tmp/kaggle/ppo_selected_final_evaluations_20261023/campaign.json, monitoring.json PID92060.
scripts.kaggle.watch_selected_ppo_preparation --wait наблюдает только локальные collected каждые1200сек,
вызывает строгий selector/builder лишь receipt COMPLETE/completeTrue; GPUjob самНЕотправляет.
preparation_status.json/log показывают готовность OWNER-пакетов.
После prepared проверить input_pins, загрузить только selected_dataset, дождаться ready, отправить
OWNER/hastycr-ppo-selected-final-eval-20261023 на2T4 сtimeout24000. evaluation_dataset ужеготов.
Не перепушивать существующийработающий kernel; приошибке сначала проверитьегоstate.
Для recovery коллекция collected_v2, source_manifest inputs/manifest.json; остальные originalfull.
15тестов watcher/selector/builder прошли. GPUevaluations ещёНЕзапущены, модели продолжаютtraining.

## Общая итоговая оценка v3 готова
Использовать tmp/kaggle/ppo_selected_evaluation_20261023_v3/ (164файла/15baselines),
SHA1a34b396650d0f0a51c5276cbc40fbb864b4f86db0019fbdc3edb874164453fb.
Одинаковая исправленная DefenseEnv._target для всех4;145из146training-файлов прежние.
Две поддерживаемыеtraining-линии: original40ba... и explicit recovery a55a...;
reports/defense_plan/ppo_selected_evaluation_readiness_v3.json.
Для recovery selector использовать source_manifest=tmp/kaggle/ppo_full_target_recovery_20261023/inputs/manifest.json
(рядом original_manifest.json/resume.pt); collected_v2 и launch_config.json в тойжеrecoveryпапке.
Для остальных3 использовать исходныйshared/manifest.json и full/OWNER/collected.
select_completed_ppo_full проверяет migration, original/resume SHA, бюджет2М/update2000;
prepare_selected_ppo_evaluation проверяет обе source-линии; controller передаёт original_training_files
в joint-loader. sim/evaluation_source_contract проверяет все17 fingerprint-файлов и
побайтово обратимость единственного выражения, обычный strictloader не отключён.
34теста и8realnetwork validationrollouts прошли; это НЕ полныйtest/приёмка.
Ничего не загружено/не отправлено наGPU для finaleval, ждать реальногоfull2М.

## Recovery2 теперь действительно обучается
Подтверждены TeslaT4 и update797+ (бывший сбой пройден), reports/defense_plan/ppo_target_recovery_gpu_proof.json.
Source a55a317b..., explicit semantic migration, не exact same-source continuation.
Не перезапускать; текущийstream_v2/collected_v2, polling1200сек.
Далее подготовить общую исправленную evalсреду и строгую поддержку migration в selector/builder,
после всех полных2М провести одинаковые val/test/15baseline сравнения всех4.

## Recovery version2 отправлен
Исправленный kernel version2 успешно отправлен; tmp/kaggle/ppo_full_target_recovery_20261023/launch_version2_receipt.json.
Первый version1 ERROR/KeyError mode доказан live trace и сохранён, SDKвыгрузка timeout без данных.
Локальные v1 observer остановлены только после подтверждённогоERROR; файлы сохранены.
Текущийmonitoring.json указывает stream_v2/collected_v2; interval1200.
Ждём actualCUDA и update797+; не отправлять повторно из-за404/SSL.

## Исправление recovery-конфига
Первый recovery version1 завершился ERROR до обучения: builder взял screening config безmode/selection.
Builder исправлен на diagnostic.launch_config.original и добавлен regression test.
Первый status/исходныйconfig/code сохранены в recoveryпапке; first_version_collection собирает kernel log.
Текущий launch_config.json/job/recover.py исправлены и проверены. Не считать version1 обучающимся.
Для version2 дождаться сохраненияterminalлогов, затем push и проверитьCUDA/рост797+.

# Актуальное продолжение: причина сбоя доказана, перенос из796

Диагностический kernel ERROR воспроизвёл тот же сбой, все96updates701–796 numerical-identical.
reports/defense_plan/ppo_script_target_failure_proof.json. Восстановлен checkpoint796/815104
SHA583c6cb5454e0d4149e3f9a5bee4760fc4314ad42e0728d1a0b9a73ef085d38b.
Zap target x=-24 внеарены. Fix: только return arena.clamp_to_arena(chosen.pos)
в DefenseEnv._target, не менять physics/позиции тел или пропускать invalidscript.
Новый явно migrated source a55a317b...,146файлов с единственным изменением выражения.
Recovery-пакет tmp/kaggle/ppo_full_target_recovery_20261023/. Private dataset ready.
Первый kernels push вернул404; authenticated kernels list подтвердил отсутствие новогоjob.
Повтор успешно создал version1 vertel0/hastycr-ppo-def796-recovery-20261023;
launch_retry_receipt.json сохранён. stream/collector запущены, polling1200,
monitoring.json содержит PID; CUDA/рост797+ ещё требуют проверки. Другие3 originalfull API RUNNING подтверждены.
Controller run_ppo_selected_full поддерживает строго проверенный migration из796;
prepare_ppo_target_recovery пакует исходники/неизменённые weights и resume.
Будущие selector/evalbuilder пока требуют исходный40ba... и НЕ принимают миграцию: нужно
явно добавить verified migration и общий исправленный evalsource для всех4, не обходить SHA.

# Hasty-CR: рабочая карта проекта

## Сейчас:3full RUNNING + диагностика сбоя второго аккаунта

Оригинальный vertel0/hastycr-ppo-coef-full-20261023 завершилсяERROR:
invalid_script_play_timeout в pair:bandit:battle_ram:zap/back/x1/left,
новыйконтекстseed49. Последнийcompleted796/815104, последнийсохранённый
checkpoint700/716800 SHA c40b573d0fe80ff01e514f2a539c370d2c6efd5c5f6479a1a012b0d5da41d433.
Всеартефакты ERROR выгружены и проверены; original/SSE остановлен только
после сохранения. Другие3GPUjobs продолжаются, не трогать/не перезапускать.
reports/defense_plan/ppo_full_economy_soft_failure.json.

Запущен ТОЛЬКОдля второго диагностический resume100updates из700:
vertel0/hastycr-ppo-def700-diagnostic-20261023 version1, реальныеT4/ростsteps
подтверждены: reports/defense_plan/ppo_full_diagnostic_launch.json.
Пакет tmp/kaggle/ppo_full_defense_recovery_20261023/, monitor/PID там.
Source тотже40ba4ec0...,RNG/optimizer/envstates восстановлены, physics не менялись.
Только read-only wrapper печатает SCRIPT_PLAY_DIAGNOSTIC приошибке;
checkpoint-every1, cap100additionalupdates (до800), сравнивает701–796с
оригинальнымиполнымиmetrics безelapsed. Диагностическийrun НЕ завершает2М.
Послеегоterminal: собрать result/train.log/latestcheckpoint (prefix selected_full),
проверить failure_telemetry и original_prefix_comparisons. Причинаконкретного
отклонения карты ещё НЕ установлена; не угадывать/не пропускатьinvalid-case.
CPU четыреэпизода seed49 сpolicy700(argmax/3sample) прошли, это неGPU-replay.

Реальныйпрогрессдиагностики уже>700 наT4, но точноеnumerical совпадение
проверитсяпослеterminal. ЕслиSSE SSLобрывается, это неостановкаGPU; один
локальныйstream переподключён, старыефайлы сохранены. Никакихповторных
push поверхработающихkernels. Старыеgeneric4-RUNNING таблицы ниже — история.


## Полный этап PPO после подбора (история запуска)

Все16 опытовseed42 и6seed43 завершены, выгружены и проверены.
Победители среднего двухseed: defense spell_precision/economy_soft;
joint economy_soft/tower_survival. Выбор безtest и безrawreward.
reports/defense_plan/ppo_coefficient_confirmed_selection.json,
ppo_coefficient_confirmed.md иppo_coefficient_confirmation_checkpoints.json.

Отправленыversion1 и фактически обучаются4 kernels
OWNER/hastycr-ppo-coef-full-20261023:
- andriashadrin: defense spell_precision;
- vertel0: defense economy_soft;
- andreiperelaiko: joint economy_soft;
- demjanklimov: joint tower_survival.
ВездеTeslaT4 и ростsteps подтверждены:
reports/defense_plan/ppo_selected_full_launch.json all_verified=True.
Пакет tmp/kaggle/ppo_coefficient_full_20261023/; PID/пути observer/collector
в OWNER/monitoring.json. Не отправлятьповторно; опрос раз1200сек.
При старте третьего восстановлен только локальныйstream после auth/SSE404;
kernel не перезапускался. СтарыеSSE первого/второго этапов остановлены только
после полнойвыгрузки; файлы сохранены.

Каждыйполныйrun:2048000 новых шагов, fixedteacher своего режима,
новыйoptimizer, seed42, sourcearchive40ba4ec0..., сохранение100updates,
trainwallclock18000сек/kernel24000сек. При incomplete_budget сохранить
последниеweights и продолжать толькоexact same-source boundary;
full_command учитывает additionalupdates defense / finalupdate joint.
Полныйбюджет и конечнаяприёмкакачества ещё НЕ достигнуты. v25 неизменна.


Готова новая EVAL-only версия пакета:
tmp/kaggle/ppo_selected_evaluation_20261023_v2/, archive0544f61b...;
reports/defense_plan/ppo_selected_evaluation_readiness_v2.json.
Все146 training-файлов неизменны. Новый sim/evaluation_terminal_damage.py
добавляет opt-in terminal attribution в sim.eval_strategy --terminal-attribution:
legacy metrics сохраняются, v2 записывает источник добивания только по
наблюдаемому HP delta во время исходного KingDestroyed. Для5 парных случаев
проверены побайтовые observations/rewards/masks/legacy metrics/snapshot.
Неизвестные источники остаютсяunknown. Никакоеtraining/reward/optimizer
не изменено. Дополнительно пройден CPU-preflight реальных306/309сетей на
2validation-контекстах/режим, с/безobserver:8эпизодов иидентичныеlegacyrecords.
reports/defense_plan/ppo_selected_evaluation_cpu_preflight.json; это только
проверкасовместимости, не полныйtest и не приёмкакачества. Пакет v1 сохранён, ниv1ниv2 не загружались/не оценивались наGPU.
36тестов observer/attack/evaljob прошли; изолированно загрузились306/309
weights, code_fingerprint иCLIflag проверены. Для новой итоговой оценки
использоватьv2. Будущий job builder должен фиксировать конечный update2000
из selected_full/result.json, а не старый selector истории best_validation:
в новых fulljobs eval-every0, полных результатов пока нет.


Фиксатор готов: python -m scripts.kaggle.select_completed_ppo_full
--collected FULL/OWNER/collected --launch-config FULL/OWNER/launch_config.json
--source-manifest tmp/kaggle/ppo_coefficient_study_20261023/shared/manifest.json
--output NEW_SELECTED_DIR. Сохраняет raw checkpoint какpolicy.pt, SHAне меняет;
schema selected-full-ppo-evaluation-input-v1. Не подставлять старыйselector
joint-strategy-selection-v1: разные контракты. Новыйbuilder готов: scripts.kaggle.prepare_selected_ppo_evaluation,
controller scripts.kaggle.run_selected_ppo_evaluation. Источникevaluation —
tmp/kaggle/ppo_selected_evaluation_20261023_v2/. Всем4одинаковые
1056val+804test и15×20duels; старые306безновых3features, joint309native.
Не создавать/не отправлять evaljobs до реальных COMPLETE/fulltarget результатов.
По первой20минпроверке все4RUNNING иactual10,95–12,50%; snapshot
reports/defense_plan/ppo_selected_full_progress_20min.json. На первом SDKstatus
былHTTPError, CLI подтвердилRUNNING в OWNER/scheduled_status_recheck.json;
при расчёте времени следующегоопроса учитывать этот свежийreceipt вместо
устаревшегоcollected.status.json. SSE теперь переподключается start-to-start1200сек.


Свежий60минsnapshot: reports/defense_plan/ppo_selected_full_progress_60min.json.
Все4API RUNNING ирастущиеактуальныеsteps; robustcollectors отработалипосле
задержки, безошибок/раннихAPIзапросов. В monitoring.json next_observation
теперь обновлён от реального свежегоstatus.checked_host_ns+1200сек.
Не отправлятьповторно kernels и не объявлять target_met до конечных2048000.


Свидетельства физики отдельно: reports/defense_plan/frozen_mechanics_evidence.json.
ПодтвержденыSHA двух старыхnativeисточников и шестьтекущихlegacy anchor-координат.
Исторические25/32по снарядам — косвенные метки, неaccuracy нынешнегодвижка;
host-timeскорость не является игровыми тиками. Native entityanchor/hitbox/terrain
соответствие остаётсянепроверенным. Не применятьгипотезурадиуса илиизменять
геометрию внутриактивногоfrozenrun; синтетическийobserver-parity не доказывает
real-game mechanics parity. Текущие4модели/сравнения относятся кзафиксированному
симулятору, конечнаяприёмка ещё требует независимыхреальныхсвидетельств.

Исторический40минsnapshot: reports/defense_plan/ppo_selected_full_progress_40min.json,
все4RUNNING/растущиеsteps, около20–23%. На третьем былReadTimeout SDK,
подтверждён CLI в OWNER/scheduled_status_recheck.json. Автоматическое
SSE переподключение сработало; эти4observer не перезапускались при40мин.


После40мин full collectors обновлены на устойчивый транспорт/cache.
В OWNER/monitoring.json новыеcollector_pid,collector_log,
collector_next_observation_host_ns иinitial_delay; первыйопрос отложен до
прежнего20минсрока, ранних API-запросов не было. Это не изменениеGPUjobs.
При частичнойвыгрузке можно сохранить прежнююпапку, затем запускатьcollector
в новый output с --cache OLD_DIR: проверяетсяkernel/status/SHA. Не удалять
частичныефайлы и не считатьreceipt.complete=False завершённоймоделью.



## Завершённый подбор коэффициентов PPO (история)


Первый подбор ЗАВЕРШЁН: все16опытов,102400шагов каждый, API COMPLETE,
все4collected/receipt.complete=True. Локальныйаудит весов/optimizer/снимков
восьми сред выполнен для всех16, v25 SHA неизменна. Отбор итаблица:
reports/defense_plan/ppo_coefficient_screen_selection.json и
reports/defense_plan/ppo_coefficient_screening.md.

Текущий этап — seed43,6опытов на3аккаунтах. Jobs version1:
andriashadrin,vertel0,andreiperelaiko/hastycr-ppo-coef-confirm-20261023.
Пакет tmp/kaggle/ppo_coefficient_confirmation_20261023/. Все3фактически
на2T4,ростsteps обоихworkers подтверждён:
reports/defense_plan/ppo_coefficient_confirmation_launch.json all_verified=True.
Не отправлятьповторно. Наблюдатели/collector PID и пути в OWNER/monitoring.json,
опрос1200сек. ВторойSSEпервоначально404, переподключён только локальныйstream,
теперь CUDA/steps подтверждены. Старыеobserver первогоэтапа остановлены только
после terminalCOMPLETE+полнойвыгрузки; удалённыеjobs не менялись.

После трёх полных confirmation collected запустить
scripts.kaggle.prepare_ppo_selected_full: повторно проверит обаseed, выберет
2defense+2joint и подготовит4полных job. Пока полныеjobs не отправлены.
Финалистыseed43: defense economy_soft/spell_precision/economy_strong;
joint economy_soft/tower_survival/survival_balanced. Никакого test-ретюна.


Новый прямой запрос пользователя: сначала102400 шагов на конфигурацию,
затем по2 лучших конфигурации полностью в режимахdefense иjoint (всего4PPO).
Не запускать новыеGRPO; предыдущие модели/замеры сохранить. Выбор только
поvalidation, не поtest и не по reward разных профилей.
Протокол/8профилей/фиксированные128-context validation: data/ppo_tuning/.
В каждом режиме одинаковый weights-only teacher: текущийvalidation-best
PPO1 (SHA0aa0648d...) илиjointPPO3 (SHAd9b0a524...). Optimizer новый.
Новый пакет: tmp/kaggle/ppo_coefficient_study_20261023/. Все4 inputs ready,
все4 kernel hastycr-ppo-coef-screen-20261023 version1 отправлены; на каждом
подтверждены2TeslaT4 и ростsteps у обоих workers. Всего16коротких trials,
по102400шагов. Доказательство: reports/defense_plan/ppo_coefficient_screening_launch.json.
Все4 study jobs API COMPLETE, все16train-бюджетов достигнуты. Идёт выгрузка:
vertel0/demjanklimov collected/receipt.complete=True; оставшиеся два collector
PID/логи вmonitoring.json, скачивание продолжалось. Не перезапускать обучение.
При сетевых сбоях коллектор сохраняет частичные файлы и повторяет загрузку;
cache использует только файлы, совпавшие с receipt SHA. Terminal collectors иSSE наблюдатели запущены,
PID/пути вOWNER/monitoring.json; status раз1200сек. Затем top3/mode seed43,
потом полное обучение2лучших/mode. Прежние eval завершены и сохранены;
200матчей+4020skill проверены: reports/defense_plan/five_model_round_robin_verified.json.

Пользователь явно разрешил регулярныйgitpush вorigin.62 накопленныхкоммита
уже отправлены вmain; source backup snapshot — branch
backup/workspace-source-20261023-101624. Токены остаются внеGit.
Все старые расчёты закончить/сохранить; подготовленный80-match parser-control
v25 job ещё НЕ отправлен и уступает приоритет подборуPPO.

## Обучение завершено, идёт полная оценка

Все4 основныхtraining kernels API COMPLETE и конечные счётчики проверены:
PPO1 2048000,GRPO2 1000650,jointPPO3 2048000,jointGRPO4 1000297.
Подтверждение: reports/defense_plan/four_training_completion.json.
Старые описания RUNNING обучения ниже — история. Ничего не переобучать
и не перепушивать поверх завершённых jobs из-за устаревшего SSE progress.
Конечные веса сохранены в collected/ и collected_final_/collected_retry_;
validation-best для общей оценки уже закреплены, v25 не изменена.

Запущены новыеfinaleval jobs: andriashadrin/hastycr-defense-final-eval-20261023
(PPO1+дополнительныйконтрольGRPO1),vertel0/hastycr-grpo2-final-eval-20261023,
andreiperelaiko/hastycr-joint-ppo-final-eval-20261023 и
demjanklimov/hastycr-joint-grpo-final-eval-20261023. Actual CUDA подтверждены.
Defence-only:708test+15×20duels на модель; joint:1056validation+804test+15×20duels.
Качество не принято, часть результатов ещё ожидается. Проверять раз20мин;
естественные SSE события читать можно. Не повторять активные GPU jobs.

Новый запрос — круговой турнир4основныхмоделей+v25 (200матчей), затем
таблица всех5. Подагент ppo300_live5 готовит отдельныйnewkernel и common
804-case skillbenchmark для всех5 на одномfrozen source. Дополнительный
GRPO1 в эти5 не входит. LiveMuMu по-прежнему0/5 и заблокирован SVIP.

## Последний запрос: PPO300 test и5 реальных игр

Пользователь явно разрешил ровно5 новых реальных матчей с
tmp/kaggle/defense_train_stage2_20261022/recovered_terminal_retry_1/ppo/best_validation_000300.pt,
SHA d4a583bc730b1b0373135c99c409f18ef475d8c3247dd29f639e74441e579cea.
Пока0/5: MuMu stopped и требует оплату истёкшего SVIP. Не покупать подписку,
не очищать данные и не создавать другой профиль вместо существующего.
Readiness и команды продолжения: tmp/live/ppo300_20261023_readiness/.
Полный mac.png содержит частный фон браузера: не публиковать/не добавлять в Git.
memory_series теперь передаёт выбранный checkpoint всем remaining runners;
никакого fallback на v25. После восстановления MuMu проверить свежие ADB/лобби
и запустить только оставшиеся5 с тем же checkpoint. Каноническую v25 не менять.

Отдельно запущена промежуточная GPU-оценка PPO300:
andriashadrin/hastycr-ppo300-test-v25-20261023,708test эпизодов и20парных игр
с v25. Это явный запрос проверить промежуточные веса, не финальная приёмка.
Пакет и результаты: tmp/kaggle/ppo300_test_v25_20261023/.
Подагент ppo300_offline_eval следит за ней; дубликат не запускать.

## Дополнительные общие политики на аккаунтах 3 и 4

Отправлены версии1: andreiperelaiko/hastycr-joint-ppo-20261023 и
demjanklimov/hastycr-joint-grpo-20261023. В каждой ОДНА политика на обе задачи,
309 признаков, task-mode joint, отдельный новый optimizer, инициализация v25.
Пакеты и submission receipts: tmp/kaggle/joint_strategy_20261023/ по владельцу.
CUDA Tesla T4 и рост шагов обоих тренеров подтверждены:
reports/defense_plan/joint_strategy_connection_check.json. Не дублировать задания.
Новые collectors используют --profile strategy --poll-seconds1200.
Все четыре модели проверять раз в20мин; пользователю только проценты.
Первоначальные два задания не изменять, секреты брать только из окружения
изолированных процессов. Это обучение смешанным навыкам, не приёмка стратегии.

## Отдельный GRPO на втором токене

По прямому запросу пользователя запущен vertel0/hastycr-grpo-account2-20261023,
версия1. Аккаунт второго токена — vertel0, индекс1 в приватном реестре
~/.kaggle/access_tokens.json. Рабочий ~/.kaggle/access_token не менять.
Аутентификация второго аккаунта передаётся KAGGLE_API_TOKEN только дочерним
локальным процессам, никогда не включать значение в команды, логи или пакеты.
Пакет/job: tmp/kaggle/grpo_account2_20261023/; исходники source_v2 закреплены,
boundary350/269282. Реальные GPU/ростsteps проверены, live_stream/progress.json
содержит свежие счётчики. Collector запущен --profile grpo-account2 с отдельным
токеном. Первый kernel НЕ остановлен; не запускать дубликаты ни на одном аккаунте.
Оценка этого GRPO требует отдельного подтверждённого результата и validation.

## Текущий запуск восстановления защиты

Исходный stage2 завершился ERROR: GRPO на378/292766 шагах, PPO на1164/1191936.
Сохранения проверены: PPO update1150/1177600, GRPO boundary350/269282.
Продолжение запущено в **andriashadrin/hastycr-defense-semantic-recovery**, версия1,
API подтвердил RUNNING. Не перезапускать и не заменять при RUNNING/QUEUED.
Это фактический slug из ответа Kaggle; желаемый dated slug не был создан.
Требуется добрать PPO2048000 иGRPO1000000, затем свежая validation selection.
Состояние: tmp/kaggle/defense_recovery_20261022/collected.status.json;
collector запущен модулем scripts.kaggle.collect_defense_recovery.
Текущий controller пишет детальные train-логи в output-файлы, не в stdout;
SSE может быть пустым, это не доказательство остановки обучения.
Последние сохранения и receipt: tmp/kaggle/defense_train_stage2_20261022/recovered_terminal_retry_1.
Источник recovery заморожен в source_v2 и отдельном приватном dataset;
исправление Princess/Firecracker escort проверено без изменения архива.
Никакие live-матчи не запускать. v25 и исходный stage2 не заменять.
Исторические сведения о RUNNING stage2 ниже устарели; этот раздел приоритетен.

## Активное большое обучение

Пакет продолжения: `tmp/kaggle/defense_recovery_20261022/source_v2`;
контроллер `scripts/kaggle/run_defense_recovery.py`. Это явная миграция
семантики среды (not_exact_resume), а не обход проверки exact resume.
В Kaggle архивы могут автоматически распаковываться: передавать
`--source-extracted-root` явно на известный каталог source/.
Самодостаточные выбранные веса пишутся в selected/ppo.pt и selected/grpo.pt.
Итоговая оценка: приватный dataset
`andriashadrin/hastycr-defense-eval-repaired-20261022` уже ready;
контроллер evaluate_stage2.py принимает --recovery-result-sha256.
Исходники recovery загружены отдельно как
`andriashadrin/hastycr-defense-recovery-source-20261022`; перед повторной
попыткой проверить текущий процесс загрузки и удалённый статус.
Полные чекпойнты stage2 нельзя получить из пустого metadata listing во
время RUNNING; terminal-only recovery watcher уже запущен.

**Уточнение:** общий kernel RUNNING, но subprocess GRPO завершился с ошибкой
на группе378/292766 шагах: invalid_script_trigger_timeout для
`pair:firecracker:hog_rider:skeletons/back/x1/right`. PPO продолжает обучение.
Не перезапускать/не заменять kernel, пока PPO работает. После завершения
нужен recovery весов из metadata outputs: parent может упасть до этапа
формирования selected_checkpoint/SHA. Подробности в
`reports/defense_plan/stage2_grpo_failure.json`.

`andriashadrin/hastycr-defense-stage2-20261022`, версия 1, RUNNING.
Цели: PPO 2048000 новых шагов, GRPO 1000000 завершённых шагов, весь
train-каталог, warmstart весов из завершённого первого этапа. Это новый
этап с расширенной средой и новым optimizer, не точный resume старых
траекторий. Лимит 21600 с каждому тренеру, kernel timeout24000 с.
Состояние: `tmp/kaggle/defense_train_stage2_20261022/monitor_status.json`;
живой прогресс: `live_stream_reconnect_4/progress.json` рядом. Первый поток
оборвался по SSL, обучение продолжалось; восстановлен только просмотр логов. Источник заморожен в
`dataset/source.tar.gz`, каждый файл проверяется по manifest.json.
**Не запускать повторно и не заменять notebook при RUNNING/QUEUED.**
При смене исходников для exact resume брать замороженный bundle этого
запуска. Рабочая v25 неизменна. Ни один живой матч не возобновлять.
После обучения предусмотрены validation и20 парных симуляционных игр
каждой модели с v25; это ещё не сравнение со всеми15 прошлыми моделями.

## Завершённый первый этап

`andriashadrin/hastycr-defense-stage1-20261022`, версия 2: COMPLETE.
PPO 200 обновлений/102400 шагов, GRPO 100 обновлений/57095 шагов.
Веса скачаны и проверены. Парная validation 24 эпизода —
`reports/defense_plan/stage1_validation_summary.json`; это не итоговая приёмка.
Первый этап повторно не запускать. Активный второй этап описан выше.

## Текущее состояние защитного RL

Источник требований — `plan.md`, выполненные этапы — `done.md`.
Каталог атак экспортируется `scripts/defense/export_catalog.py`; полный
hold-only аудит — `scripts/defense/audit_catalog.py` (это проверка среды,
не обучение). Рабочая v25 сохраняется без изменений.
Приватный защитный CUDA smoke: `andriashadrin/hastycr-defense-smoke-20261022`.
Версия 1: тесты и оба тренера завершились с кодом 0, по 5 итераций;
финальная проверка артефактов упала на отсутствующем sys.path модуля sim.
Исправление загрузчика и обновлённый пакет данных находятся в
`tmp/kaggle/defense_smoke_20261022/`. Не запускать вторую копию без проверки
удалённого статуса. Контрольная сумма v25 проверяется упаковщиком.
Устаревшие состояния более ранних заданий ниже являются историей.

## План и журнал выполнения

Пользовательский план: `plan.md`. Подтверждённый прогресс и расположение
кода/отчётов: `done.md`. После каждого выполненного этапа обновлять `done.md`:
что изменено, где лежит, чем проверено, что осталось. Не отмечать подготовку
как успешное обучение. Секреты в журнал не записывать.
Уточнение терминального условия: смерть любого короля немедленно завершает
траекторию. Обычное доигрывание без новых карт применяется, пока короли живы.

## Приватный тест CUDA на Kaggle

Пользователь разрешил тестовое обучение на 1–5 итераций. Отправлено приватное
задание `andriashadrin/hastycr-cuda-smoke-20261022`: ровно5 PPO rollout/update
итераций (2envs ×8rollout ×5 =80steps), CUDA обязательна, исходная копияv25.
Статус/результаты: `tmp/kaggle/smoke_20261022/monitor_status.json`, код задачи
`tmp/kaggle/smoke_20261022/job/smoke_train.py`, загрузки `output/` рядом.
Первый запуск подтвердил CUDA=true и2TeslaT4, GPU matmul finite=true. Затем упал до обучения: Kaggle распаковал source.tar.gz, старый загрузчик искал архив. Исправлено разрешение распакованной структуры; отправлена версия2 того же kernel. Актуальный монитор/результаты второй версии: `tmp/kaggle/smoke_20261022/v2/monitor_status.json` и `v2/output/`. На момент повторной отправки версия2 QUEUED; пять PPO итераций ещё не подтверждены.
Конечный monitor.py проверяет статус и скачивает логи/чекпойнт при завершении.
CLI: `/Users/andriashadrin/.local/share/hastycr-kaggle-cli/bin/kaggle`.
Токен хранится вне репозитория; не печатать, не копировать в dataset/notebook.
Не отправлять второе задание поверх этого без проверки статуса. Лимит600сек
выполнения; рабочийv25 не заменять. Бои на Mac остаются остановленными.

## Остановка пользователем

Пользователь остановил текущую серию. Бот и очередь завершены: 25 матчей,
12 побед,13 поражений,48% winrate. Журнал:
`tmp/live/memory_v25_20261022_series30/series.json`, status=stopped_by_user.
**Не возобновлять оставшиеся пять и не запускать новые матчи без нового
явного запроса.** Описания активной серии ниже сохранены как история.

## Активная серия v25 с памятью, 22 октября 2026

По новому запросу пользователя запущены **30 матчей**. Первый — исправленный
матч из `tmp/live/memory_v25_20261022_final3`; два прежних поражения прототипа
в эти 30 не входят. **Не запускать второй процесс или новую серию** без сверки
`tmp/live/memory_v25_20261022_series30/series.json` и процессов `memory_play` /
`memory_series`. Супервизор `scripts/gamestate/memory_series.py` переводит игру
через проверенную кнопку OK в лобби, ждёт `no_battle`, нажимает Battle и ждёт
`MATCH_START`. После первых трёх завершений он запускает runner только на
оставшиеся 27; суммарный лимит остаётся 30. При ошибке меню не повторяет
нажатия вслепую. Итоговые снимки и счётчик побед/поражений в папке серии;
неясный результат отмечается unknown, а не угадывается.

Активный путь: `scripts/gamestate/memory_play.py`, v25, native v8,
`coordinate_mode=physical_live`, без CV в наблюдении. Старый `run-macos.sh`
по-прежнему использует CV и по умолчанию v11: **не использовать для этого
запуска**. Чекпойнт v25 не менялся. Native v8 опционально читает свою руку,
цикл, баланс эликсира и игровые часы (`--own-state`); v6 API не изменён.
Подробности и ограничения в `scripts/gamestate/agent.md`.

В первых двух пробах были ошибки интеграции: остановка при отсутствии
подтверждения карты и переворот Y, отключавший защитные действия. Исправлены
таймаут (другие слоты рассматриваются по свежему состоянию) и явный режим
physical_live. Это не доказывает качество самой стратегии: полная история
карт соперника ещё отсутствует, неизвестные tactical profiles сохраняют
измеренные HP/позицию, но не вымышленные характеристики. Такие случаи
отмечаются INPUT_WARNING. Данные о старом мосте ниже — история разработки.

## Текущее изменение: переход на ARM64-парсер (22 октября 2026)

Пользователь уточнил: сохранить **v25**. Проверенная копия чекпойнта и архив
старого кода/BuildABot лежат вне проекта:
`/Users/andriashadrin/projects/Hasty-CR-preserved-v25-20261022-131231/`.
SHA-256 сохранён в `manifest.json`, копия чекпойнта сверена с оригиналом.
Очистка CV и переключение активного бота ещё не выполнены: описание старого
live-контура ниже пока соответствует коду, а не целевому устройству системы.

Добавлен математический счётчик своего эликсира
`scripts/gamestate/elixir.py` и `CurrentStateReader.query_with_elixir()`.
Поддерживаются подтверждённый доход, пересчёт запоздавших событий,
резервирование команд до подтверждения, отмена при известном отказе,
прогноз баланса и ожидания стоимости. В ответе API есть отдельный доступный
бюджет с учётом резервов; профильные тесты: 39 passed.
Инструкция и ограничения — в `scripts/gamestate/agent.md`: счётчику нужны
игровые часы и подтверждённые расходы, их native API пока не возвращает.
Не подставлять host-time или первые появления юнитов вместо этих данных.
Новые бои для проверки не запускались.

### Согласование парсера и симулятора

Отчёт: `reports/parser_sim_alignment_20261022/report.md`.
`scripts/gamestate/model_bridge.py` даёт вход v25 через общий с обучением
`sim.env.observe`/`legal_mask`, с проверкой сторон, реальных HP и собственных
входных данных; `MemoryPolicy` выполняет inference без нажатий. Старый бот
и launcher ещё не переключены на память. Запасной public-loader исправлен:
он больше не переводит Speed в секунды второй раз; `UnitSpec` хранит raw
Speed, `make_unit` переводит его в mt/s. Скоростной признак старого RL-моста
тоже согласован с Entity.

Обнаружены несогласованные координаты башен и старое отражение Y наблюдения
относительно действий. Для v25 этот контракт не менять молча. Снарядный
аудит старого повтора дал 32 предварительно сравнимых попадания: 25 совпадений
при выборе цели из снимка; гипотеза учёта радиуса атакующего даёт 32/32.
Это косвенные метки, не доказанная общая точность. Основная дальность атаки
и таргетинг не изменены. Клиентского дампа по-прежнему нет, readiness=false;
широкий контрольный прогон содержит 120 старых падений после исправления
скорости против 126 до, новых падений в выбранных пяти файлах не выявлено.
Точные результаты и границы выводов сохранять вместе с отчётом.

Этот файл — короткий маршрут для человека или агента, продолжающего разработку на другом устройстве. Основное описание и установка — в [README.md](README.md). Репозиторий пользователя: `origin` (`andria909/Hasty-CR`); исходный проект: `upstream` (`hastylmao/Hasty-CR`). Пушить только в `origin`, не в `upstream`.

## Что где лежит

| Путь | Назначение |
| --- | --- |
| `sim/` | Детерминированный симулятор, физика, окружение PPO, тренировка и парное сравнение чекпойнтов. |
| `sim/env.py` | Наблюдения, действия, награды и задержка действий в симуляции. |
| `sim/public_state.py` | Публичные признаки оппонента: карты, эликсир, скорость/позиции и угрозы. |
| `sim/meta_decks.py` | Пул колод соперника для `--opponent meta`; включает приближённые live-кейсы. |
| `scripts/brain/` | Политика, RL-мост, ограничения безопасности, оценка заклинаний и трекинг боя. |
| `scripts/cr_bot.py` | Живой цикл: снимок экрана → распознавание → решение → ADB-нажатие → журнал. |
| `run-macos.sh` | Запуск на macOS: наблюдение, ручной бой или автоматическая очередь. |
| `data/live_cases/` | Проверенные описания проигранных матчей и *синтетические* колоды для обучения. |
| `tests/` | Регрессии симуляции и живого RL-моста. |
| `reports/` | Результаты замеров. `tmp/` — локальные журналы/обучение, Git его игнорирует. |

## Текущий игровой контекст

Наша фактическая колода — Hog Rider, Cannon, Musketeer, Fireball, Knight, Electro Spirit, Skeletons, Zap (не классические Ice Golem / Ice Spirit / Log). Цель — надёжная оборона в центре с пушкой и мушкетёром, дешёвый кайт по типу цели, затем темповая атака Hog; не тратить весь эликсир без угрозы. Заклинания должны попадать по прогнозируемому положению цели и давать выгодный размен или добивать башню.

Последний зафиксированный матч перед остановкой бота — `20261013_225020_m009`, поражение 1:3 на `portable_v25_cannon_economy_best.pt`. Его журнал сохранён в `data/live_cases/20261013_225020_m009_source.json`, описание — в `data/live_cases/20261013_225020_m009.json`. В журнале есть действия и редкие снимки HP башен, но **нет** полного наблюдения перед каждым решением. Поэтому это не готовая экспертная PPO-траектория: синтетическая колода лишь добавляет похожее давление в пул `meta`. Восемь карт взяты из наблюдавшихся угроз, но точный состав колоды соперника не подтверждён.

Версии v26–v29 и JSON-отчёты — эксперименты, а не автоматически улучшенный live-релиз. Для v29 есть отрицательный разбор в `reports/portable_v29_livecase_probe.md`: на независимом блоке результат хуже v25. Не заменяйте рабочий чекпойнт только по результату симуляции: сравните на одинаковых seed с `brain,meta`, затем наблюдайте короткие live-тесты. Последняя остановка была по запросу пользователя; самовольно не возобновляйте бои.

## Быстрый старт на другом Mac

```bash
python3.12 -m venv .venv-live
.venv-live/bin/pip install -e '.[live,dev]'
.venv-live/bin/python -m pytest tests -q
```

Для live нужны установленный Clash Royale в Android-эмуляторе, ADB-подключение, локальные `vendor/`-зависимости и чекпойнт. `vendor/`, `tmp/`, локальные конфиги `scripts/brain/*.json`, VPN и учётные данные в Git не входят; сверяйтесь с разделом macOS в README и `vendor/manifest.json`. Проверка устройства: `adb devices`; при нескольких устройствах передать `--serial`.

```bash
./run-macos.sh --serial DEVICE_ID --checkpoint checkpoints/portable_v25/portable_v25_cannon_economy_best.pt --shadow
```

`--shadow` только наблюдает (по умолчанию), `--tap-friendly` нажимает карты, но не кнопку Battle, `--ladder --matches N` сам запускает рейтинговые игры; это риск блокировки аккаунта. Не запускать `--ladder` без явного запроса пользователя. Остановить живой цикл можно Ctrl-C; проверить отсутствие процесса перед новым запуском.

## Обучение и проверка

```bash
.venv-live/bin/python -m sim.train_ppo --help
.venv-live/bin/python -m sim.train_ppo --opponent meta --brain-share 0.5 --steps 200000 --envs 6 --init checkpoints/portable_v25/portable_v25_cannon_economy_best.pt --name experiment
.venv-live/bin/python -m sim.eval_checkpoints --baseline checkpoints/portable_v25/portable_v25_cannon_economy_best.pt --candidate tmp/rl/experiment_best.pt --opponents brain,meta --episodes 80 --output reports/experiment_eval.json
```

Тренер пишет веса в `tmp/rl/`; проверьте созданный файл перед оценкой. Не считайте одиночную победу, reward или win-rate по одному сопернику доказательством прогресса. Проверяйте задержку от наблюдения до нажатия и отдельные случаи: ранний Hog врага, тяжёлый пуш, защитный Fireball, здание для Hog, удержание эликсира, добивание башни.

Перед коммитом: `git status --short`, `git diff --check`, тесты, проверка на токены/ключи. `*.pt`, `checkpoints/` и `tmp/` игнорируются; для передачи выбранного чекпойнта на другое устройство добавляйте только конкретный проверенный файл (`git add -f <точный путь>`), а не весь `tmp/`. Не включайте VPN-ключи, данные аккаунта и cookies.

## Правила этой разработки

На длинных задачах сохранять не менее 50% каждого активного окна квоты Codex: проверять через `$usage` до начала и после этапов, при остатке ≤50% остановиться и спросить пользователя. Ничего не удалять и не чистить без отдельного разрешения на конкретные файлы. Изменения лучше делать обратимыми. Новую модель продвигать в live только после тестов и явного сравнения с базой.

## Проверка live CV, 21 октября 2026

Текущие эталоны восьми карт находятся в `data/card_references/current_20261021`; карты руки и превью читает `scripts/brain/cards.py`. Исправления поля подключены через `scripts/live_perception.py`: фильтр границ текущей арены, исключение почти идентичных гипотез, ограниченное уточнение слабых зданий теми же ONNX-весами и проверка стороны здания по полоске HP. Профильные проверки: 110 passed, 5 skipped; новые бои и обучение не запускались.

Результаты поля и ограничения: `reports/board_cv_fix_20261021_final/report.txt`. Переносимые реальные кадры и предварительная разметка: `data/validation/board_regression_20261021/`. Это диагностические связанные кадры, не независимая оценка всего детектора. Оставшиеся ошибки сохранены в `unresolved.json`: свой Hog как вражеский Knight, эффект смерти как Mini PEKKA, полный пропуск войск под эффектами. Не заявлять идеальное распознавание поля. Повторный офлайн-прогон: `.venv-live/bin/python scripts/audit_board_cv.py --output reports/NEW_BOARD_AUDIT`.

KataCR проверен отдельно офлайн: `scripts/brain/katacr.py`, `scripts/eval_katacr.py`, отчёт `reports/katacr_eval_20261021/report.txt`. Оба опубликованных v0.7.13 скачаны в `tmp/katacr_eval_20261021`; источники/хэши — `data/detectors/katacr_v0713.json`. На MPS: 148 кадров, среднее 122 мс. На 4 размеченных кадрах — 9/9 объектов, но в общем прогоне 59 противоречий своей колоде, включая ложные классы на надписях и ошибки стороны. В live KataCR не подключён. При проверке исправлена разметка кадра 3: враг mega_minion, а не knight; `labels_v2.json` и пересчитанные метрики в отчёте KataCR заменяют прежние проценты на labels v1. Оригиналы сохранены. Проверки адаптера/метрики/поля/покрытия: 36 passed, 1 skipped.


Офлайн-проверка исходной пары KataCR v0.7.13 на авторском датасете (22 октября 2026): `scripts/eval_katacr_original.py`, полный список `images/part2/yolo_annotation.txt` — 6939 кадров, 117026 объектов. Отчёт `reports/katacr_original_20261022_full/report.txt`; юниты: precision 92.82%, recall 75.89%, mAP50 87.62%, mAP50-95 67.95%; сторона верна у 97.86% правильно найденных юнитов. Это родная проверка автора, возможны общие исходники с обучением; не независимая оценка новой игры. Отдельный последовательный замер 64 кадров при работающем MuMu: среднее 254 мс, p95 349 мс. Нового обучения, боёв и продвижения в live не было.

## ARM64-парсер памяти и живой замер, 22 октября 2026

Инструкция: [scripts/gamestate/agent.md](scripts/gamestate/agent.md). Скрипт `scripts/gamestate/query_arm64.py` возвращает текущие ID/имя/сторону/HP/координаты войск, зданий и башен без скриншота. Native-reader `scripts/gamestate/native/arm64_entities.c` читает память только через `pread`; Python проверяет root, ARM64, versionCode160402020 и закреплённый SHA-256. Мир боя находится автоматически при каждом запросе. Не отключать проверки сборки и не смешивать с прежним x86-reader.

Один запрос из корня проекта после сборки/установки native-reader по инструкции:

```bash
.venv-live/bin/python -m scripts.gamestate.query_arm64 --serial 127.0.0.1:26625
```

Живой матч с разрешения пользователя: v25, ровно1 бой, 35 розыгрышей, поражение0:1; бот штатно завершился по `max_matches=1`. Парсер работал параллельно, бот всё ещё использовал BuildABot/CV. Из600 запросов в начале матча для итоговой метрики взяты543 после подтверждённого MATCH_START: median21,71мс, p95=44,10мс, max57,31мс, 543/543≤100мс, отклонений0;6–18 объектов включая башни/снаряды. Это полный запрос с Mac до разобранного состояния, без первоначальной проверки версии/хэша. Отчёт: [reports/arm64_live_query_20261022.json](reports/arm64_live_query_20261022.json); локальные исходные JSONL и лог матча указаны внутри.

Замер ограничен первой минутой одного живого матча. Это текущие позиции тел, не доказанный полный поток постановок карт: первые появления, групповые карты, дочерние юниты и мгновенные заклинания требуют отдельного учёта. Side0/1 не означает фиксированно «мы/враг». Скорость парсера не является скоростью нынешнего CV-бота. Профильные тесты:10 passed (`test_arm64_entities.py`, `test_query_arm64.py`). Не возобновлять бои автоматически после завершённого теста.
