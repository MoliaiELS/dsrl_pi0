# SFT-A execution supplement (2026-10-28)

This is the single conditional next batch selected after completed first_batch
18584. It does not start SFT-S, PPO, stage estimation or recovery experiments.
Scientific configuration is `configs/sft_a_v1.json`; source demonstration hashes
and exact revisions are in `configs/sft_a_sources.json`.

## Evidence and scope

DSRL first-grasp A is 80/80 for all three controllers, while B is 0/80.
Base B first-grasp is also weak (4/80). B ever-success often follows wrong-goal
A completion. This supports testing balanced genuine A/B demonstration training
to separate a weak goal interface/B skill from later stage-selection questions.
It does not establish forgetting in a frozen VLA.

## Demonstration timing and image conventions

The pinned LIBERO `scripts/create_dataset.py` appends observations **after**
`env.step(actions[j])`, while retained `states[j]` is before that action.
Across all 100 files' first frames, `states[1][1:10]` exactly equals the joint and
gripper observation at index 0. The causal target for obs[j] is therefore
actions[j+1:j+51]. This offset was fixed before any parameter update. It is an
explicit dataset adaptation, not an unreported reuse of same-index supervision.
No padding or episode crossing is accepted; there are T-50 complete windows.
Terminal recorded states satisfy the correct original official goal in all 100
demonstrations. Data stage 18643 passed with 0 simulator steps and 0 model calls.

Both HDF5 cameras require a 180-degree rotation to match the existing runtime
camera convention. The adapter uses the author's actual
`openpi_client.image_tools.resize_with_pad`, preserves extra DeltaActions on the
first six dimensions, pads to 32 and uses the original mean/std and epsilon.
Actual transformed images are checked pixel-for-pixel; states/actions are checked
against the original OpenPI transform. Camera re-render orientation comparisons
are approximate historical render checks, not a claim of bitwise sensor replay.
Original first_batch replay tolerances remain unchanged.

## Implementation and gates

The pinned RLinf native Pi0 model, selective mixed precision, PEFT all-linear
injection and native flow-matching loss are reused. This project's single-GPU
sampler/checkpoint/validation wrapper is **not** the default distributed RLinf SFT
pipeline (its VLA validation path is unimplemented at this revision).

1. Data: whole trajectories 40 train/10 validation per true goal, split 99023.
2. Reference: 10 saved development observations x 6 A/B/stage texts x 2 explicit
   z arrays, with repeated JAX and Torch outputs. No reserved observations.
3. Alignment: per-dimension/per-prompt normalized RMSE and max error; .01/.1
   alarms. Then all 20 original Base development cases are repeated in native
   Torch, with exact initial-state/image checks. A predeclared engineering gate
   requires >=90% first-grasp and goal-vector agreement separately; this is not a
   statistical equivalence test. Differences below static alarms still require
   review. No SFT update opens on a failed alignment gate.
4. Profile: two real accumulated updates, micro1/global8, base hashes before and
   after, LoRA coverage/deltas, finite gradients, checkpoint and a fresh process
   restore at identical observation/z. These two updates count towards 500/3000.
5. Training: seed17, pilot500, max3000 including pilot/profile, saves every500.
   Fixed 8 real validation windows per held-out trajectory with fixed noise/time
   seed99024; equal-window within trajectory, equal-trajectory within goal, equal
   goals. Mean loss across all 32 modeled dimensions is the native objective;
   real7 is a diagnostic. Continue beyond500 only if held-out mean loss and B
   real7 loss are below initial; otherwise stop and review. Select checkpoint by
   held-out mean loss, never behavior. LoRA rank8/alpha8/drop0, LR1e-4,
   betas(.9,.95), eps1e-8, decay1e-10, clip1, warmup100.
6. Behavior: selected adapter and native unadapted model, same new resets
   3300-3319 x A/B, 80 episodes/32000 policy steps, physical A scene, 400 steps,
   chunk20/horizon50/10 denoise, no early done. Save raw action/noise/prompt/event
   traces and preselected videos. Report first grasp, official ever-goal and
   selective first completion separately. No old confirmation retuning.

The pinned template's `warmup_style=cosine` is not read by its FSDP scheduler
builder; absent `lr_scheduler`, that builder uses constant-with-warmup. This
batch explicitly chooses `lr_scheduler=cosine`, with 100-update warmup, one half
cycle and minimum0, to resolve the documented cosine intention. The wrapper
calls the same public Transformers scheduler as RLinf's cosine branch. This is
an explicit schedule resolution, not a claim to keep an effective constant
schedule unchanged.

## Accounting and runtime

### Migration review and attribution scope (2026-10-28)

Run18655 completed20 original Base development episodes (8000 physics steps,
zero technical retries/errors) with the original XML/physics/proprio/RGB reset
checks and the actual saved z arrays. First-grasp/goal-vector agreement is
60%/70%, below the declared90% gate. Static alarms remain23/120 after source-backed
compatibility corrections (meanRMSE .011173, maxRMSE .088213, maxabsolute2.0438).
The original JAX model itself differs across4090/3090 with33/120 alarms,
maxRMSE .087781 and maxabsolute2.0403 on identical arrays. This documents numeric
sensitivity; it does **not** establish cross-framework equivalence.

The original strict migration gate stays **failed**. Under section11.1's explicit
fallback, this batch proceeds only as a new **within-native-runtime** SFT-A
experiment, with paired native-unadapted/native-adapted behavior checks. It cannot
be attributed as a one-variable change relative to the old JAX DSRL results.
Native B correct-first is1/10, wrong-A-first4/10, B-ever2/10; A correct-first and
ever are2/10. Genuine A/B data first-grasp and official terminal success are each
50/50. These support testing the same targeted adaptation in the native runtime.

`configs/sft_a_migration_review.json` locks the actual evidence hashes and limited
decision. `tools/sft_migration_review.py` verifies complete trajectories/artifacts,
exact reset evidence, isolated80/20 data and the genuine demo audit before creating
`SFT_BASELINE_READY.json` with `cross_framework_numerical_equivalence=false`.
It never creates `SFT_ALIGNMENT_PASS.json` or changes failed source metrics.
The scope file follows profile/checkpoint/train/evaluation. Original hyperparameters,
data split, new behavior seeds, pilot gate and shared3000-update cap remain fixed.
No SFT-S or PPO starts from this review. Trajectory validation is held out from
adapter training; overlap with the base model's historical pretraining is unverified,
so no never-seen-task/data or OOD claim is justified.

The new environment is isolated at
`/mnt/slurmfs-4090node1/homes/yguo704/miniforge3/envs/dsrl_sft_native_20261028`.
The first NFS-target installation was stopped during slow cross-filesystem file
copy, before any model training; its partial prefix and logs are retained. The
new prefix reuses the same downloaded packages on the home filesystem. Arrow20
is pinned for the resolved datasets API: Arrow21 removed PyExtensionType
([official release](https://arrow.apache.org/release/21.0.0.html)). Original
dsrl_pi0 is unchanged. Each GPU stage requires pip-check/import completion and
the exact final requirements hash, and saves its native and simulation versions.

Separate from the closed first_batch ledger: offline demonstrations100,
training updates<=3000, migration development rollout8000 physical steps,
post-SFT paired checks32000 steps, static model calls and failed engineering
attempts recorded separately. Slurm job wall time and resource traces stay in
each `runs/job-ID`. A failed installation or data audit is an engineering failure,
not evidence against SFT. No full RL training is authorized by this supplement.

Use `tools/remote.ps1 sft-data/sft-ref/sft-align/sft-smoke/sft-train/sft-eval`.
All prerequisite flags and config hashes must match. Long stages run on compute
nodes; node1 data/reference/profile debugging is limited to20 minutes.
