# Handoff: LIBERO Diffusion Policy

_Last updated: 2026-06-21 · Branch: hpc_version @ 2382c82_

## Goal
Add a Diffusion Policy implementation adapted from `docs/diffusion_policy_vision_pusht_demo.ipynb` so `libero/lifelong/main.py` can train directly on LIBERO HDF5 data, especially `/scratch/prj/eng_demt_robot_learning/dataset/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5`.

## Current Progress
Implemented and registered a new `DiffusionPolicy`:

- Added `libero/lifelong/models/diffusion_policy.py`.
- Added `libero/configs/policy/diffusion_policy.yaml`.
- Registered the policy in `libero/lifelong/models/__init__.py`.
- Added compatible dependencies to `requirements.txt`: `diffusers==0.11.1`, `huggingface-hub==0.13.4`, `safetensors==0.4.5`.
- Added `LIBERO_ALPHABET_SOUP` in `libero/libero/benchmark/__init__.py` for the single alphabet-soup object task.
- Updated `libero/lifelong/algos/base.py` so policies with `update_ema()` get an EMA hook after optimizer step.
- Updated `libero/lifelong/algos/base.py` so `eval.eval=false` actually skips rollout success evaluation and still saves a checkpoint.
- Updated `libero/lifelong/evaluate.py` so the standalone evaluator can use `libero_alphabet_soup`, compute task embeddings for the benchmark's actual task count, and reject out-of-range `task_id` values.
- Read `/scratch/users/k23114984/code/arcap_policy/STEP2_train_policy` and compared its robomimic 3D Diffusion Policy implementation against this 2D LIBERO DP. Useful ideas were migrated without adding pointcloud inputs:
  - Training and inference schedulers are now separate.
  - Training still uses `DDPMScheduler` for `add_noise()`.
  - Inference can use `DDIMScheduler`; config default is `inference_scheduler=ddim` and `num_inference_iters=10`.
  - Scheduler parameters are configurable: `num_train_timesteps`, `beta_schedule`, `prediction_type`, `ddim_set_alpha_to_one`, `ddim_steps_offset`.
  - The first training batch checks that actions are normalized to `[-1, 1]`; set `policy.enforce_action_bounds=false` only for debugging.

The policy uses:

- Two ResNet18+GroupNorm visual encoders for `agentview_rgb` and `eye_in_hand_rgb`.
- Low-dimensional proprio from `gripper_states` and `joint_states`.
- Conditional 1D U-Net over action sequences.
- `DDPMScheduler` for training noise and configurable DDPM/DDIM inference sampling.
- `data.seq_len=16`, `obs_horizon=2`, `pred_horizon=16`, `action_horizon=8`.

Validation completed on an interactive GPU node:

```bash
python libero/lifelong/main.py \
  benchmark_name=LIBERO_ALPHABET_SOUP \
  policy=diffusion_policy \
  lifelong=base \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets \
  data.seq_len=16 \
  train.n_epochs=1 \
  train.batch_size=16 \
  train.num_workers=0 \
  eval.eval=false
```

This completed successfully before the arcap-inspired scheduler split:

```text
[info] policy has 38.7 GFLOPs and 115.4 MParams
[info] Epoch:   0 | train loss:  1.07 | time: 1.49
[info] Epoch:   1 | train loss:  0.14 | time: 2.56
[info] finished learning
```

After migrating DDIM inference and action-bound checking, the user reran the same no-rollout command on GPU. It completed successfully:

```text
[info] policy has 38.7 GFLOPs and 115.4 MParams
[info] Epoch:   0 | train loss:  1.06 | time: 1.48
[info] Epoch:   1 | train loss:  0.17 | time: 2.56
[info] finished learning
```

This confirms the action-bound check passes for the LIBERO alphabet-soup HDF5 and the new config loads correctly.

Rollout smoke test also reached evaluation successfully:

```bash
python libero/lifelong/main.py \
  benchmark_name=LIBERO_ALPHABET_SOUP \
  policy=diffusion_policy \
  lifelong=base \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets \
  data.seq_len=16 \
  train.n_epochs=1 \
  train.batch_size=16 \
  train.num_workers=0 \
  eval.eval=true \
  eval.n_eval=5 \
  eval.eval_every=1 \
  eval.use_mp=false \
  policy.num_diffusion_iters=10
```

It completed epoch 0 rollout without shape/device/action-format errors:

```text
[info] evaluate task 0 takes 142.8 seconds
[info] Epoch:   0 | succ: 0.00 ± 0.00 | best succ: 0.0 | succ. AoC 0.00
[info] Epoch:   1 | train loss:  0.14
```

Success is expected to be 0 for an effectively untrained 1-epoch smoke test.

## What Worked
The existing LIBERO HDF5 `SequenceDataset` already provides `(B,T,...)` batches compatible with diffusion sequence training when `data.seq_len=16`.

`diffusers==0.11.1` is compatible with the current environment:

```text
torch 1.11.0+cu113
torchvision 0.12.0+cu113
transformers 4.21.1
diffusers 0.11.1
huggingface_hub 0.13.4
```

`policy.num_diffusion_iters=10` is useful for fast smoke tests. The config default remains `100`.

The arcap 3D DP reference in `/scratch/users/k23114984/code/arcap_policy/STEP2_train_policy/robomimic/algo/diffusion_policy.py` uses the same `diffusers==0.11.1` scheduler APIs. Its strongest transferable idea for this repo was DDPM training with faster DDIM inference; pointcloud and Dex/action weighting branches were intentionally not ported.

`policy.inference_scheduler=ddim policy.num_inference_iters=10` is now the default fast rollout path. Use `policy.inference_scheduler=ddpm policy.num_inference_iters=100` if a final comparison with the older DDPM sampling behavior is needed.

`eval.use_mp=false` is useful for rollout debugging because it avoids noisy multi-process env startup.

## What Didn't Work
`diffusers==0.18.2` was tried first because the official notebook uses it, but it failed with current `transformers==4.21.1`:

```text
ImportError: cannot import name 'SAFE_WEIGHTS_NAME' from transformers.utils
```

Do not upgrade `transformers` casually because existing LIBERO task embedding code depends on the current stack. Use `diffusers==0.11.1` for now.

`train.num_workers=4` failed with:

```text
TypeError: h5py objects cannot be pickled
```

The robomimic `SequenceDataset` holds h5py objects, so use `train.num_workers=0` unless the dataset is refactored so each worker opens its own HDF5 file.

The first U-Net up path implementation had a channel mismatch in `compute_flops()`:

```text
expected input[1, 1536, 8] to have 2048 channels
```

Fixed by constructing the up residual block with `dim_out + dim_in`.

The base training loop originally ignored `eval.eval=false` and always ran rollout evaluation at epoch 0. Fixed in `base.py`.

## Key Files & Commands
Important files:

- `libero/lifelong/models/diffusion_policy.py`
- `libero/configs/policy/diffusion_policy.yaml`
- `libero/lifelong/models/__init__.py`
- `libero/lifelong/algos/base.py`
- `libero/lifelong/evaluate.py`
- `libero/libero/benchmark/__init__.py`
- `requirements.txt`
- `docs/note_cn.md`
- `docs/diffusion_policy_vision_pusht_demo.ipynb`

Current working-tree status to preserve:

- Modified tracked files: `libero/libero/benchmark/__init__.py`, `libero/lifelong/algos/base.py`, `libero/lifelong/evaluate.py`, `libero/lifelong/models/__init__.py`, `requirements.txt`.
- Untracked DP files: `libero/configs/policy/diffusion_policy.yaml`, `libero/lifelong/models/diffusion_policy.py`.
- Untracked handoff directory: `handoffs/`.
- several docs under `docs/`
- existing `handoffs/single-task-algo.md`

Fast no-rollout smoke test:

```bash
python libero/lifelong/main.py \
  benchmark_name=LIBERO_ALPHABET_SOUP \
  policy=diffusion_policy \
  lifelong=base \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets \
  data.seq_len=16 \
  train.n_epochs=1 \
  train.batch_size=16 \
  train.num_workers=0 \
  eval.eval=false
```

Rollout smoke test:

```bash
python libero/lifelong/main.py \
  benchmark_name=LIBERO_ALPHABET_SOUP \
  policy=diffusion_policy \
  lifelong=base \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets \
  data.seq_len=16 \
  train.n_epochs=1 \
  train.batch_size=16 \
  train.num_workers=0 \
  eval.eval=true \
  eval.n_eval=5 \
  eval.eval_every=1 \
  eval.use_mp=false
```

Potential formal training command:

```bash
python libero/lifelong/main.py \
  benchmark_name=LIBERO_ALPHABET_SOUP \
  policy=diffusion_policy \
  lifelong=base \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets \
  data.seq_len=16 \
  train.n_epochs=50 \
  train.batch_size=16 \
  train.num_workers=0 \
  eval.eval=true \
  eval.n_eval=20 \
  eval.eval_every=5
```

## Next Steps
1. Run the rollout smoke test with the new default DDIM 10-step inference and compare `evaluate task 0 takes ... seconds` against the earlier ~142.8s rollout.
2. If rollout action format remains correct, run a longer training job and monitor rollout success.
3. Consider adding batch-level progress logging or a tqdm in `learn_one_task()` because DP epochs are slow and currently quiet.
4. Optional: compare `policy.inference_scheduler=ddpm policy.num_inference_iters=100` against default DDIM 10-step for final evaluation quality/speed.
5. Optional: refactor dataset loading if multi-worker training is needed; otherwise keep `train.num_workers=0`.

## Open Questions
Should formal evaluation use DDIM with 10 or 20 inference steps, or keep DDPM 100-step sampling for final numbers? Default is now DDIM 10-step for speed.

Should the default DP config remain large (`vision_feature_dim=512`, `down_dims=[256,512,1024]`) or include a smaller debug config for faster iteration?

## Changelog
- 2026-06-20: Created handoff after DP training and rollout smoke tests reached the main training/evaluation paths.
- 2026-06-20: Updated repo-state notes to include the current alphabet-soup benchmark/evaluator changes and untracked DP files.
- 2026-06-21: Captured arcap 3D DP comparison, migrated DDIM inference/action-bound checking into the 2D DP, and recorded the successful post-change no-rollout GPU smoke test.
