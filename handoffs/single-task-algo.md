# Handoff: Single-Task Algo

_Last updated: 2026-06-20 · Branch: hpc_version @ 2382c82_

## Goal
Continue helping the user develop a custom single-task learning algorithm for LIBERO, using `LIBERO_ALPHABET_SOUP` and the alphabet soup HDF5 dataset as the first smoke-test target. The user wants guidance and review while writing the code themselves; do not implement the algo for them unless they explicitly change that instruction.

## Current Progress
Read and summarized `docs/note_cn.md`, `docs/learning.md`, and `docs/new_benchmark.md`. Added new documentation to `docs/note_cn.md`: chapter 8 records the `SingleTask` debug session, and chapter 9 now records a single-task-only plan for writing a custom algo. Chapter 9 was revised after the user clarified that it should not be framed as multitask or lifelong learning, even though the project directory and Hydra config group are named `libero/lifelong/...` and `lifelong=...`.

The session verified the `SingleTask` call chain: `main.py` reads `lifelong=single_task`, `configs/lifelong/single_task.yaml` sets `algo: SingleTask`, `get_algo_class("SingleTask")` instantiates it, and `algo.learn_one_task(...)` runs the inherited `Sequential` training loop. `SingleTask` itself only saves the initial policy and resets to it in `start_task()`.

Repository state observed during handoff creation:

```text
 M libero/libero/benchmark/__init__.py
 M libero/lifelong/algos/base.py
 M libero/lifelong/evaluate.py
?? docs/data_info.md
?? docs/diffusion_policy_vision_pusht_demo.ipynb
?? docs/learning.md
?? docs/new_benchmark.md
?? docs/note_cn.md
```

Only `docs/note_cn.md` and this handoff were edited in this session by the assistant. Other modified/untracked files existed in the worktree and should not be reverted without user approval.

## What Worked
`LIBERO_ALPHABET_SOUP` correctly resolves to the alphabet soup task and loads:

```text
/scratch/prj/eng_demt_robot_learning/dataset/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
```

The user ran `SingleTask` on that benchmark. The run successfully reached and completed:

```text
epoch 0 loss
epoch 0 rollout
epoch 1 training
epoch 1 rollout
```

Observed output included:

```text
[info] Epoch:   0 | train loss:  5.49 | time: 1.59
[info] evaluate task 0 takes 55.9 seconds
[info] Epoch:   0 | succ: 0.00 ± 0.00 | best succ: 0.0 | succ. AoC 0.00
[info] Epoch:   1 | train loss: -9.13 | time: 3.16
[info] evaluate task 0 takes 50.8 seconds
[info] Epoch:   1 | succ: 0.00 ± 0.00 | best succ: 0.0 | succ. AoC 0.00
```

`succ=0.00` is expected for a tiny smoke test and should not be treated as an algorithm result.

## What Didn't Work
`train.num_workers=0` initially failed because `Sequential.learn_one_task()` used `persistent_workers=True` unconditionally:

```text
ValueError: persistent_workers option needs num_workers > 0
```

The stable pattern is:

```python
persistent_workers=self.cfg.train.num_workers > 0
```

Using `train.num_workers=1` failed because the Robomimic/HDF5 dataset contains h5py objects that cannot be pickled by PyTorch multiprocessing workers:

```text
TypeError: h5py objects cannot be pickled
```

After training and epoch rollout completed, final global evaluation failed for the same h5py reason because default `eval.num_workers=4` made `evaluate_loss()` use multiprocessing. Use `eval.num_workers=0` for full evaluation, or `eval.eval=false` to skip the final `main.py` evaluation while debugging.

The repeated robosuite, Gym, and thop messages during rollout are warnings/import noise from evaluation subprocesses, not the primary failure.

## Key Files & Commands
Important files:

```text
docs/note_cn.md
docs/learning.md
docs/new_benchmark.md
libero/lifelong/algos/base.py
libero/lifelong/algos/single_task.py
libero/lifelong/algos/__init__.py
libero/configs/lifelong/single_task.yaml
libero/lifelong/main.py
libero/lifelong/metric.py
```

Recommended no-rollout single-task smoke test:

```shell
cd /scratch/prj/eng_demt_robot_learning/libero_demt

python libero/lifelong/main.py \
  benchmark_name=LIBERO_ALPHABET_SOUP \
  policy=bc_transformer_policy \
  lifelong=single_task \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets \
  train.n_epochs=1 \
  train.batch_size=32 \
  train.num_workers=0 \
  eval.eval=false \
  eval.eval_every=999
```

Recommended single-task training plus small rollout, skipping final global eval:

```shell
python libero/lifelong/main.py \
  benchmark_name=LIBERO_ALPHABET_SOUP \
  policy=bc_transformer_policy \
  lifelong=single_task \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets \
  train.n_epochs=1 \
  train.batch_size=32 \
  train.num_workers=0 \
  eval.eval=false \
  eval.eval_every=1 \
  eval.n_eval=2 \
  eval.use_mp=false
```

Recommended full small evaluation, avoiding h5py multiprocessing:

```shell
python libero/lifelong/main.py \
  benchmark_name=LIBERO_ALPHABET_SOUP \
  policy=bc_transformer_policy \
  lifelong=single_task \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets \
  train.n_epochs=1 \
  train.batch_size=32 \
  train.num_workers=0 \
  eval.eval=true \
  eval.eval_every=1 \
  eval.n_eval=2 \
  eval.num_workers=0 \
  eval.use_mp=false
```

For a future custom algo, chapter 9 currently uses this placeholder naming:

```text
libero/lifelong/algos/my_single_task_algo.py
libero/configs/lifelong/my_single_task_algo.yaml
algo: MySingleTaskAlgo
lifelong=my_single_task_algo
```

## Next Steps
1. Ask the user what concrete single-task algorithm idea they want to implement first, or whether they want to start with a behavior-equivalent `MySingleTaskAlgo` skeleton.
2. Have the user create the minimal algo file, config file, and import. Provide guidance only unless asked to edit.
3. Check that `Available algorithms` includes `mysingletaskalgo`.
4. Run the no-rollout smoke test with `LIBERO_ALPHABET_SOUP`.
5. Add one algorithm feature at a time, then rerun the same smoke test.
6. Only after basic training works, run the small rollout command and later the full small evaluation command.

## Open Questions
What custom single-task training behavior should the new algo implement first: extra loss, sample reweighting, optimizer-step changes, freezing/unfreezing modules, buffer use, or something else?

## Changelog
- 2026-06-20: Created handoff for the single-task algo planning and `SingleTask`/`LIBERO_ALPHABET_SOUP` debug session.
