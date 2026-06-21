# Customizing Learning Models

LIBERO training is driven by Hydra configs and a small policy registry. The usual path is to change config values first, then create a new policy class only when the existing `BCRNNPolicy`, `BCTransformerPolicy`, or `BCViLTPolicy` cannot express the model you need.

## Training Entry Point

Run lifelong behavior cloning from `libero/lifelong/main.py`:

```shell
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

python libero/lifelong/main.py \
  seed=10000 \
  benchmark_name=LIBERO_SPATIAL \
  policy=bc_transformer_policy \
  lifelong=base
```

Main choices:

| Override | Options |
| --- | --- |
| `benchmark_name` | `LIBERO_SPATIAL`, `LIBERO_OBJECT`, `LIBERO_GOAL`, `LIBERO_90`, `LIBERO_10` |
| `policy` | `bc_rnn_policy`, `bc_transformer_policy`, `bc_vilt_policy` |
| `lifelong` | `base`, `er`, `ewc`, `packnet`, `multitask`, `single_task` |

The base config is `libero/configs/config.yaml`. It includes default config groups from `libero/configs/data`, `libero/configs/policy`, `libero/configs/train`, `libero/configs/eval`, and `libero/configs/lifelong`.

## Fast Customization with Hydra Overrides

Most experiments only need command-line overrides:

```shell
python libero/lifelong/main.py \
  benchmark_name=LIBERO_OBJECT \
  policy=bc_rnn_policy \
  lifelong=base \
  train.n_epochs=100 \
  train.batch_size=16 \
  data.seq_len=20 \
  eval.eval=false \
  folder=/absolute/path/to/datasets
```

Common knobs:

| Config key | Meaning |
| --- | --- |
| `folder` | Root dataset folder. Defaults to `get_libero_path("datasets")`. |
| `device` | Training device, usually `cuda` or `cpu`. |
| `data.seq_len` | Temporal sequence length sampled from each trajectory. |
| `data.obs.modality.rgb` | Image observation keys loaded from HDF5. |
| `data.obs.modality.low_dim` | Low-dimensional observation keys loaded from HDF5. |
| `train.n_epochs` | Epochs per lifelong task. |
| `train.batch_size` | Batch size for the PyTorch dataloader. |
| `train.num_workers` | Dataloader worker count. |
| `train.use_augmentation` | Enables image augmentation in `BasePolicy.preprocess_input`. |
| `eval.eval` | Enables rollout evaluation during training. |
| `pretrain_model_path` | Optional checkpoint loaded into the policy before learning. |

## Matching a Policy to a Dataset

Before training on a custom dataset, make sure the config observation keys exist in the HDF5 file:

```shell
python scripts/get_dataset_info.py --dataset /path/to/task_demo.hdf5
```

Default visual training expects:

```yaml
data:
  obs:
    modality:
      rgb: ["agentview_rgb", "eye_in_hand_rgb"]
      depth: []
      low_dim: ["gripper_states", "joint_states"]
```

For a single-camera model:

```shell
python libero/lifelong/main.py \
  policy=bc_transformer_policy \
  data.obs.modality.rgb='[agentview_rgb]'
```

For RGB-only training without proprioception:

```shell
python libero/lifelong/main.py \
  policy=bc_transformer_policy \
  data.obs.modality.low_dim='[]'
```

If you add a new HDF5 observation key, add it to the right modality list. Robomimic uses these modality groups to infer shapes and preprocessing.

For a concrete pre-training check, inspect and analyze the alphabet soup object-task dataset:

```shell
python scripts/get_dataset_info.py \
  --dataset /scratch/prj/eng_demt_robot_learning/dataset/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5

python /scratch/prj/eng_demt_robot_learning/libero_demt/scripts/analyze_trajectories.py \
  --dataset /scratch/prj/eng_demt_robot_learning/dataset/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \
  --output /tmp/alphabet_soup_trajectory_analysis.png
```

This example confirms the default policy inputs are available: `agentview_rgb`, `eye_in_hand_rgb`, `gripper_states`, and `joint_states`. It also confirms `actions` has shape `(T, 7)`, so the policy head will be sized to seven robot action dimensions through `shape_meta["ac_dim"]`.

Use the trajectory statistics before training:

| Check | Why it matters |
| --- | --- |
| Trajectory length mean around `156` steps | Helps estimate sequence count, epoch time, and whether `data.seq_len=10` is reasonable. |
| RGB frames are `128x128` | Matches `data.img_h=128` and `data.img_w=128`. |
| Actions are within `[-1, 1]` | Matches the expected normalized controller action range. |
| Gripper action dim reaches `-1` and `1` | Confirms grasp open/close commands are represented. |
| Start and end EEF bounds are separated | Confirms the demonstrations contain meaningful pick-and-place motion rather than static frames. |

## Editing Existing Policy Configs

Policy YAML files live in `libero/configs/policy/`.

Examples:

| File | Use |
| --- | --- |
| `bc_rnn_policy.yaml` | Recurrent policy over image, language, and low-dimensional embeddings. |
| `bc_transformer_policy.yaml` | Temporal transformer policy. |
| `bc_vilt_policy.yaml` | Vision-language transformer style policy. |

For `bc_transformer_policy.yaml`, common fields include:

```yaml
policy_type: BCTransformerPolicy
embed_size: 64
transformer_num_layers: 4
transformer_num_heads: 6
transformer_mlp_hidden_size: 256
transformer_dropout: 0.1
transformer_max_seq_len: 10
```

When changing temporal length, keep these aligned:

```shell
python libero/lifelong/main.py \
  policy=bc_transformer_policy \
  data.seq_len=20 \
  policy.transformer_max_seq_len=20
```

Image encoder, language encoder, augmentation, position encoding, and policy head are nested config groups under `libero/configs/policy/`.

## Creating a New Policy Class

Use the existing policies as templates:

- `libero/lifelong/models/bc_rnn_policy.py`
- `libero/lifelong/models/bc_transformer_policy.py`
- `libero/lifelong/models/bc_vilt_policy.py`

Minimum steps:

1. Create a file such as `libero/lifelong/models/my_policy.py`.
2. Define a class that inherits `BasePolicy`.
3. Implement `forward(self, data, train_mode=True)` for training.
4. Implement `get_action(self, data)` for evaluation rollouts.
5. Build `self.policy_head` with output size `shape_meta["ac_dim"]`.
6. Import the class in `libero/lifelong/models/__init__.py`.
7. Add a YAML config in `libero/configs/policy/my_policy.yaml`.

The registry is automatic. Any subclass of `BasePolicy` is registered by class name through the metaclass in `libero/lifelong/models/base_policy.py`. If your class is named `MyPolicy`, set:

```yaml
policy_type: MyPolicy
```

Then launch with:

```shell
python libero/lifelong/main.py policy=my_policy
```

## Data Format Seen by a Policy

The dataloader returns a dictionary from Robomimic, then `SequenceVLDataset` adds a task embedding:

```text
data
|-- obs
|   |-- agentview_rgb
|   |-- eye_in_hand_rgb
|   |-- gripper_states
|   `-- joint_states
|-- actions
`-- task_emb
```

Training calls:

```python
loss = self.policy.compute_loss(data)
```

`BasePolicy.compute_loss` applies image augmentation when enabled, calls `forward`, and computes the action loss through `self.policy_head.loss_fn(dist, data["actions"], reduction)`.

During evaluation, `get_action` receives a single step of observations plus a task embedding. Existing policies call `preprocess_input(data, train_mode=False)` to add the time dimension expected by sequence models.

## Customizing the Action Head

The default policy head config is `libero/configs/policy/policy_head/gmm_head.yaml`. The policy classes set:

```python
policy_head_kwargs.output_size = shape_meta["ac_dim"]
```

This means the action dimension is inferred from `data/demo_i/actions` in the HDF5 file. If your robot has a different action space, regenerate the dataset with the correct `actions` shape and make sure the environment accepts the same action dimension during evaluation.

## Adding a New Training Algorithm

Algorithms live in `libero/lifelong/algos/` and are registered similarly to policies.

To add one:

1. Create `libero/lifelong/algos/my_algo.py`.
2. Subclass `Sequential` from `libero/lifelong/algos/base.py` unless you need a different training loop.
3. Override `start_task`, `observe`, `end_task`, or `learn_one_task` as needed.
4. Import the class in `libero/lifelong/algos/__init__.py`.
5. Add `libero/configs/lifelong/my_algo.yaml`.

Launch with:

```shell
python libero/lifelong/main.py lifelong=my_algo
```

## Training on a Custom Task or Dataset

For an existing LIBERO benchmark task, place the HDF5 file under the dataset root using the benchmark's expected relative path:

```text
<folder>/<suite>/<task_name>_demo.hdf5
```

Then run with:

```shell
python libero/lifelong/main.py \
  benchmark_name=LIBERO_SPATIAL \
  folder=/absolute/path/to/datasets
```

For a fully new task, the benchmark must know about it. Add or modify the relevant BDDL task files under `libero/libero/bddl_files/`, update benchmark task registration if needed, then make sure `benchmark.get_task_demonstration(i)` resolves to your HDF5 file.

## Single-Task Alphabet Soup Benchmark

This repo also registers a convenience benchmark for training and evaluating only the existing alphabet soup task:

```text
LIBERO_ALPHABET_SOUP
```

It reuses the original `libero_object` assets:

| Asset | Path resolved by the benchmark |
| --- | --- |
| Dataset | `libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5` |
| BDDL | `libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl` |
| Init states | `libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.pruned_init` |

Train only this task:

```shell
python libero/lifelong/main.py \
  benchmark_name=LIBERO_ALPHABET_SOUP \
  policy=bc_transformer_policy \
  lifelong=base \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets \
  train.n_epochs=50 \
  train.batch_size=32 \
  eval.eval=true
```

For a quick smoke test:

```shell
python libero/lifelong/main.py \
  benchmark_name=LIBERO_ALPHABET_SOUP \
  policy=bc_transformer_policy \
  lifelong=base \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets \
  train.n_epochs=1 \
  train.batch_size=2 \
  train.num_workers=0 \
  eval.eval=false
```

Since this benchmark has one task, the only task id is `0`.

## Practical Debug Loop

1. Inspect the dataset:

   ```shell
   python scripts/get_dataset_info.py --dataset /path/to/task_demo.hdf5
   ```

2. Run a short CPU or single-GPU smoke test:

   ```shell
   python libero/lifelong/main.py \
     benchmark_name=LIBERO_SPATIAL \
     policy=bc_transformer_policy \
     train.n_epochs=1 \
     train.batch_size=2 \
     train.num_workers=0 \
     eval.eval=false
   ```

3. If the model fails at shape inference, check `data.obs.modality` against the printed HDF5 keys.
4. If loss runs but evaluation fails, check that `env_args`, `bddl_file_name`, and action dimensions match the task environment.
5. Once the smoke test passes, restore evaluation and scale epochs, batch size, and workers.

## Output Files

`libero/lifelong/main.py` creates an experiment directory through `create_experiment_dir(cfg)`. It writes:

| File | Purpose |
| --- | --- |
| `config.json` | Resolved training config for the run. |
| `task{i}_model.pth` | Best checkpoint for lifelong task `i`, selected by evaluation success. |
| `result.pt` | Loss and success matrices plus optional saved simulation states. |

Keep the generated `config.json` with checkpoints. It records observation keys, policy type, benchmark, dataset folder, and training settings needed to reproduce a run.
