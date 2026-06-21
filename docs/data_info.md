# LIBERO HDF5 Dataset Structure

This repository stores robot demonstrations as HDF5 files. Training uses the converted dataset format produced by `scripts/create_dataset.py` and loaded by `libero/lifelong/datasets.py`.

## File Layout

A standard training dataset has this high-level structure:

```text
task_demo.hdf5
`-- data
    |-- attrs
    |   |-- env_name
    |   |-- env_args
    |   |-- problem_info
    |   |-- bddl_file_name
    |   |-- bddl_file_content
    |   |-- macros_image_convention
    |   |-- num_demos
    |   `-- total
    |-- demo_0
    |   |-- attrs
    |   |   |-- num_samples
    |   |   |-- model_file
    |   |   `-- init_state
    |   |-- obs
    |   |   |-- agentview_rgb
    |   |   |-- eye_in_hand_rgb
    |   |   |-- gripper_states
    |   |   |-- joint_states
    |   |   |-- ee_states
    |   |   |-- ee_pos
    |   |   `-- ee_ori
    |   |-- actions
    |   |-- states
    |   |-- robot_states
    |   |-- rewards
    |   `-- dones
    `-- demo_1
        `-- ...
```

Optional keys can appear depending on conversion flags. If `--use-depth` is passed to `scripts/create_dataset.py`, each demo also stores `obs/agentview_depth` and `obs/eye_in_hand_depth`. If `--no-proprio` is passed, proprioceptive observation keys such as `gripper_states`, `joint_states`, and `ee_*` are omitted.

## Root Attributes

The `data` group stores metadata needed to reconstruct the task and evaluation environment:

| Attribute | Purpose |
| --- | --- |
| `env_name` | Robosuite or LIBERO environment name. |
| `env_args` | JSON string with environment type, task name, BDDL file, and environment keyword arguments. |
| `problem_info` | JSON string with task metadata, including `language_instruction`. |
| `bddl_file_name` | Path to the BDDL task definition used to create the dataset. |
| `bddl_file_content` | BDDL file text stored inside the dataset for reproducibility. |
| `macros_image_convention` | Image convention active when the dataset was created. |
| `num_demos` | Number of demonstration groups under `data`. |
| `total` | Total number of recorded transitions across all demonstrations. |

The training script reads benchmark task metadata separately, but the dataset still needs these attributes for inspection, replay, and reproducible environment construction.

## Demonstration Groups

Each `data/demo_i` group is one trajectory after playback and trimming. `scripts/create_dataset.py` skips the first few raw controller steps with `cap_index = 5`, then writes aligned observations, actions, states, rewards, and done flags.

Common per-demo attributes:

| Attribute | Purpose |
| --- | --- |
| `num_samples` | Number of usable transitions in this trajectory. |
| `model_file` | Mujoco XML model for resetting or replaying the episode. |
| `init_state` | Flattened simulator state used at the start of the converted episode. |

Common per-demo datasets:

| Dataset | Typical shape | Meaning |
| --- | --- | --- |
| `actions` | `(T, A)` | Normalized robot actions. `scripts/get_dataset_info.py` expects values in `[-1, 1]`. |
| `states` | `(T, S)` | Flattened Mujoco simulator states. |
| `robot_states` | `(T, R)` | Robot state vector returned by the environment. |
| `rewards` | `(T,)` | Sparse success labels written by conversion. The final step is `1`. |
| `dones` | `(T,)` | Episode end flags. The final step is `1`. |

`T` is the trajectory length after trimming. Action dimension `A` is inferred by Robomimic and stored in `shape_meta["ac_dim"]` during loading.

## Observation Keys

Default training observations are configured in `libero/configs/data/default.yaml`:

```yaml
obs:
  modality:
    rgb: ["agentview_rgb", "eye_in_hand_rgb"]
    depth: []
    low_dim: ["gripper_states", "joint_states"]
```

The corresponding HDF5 paths are under each demo's `obs` group:

| HDF5 path | Description |
| --- | --- |
| `obs/agentview_rgb` | Third-person RGB camera frames, usually `(T, 128, 128, 3)`. |
| `obs/eye_in_hand_rgb` | Wrist camera RGB frames, usually `(T, 128, 128, 3)`. |
| `obs/gripper_states` | Gripper joint positions. |
| `obs/joint_states` | Robot arm joint positions. |
| `obs/ee_states` | End-effector position plus axis-angle orientation. |
| `obs/ee_pos` | End-effector xyz position. |
| `obs/ee_ori` | End-effector axis-angle orientation. |
| `obs/agentview_depth` | Optional third-person depth frames. |
| `obs/eye_in_hand_depth` | Optional wrist-camera depth frames. |

Only keys listed in `cfg.data.obs.modality` are loaded for training. If you add or remove observation datasets, update the config so `libero/lifelong/datasets.py:get_dataset` can pass the correct `obs_keys` to Robomimic.

## How Training Reads HDF5 Files

`libero/lifelong/main.py` resolves each task's dataset as:

```python
os.path.join(cfg.folder, benchmark.get_task_demonstration(i))
```

By default, `cfg.folder` is `get_libero_path("datasets")`. You can override it from the command line:

```shell
python libero/lifelong/main.py \
  benchmark_name=LIBERO_SPATIAL \
  folder=/absolute/path/to/datasets
```

`libero/lifelong/datasets.py:get_dataset` then:

1. Initializes Robomimic observation utilities from `cfg.data.obs.modality`.
2. Reads shape metadata with `FileUtils.get_shape_metadata_from_dataset`.
3. Builds a `robomimic.utils.dataset.SequenceDataset`.
4. Loads `dataset_keys=["actions"]` and the configured observation keys.
5. Returns the dataset plus `shape_meta`, which is later used to size the policy action head and image encoders.

The default temporal window is `data.seq_len: 10`; each training item contains a short sequence of observations and actions.

## Creating Converted Datasets

Raw collection scripts can produce a lighter raw HDF5 file with only `states`, `actions`, and per-demo `model_file`. Convert it to the training format with:

```shell
python scripts/create_dataset.py \
  --demo-file /path/to/raw/demo.hdf5 \
  --use-camera-obs \
  --output-file /path/to/output/task_demo.hdf5
```

Useful flags:

| Flag | Effect |
| --- | --- |
| `--use-camera-obs` | Records `agentview_rgb` and `eye_in_hand_rgb`; required for the default visual policies. |
| `--use-depth` | Also records depth observations. |
| `--no-proprio` | Omits proprioceptive observation keys. |
| `--output-file` | Writes to an explicit HDF5 path instead of the default LIBERO dataset folder. |
| `--overwrite` | Allows replacing an existing output file. |

## Inspecting and Validating a Dataset

Print statistics, metadata, and the first demo structure:

```shell
python scripts/get_dataset_info.py --dataset /path/to/task_demo.hdf5
```

Print every demo structure:

```shell
python scripts/get_dataset_info.py --dataset /path/to/task_demo.hdf5 --verbose
```

Visualize RGB frames from one demo:

```shell
python scripts/visualize_dataset_demo.py \
  --dataset /path/to/task_demo.hdf5 \
  --demo-id 0 \
  --camera agentview_rgb
```

Replay simulator states and actions:

```shell
python scripts/replay_libero_demo.py \
  --dataset /path/to/task_demo.hdf5 \
  --demo-id 0
```

Analyze end-effector trajectories:

```shell
python scripts/analyze_trajectories.py --dataset /path/to/task_demo.hdf5
```

## Worked Example: Alphabet Soup Dataset

This repository includes a trajectory analysis utility that can summarize end-effector paths, gripper states, and action ranges from a LIBERO HDF5 file. For example, analyze the alphabet soup object-task dataset with:

```shell
python /scratch/prj/eng_demt_robot_learning/libero_demt/scripts/analyze_trajectories.py \
  --dataset /scratch/prj/eng_demt_robot_learning/dataset/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \
  --output /tmp/alphabet_soup_trajectory_analysis.png
```

The script reads:

- `data/demo_i/obs/ee_pos` for 3D and top-down end-effector trajectories.
- `data/demo_i/obs/gripper_states` for gripper opening and closing ranges.
- `data/demo_i/actions` for action saturation and per-dimension control ranges.

On the alphabet soup dataset, the dataset inspection command:

```shell
python scripts/get_dataset_info.py \
  --dataset /scratch/prj/eng_demt_robot_learning/dataset/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
```

reports:

| Field | Value |
| --- | --- |
| Total transitions | `7808` |
| Total trajectories | `50` |
| Trajectory length | min `136`, max `196`, mean `156.16`, std `13.72` |
| Action range | `[-1.0, 1.0]` |
| Language instruction | `pick up the alphabet soup and place it in the basket` |
| Environment | `Libero_Floor_Manipulation` |
| Action shape | `(T, 7)` |
| RGB shape | `(T, 128, 128, 3)` for `agentview_rgb` and `eye_in_hand_rgb` |
| Proprio shape | `gripper_states: (T, 2)`, `joint_states: (T, 7)`, `ee_pos: (T, 3)` |
| Simulator state shape | `states: (T, 110)` |

The trajectory analysis output for the same file is:

| Statistic | Value |
| --- | --- |
| Plotted trajectories | `50` |
| Trajectory length | min `136`, max `196`, mean `156.2`, std `13.7` |
| Start EEF X bounds | `[-0.1632, -0.1245]` |
| Start EEF Y bounds | `[-0.0316, 0.0219]` |
| Start EEF Z bounds | `[0.2500, 0.2808]` |
| End EEF X bounds | `[-0.0624, 0.0900]` |
| End EEF Y bounds | `[0.2212, 0.2992]` |
| End EEF Z bounds | `[0.1351, 0.2819]` |
| Left gripper range | `[0.0007, 0.0405]` |
| Right gripper range | `[-0.0412, -0.0010]` |

Action dimension ranges:

| Action dim | Range |
| --- | --- |
| `0` | `[-0.7929, 0.7795]` |
| `1` | `[-0.9375, 0.8545]` |
| `2` | `[-0.9375, 0.9375]` |
| `3` | `[-0.1029, 0.1661]` |
| `4` | `[-0.1693, 0.2250]` |
| `5` | `[-0.3096, 0.1264]` |
| `6` | `[-1.0000, 1.0000]` |

Use this example as a sanity-check template for new datasets. The trajectory plot should show a coherent start region near the object and an end region near the basket. The action ranges should be mostly inside `[-1, 1]`; repeated hard saturation can indicate controller scaling or collection issues. The gripper range should show both open and closed states for a pick-and-place task.

## Common Dataset Issues

- Missing `data` group: the file is not in the expected LIBERO training format.
- Missing `obs/agentview_rgb` or `obs/eye_in_hand_rgb`: either reconvert with `--use-camera-obs` or remove that key from `cfg.data.obs.modality.rgb`.
- Missing proprio keys: either reconvert without `--no-proprio` or remove missing keys from `cfg.data.obs.modality.low_dim`.
- Actions outside `[-1, 1]`: training can become unstable, and `scripts/get_dataset_info.py` will raise an error.
- Wrong dataset folder layout: benchmark loading expects paths like `<folder>/<suite>/<task_name>_demo.hdf5`, because `benchmark.get_task_demonstration(i)` returns a path relative to `cfg.folder`.
- Inconsistent camera size: update `data.img_h`, `data.img_w`, and model/image encoder settings if you intentionally change image resolution.
