# How to Manually Create a LIBERO Task and Demo Dataset

This note explains how to create a LIBERO task like:

`datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5`

The important point is that the HDF5 file is not the task definition. It is the recorded demonstration dataset. A complete LIBERO task has three parts:

1. A BDDL task file, which defines the scene, objects, initial symbolic state, language, and goal.
2. Init-state files, usually `.init` and `.pruned_init`, which store simulator states used for benchmark resets.
3. A demo HDF5 file, which stores successful trajectories for imitation learning.

For the alphabet soup task, the core files are:

```text
LIBERO/libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl
LIBERO/libero/libero/init_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.init
LIBERO/libero/libero/init_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.pruned_init
datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
```

## 1. Understand the Existing Task

The existing BDDL file says:

```lisp
(:language Pick the alphabet soup and place it in the basket)
```

It uses the floor manipulation environment:

```lisp
(define (problem LIBERO_Floor_Manipulation)
  (:domain robosuite)
```

It defines regions on the floor:

```lisp
(bin_region
  (:target floor)
  (:ranges ((-0.01 0.25 0.01 0.27)))
)

(target_object_region
  (:target floor)
  (:ranges ((-0.145 -0.265 -0.095 -0.215)))
)
```

It defines movable objects:

```lisp
(:objects
  alphabet_soup_1 - alphabet_soup
  basket_1 - basket
  salad_dressing_1 - salad_dressing
  cream_cheese_1 - cream_cheese
  milk_1 - milk
  tomato_sauce_1 - tomato_sauce
  butter_1 - butter
)
```

It marks the real task objects:

```lisp
(:obj_of_interest
  alphabet_soup_1
  basket_1
)
```

It sets the initial symbolic placement:

```lisp
(:init
  (On alphabet_soup_1 floor_target_object_region)
  (On basket_1 floor_bin_region)
)
```

And it defines success:

```lisp
(:goal
  (And (In alphabet_soup_1 basket_1_contain_region))
)
```

So the task is: spawn alphabet soup, basket, and distractor objects on the floor; succeed when `alphabet_soup_1` is inside `basket_1_contain_region`.

## 2. Check That the Object Assets Exist

Before writing a new task, make sure every object category exists in LIBERO.

For this task, object names such as `alphabet_soup`, `basket`, `salad_dressing`, `cream_cheese`, `milk`, `tomato_sauce`, and `butter` are registered through:

```text
LIBERO/libero/libero/envs/objects/
LIBERO/libero/libero/assets/
```

The loader path is:

```python
from libero.libero.envs.objects import get_object_fn

get_object_fn("alphabet_soup")
get_object_fn("basket")
```

If you invent a new object category, you must add its MJCF/XML/mesh assets and register it in the object system. If you only want a new task using existing objects, you only need a new BDDL file.

## 3. Create the BDDL File

For a task like this, the fastest manual method is to copy the existing BDDL and edit it.

Example target path:

```text
LIBERO/libero/libero/bddl_files/libero_object/pick_up_the_my_object_and_place_it_in_the_basket.bddl
```

Minimum fields to edit:

```lisp
(:language Pick the my object and place it in the basket)

(:objects
  my_object_1 - my_object
  basket_1 - basket
  ...
)

(:obj_of_interest
  my_object_1
  basket_1
)

(:init
  (On my_object_1 floor_target_object_region)
  (On basket_1 floor_bin_region)
  ...
)

(:goal
  (And (In my_object_1 basket_1_contain_region))
)
```

For the exact alphabet soup task, the BDDL already exists:

```text
LIBERO/libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl
```

If you want to recreate it manually, copy that file content exactly.

## 4. Register the Task in the Benchmark Map

If the task should appear in the standard LIBERO benchmark API, add its base name to:

```text
LIBERO/libero/libero/benchmark/libero_suite_task_map.py
```

For the alphabet soup task, it is already listed under `libero_object`:

```python
"pick_up_the_alphabet_soup_and_place_it_in_the_basket",
```

The benchmark loader then expects:

```text
bddl_files/libero_object/<task_name>.bddl
init_files/libero_object/<task_name>.pruned_init
datasets/libero_object/<task_name>_demo.hdf5
```

If you do not register the task, you can still create an environment directly by passing `bddl_file_name` to `OffScreenRenderEnv` or the matching task class.

## 5. Test That the BDDL Loads

In a working LIBERO environment:

```bash
cd /home/endongsun/robot_learning/LIBERO
conda activate libero
pip install -e .
```

Then run a smoke test:

```python
import os
from libero.libero.envs import OffScreenRenderEnv

bddl = "/home/endongsun/robot_learning/LIBERO/libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl"

env = OffScreenRenderEnv(
    bddl_file_name=bddl,
    camera_heights=128,
    camera_widths=128,
)
env.seed(0)
obs = env.reset()
for _ in range(10):
    obs, reward, done, info = env.step([0.0] * 7)
env.close()
```

If this fails, fix the BDDL/object names before collecting data. Common errors are:

- object category not registered;
- object instance name mismatch, such as `alphabet_soup_1` vs `alphabet_soup`;
- region name mismatch, such as `floor_target_object_region`;
- missing affordance region, such as `basket_1_contain_region`.

## 6. Create Init States

The `.init` and `.pruned_init` files store simulator states, not readable text. LIBERO uses the `.pruned_init` file during benchmark evaluation:

```python
task_suite.get_task_init_states(task_id)
```

For a new task, you need to generate many reset states and keep valid ones. A practical workflow is:

1. Load the BDDL environment.
2. Repeatedly call `env.reset()` with different seeds.
3. Save `env.sim.get_state().flatten()` or the state returned by the LIBERO helper into a tensor/array.
4. Remove states where objects collide, spawn out of reach, or make the task impossible.
5. Save the final set as:

```text
LIBERO/libero/libero/init_files/libero_object/<task_name>.pruned_init
```

For cloning the alphabet soup task exactly, reuse:

```text
LIBERO/libero/libero/init_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.pruned_init
```

For a new object or new region layout, do not reuse the old init states. They encode the old simulator layout.

## 7. Collect Successful Demonstrations

Use the LIBERO teleoperation script:

```bash
cd /home/endongsun/robot_learning/LIBERO
conda activate libero

python scripts/collect_demonstration.py \
  --bddl-file /home/endongsun/robot_learning/LIBERO/libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl \
  --directory demonstration_data \
  --robots Panda \
  --controller OSC_POSE \
  --device keyboard \
  --num-demonstration 50
```

You can use `--device spacemouse` if a SpaceMouse is configured.

This script opens an interactive robosuite viewer and records successful episodes. It writes a raw `demo.hdf5` under a timestamped folder in `demonstration_data/`. The raw file stores mostly simulator states and actions.

Important: only keep task-successful demonstrations. The collection script waits until `env._check_success()` stays true briefly before saving the episode.

## 8. Convert Raw Demo to Training HDF5

The large dataset file like:

```text
datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
```

is created by converting the raw demonstration file into a training dataset with RGB observations, proprioception, rewards, and dones.

Run:

```bash
cd /home/endongsun/robot_learning/LIBERO
conda activate libero

python scripts/create_dataset.py \
  --demo-file /path/to/raw/demo.hdf5 \
  --use-camera-obs
```

The script determines the output path from the raw file's BDDL metadata. For the alphabet soup BDDL, it writes:

```text
<LIBERO dataset path>/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
```

In this workspace, the already downloaded dataset lives at:

```text
/home/endongsun/robot_learning/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
```

If your LIBERO config points to a different dataset root, move or symlink the generated file into the expected dataset directory.

## 9. Expected HDF5 Structure

The final training HDF5 should contain:

```text
data/
  attrs:
    env_name
    problem_info
    env_args
    bddl_file_name
    bddl_file_content
    num_demos
    total

  demo_0/
    attrs:
      num_samples
      model_file
      init_state
    actions
    states
    robot_states
    rewards
    dones
    obs/
      agentview_rgb
      eye_in_hand_rgb
      gripper_states
      joint_states
      ee_states
      ee_pos
      ee_ori
```

Use this command in a proper LIBERO Python environment:

```bash
python LIBERO/scripts/get_dataset_info.py \
  --dataset datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
```

This workspace currently lacks `h5py`, so local inspection from the base shell will fail unless dependencies are installed.

## 10. Replay and Verify the Dataset

After creating the HDF5, replay at least a few demos:

```bash
cd /home/endongsun/robot_learning
conda activate libero

python LIBERO/scripts/replay_libero_demo.py \
  --dataset datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \
  --suite libero_object \
  --task-id 0 \
  --demo-id 0 \
  --save-video
```

Note: `task-id` depends on LIBERO's task order. In the default `libero_object` order, alphabet soup is task id `0`. For direct debugging, it is often clearer to load the BDDL path explicitly in a small script rather than rely on benchmark order.

A valid dataset should:

- reset to the stored initial simulator state;
- replay actions without large simulator divergence;
- reach the BDDL goal `(In alphabet_soup_1 basket_1_contain_region)`;
- have final `rewards[-1] == 1` and final `dones[-1] == 1`;
- contain RGB images if you used `--use-camera-obs`.

## 11. Minimal Checklist for Creating a New Similar Task

For a new task `pick_up_the_X_and_place_it_in_the_basket`:

1. Confirm object category `X` exists in `LIBERO/libero/libero/envs/objects/`.
2. Create `LIBERO/libero/libero/bddl_files/libero_object/pick_up_the_X_and_place_it_in_the_basket.bddl`.
3. Set `(:language ...)`, `(:objects ...)`, `(:obj_of_interest ...)`, `(:init ...)`, and `(:goal ...)`.
4. Add the task name to `LIBERO/libero/libero/benchmark/libero_suite_task_map.py` if you want benchmark API support.
5. Generate and prune init states into `LIBERO/libero/libero/init_files/libero_object/<task_name>.pruned_init`.
6. Smoke-test environment reset and random steps.
7. Collect successful raw demonstrations with `scripts/collect_demonstration.py`.
8. Convert raw demos with `scripts/create_dataset.py --use-camera-obs`.
9. Put the final file at `datasets/libero_object/<task_name>_demo.hdf5`.
10. Replay multiple demos and verify success.

## 12. Connection to the DEMT Notes

The notes in `Thought_cn.md` emphasize that demonstrations should be successful but may differ in human-controllable structures such as approach path, final contact pose, gripper timing, release timing, and speed profile.

For this task, the clean experimental unit is:

```text
same BDDL task + matched init-state coverage + successful demos only + controlled variation in teaching structure
```

That means if you create perturbation datasets for alphabet soup, do not change the BDDL or initial-state distribution when testing guidance effects. Change only the intended demonstration factor, then replay/revalidate every trajectory before training.
