# 在 LIBERO 中加入 Sawyer / KUKA IIWA 的设计说明

本文档说明如何设计并接入 Sawyer 和 KUKA IIWA，使它们能像当前 Franka Panda 一样用于 LIBERO 任务。

当前结论：

```text
robosuite 支持 Sawyer 和 IIWA
LIBERO 当前只注册了 MountedPanda / OnTheGroundPanda
```

所以不能只把采集命令改成：

```bash
--robots Sawyer
```

或：

```bash
--robots IIWA
```

因为 LIBERO 的 problem class 会自动把机器人名改成：

```text
Mounted<robot_name>
OnTheGround<robot_name>
```

例如：

```text
--robots Panda
```

在 floor / living room / coffee table 任务中会变成：

```text
OnTheGroundPanda
```

在 tabletop / kitchen / study 任务中会变成：

```text
MountedPanda
```

因此要支持 Sawyer 和 IIWA，需要新增：

```text
MountedSawyer
OnTheGroundSawyer
MountedIIWA
OnTheGroundIIWA
```

## 1. 当前 Panda 支持方式

现有文件：

```text
LIBERO/libero/libero/envs/robots/mounted_panda.py
LIBERO/libero/libero/envs/robots/on_the_ground_panda.py
LIBERO/libero/libero/envs/robots/__init__.py
```

`MountedPanda` 使用：

```python
xml_path_completion("robots/panda/robot.xml")
default_mount = "RethinkMount"
default_gripper = "PandaGripper"
default_controller_config = "default_panda"
```

`OnTheGroundPanda` 使用：

```python
xml_path_completion("robots/panda/robot.xml")
default_mount = None
default_gripper = "PandaGripper"
default_controller_config = "default_panda"
```

注册位置：

```python
ROBOT_CLASS_MAPPING.update(
    {
        "MountedPanda": SingleArm,
        "OnTheGroundPanda": SingleArm,
    }
)
```

## 2. robosuite 中已有的 Sawyer / IIWA 信息

Sawyer 在 robosuite 中已有：

```text
robosuite/models/robots/manipulators/sawyer_robot.py
```

关键配置：

```python
xml_path_completion("robots/sawyer/robot.xml")
default_mount = "RethinkMount"
default_gripper = "RethinkGripper"
default_controller_config = "default_sawyer"
init_qpos = np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, -1.57])
```

IIWA 在 robosuite 中已有：

```text
robosuite/models/robots/manipulators/iiwa_robot.py
```

关键配置：

```python
xml_path_completion("robots/iiwa/robot.xml")
default_mount = "RethinkMount"
default_gripper = "Robotiq140Gripper"
default_controller_config = "default_iiwa"
init_qpos = np.array([0.000, 0.650, 0.000, -1.890, 0.000, 0.600, 0.000])
```

控制器配置也已经存在：

```text
default_sawyer.json
default_iiwa.json
```

所以主要工作不是从零建机器人模型，而是在 LIBERO 中建立合适的 wrapper、base pose、workspace offset 和验证流程。

## 3. 推荐新增文件

建议新增：

```text
LIBERO/libero/libero/envs/robots/mounted_sawyer.py
LIBERO/libero/libero/envs/robots/on_the_ground_sawyer.py
LIBERO/libero/libero/envs/robots/mounted_iiwa.py
LIBERO/libero/libero/envs/robots/on_the_ground_iiwa.py
```

也可以把多个类放在一个文件里，但按 Panda 当前风格，单独文件更清楚。

## 4. MountedSawyer 模板

```python
import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class MountedSawyer(ManipulatorModel):
    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/sawyer/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "RethinkGripper"

    @property
    def default_controller_config(self):
        return "default_sawyer"

    @property
    def init_qpos(self):
        return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, -1.57])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
            "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
```

## 5. OnTheGroundSawyer 模板

```python
import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class OnTheGroundSawyer(ManipulatorModel):
    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/sawyer/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return None

    @property
    def default_gripper(self):
        return "RethinkGripper"

    @property
    def default_controller_config(self):
        return "default_sawyer"

    @property
    def init_qpos(self):
        return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, -1.57])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "coffee_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.41),
            "living_room_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.42),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
```

## 6. MountedIIWA 模板

```python
import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class MountedIIWA(ManipulatorModel):
    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/iiwa/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "Robotiq140Gripper"

    @property
    def default_controller_config(self):
        return "default_iiwa"

    @property
    def init_qpos(self):
        return np.array([0.000, 0.650, 0.000, -1.890, 0.000, 0.600, 0.000])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
            "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
```

## 7. OnTheGroundIIWA 模板

```python
import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class OnTheGroundIIWA(ManipulatorModel):
    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/iiwa/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return None

    @property
    def default_gripper(self):
        return "Robotiq140Gripper"

    @property
    def default_controller_config(self):
        return "default_iiwa"

    @property
    def init_qpos(self):
        return np.array([0.000, 0.650, 0.000, -1.890, 0.000, 0.600, 0.000])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "coffee_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.41),
            "living_room_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.42),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
```

## 8. 注册机器人

修改：

```text
LIBERO/libero/libero/envs/robots/__init__.py
```

加入 import：

```python
from .mounted_sawyer import MountedSawyer
from .on_the_ground_sawyer import OnTheGroundSawyer
from .mounted_iiwa import MountedIIWA
from .on_the_ground_iiwa import OnTheGroundIIWA
```

加入 mapping：

```python
ROBOT_CLASS_MAPPING.update(
    {
        "MountedPanda": SingleArm,
        "OnTheGroundPanda": SingleArm,
        "MountedSawyer": SingleArm,
        "OnTheGroundSawyer": SingleArm,
        "MountedIIWA": SingleArm,
        "OnTheGroundIIWA": SingleArm,
    }
)
```

注意：`ManipulatorModel` 子类会通过 robosuite 的 metaclass 自动注册到 robot model registry；`ROBOT_CLASS_MAPPING` 是告诉 robosuite 这些机器人用 `SingleArm` runtime wrapper。

## 9. 使用命令

注册完成后，可以尝试：

```bash
cd /home/endongsun/robot_learning/LIBERO

conda run -n libero_cpu env \
  NUMBA_DISABLE_JIT=1 \
  MPLCONFIGDIR=/tmp/matplotlib_libero \
  python scripts/collect_demonstration.py \
  --bddl-file /home/endongsun/robot_learning/LIBERO/libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl \
  --directory /home/endongsun/robot_learning/LIBERO/demonstration_data \
  --robots Sawyer \
  --controller OSC_POSE \
  --device keyboard \
  --num-demonstration 1
```

或：

```bash
--robots IIWA
```

对于 alphabet soup 这个 floor task：

```text
Sawyer -> OnTheGroundSawyer
IIWA   -> OnTheGroundIIWA
```

对于 tabletop / kitchen / study task：

```text
Sawyer -> MountedSawyer
IIWA   -> MountedIIWA
```

## 10. 必须验证的内容

新增 wrapper 后，不要直接采数据。先做 smoke test：

```python
from robosuite import load_controller_config
from libero.libero.envs import TASK_MAPPING
import libero.libero.envs.bddl_utils as BDDLUtils

bddl = "/home/endongsun/robot_learning/LIBERO/libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl"
problem_info = BDDLUtils.get_problem_info(bddl)
problem_name = problem_info["problem_name"]

config = {
    "robots": ["Sawyer"],
    "controller_configs": load_controller_config(default_controller="OSC_POSE"),
}

env = TASK_MAPPING[problem_name](
    bddl_file_name=bddl,
    **config,
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_heights=128,
    camera_widths=128,
    ignore_done=True,
    control_freq=20,
)
obs = env.reset()
for _ in range(10):
    obs, reward, done, info = env.step([0.0] * env.action_dim)
env.close()
```

分别测试：

```text
Sawyer + floor task
IIWA + floor task
Sawyer + tabletop task
IIWA + tabletop task
```

## 11. 风险点

### 11.1 base pose 不一定合适

上面模板先复用 Panda 的 `base_xpos_offset`。这只是第一版设计，不保证 Sawyer / IIWA 的 reachability 最优。

需要检查：

- 末端初始位置是否在 workspace 附近；
- 机械臂是否和桌子、地面、墙、篮子碰撞；
- 是否能到达目标物体；
- 是否能到达 basket / plate / drawer 等 goal region；
- 相机视角是否仍然能看到关键动作。

如果 reachability 不好，优先调：

```python
base_xpos_offset
init_qpos
```

### 11.2 gripper 几何不同

Panda 使用：

```text
PandaGripper
```

Sawyer 使用：

```text
RethinkGripper
```

IIWA 使用：

```text
Robotiq140Gripper
```

夹爪宽度、接触点、闭合动作不同。原 Panda 成功的 grasp pose 不一定适用于 Sawyer / IIWA。

对于 DEMT 或 imitation learning，不能混用 Panda demo 和 Sawyer / IIWA demo。动作维度和机器人动力学都可能不同。

### 11.3 init states 不能直接复用

LIBERO 的 `.pruned_init` 存的是 simulator state。它和机器人模型、关节数、body tree 有关。

如果换机器人，旧的 Panda init states 很可能不能直接用于 Sawyer / IIWA。

需要重新生成：

```text
LIBERO/libero/libero/init_files/<suite>/<task_name>_<robot>.pruned_init
```

或者建立新的 benchmark entry，明确这是 Sawyer/IIWA 版本。

不建议覆盖原始 Panda `.pruned_init`。

### 11.4 dataset 文件不能覆盖 Panda 原数据

原数据：

```text
datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
```

是 Panda 数据。Sawyer / IIWA 数据建议另存为：

```text
datasets/libero_object_sawyer/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
datasets/libero_object_iiwa/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
```

或：

```text
datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_sawyer_demo.hdf5
datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_iiwa_demo.hdf5
```

核心原则：不要让训练代码误以为 Sawyer / IIWA 数据是原始 Panda benchmark 数据。

## 12. 推荐实施顺序

1. 新增 `MountedSawyer` 和 `OnTheGroundSawyer`。
2. 在 `robots/__init__.py` 注册 Sawyer。
3. 只测试一个简单 floor pick-place task。
4. 调 `base_xpos_offset` 和 `init_qpos`，直到无碰撞且能遥操作成功。
5. 采 1 条 Sawyer demo。
6. 转换 HDF5 并 replay。
7. 再做 IIWA。
8. 最后扩展到 tabletop / kitchen / study tasks。

不要一开始就改全部 suite。先让一个 task 稳定成功。

## 13. 推荐命名

为了不破坏原 benchmark，建议新增 robot-specific suite 名：

```text
libero_object_sawyer
libero_object_iiwa
```

也可以先不改 benchmark，只用 BDDL path 直接采集和 replay。等机器人稳定后，再正式加入 benchmark map。

## 14. 成功标准

一个新机器人接入可以认为初步成功，至少要满足：

```text
环境能 reset
无初始碰撞
随机 action 能 step
键盘遥操作能控制末端
能完成至少一条成功 demo
raw demo 能转换成训练 HDF5
HDF5 replay 能再次达到 success
```

只有完成 replay success，才说明这条数据真正可用。

## 15. 结论

Sawyer 和 KUKA IIWA 可以作为设计目标接入 LIBERO，但它们不是当前 LIBERO benchmark 的即插即用机器人。

最小实现是新增四个 wrapper：

```text
MountedSawyer
OnTheGroundSawyer
MountedIIWA
OnTheGroundIIWA
```

然后注册到：

```text
LIBERO/libero/libero/envs/robots/__init__.py
```

第一版可以复用 robosuite 默认 robot XML、gripper、controller config，并临时复用 Panda 的 base offset。真正用于实验前，必须重新验证 reachability、init states、demo replay 和 dataset 命名。
