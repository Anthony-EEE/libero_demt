# LIBERO 数据、训练与验证全流程笔记

这份笔记按完整实验流程组织：先确认 HDF5 数据，再启动学习训练，最后做验证和 rollout evaluation。

## 1. 数据阶段

LIBERO 的训练数据是 HDF5 文件。一个标准训练数据集大致长这样：

```text
task_demo.hdf5
`-- data
    |-- demo_0
    |   |-- obs
    |   |   |-- agentview_rgb
    |   |   |-- eye_in_hand_rgb
    |   |   |-- gripper_states
    |   |   |-- joint_states
    |   |   |-- ee_pos
    |   |   `-- ee_ori
    |   |-- actions
    |   |-- states
    |   |-- rewards
    |   `-- dones
    |-- demo_1
    `-- ...
```

每个 `demo_i` 是一条机器人示范轨迹。

| 字段 | 作用 |
| --- | --- |
| `obs/agentview_rgb` | 第三视角 RGB 图像 |
| `obs/eye_in_hand_rgb` | 机械臂腕部相机图像 |
| `obs/gripper_states` | 夹爪状态 |
| `obs/joint_states` | 机械臂关节状态 |
| `obs/ee_pos` | 末端执行器位置 |
| `actions` | 机器人动作，通常 shape 是 `(T, 7)` |
| `states` | Mujoco 仿真状态 |
| `rewards` | 成功奖励，通常最后一步为 `1` |
| `dones` | episode 结束标志 |

先检查数据集：

```shell
python scripts/get_dataset_info.py \
  --dataset /scratch/prj/eng_demt_robot_learning/dataset/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
```

这个命令会检查：

- demonstration 数量
- 每条轨迹长度
- action 是否在 `[-1, 1]`
- observation keys 是否完整
- 图像尺寸
- action shape
- environment metadata

alphabet soup 数据集的例子：

```text
total trajectories: 50
total transitions: 7808
traj length mean: 156.16
action min: -1.0
action max: 1.0
language instruction: pick up the alphabet soup and place it in the basket
actions shape: (T, 7)
agentview_rgb shape: (T, 128, 128, 3)
eye_in_hand_rgb shape: (T, 128, 128, 3)
```

再分析轨迹质量：

```shell
python /scratch/prj/eng_demt_robot_learning/libero_demt/scripts/analyze_trajectories.py \
  --dataset /scratch/prj/eng_demt_robot_learning/dataset/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \
  --output /tmp/alphabet_soup_trajectory_analysis.png
```

这个脚本会画：

- 3D end-effector 轨迹
- top-down 轨迹投影
- 起点和终点分布
- gripper range
- action 每个维度的范围

数据阶段的目标不是训练，而是确认数据合理。

| 检查项 | 为什么重要 |
| --- | --- |
| action 在 `[-1, 1]` | 训练默认假设动作已经 normalize |
| RGB shape 是 `128x128x3` | 匹配默认模型输入 |
| `agentview_rgb` 和 `eye_in_hand_rgb` 存在 | 默认 visual policy 会读取这两个相机 |
| `gripper_states` 和 `joint_states` 存在 | 默认 low-dimensional 输入会读取它们 |
| 轨迹起点和终点有明显移动 | 说明 demonstration 不是静态或失败轨迹 |
| gripper 有开合变化 | pick-and-place 任务必须有 grasp 行为 |

## 2. 数据如何进入训练

训练入口是：

```text
libero/lifelong/main.py
```

它会根据 benchmark 找到对应数据集：

```python
os.path.join(cfg.folder, benchmark.get_task_demonstration(i))
```

如果设置：

```shell
folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets
benchmark_name=LIBERO_OBJECT
```

训练会加载类似这个路径：

```text
/scratch/prj/eng_demt_robot_learning/dataset/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
```

数据加载逻辑在：

```text
libero/lifelong/datasets.py
```

核心函数是：

```python
get_dataset(...)
```

它会：

1. 读取 config 里的 observation keys。
2. 用 Robomimic 读取 HDF5。
3. 构建 `SequenceDataset`。
4. 返回 dataset 和 `shape_meta`。
5. 用 `shape_meta["ac_dim"]` 决定模型 action head 输出维度。

默认 observation config 在：

```text
libero/configs/data/default.yaml
```

默认内容：

```yaml
obs:
  modality:
    rgb: ["agentview_rgb", "eye_in_hand_rgb"]
    depth: []
    low_dim: ["gripper_states", "joint_states"]
```

因此 HDF5 里必须有：

```text
obs/agentview_rgb
obs/eye_in_hand_rgb
obs/gripper_states
obs/joint_states
actions
```

否则训练会在数据加载或 shape inference 阶段失败。

## 3. 学习训练阶段

基本训练命令：

```shell
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

python libero/lifelong/main.py \
  benchmark_name=LIBERO_OBJECT \
  policy=bc_transformer_policy \
  lifelong=base \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets
```

关键参数：

| 参数 | 含义 |
| --- | --- |
| `benchmark_name` | 选择任务集合，比如 `LIBERO_OBJECT` |
| `policy` | 选择模型结构 |
| `lifelong` | 选择 lifelong learning 算法 |
| `folder` | 数据集根目录 |
| `train.n_epochs` | 每个任务训练多少 epoch |
| `train.batch_size` | batch size |
| `data.seq_len` | 每个训练样本包含多少连续时间步 |
| `eval.eval` | 是否训练中 rollout 评估 |

可选 policy：

```text
bc_rnn_policy
bc_transformer_policy
bc_vilt_policy
```

可选 lifelong algorithm：

```text
base
er
ewc
packnet
multitask
single_task
```

先跑一个 smoke test：

```shell
python libero/lifelong/main.py \
  benchmark_name=LIBERO_OBJECT \
  policy=bc_transformer_policy \
  lifelong=base \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets \
  train.n_epochs=1 \
  train.batch_size=2 \
  train.num_workers=0 \
  eval.eval=false
```

这个测试用来确认：

- 数据能不能读
- 模型能不能 build
- loss 能不能 forward/backward
- action head shape 是否正确
- observation key 没有 mismatch

正式训练：

```shell
python libero/lifelong/main.py \
  benchmark_name=LIBERO_OBJECT \
  policy=bc_transformer_policy \
  lifelong=base \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets \
  train.n_epochs=50 \
  train.batch_size=32 \
  eval.eval=true
```

如果你只想训练 alphabet soup 一个任务，可以使用单任务 benchmark：

```text
LIBERO_ALPHABET_SOUP
```

这个 benchmark 只包含一个任务：

```text
pick_up_the_alphabet_soup_and_place_it_in_the_basket
```

它复用已有的 `libero_object` 文件：

| 资源 | 路径 |
| --- | --- |
| Dataset | `libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5` |
| BDDL | `libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl` |
| Init states | `libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.pruned_init` |

单任务 smoke test：

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

单任务正式训练和 rollout：

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

因为这个 benchmark 只有一个任务，所以 task id 永远是 `0`。

## 4. 模型内部训练逻辑

训练主循环大致是：

```text
main.py
 -> load benchmark
 -> load HDF5 datasets
 -> get task language embeddings
 -> create policy
 -> create lifelong algorithm
 -> train task by task
 -> evaluate
 -> save checkpoint
```

模型注册在：

```text
libero/lifelong/models/__init__.py
```

policy 基类在：

```text
libero/lifelong/models/base_policy.py
```

训练时调用：

```python
loss = self.policy.compute_loss(data)
```

`compute_loss` 做三件事：

1. 对图像做 augmentation。
2. 调用 policy 的 `forward(data)`。
3. 用 `policy_head.loss_fn(...)` 计算 action prediction loss。

输入 data 大致是：

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

输出是动作分布，然后和 dataset 里的 `actions` 做 imitation learning loss。

## 5. 如何自定义学习模型

最简单的方式是改 config。

改变 transformer 长度：

```shell
python libero/lifelong/main.py \
  policy=bc_transformer_policy \
  data.seq_len=20 \
  policy.transformer_max_seq_len=20
```

改变 batch size 和 epoch：

```shell
python libero/lifelong/main.py \
  train.batch_size=16 \
  train.n_epochs=100
```

只用一个相机：

```shell
python libero/lifelong/main.py \
  data.obs.modality.rgb='[agentview_rgb]'
```

不用 proprio：

```shell
python libero/lifelong/main.py \
  data.obs.modality.low_dim='[]'
```

如果要写新模型：

1. 新建文件：

   ```text
   libero/lifelong/models/my_policy.py
   ```

2. 继承 `BasePolicy`：

   ```python
   from libero.lifelong.models.base_policy import BasePolicy

   class MyPolicy(BasePolicy):
       def forward(self, data, train_mode=True):
           ...

       def get_action(self, data):
           ...
   ```

3. 在 `libero/lifelong/models/__init__.py` 里 import：

   ```python
   from libero.lifelong.models.my_policy import MyPolicy
   ```

4. 新建 config：

   ```text
   libero/configs/policy/my_policy.yaml
   ```

5. 在 config 里写：

   ```yaml
   policy_type: MyPolicy
   ```

6. 启动：

   ```shell
   python libero/lifelong/main.py policy=my_policy
   ```

## 6. 验证阶段

验证分三层。

第一层是数据验证：

```shell
python scripts/get_dataset_info.py --dataset /path/to/demo.hdf5
python scripts/analyze_trajectories.py --dataset /path/to/demo.hdf5
python scripts/visualize_dataset_demo.py --dataset /path/to/demo.hdf5 --demo-id 0 --camera agentview_rgb
```

确认数据本身没有问题。

第二层是训练 smoke test：

```shell
python libero/lifelong/main.py \
  benchmark_name=LIBERO_OBJECT \
  policy=bc_transformer_policy \
  lifelong=base \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets \
  train.n_epochs=1 \
  train.batch_size=2 \
  train.num_workers=0 \
  eval.eval=false
```

确认训练代码可以跑通。

第三层是 rollout success evaluation。

如果训练时设置：

```shell
eval.eval=true
```

`main.py` 会在训练过程中调用 evaluation，计算任务成功率。

也可以单独评估 checkpoint：

```shell
python libero/lifelong/evaluate.py \
  --benchmark LIBERO_OBJECT \
  --task_id 0 \
  --algo base \
  --policy bc_transformer_policy \
  --seed 10000 \
  --ep 50 \
  --load_task 0 \
  --device_id 0
```

训练输出通常包括：

| 文件 | 作用 |
| --- | --- |
| `config.json` | 本次实验完整配置 |
| `task0_model.pth` | task 0 最佳模型 |
| `task1_model.pth` | task 1 最佳模型 |
| `result.pt` | loss / success 结果矩阵 |

## 7. 推荐实际工作流

建议按这个顺序做：

```text
Step 1: 检查 HDF5
Step 2: 分析轨迹
Step 3: 可视化 demo 视频
Step 4: 跑 1 epoch smoke test
Step 5: 正式训练
Step 6: rollout evaluation
Step 7: 看失败 case，再回到数据分析
```

对应命令：

```shell
python scripts/get_dataset_info.py --dataset DATASET.hdf5
```

```shell
python scripts/analyze_trajectories.py --dataset DATASET.hdf5
```

```shell
python scripts/visualize_dataset_demo.py \
  --dataset DATASET.hdf5 \
  --demo-id 0 \
  --camera agentview_rgb
```

```shell
python libero/lifelong/main.py \
  benchmark_name=LIBERO_OBJECT \
  policy=bc_transformer_policy \
  lifelong=base \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets \
  train.n_epochs=1 \
  train.batch_size=2 \
  train.num_workers=0 \
  eval.eval=false
```

```shell
python libero/lifelong/main.py \
  benchmark_name=LIBERO_OBJECT \
  policy=bc_transformer_policy \
  lifelong=base \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets \
  train.n_epochs=50 \
  train.batch_size=32 \
  eval.eval=true
```

核心流程是：

```text
HDF5 demonstrations
 -> Robomimic SequenceDataset
 -> LIBERO policy model
 -> imitation learning loss on actions
 -> checkpoint
 -> rollout in simulator
 -> success rate
```

这个 pipeline 不能只看 training loss。最终判断标准是 policy 在 LIBERO 环境里 rollout 是否成功完成任务。

## 8. SingleTask 单任务训练调试记录

这一节记录一次实际调试：只用 alphabet soup 的 HDF5 数据，在 `LIBERO_ALPHABET_SOUP` 单任务 benchmark 上跑 `single_task` learning。

目标 dataset：

```text
/scratch/prj/eng_demt_robot_learning/dataset/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
```

目标 benchmark：

```text
LIBERO_ALPHABET_SOUP
```

目标算法：

```text
lifelong=single_task
```

### 8.1 SingleTask 是怎么进入 learning 的

`single_task.py` 很短，因为大部分训练逻辑都继承自 `Sequential`：

```text
libero/lifelong/algos/base.py
```

调用链是：

```text
libero/lifelong/main.py
 -> Hydra 读取 lifelong=single_task
 -> configs/lifelong/single_task.yaml 里设置 algo: SingleTask
 -> get_algo_class("SingleTask")
 -> 实例化 SingleTask(n_tasks, cfg)
 -> 对每个 task 调用 algo.learn_one_task(...)
```

`SingleTask` 能被 `get_algo_class` 找到，是因为：

```text
libero/lifelong/algos/__init__.py
```

里 import 了：

```python
from libero.lifelong.algos.single_task import SingleTask
```

import 时，`AlgoMeta` metaclass 会把 `SingleTask` 注册到 `REGISTERED_ALGOS`。

`SingleTask` 只改变一个行为：每个新 task 开始前，把 policy 重置回初始随机模型。

```python
self.init_pi = copy.deepcopy(self.policy)
```

保存初始 policy。

```python
self.policy = copy.deepcopy(self.init_pi)
super().start_task(task)
```

每个 task 开始时重置 policy，然后复用 `Sequential.start_task()` 创建 optimizer 和 scheduler。

所以区别是：

```text
Sequential/base:
task 0 训练完继续训练 task 1

SingleTask:
每个 task 都从同一个初始 policy 重新训练
```

对于 `LIBERO_ALPHABET_SOUP`，只有一个 task，所以 `SingleTask` 和普通训练在任务数量上没有差别；它主要用于验证算法注册、训练循环和 rollout 流程。

### 8.2 本次已经跑通的部分

使用命令：

```shell
python libero/lifelong/main.py \
  benchmark_name=LIBERO_ALPHABET_SOUP \
  policy=bc_transformer_policy \
  lifelong=single_task \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets \
  train.n_epochs=1 \
  train.batch_size=2 \
  train.num_workers=0 \
  eval.eval=true \
  eval.n_eval=5 \
  eval.eval_every=1
```

日志确认已经跑通：

```text
Hydra config 加载
SingleTask algo 注册并选中
LIBERO_ALPHABET_SOUP benchmark 加载
目标 HDF5 读取
50 demos / 7808 sequences
policy build
进入 task 0 training
epoch 0 loss
epoch 0 rollout
epoch 1 training
epoch 1 rollout
```

实际输出示例：

```text
[info] Epoch:   0 | train loss:  5.49 | time: 1.59
[info] evaluate task 0 takes 55.9 seconds
[info] Epoch:   0 | succ: 0.00 ± 0.00 | best succ: 0.0 | succ. AoC 0.00
[info] Epoch:   1 | train loss: -9.13 | time: 3.16
[info] evaluate task 0 takes 50.8 seconds
[info] Epoch:   1 | succ: 0.00 ± 0.00 | best succ: 0.0 | succ. AoC 0.00
```

`succ=0.00` 在这个 smoke test 里正常，因为只训练了 1 epoch，而且 batch size 很小。这个测试的目的不是获得好结果，而是确认训练和 rollout pipeline 能跑通。

### 8.3 第一个坑：persistent_workers 和 num_workers

第一次用：

```text
train.num_workers=0
```

报错：

```text
ValueError: persistent_workers option needs num_workers > 0
```

原因在 `Sequential.learn_one_task()` 的 `DataLoader`：

```python
train_dataloader = DataLoader(
    dataset,
    batch_size=self.cfg.train.batch_size,
    num_workers=self.cfg.train.num_workers,
    sampler=RandomSampler(dataset),
    persistent_workers=True,
)
```

PyTorch 不允许：

```text
num_workers=0
persistent_workers=True
```

更稳的写法是：

```python
train_dataloader = DataLoader(
    dataset,
    batch_size=self.cfg.train.batch_size,
    num_workers=self.cfg.train.num_workers,
    sampler=RandomSampler(dataset),
    persistent_workers=self.cfg.train.num_workers > 0,
)
```

这样 `train.num_workers=0` 时不会启用 persistent workers。

### 8.4 第二个坑：h5py objects cannot be pickled

尝试把：

```text
train.num_workers=1
```

之后又报错：

```text
TypeError: h5py objects cannot be pickled
```

原因是 PyTorch `DataLoader` 使用 multiprocessing worker 时，需要 pickle dataset。但 Robomimic / HDF5 dataset 内部持有 h5py 对象，h5py 对象不能被 pickle。

所以训练阶段最稳的设置是：

```text
train.num_workers=0
```

同时确保 `persistent_workers` 根据 `num_workers > 0` 自动关闭。

### 8.5 第三个坑：eval.num_workers 也会触发 h5py pickle

训练和 epoch 内 rollout 都跑完后，最后又在：

```text
main.py -> evaluate_loss(...)
```

报错：

```text
TypeError: h5py objects cannot be pickled
```

这次原因不是 `train.num_workers`，而是默认：

```text
eval.num_workers=4
```

`evaluate_loss()` 也会创建 `DataLoader`。如果 `eval.num_workers > 0`，同样会触发 h5py pickle 问题。

如果要完整跑 task 结束后的 loss / success matrix evaluation，需要显式设置：

```text
eval.num_workers=0
```

如果只是调试 learning algorithm，可以先跳过最后全局 evaluation：

```text
eval.eval=false
```

注意：`eval.eval=false` 只跳过 `main.py` 里 task 结束后的 `evaluate_loss/evaluate_success`。`learn_one_task()` 内部的 epoch rollout 由下面条件控制：

```python
if epoch % self.cfg.eval.eval_every == 0:
    evaluate_one_task_success(...)
```

所以如果想完全避开 rollout，可以把：

```text
eval.eval_every=999
```

设成大于当前训练 epoch 数。

### 8.6 推荐调试命令

只测试训练，不做 rollout：

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
  eval.eval_every=999
```

测试训练和少量 single benchmark rollout，但跳过最后全局 eval：

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

完整测试训练、epoch rollout、最后全局 eval：

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

### 8.7 当前结论

`SingleTask + LIBERO_ALPHABET_SOUP + alphabet_soup_demo.hdf5` 已确认可以完成：

```text
加载数据
构建 policy
训练
rollout evaluation
```

目前遇到的问题主要来自 `DataLoader` multiprocessing 和 h5py 不兼容，不是 `SingleTask`、benchmark 或 HDF5 数据本身的问题。

## 9. 自己写 single-task algo 的计划

这一章记录下一步自己实现 single-task learning algorithm 的计划。目标是在现有训练入口里，只针对一个任务、一个 HDF5 数据集、一个 benchmark 写和测试自己的算法。

当前默认测试任务是：

```text
LIBERO_ALPHABET_SOUP
```

当前默认测试数据是：

```text
/scratch/prj/eng_demt_robot_learning/dataset/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
```

虽然项目里的目录名仍然是：

```text
libero/lifelong/algos/
libero/configs/lifelong/
```

但第 9 章只讨论 single-task algo：先写一个只服务单任务训练的算法。也就是说，先只关心：

```text
task_id = 0
load_task = 0
一个 dataset
一个 policy
一个训练循环
一个 rollout success rate
```

原则：先写一个最小可运行版本，行为尽量接近 `SingleTask`，确认注册、配置、训练入口都通了，再逐步加入自己的单任务训练逻辑。

### 9.1 先读懂 single-task 相关结构

优先读这些文件：

```text
libero/lifelong/algos/base.py
libero/lifelong/algos/single_task.py
libero/lifelong/algos/__init__.py
libero/lifelong/main.py
libero/configs/lifelong/single_task.yaml
```

重点理解：

```text
Sequential.__init__()
Sequential.start_task()
Sequential.observe()
Sequential.eval_observe()
Sequential.learn_one_task()
Sequential.end_task()
```

其中 `learn_one_task()` 是默认单任务训练流程，里面负责：

```text
创建 DataLoader
epoch loop
调用 observe 做训练
调用 eval_observe 做 loss evaluation
调用 evaluate_one_task_success 做 rollout
保存最佳 checkpoint
调用 end_task 做算法后处理
```

第一版 single-task algo 不需要一开始重写 `learn_one_task()`。先复用它，把改动集中在 `start_task()`、`observe()` 或 `end_task()`。

### 9.2 决定 override 哪个函数

写新 single-task algo 前，先判断自己的算法改的是哪一层。

如果只是改变每个 batch 的训练 loss 或 gradient update，优先 override：

```text
observe(data)
```

例如：

```text
加 regularization loss
加 replay batch
改 backward 方式
做 gradient projection
改 optimizer step 前后的逻辑
```

如果需要在单任务开始时准备状态，override：

```text
start_task(task)
```

例如：

```text
冻结部分参数
初始化 buffer
重建 optimizer
```

如果需要在单任务结束后保存或更新信息，override：

```text
end_task(dataset, task_id, benchmark, env=None)
```

例如：

```text
保存旧模型 snapshot
保存训练统计
保存 buffer
保存额外评估结果
```

只有当默认 epoch loop 完全不适合时，才考虑 override：

```text
learn_one_task(dataset, task_id, benchmark, result_summary)
```

重写 `learn_one_task()` 风险更高，因为要自己处理 training、evaluation、checkpoint、scheduler、result summary 等逻辑。单任务第一版先不要这样做。

### 9.3 最小文件结构

新增算法文件，例如：

```text
libero/lifelong/algos/my_single_task_algo.py
```

新增 Hydra config：

```text
libero/configs/lifelong/my_single_task_algo.yaml
```

更新算法 import：

```text
libero/lifelong/algos/__init__.py
```

最小 config 先保持简单：

```yaml
algo: MySingleTaskAlgo
```

`MySingleTaskAlgo` 的 class name 要和 config 里的 `algo` 对应。注册时会转成小写 key，所以：

```text
MySingleTaskAlgo -> mysingletaskalgo
SingleTask -> singletask
Sequential -> sequential
```

### 9.4 最小实现策略

第一版不要直接写复杂算法。建议先写一个行为等价于 `SingleTask` 或 `Sequential` 的版本，只继承并调用父类逻辑。

目标不是创新，而是确认这几件事：

```text
algo 文件能 import
AlgoMeta 能自动注册
Hydra config 能加载
get_algo_class 能找到新类
policy 能构建
learn_one_task 能进入
observe 能被调用
loss 能 forward/backward
checkpoint 和 rollout 不破
```

确认第一版跑通后，再逐步加入自己的单任务算法逻辑。

### 9.5 推荐开发顺序

Step 1：创建最小 single-task algo 文件。

```text
libero/lifelong/algos/my_single_task_algo.py
```

先只继承 `Sequential`，必要时 override 一个很薄的 `observe()` 或 `start_task()`。如果想完全沿用 `SingleTask` 的“每次从初始 policy 开始”行为，也可以参考 `SingleTask` 的 `start_task()`。

Step 2：在 `algos/__init__.py` import 新 class。

```text
from libero.lifelong.algos.my_single_task_algo import MySingleTaskAlgo
```

Step 3：创建 config。

```text
libero/configs/lifelong/my_single_task_algo.yaml
```

内容先写：

```yaml
algo: MySingleTaskAlgo
```

Step 4：做 Python 编译检查。

```shell
python -m py_compile \
  libero/lifelong/algos/my_single_task_algo.py \
  libero/lifelong/algos/__init__.py
```

Step 5：确认算法被注册。

运行任意短命令时，启动日志会打印：

```text
Available algorithms:
```

里面应该出现新算法，例如：

```text
'mysingletaskalgo': <class 'libero.lifelong.algos.my_single_task_algo.MySingleTaskAlgo'>
```

Step 6：先跑只训练、不 rollout 的 smoke test。

```shell
python libero/lifelong/main.py \
  benchmark_name=LIBERO_ALPHABET_SOUP \
  policy=bc_transformer_policy \
  lifelong=my_single_task_algo \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets \
  train.n_epochs=1 \
  train.batch_size=32 \
  train.num_workers=0 \
  eval.eval=false \
  eval.eval_every=999
```

Step 7：再跑少量 rollout，但跳过最后全局 eval。

```shell
python libero/lifelong/main.py \
  benchmark_name=LIBERO_ALPHABET_SOUP \
  policy=bc_transformer_policy \
  lifelong=my_single_task_algo \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets \
  train.n_epochs=1 \
  train.batch_size=32 \
  train.num_workers=0 \
  eval.eval=false \
  eval.eval_every=1 \
  eval.n_eval=2 \
  eval.use_mp=false
```

Step 8：如果需要完整 evaluation，再显式关闭 eval dataloader multiprocessing。

```shell
python libero/lifelong/main.py \
  benchmark_name=LIBERO_ALPHABET_SOUP \
  policy=bc_transformer_policy \
  lifelong=my_single_task_algo \
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

### 9.6 加 single-task 算法逻辑的顺序

最小版本跑通后，再按风险从低到高添加功能。

第一层：只加额外 logging。

```text
打印 current_task
打印 loss 分量
打印 buffer size
打印 grad norm
```

第二层：加不改变训练行为的状态保存。

```text
保存旧 policy snapshot
保存 task 0 的训练统计
保存 dataset 统计量
```

第三层：加轻量 loss modification。

```text
loss = bc_loss + lambda * aux_loss
```

第四层：加单任务训练机制。

```text
样本重加权
额外 action loss
额外 representation loss
数据增强开关
冻结或解冻部分网络
自定义 optimizer step
```

第五层：如果默认训练循环不够，再考虑改 `learn_one_task()`。

每加一层，都先跑同一个单任务 smoke test：

```text
LIBERO_ALPHABET_SOUP
train.n_epochs=1
train.batch_size=32
train.num_workers=0
eval.eval=false
eval.eval_every=999
```

确认不破坏基本训练后，再跑 rollout。

### 9.7 写 algo 时要避免的坑

不要一开始重写整个 `learn_one_task()`。默认实现已经处理了很多细节：

```text
optimizer
scheduler
checkpoint
rollout eval
best model reload
result summary
```

不要在 dataloader 上随意打开 multiprocessing。当前 HDF5 / h5py dataset 和 multiprocessing worker 不兼容，调试阶段固定用：

```text
train.num_workers=0
eval.num_workers=0
```

如果复用 `Sequential.learn_one_task()`，确保 `persistent_workers` 不会在 `num_workers=0` 时开启。

不要只看 training loss。最终还是要跑 rollout success：

```text
policy 在 LIBERO 环境里是否完成任务
```

不要把 single-task smoke test 的 success rate 当成算法结论。`train.n_epochs=1`、`eval.n_eval=2` 只能说明 pipeline 是否工作，不能说明算法好坏。

### 9.8 当前推荐工作方式

下一步实现 single-task algo 时，采用这个节奏：

```text
1. 写最小 MySingleTaskAlgo
2. 注册并确认 Available algorithms 里能看到
3. 用 LIBERO_ALPHABET_SOUP 跑 no-rollout smoke test
4. 跑少量 rollout smoke test
5. 加一个最小单任务算法特性
6. 重复 smoke test
7. 最后再扩大 epochs、batch size、n_eval
```

这个流程可以把问题分开：

```text
注册问题
Hydra config 问题
DataLoader 问题
policy loss 问题
rollout env 问题
单任务算法逻辑问题
```

每次只改一层，出错时更容易定位。
