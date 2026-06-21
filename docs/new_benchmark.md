# 新建单任务 Benchmark 工作流

这份文档记录如何从已有 LIBERO benchmark 中抽出一个任务，注册成新的单任务 benchmark。这样下次换任务时，只需要告诉 Codex 任务名或 HDF5 文件路径，就可以按同一流程替换 benchmark。

## 目标

把已有 benchmark 里的一个任务，例如：

```text
libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket
```

包装成一个新的单任务 benchmark：

```text
LIBERO_ALPHABET_SOUP
```

训练和评估时只跑这一个任务：

```text
1 task dataset
 -> train one policy
 -> rollout only this task
 -> report success rate
```

这个方式不会复制 BDDL、dataset 或 init states。新的 benchmark 只是引用已有任务资产。

## 为什么这样做

`libero/lifelong/main.py`、`evaluate_success`、task embedding、init state loading 都围绕 benchmark API 写好了。

如果直接写独立训练脚本，需要重新处理：

- task language
- BDDL 路径
- HDF5 路径
- init states 路径
- rollout environment
- result summary shape
- checkpoint naming

注册单任务 benchmark 更干净，因为可以继续使用原本的训练和评估管线。

## Benchmark 需要提供什么

一个完整 LIBERO benchmark 至少需要：

| 内容 | 用途 |
| --- | --- |
| task name | 用来定位 dataset、BDDL、init states |
| task language | 用来生成 task embedding |
| problem folder | 例如 `libero_object` |
| BDDL file | rollout 时创建环境 |
| HDF5 demo file | behavior cloning 训练数据 |
| pruned init states | rollout evaluation 的固定初始状态 |

对于已有 LIBERO 任务，这些通常已经存在。

## 相关代码文件

主要修改：

```text
libero/libero/benchmark/__init__.py
```

如果要支持 standalone evaluation，还要检查：

```text
libero/lifelong/evaluate.py
```

任务列表来源：

```text
libero/libero/benchmark/libero_suite_task_map.py
```

已有 BDDL：

```text
libero/libero/bddl_files/<suite>/<task>.bddl
```

已有 init states：

```text
libero/libero/init_files/<suite>/<task>.pruned_init
```

已有 dataset：

```text
<dataset_root>/<suite>/<task>_demo.hdf5
```

例如：

```text
/scratch/prj/eng_demt_robot_learning/dataset/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
```

## 当前已实现的例子

当前已经注册：

```text
LIBERO_ALPHABET_SOUP
```

它只包含一个已有任务：

```text
pick_up_the_alphabet_soup_and_place_it_in_the_basket
```

在代码里：

```python
@register_benchmark
class LIBERO_ALPHABET_SOUP(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_alphabet_soup"
        self.tasks = [
            task_maps["libero_object"][
                "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
            ]
        ]
        self.n_tasks = len(self.tasks)
```

它复用：

| 资源 | 路径 |
| --- | --- |
| Dataset | `libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5` |
| BDDL | `libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl` |
| Init states | `libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.pruned_init` |

## 下次换任务时需要给 Codex 什么

最方便的是给 HDF5 dataset 路径，例如：

```text
/scratch/prj/eng_demt_robot_learning/dataset/datasets/libero_object/pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5
```

Codex 可以从路径推断：

```text
suite = libero_object
task = pick_up_the_milk_and_place_it_in_the_basket
```

也可以直接给：

```text
suite: libero_object
task: pick_up_the_milk_and_place_it_in_the_basket
new benchmark name: LIBERO_MILK
```

推荐提供：

| 信息 | 示例 |
| --- | --- |
| suite | `libero_object` |
| task name | `pick_up_the_milk_and_place_it_in_the_basket` |
| benchmark class name | `LIBERO_MILK` |
| dataset root | `/scratch/prj/eng_demt_robot_learning/dataset/datasets` |

## Codex 下次应执行的检查

换任务时，先检查这几项。

### 1. 任务是否在 task map 里

```shell
rg -n "pick_up_the_milk_and_place_it_in_the_basket" \
  libero/libero/benchmark/libero_suite_task_map.py
```

如果找不到，说明这个任务没有注册在已有 LIBERO suite 里，需要先添加到 `libero_suite_task_map.py` 或确认 task name 是否写错。

### 2. BDDL 是否存在

```shell
ls libero/libero/bddl_files/libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl
```

如果 BDDL 不存在，rollout evaluation 无法创建环境。

### 3. Init states 是否存在

```shell
ls libero/libero/init_files/libero_object/pick_up_the_milk_and_place_it_in_the_basket.pruned_init
```

如果 `.pruned_init` 不存在，训练可以读 HDF5，但 rollout evaluation 会失败。

### 4. Dataset 是否存在

```shell
ls /scratch/prj/eng_demt_robot_learning/dataset/datasets/libero_object/pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5
```

如果 dataset 不存在，需要先下载或生成 HDF5。

### 5. Dataset 结构是否匹配默认 policy

```shell
python scripts/get_dataset_info.py \
  --dataset /scratch/prj/eng_demt_robot_learning/dataset/datasets/libero_object/pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5
```

默认 policy 需要：

```text
obs/agentview_rgb
obs/eye_in_hand_rgb
obs/gripper_states
obs/joint_states
actions
```

## 修改 benchmark registry

在 `libero/libero/benchmark/__init__.py` 添加新的 class。

模板：

```python
@register_benchmark
class LIBERO_MY_TASK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_my_task"
        self.tasks = [
            task_maps["<suite>"][
                "<task_name>"
            ]
        ]
        self.n_tasks = len(self.tasks)
```

例子：

```python
@register_benchmark
class LIBERO_MILK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_milk"
        self.tasks = [
            task_maps["libero_object"][
                "pick_up_the_milk_and_place_it_in_the_basket"
            ]
        ]
        self.n_tasks = len(self.tasks)
```

注意：

- class name 使用大写，例如 `LIBERO_MILK`。
- `self.name` 使用小写，例如 `libero_milk`。
- `task_maps["<suite>"]["<task_name>"]` 必须存在。
- 不要调用 `_make_benchmark()`，因为默认 `_make_benchmark()` 会套用 10-task ordering。
- 单任务 benchmark 应该直接设置 `self.tasks` 和 `self.n_tasks`。

## 修改 standalone evaluate.py

如果需要使用：

```shell
python libero/lifelong/evaluate.py ...
```

需要在 `libero/lifelong/evaluate.py` 里增加映射。

例如：

```python
benchmark_map = {
    ...
    "libero_milk": "LIBERO_MILK",
}
```

同时在 argparse choices 里加入：

```python
"libero_milk",
```

如果 evaluate 脚本对 task 数量有硬编码，比如：

```python
range(10)
```

要改成根据 benchmark 的 `n_tasks` 处理，或者为单任务 benchmark 限制：

```text
task_id = 0
load_task = 0
```

当前 `evaluate.py` 已经为 `libero_alphabet_soup` 做了兼容。

## 训练命令模板

Smoke test：

```shell
python libero/lifelong/main.py \
  benchmark_name=LIBERO_MY_TASK \
  policy=bc_transformer_policy \
  lifelong=base \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets \
  train.n_epochs=1 \
  train.batch_size=2 \
  train.num_workers=0 \
  eval.eval=false
```

正式训练和 rollout：

```shell
python libero/lifelong/main.py \
  benchmark_name=LIBERO_MY_TASK \
  policy=bc_transformer_policy \
  lifelong=base \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets \
  train.n_epochs=50 \
  train.batch_size=32 \
  eval.eval=true
```

对于单任务 benchmark：

```text
task_id = 0
load_task = 0
```

## 当前 alphabet soup 命令

Smoke test：

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

正式训练：

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

## 验证修改

至少运行：

```shell
python -m py_compile \
  libero/libero/benchmark/__init__.py \
  libero/lifelong/evaluate.py
```

如果当前 Python 环境有依赖，可以进一步检查：

```shell
LIBERO_CONFIG_PATH=/tmp/libero_config_test python - <<'PY'
from libero.libero.benchmark import get_benchmark

bench = get_benchmark("LIBERO_ALPHABET_SOUP")()
print(bench.n_tasks)
print(bench.get_task_names())
print(bench.get_task_demonstration(0))
print(bench.get_task_bddl_file_path(0))
print(bench.get_task(0).init_states_file)
print(bench.get_task(0).language)
PY
```

期望：

```text
n_tasks = 1
demo path = <suite>/<task>_demo.hdf5
bddl path = <suite>/<task>.bddl
init states = <task>.pruned_init
```

## 常见错误

### 1. task name 和文件名不一致

`benchmark.get_task_demonstration(i)` 会生成：

```text
<problem_folder>/<task_name>_demo.hdf5
```

所以 task name 必须和 dataset 文件名去掉 `_demo.hdf5` 后一致。

### 2. BDDL 存在但 init states 不存在

训练可以开始，但 rollout evaluation 会在加载 `.pruned_init` 时失败。

### 3. 单任务 benchmark 调用了 `_make_benchmark()`

不要这样做。默认 `_make_benchmark()` 会用 10-task order：

```python
self.tasks = [tasks[i] for i in task_orders[self.task_order_index]]
```

单任务 suite 只有一个任务，会 index out of range。

### 4. evaluate.py 假设 10 tasks

如果 standalone evaluate 用 `range(10)` 或 `Sequential(10, cfg)`，单任务 benchmark 会不匹配。应该根据 `benchmark.n_tasks` 创建 algo 和 task embeddings。

### 5. dataset root 设置错

训练命令里的：

```shell
folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets
```

必须是 suite 文件夹的上一级。

也就是说实际 dataset 应该在：

```text
<folder>/<suite>/<task>_demo.hdf5
```

## 下次让 Codex 换任务时的推荐请求

可以直接这样说：

```text
请把单任务 benchmark 从 LIBERO_ALPHABET_SOUP 换成 milk 任务：
dataset=/scratch/prj/eng_demt_robot_learning/dataset/datasets/libero_object/pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5
new benchmark name=LIBERO_MILK
并更新 docs/new_benchmark.md 里的当前命令。
```

Codex 应该执行：

1. 从 dataset path 推断 suite 和 task name。
2. 检查 task map、BDDL、init states、dataset。
3. 修改 `libero/libero/benchmark/__init__.py`。
4. 如需 standalone evaluate，修改 `libero/lifelong/evaluate.py`。
5. 更新文档里的命令。
6. 运行 `py_compile`。
7. 如果依赖可用，运行 benchmark import smoke test。
