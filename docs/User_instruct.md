# LIBERO 手动采集一条数据用户手册

本文档说明如何在当前仓库中，手动采集一条类似下面任务的数据：

```text
pick_up_the_alphabet_soup_and_place_it_in_the_basket
```

对应目标是：用机器人夹起 `alphabet_soup`，放进 `basket`。

## 1. 当前环境

仓库路径：

```bash
/home/endongsun/robot_learning
```

LIBERO 代码路径：

```bash
/home/endongsun/robot_learning/LIBERO
```

已有数据集路径：

```bash
/home/endongsun/robot_learning/datasets/libero_object/
```

可用 conda 环境：

```bash
libero_cpu
```

当前机器上 `base` 环境没有 LIBERO 依赖，所以不要直接用 `python` 跑 LIBERO。使用：

```bash
conda run -n libero_cpu ...
```

或：

```bash
conda activate libero_cpu
```

## 2. 采集前检查

先进入 LIBERO 目录：

```bash
cd /home/endongsun/robot_learning/LIBERO
```

检查 BDDL 任务文件是否存在：

```bash
ls libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl
```

这个文件定义了任务本身，包括语言指令、物体、初始位置区域、目标条件。

检查当前 shell 是否能显示 GUI：

```bash
echo $DISPLAY
```

正常情况下应看到类似：

```text
:0
```

如果没有 `DISPLAY`，交互式 viewer 可能无法打开。

## 3. 重要环境变量

当前 `libero_cpu` 环境导入 robosuite 时会遇到 numba cache 问题。运行采集脚本时必须带上：

```bash
NUMBA_DISABLE_JIT=1
```

建议同时设置 matplotlib 缓存目录：

```bash
MPLCONFIGDIR=/tmp/matplotlib_libero
```

完整命令中会写成：

```bash
env NUMBA_DISABLE_JIT=1 MPLCONFIGDIR=/tmp/matplotlib_libero ...
```

## 4. 安装过的额外依赖

键盘遥操作需要 `pynput`。当前已经装入 `libero_cpu`：

```bash
conda run -n libero_cpu pip install pynput
```

如果以后换环境，缺这个包时会报：

```text
ModuleNotFoundError: No module named 'pynput'
```

重新安装即可。

## 5. 本仓库已做的兼容补丁

当前 robosuite 的 OpenCV viewer 与 LIBERO 原采集脚本的键盘 callback API 不完全一致。

已修改：

```text
LIBERO/scripts/collect_demonstration.py
LIBERO/scripts/libero_100_collect_demonstrations.py
```

修改目的：当 viewer 没有 `add_keyup_callback` 时，不再注册旧版 callback，改为依赖 `pynput` 的全局键盘监听。

如果以后重新 clone LIBERO，可能需要再次做这个补丁，否则采集时可能报：

```text
TypeError: add_keypress_callback() takes 2 positional arguments but 3 were given
```

## 6. 启动采集一条数据

使用键盘采集一条 demonstration：

```bash
cd /home/endongsun/robot_learning/LIBERO

conda run -n libero_cpu env \
  NUMBA_DISABLE_JIT=1 \
  MPLCONFIGDIR=/tmp/matplotlib_libero \
  python scripts/collect_demonstration.py \
  --bddl-file /home/endongsun/robot_learning/LIBERO/libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl \
  --directory /home/endongsun/robot_learning/LIBERO/demonstration_data \
  --robots Panda \
  --controller OSC_POSE \
  --device keyboard \
  --num-demonstration 1
```

启动后会出现一个 viewer 窗口。终端里会打印任务语言：

```text
pick the alphabet soup and place it in the basket
```

并打印键盘控制说明。

## 7. 键盘控制

当前 robosuite keyboard 控制如下：

```text
w / s      沿 x 方向移动末端
a / d      沿 y 方向移动末端
r / f      上升 / 下降末端
space      开合夹爪
z / x      绕 x 轴旋转末端
t / g      绕 y 轴旋转末端
c / v      绕 z 轴旋转末端
q          放弃当前 episode 并重置
ESC        退出
```

实际操作建议：

1. 点一下 viewer 窗口，确保键盘焦点在窗口或当前 X session。
2. 用 `w/s/a/d` 把夹爪移动到 alphabet soup 上方。
3. 用 `f` 缓慢下降到可夹取高度。
4. 按 `space` 关闭夹爪。
5. 用 `r` 抬高物体。
6. 用 `w/s/a/d` 移动到 basket 上方。
7. 用 `f` 降低到 basket 内或 basket 上方。
8. 按 `space` 打开夹爪释放物体。
9. 等待几帧，让环境检测成功。

如果动作做坏了，按 `q` 重来。失败轨迹不会作为成功数据保存。

## 8. 成功保存的判定

采集脚本内部会检查：

```python
env._check_success()
```

只有任务成功并保持短暂时间后，脚本才会保存这一条。

保存成功后，`--num-demonstration 1` 会满足，脚本会结束。

原始 demonstration 会保存在类似目录：

```text
/home/endongsun/robot_learning/LIBERO/demonstration_data/
```

里面会有一个带时间戳的子目录，子目录中包含：

```text
demo.hdf5
```

注意：这个是 raw demo 文件，主要包含 simulator states 和 actions，还不是最终训练用的大 HDF5。

## 9. 查找刚采到的 raw demo

采集结束后运行：

```bash
find /home/endongsun/robot_learning/LIBERO/demonstration_data \
  -name demo.hdf5 \
  -printf "%T@ %p\n" | sort -n | tail -5
```

最后一行通常就是最新采集的一条 raw demo。

## 10. 转换成训练 HDF5

假设最新 raw 文件是：

```text
/home/endongsun/robot_learning/LIBERO/demonstration_data/<some_folder>/demo.hdf5
```

转换命令：

```bash
cd /home/endongsun/robot_learning/LIBERO

conda run -n libero_cpu env \
  PYTHONUNBUFFERED=1 \
  NUMBA_DISABLE_JIT=1 \
  MPLCONFIGDIR=/tmp/matplotlib_libero \
  python scripts/create_dataset.py \
  --demo-file /home/endongsun/robot_learning/LIBERO/demonstration_data/<some_folder>/demo.hdf5 \
  --use-camera-obs \
  --output-file /home/endongsun/robot_learning/datasets_manual/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_manual_demo.hdf5
```

转换后会生成训练用 HDF5。手动采集的数据建议放在：

```text
/home/endongsun/robot_learning/datasets_manual/
```

不要把手动采集数据写到：

```text
/home/endongsun/robot_learning/datasets/libero_object/
```

这个目录是原始 LIBERO benchmark 数据目录。直接使用默认输出路径可能覆盖下载好的 benchmark HDF5。

如果转换命令还在运行，不要同时运行 `analyze_trajectories.py` 或其他 HDF5 读取脚本。否则可能出现：

```text
BlockingIOError: unable to lock file
```

这表示 HDF5 文件仍被写入进程占用。等待转换命令自然结束后再分析。

## 11. 用脚本自动采集 alphabet soup 数据

如果不想用键盘人工采集，可以先用当前仓库里的 scripted collector：

```text
/home/endongsun/robot_learning/LIBERO/scripts/collect_scripted_alphabet_soup.py
```

这个脚本只针对这个任务：

```text
pick_up_the_alphabet_soup_and_place_it_in_the_basket
```

它会根据 BDDL 初始化场景，读取 `alphabet_soup_1` 和 `basket_1` 的位置，然后用 waypoint controller 自动执行：

```text
接近 alphabet soup -> 下降 -> 闭合夹爪 -> 抬起 -> 移动到 basket -> 下降 -> 松开 -> 撤离
```

采集 raw demo：

```bash
cd /home/endongsun/robot_learning/LIBERO

conda run -n libero_cpu env \
  PYTHONUNBUFFERED=1 \
  NUMBA_DISABLE_JIT=1 \
  MPLCONFIGDIR=/tmp/matplotlib_libero \
  python scripts/collect_scripted_alphabet_soup.py \
  --output /home/endongsun/robot_learning/LIBERO/demonstration_data/scripted_alphabet_soup/demo.hdf5 \
  --num-demonstration 1 \
  --overwrite \
  --verbose
```

成功时会看到类似：

```text
[attempt 1] success=True steps=301 collected=0/1
[done] saved 1 scripted demos to ...
```

然后把 scripted raw demo 转换成训练 HDF5：

```bash
cd /home/endongsun/robot_learning/LIBERO

conda run -n libero_cpu env \
  PYTHONUNBUFFERED=1 \
  NUMBA_DISABLE_JIT=1 \
  MPLCONFIGDIR=/tmp/matplotlib_libero \
  python scripts/create_dataset.py \
  --demo-file /home/endongsun/robot_learning/LIBERO/demonstration_data/scripted_alphabet_soup/demo.hdf5 \
  --use-camera-obs \
  --output-file /home/endongsun/robot_learning/datasets_manual/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_scripted_demo.hdf5 \
  --overwrite
```

当前已经验证过一个 scripted 数据集：

```text
/home/endongsun/robot_learning/datasets_manual/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_scripted_demo.hdf5
```

它包含：

```text
data/demo_0/actions        (296, 7)
data/demo_0/states         (296, 110)
data/demo_0/obs/ee_pos     (296, 3)
data/demo_0/obs/agentview_rgb
data/demo_0/obs/eye_in_hand_rgb
```

注意：这是一个 task-specific scripted policy，不是 LIBERO 所有任务的通用 oracle。换任务后通常要重新写 waypoint 或使用更强的 planner。

## 12. 最终 HDF5 应包含什么

最终训练 HDF5 通常包含：

```text
data/
  demo_0/
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

其中：

- `actions` 是控制动作；
- `states` 是 MuJoCo simulator state；
- `obs/agentview_rgb` 是第三人称相机图像；
- `obs/eye_in_hand_rgb` 是腕部相机图像；
- `rewards[-1]` 应该为 `1`；
- `dones[-1]` 应该为 `1`。

## 13. 检查数据集信息

使用：

```bash
cd /home/endongsun/robot_learning

conda run -n libero_cpu python LIBERO/scripts/get_dataset_info.py \
  --dataset datasets_manual/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_manual_demo.hdf5
```

它会打印：

- trajectory 数量；
- transition 总数；
- 每条 trajectory 长度；
- action 范围；
- language instruction；
- HDF5 内部结构。

## 14. 回放检查

用现有 replay 脚本检查 demo 是否能成功：

```bash
cd /home/endongsun/robot_learning

conda run -n libero_cpu env \
  NUMBA_DISABLE_JIT=1 \
  MPLCONFIGDIR=/tmp/matplotlib_libero \
  python LIBERO/scripts/replay_libero_demo.py \
  --dataset datasets_manual/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_manual_demo.hdf5 \
  --suite libero_object \
  --task-id 0 \
  --demo-id 0 \
  --save-video
```

默认 `libero_object` 顺序下，alphabet soup 是 `task-id 0`。

成功时终端会显示类似：

```text
success: True
```

如果保存视频，会生成一个 replay mp4。

## 15. 画末端轨迹图

注意：只能对转换后的训练 HDF5 画 `ee_pos` 轨迹。raw demo：

```text
/home/endongsun/robot_learning/LIBERO/demonstration_data/<some_folder>/demo.hdf5
```

通常只有：

```text
states
actions
```

没有：

```text
obs/ee_pos
```

所以 raw demo 不能直接用 `scripts/analyze_trajectories.py` 画末端轨迹。必须先完成第 10 步转换。

转换后的 HDF5 里应该有：

```text
data/demo_0/obs/ee_pos
```

画图命令：

```bash
cd /home/endongsun/robot_learning/LIBERO

conda run -n libero_cpu env \
  MPLCONFIGDIR=/tmp/matplotlib_libero \
  python scripts/analyze_trajectories.py \
  --dataset /home/endongsun/robot_learning/datasets_manual/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_manual_demo.hdf5 \
  --output /home/endongsun/robot_learning/datasets_manual/libero_object/alphabet_soup_manual_eef_trajectories.png
```

输出文件：

```text
/home/endongsun/robot_learning/datasets_manual/libero_object/alphabet_soup_manual_eef_trajectories.png
```

这个图包含：

- 左侧：3D end-effector trajectory；
- 右侧：XY top-down projection；
- 绿色点：轨迹起点；
- 红色星号：轨迹终点。

脚本还会打印统计量，例如：

```text
Total trajectories amount
Trajectory length
Start EEF Position Bounds
End EEF Position Bounds
Gripper States Range
Action Limits
```

如果出现：

```text
BlockingIOError: unable to lock file
```

说明 HDF5 仍在被 `create_dataset.py` 写入。等待转换命令完全结束后再画图。

如果出现：

```text
KeyError: obs/ee_pos
```

说明你传入的是 raw demo，或者转换时没有生成 proprioception。请使用 `datasets_manual/...manual_demo.hdf5` 这种转换后的文件。

当前已经成功画过的文件是：

```text
/home/endongsun/robot_learning/datasets_manual/libero_object/alphabet_soup_manual_eef_trajectories.png
```

对应数据集：

```text
/home/endongsun/robot_learning/datasets_manual/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_manual_demo.hdf5
```

里面有两条轨迹：

```text
demo_0: 381 steps
demo_1: 426 steps
```

## 16. 常见问题

### 15.1 没有 pynput

报错：

```text
ModuleNotFoundError: No module named 'pynput'
```

解决：

```bash
conda run -n libero_cpu pip install pynput
```

### 15.2 numba cache 报错

报错包含：

```text
RuntimeError: cannot cache function 'mat2quat'
```

解决：命令里加：

```bash
NUMBA_DISABLE_JIT=1
```

### 15.3 keyboard callback API 报错

报错：

```text
TypeError: add_keypress_callback() takes 2 positional arguments but 3 were given
```

解决：确认本仓库的采集脚本包含兼容补丁。当前仓库已经修改过：

```text
LIBERO/scripts/collect_demonstration.py
LIBERO/scripts/libero_100_collect_demonstrations.py
```

### 15.4 viewer 打不开

先检查：

```bash
echo $DISPLAY
```

如果为空，需要在有图形界面的终端运行，或配置 X11 转发/远程桌面。

### 15.5 键盘没反应

先点一下 viewer 窗口。  
如果仍然没反应，确认 `pynput` 已安装，并且当前系统允许程序监听键盘。

### 15.6 放错或夹取失败

按：

```text
q
```

重置当前 episode，重新做。不要用失败轨迹训练。

## 17. 一次完整流程总结

最短流程：

```bash
cd /home/endongsun/robot_learning/LIBERO

conda run -n libero_cpu env \
  NUMBA_DISABLE_JIT=1 \
  MPLCONFIGDIR=/tmp/matplotlib_libero \
  python scripts/collect_demonstration.py \
  --bddl-file /home/endongsun/robot_learning/LIBERO/libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl \
  --directory /home/endongsun/robot_learning/LIBERO/demonstration_data \
  --robots Panda \
  --controller OSC_POSE \
  --device keyboard \
  --num-demonstration 1
```

操作成功后，查找 raw demo：

```bash
find /home/endongsun/robot_learning/LIBERO/demonstration_data \
  -name demo.hdf5 \
  -printf "%T@ %p\n" | sort -n | tail -1
```

然后把 raw demo 转换成训练数据：

```bash
conda run -n libero_cpu env \
  PYTHONUNBUFFERED=1 \
  NUMBA_DISABLE_JIT=1 \
  MPLCONFIGDIR=/tmp/matplotlib_libero \
  python scripts/create_dataset.py \
  --demo-file <刚找到的 demo.hdf5> \
  --use-camera-obs \
  --output-file /home/endongsun/robot_learning/datasets_manual/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_manual_demo.hdf5
```

然后画末端轨迹：

```bash
conda run -n libero_cpu env \
  MPLCONFIGDIR=/tmp/matplotlib_libero \
  python scripts/analyze_trajectories.py \
  --dataset /home/endongsun/robot_learning/datasets_manual/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_manual_demo.hdf5 \
  --output /home/endongsun/robot_learning/datasets_manual/libero_object/alphabet_soup_manual_eef_trajectories.png
```

最后用 replay 检查是否成功。

## 18. 给 DEMT 实验的注意事项

如果你要研究 demonstration guidance，不要只看是否成功。每条 demonstration 都应成功，但还要记录或控制：

- 接近路径；
- 接触姿态；
- 夹爪闭合时机；
- 抬升时机；
- 释放位置；
- 释放高度；
- 速度曲线；
- 动作是否平滑。

做对照实验时，保持：

```text
同一个 BDDL 任务
同一组 init-state 覆盖
所有 demonstration 都成功
只改变一个教学结构因素
```

这样才能判断 learner performance 的变化来自 demonstration structure，而不是任务或初始状态分布改变。
