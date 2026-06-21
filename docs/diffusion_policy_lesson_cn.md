# LIBERO Diffusion Policy 代码解构教案

这份教案用于解释当前仓库新增的 Diffusion Policy 实现。目标不是复述论文，而是把 Diffusion Policy 的核心原理映射到我们已经改过的 LIBERO 代码上，让读者能从训练命令一路追到模型的 `compute_loss()` 和 `get_action()`。

## 1. 本节课目标

学完后应该能回答四个问题：

1. Diffusion Policy 在机器人控制里到底预测什么。
2. LIBERO 的 HDF5 序列数据如何变成条件扩散模型的训练样本。
3. 我们新增的 `DiffusionPolicy` 代码里，视觉编码器、1D U-Net、scheduler、EMA 分别承担什么职责。
4. 为什么训练用 DDPM，推理默认改成 DDIM 10 step，以及这会怎样影响 rollout 速度。

相关文件：

| 文件 | 作用 |
| --- | --- |
| `libero/lifelong/models/diffusion_policy.py` | 新增的 Diffusion Policy 主实现 |
| `libero/configs/policy/diffusion_policy.yaml` | policy 超参数和默认 scheduler 配置 |
| `libero/lifelong/models/__init__.py` | 注册 `DiffusionPolicy`，使 Hydra 可以通过 `policy=diffusion_policy` 加载 |
| `libero/lifelong/algos/base.py` | 训练循环，加入 EMA 更新和 `eval.eval=false` 跳过 rollout 的逻辑 |
| `libero/libero/benchmark/__init__.py` | 新增 `LIBERO_ALPHABET_SOUP` 单任务 benchmark |
| `libero/lifelong/evaluate.py` | standalone evaluation 适配单任务 benchmark |

## 2. Diffusion Policy 的一句话理解

传统 behavior cloning 通常直接学：

```text
observation -> action
```

Diffusion Policy 学的是：

```text
observation condition + noisy action sequence + diffusion timestep -> action noise
```

训练时，我们从专家动作序列 `a_0` 出发，随机采样一个扩散步 `t`，给动作序列加噪声得到 `a_t`。模型看到当前观测条件、带噪动作序列 `a_t` 和时间步 `t`，学习预测当初加进去的噪声 `epsilon`。

推理时，没有专家动作。模型先随机生成一段纯噪声动作计划，然后反复去噪，得到一段可执行的动作序列。

核心训练目标是：

```text
loss = MSE(predicted_noise, true_noise)
```

对应代码在 `DiffusionPolicy.compute_loss()`：

```python
action = self._training_actions(data)
noise = torch.randn_like(action)
timesteps = torch.randint(...)
noisy_action = self.train_noise_scheduler.add_noise(action, noise, timesteps)
noise_pred = self.noise_pred_net(noisy_action, timesteps, global_cond=self._obs_cond(data))
return F.mse_loss(noise_pred, noise)
```

## 3. 数据在本实现里的形状

LIBERO 的 `SequenceDataset` 输出的是一段时间序列。当前 DP 默认配置是：

```yaml
obs_horizon: 2
pred_horizon: 16
action_horizon: 8
data.seq_len: 16
```

可以把它理解成：

| 名称 | 含义 | 当前默认 |
| --- | --- | --- |
| `obs_horizon` | 用多少帧历史观测作为条件 | 2 |
| `pred_horizon` | 一次生成多长的动作计划 | 16 |
| `action_horizon` | 每次实际执行多少个动作，然后重新规划 | 8 |
| `data.seq_len` | 数据集中采样的序列长度 | 16 |

一个训练 batch 里，动作大致是：

```text
actions: (B, 16, action_dim)
```

在 alphabet soup 任务里，`action_dim` 来自 `shape_meta["ac_dim"]`，通常是 7。

观测包含两类：

```text
RGB:
  agentview_rgb
  eye_in_hand_rgb

Low-dimensional:
  gripper_states
  joint_states
```

视觉输入进入 ResNet18 编码器，低维输入直接拼接。最后取前 `obs_horizon=2` 帧作为条件，flatten 成一个全局条件向量。

对应代码：

```python
obs_features = self._encode_obs(data)
obs_features = obs_features[:, : self.obs_horizon]
global_cond = obs_features.flatten(start_dim=1)
```

## 4. 模型结构拆解

当前 `DiffusionPolicy` 可以拆成三层：

```text
LIBERO batch
  -> observation encoder
  -> conditional 1D U-Net
  -> diffusion scheduler
```

### 4.1 视觉编码器

每个 RGB 相机使用一个 ResNet18：

```python
self.image_encoders[name] = ResNet18Encoder(
    output_dim=policy_cfg.vision_feature_dim,
    pretrained=policy_cfg.pretrained_resnet,
)
```

默认 `vision_feature_dim=512`，两个相机就产生 `512 * 2` 维视觉特征。代码还把 ResNet18 里的 BatchNorm 替换成 GroupNorm：

```python
replace_bn_with_gn(resnet)
```

原因是机器人训练常用 batch size 较小，BatchNorm 统计不稳定；GroupNorm 不依赖 batch 维度，通常更稳。

### 4.2 低维状态拼接

如果配置打开：

```python
cfg.data.use_gripper
cfg.data.use_joint
cfg.data.use_ee
```

模型会把 `gripper_states`、`joint_states`、可选 `ee_states` 拼到视觉特征后面。当前主要使用 gripper 和 joint。

### 4.3 Conditional 1D U-Net

动作序列不是图像，所以这里不是 2D U-Net，而是沿时间维做卷积的 1D U-Net：

```python
self.noise_pred_net = ConditionalUnet1D(
    input_dim=self.action_dim,
    global_cond_dim=obs_dim * self.obs_horizon,
    down_dims=(256, 512, 1024),
)
```

输入形状可以理解为：

```text
noisy_action: (B, pred_horizon, action_dim)
```

进入 U-Net 前会把 action_dim 移到 channel 维：

```python
x = sample.moveaxis(-1, -2)
```

也就是：

```text
(B, T, action_dim) -> (B, action_dim, T)
```

这样 `Conv1d` 就可以沿时间维处理整段动作计划。

### 4.4 条件注入

U-Net 每个 residual block 都会接收一个条件向量：

```text
condition = diffusion timestep embedding + observation embedding
```

代码里先对 diffusion timestep 做 sinusoidal embedding：

```python
self.diffusion_step_encoder(timestep)
```

再与观测条件拼接：

```python
global_feature = torch.cat(
    [self.diffusion_step_encoder(timestep), global_cond], dim=-1
)
```

每个 `ConditionalResidualBlock1D` 会把这个条件向量变成 scale 和 bias：

```python
embed = self.cond_encoder(cond).reshape(B, 2, out_channels, 1)
out = embed[:, 0] * out + embed[:, 1]
```

这相当于告诉 U-Net：在这个观测条件下、这个扩散步上，应该怎样修正当前带噪动作。

## 5. 训练流程

训练入口仍然是：

```bash
python libero/lifelong/main.py \
  benchmark_name=LIBERO_ALPHABET_SOUP \
  policy=diffusion_policy \
  lifelong=base \
  folder=/scratch/prj/eng_demt_robot_learning/dataset/datasets \
  data.seq_len=16 \
  train.n_epochs=1 \
  train.batch_size=16 \
  train.num_workers=0
```

调用路径是：

```text
main.py
  -> Sequential.learn_one_task()
  -> Sequential.observe()
  -> DiffusionPolicy.compute_loss()
```

`Sequential.observe()` 负责标准训练步骤：

```python
self.optimizer.zero_grad()
loss = self.policy.compute_loss(data)
loss.backward()
self.optimizer.step()
self.policy.update_ema()
```

我们对 `base.py` 的关键改动是：如果 policy 有 `update_ema()`，每次 optimizer step 后调用它。

EMA 的作用是维护一份更平滑的模型参数。训练时用实时参数更新，推理时临时切换到 EMA 参数：

```python
with torch.no_grad(), self.ema_scope():
    ...
```

## 6. 训练为什么要求 action 在 [-1, 1]

Diffusion scheduler 的 `clip_sample=true` 和噪声建模都默认数据尺度比较规整。当前实现明确检查第一批训练动作：

```python
in_range = ((action >= -1.0) & (action <= 1.0)).all().item()
```

如果动作超出范围，会报错提示：

```text
DiffusionPolicy expects normalized actions in [-1, 1]
```

这个检查很重要，因为扩散模型不是简单回归。动作尺度不对时，模型学习的是错误噪声分布，rollout 中生成的动作也会不稳定。

如果只是临时 debug，可以设置：

```bash
policy.enforce_action_bounds=false
```

正式训练不建议关闭。

## 7. 推理流程

rollout 时环境每一步都会调用：

```python
DiffusionPolicy.get_action(data)
```

推理逻辑分三步。

第一步，维护观测队列：

```python
self._append_obs(data)
```

一开始队列不够长时，会复制当前观测直到达到 `obs_horizon`。

第二步，如果 action queue 空了，就重新采样一段动作计划：

```python
self.action_queue = self._sample_action_plan(self._stack_obs_queue())
```

第三步，每次只吐出队列里的第一个动作：

```python
action = self.action_queue[:, 0]
self.action_queue = self.action_queue[:, 1:]
```

所以 Diffusion Policy 不是每个环境 step 都完整去噪一次，而是每次规划 `action_horizon=8` 个动作，逐步执行，队列用完再规划。

## 8. 从噪声生成动作计划

`_sample_action_plan()` 是推理阶段的核心：

```python
action = torch.randn(b, self.pred_horizon, self.action_dim, device=self.cfg.device)
self.inference_noise_scheduler.set_timesteps(self.num_inference_iters)
for timestep in self.inference_noise_scheduler.timesteps.to(action.device):
    noise_pred = self.noise_pred_net(action, timestep.expand(b), global_cond=self._obs_cond(data))
    action = self.inference_noise_scheduler.step(noise_pred, timestep, action).prev_sample
```

它做的事是：

1. 从纯高斯噪声初始化动作序列。
2. 按 scheduler 给出的时间步倒序迭代。
3. 每一步用 U-Net 预测噪声。
4. scheduler 根据预测噪声更新动作序列。
5. 最后截取可执行部分：

```python
start = self.obs_horizon - 1
end = start + self.action_horizon
return action[:, start:end].clamp(-1, 1)
```

这里从 `obs_horizon - 1` 开始，是因为前几帧观测对应的动作上下文已经在历史里，真正要执行的是从当前时刻对齐后的未来动作。

## 9. 为什么训练用 DDPM，推理默认用 DDIM

当前配置：

```yaml
num_train_timesteps: 100
inference_scheduler: ddim
num_inference_iters: 10
```

训练仍然使用 `DDPMScheduler`：

```python
self.train_noise_scheduler = DDPMScheduler(...)
```

原因是训练只需要 `add_noise()`，DDPM 的训练目标稳定、标准，和 Diffusion Policy notebook 保持一致。

推理默认使用 `DDIMScheduler`：

```python
self.inference_noise_scheduler = DDIMScheduler(...)
```

原因是 rollout 里每次规划都要多次调用 U-Net。DDPM 100 step 会很慢，DDIM 可以用更少步数生成动作计划。当前默认 `num_inference_iters=10`，用于快速 smoke test 和初步训练验证。

如果要和旧行为对比，可以覆盖为：

```bash
policy.inference_scheduler=ddpm policy.num_inference_iters=100
```

如果要测试质量和速度折中，可以试：

```bash
policy.inference_scheduler=ddim policy.num_inference_iters=20
```

## 10. 我们在 LIBERO 里做了哪些适配

### 10.1 新增 policy 配置

`libero/configs/policy/diffusion_policy.yaml` 定义了 Hydra 可加载的 policy：

```yaml
policy_type: DiffusionPolicy
obs_horizon: 2
pred_horizon: 16
action_horizon: 8
inference_scheduler: ddim
num_inference_iters: 10
```

训练时使用：

```bash
policy=diffusion_policy
```

Hydra 会加载这个 yaml，然后 `policy_type` 决定实例化哪个 Python class。

### 10.2 注册 policy

在 `libero/lifelong/models/__init__.py` 加入：

```python
from libero.lifelong.models.diffusion_policy import DiffusionPolicy
```

这样 `get_policy_class("DiffusionPolicy")` 才能找到它。

### 10.3 单任务 benchmark

新增 `LIBERO_ALPHABET_SOUP`，只包含：

```text
pick_up_the_alphabet_soup_and_place_it_in_the_basket
```

这样可以用一个小任务快速验证 DP，而不是每次跑完整 `LIBERO_OBJECT`。

训练命令中使用：

```bash
benchmark_name=LIBERO_ALPHABET_SOUP
```

### 10.4 `eval.eval=false` 的训练路径

原始 base loop 在 epoch 0 仍然容易进入 rollout。我们改成只有：

```python
if self.cfg.eval.eval and epoch % self.cfg.eval.eval_every == 0:
```

才做 rollout evaluation。

这让 no-rollout smoke test 可以只验证数据加载、forward、loss、backward 和 checkpoint 保存，不被仿真环境阻塞。

### 10.5 `train.num_workers=0`

当前 HDF5 dataset 持有 h5py 对象，多个 dataloader worker 会触发 pickling 错误：

```text
TypeError: h5py objects cannot be pickled
```

所以现阶段命令里固定：

```bash
train.num_workers=0
```

以后如果要多 worker，需要让每个 worker 自己打开 HDF5 文件，而不是 pickle 已打开的 h5py handle。

## 11. 建议课堂讲解顺序

可以按这个顺序讲：

1. 先展示 HDF5 里一条 demo 的 `obs` 和 `actions`。
2. 说明 DP 不直接预测单步 action，而是生成未来动作序列。
3. 打开 `diffusion_policy.yaml`，解释 `obs_horizon`、`pred_horizon`、`action_horizon`。
4. 打开 `_encode_obs()`，说明两个相机和低维状态如何变成条件向量。
5. 打开 `ConditionalUnet1D`，说明为什么用 1D U-Net。
6. 打开 `compute_loss()`，逐行讲训练目标。
7. 打开 `_sample_action_plan()` 和 `get_action()`，讲 rollout 中如何从噪声动作变成 action queue。
8. 最后解释 DDIM 10 step smoke test 的意义：主要验证代码路径和速度，不期待 1 epoch 有成功率。

## 12. 当前推荐验证命令

no-rollout 训练 smoke test：

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

rollout smoke test：

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

观察重点：

```text
[info] policy has ...
[info] evaluate task 0 takes ... seconds
[info] Epoch:   0 | succ: ...
[info] Epoch:   1 | train loss: ...
[info] finished learning
```

1 epoch 的成功率为 0 是正常的。这个测试重点是确认动作形状、设备、scheduler、仿真 step 和 action queue 都没有错误。

## 13. 常见问题

### 为什么 `forward()` 里 timestep 是 0，但训练用 `compute_loss()`？

LIBERO 里很多工具会调用 policy 的 `forward()` 来做 FLOPs 或形状检查。真正训练使用的是 `compute_loss()`。所以 `forward()` 保持一个简单可运行路径，训练逻辑集中在 `compute_loss()`。

### 为什么 `pred_horizon` 要和 `data.seq_len` 对齐？

当前实现里：

```python
self.pred_horizon = min(policy_cfg.pred_horizon, cfg.data.seq_len)
```

因为训练动作来自 batch 的时间序列。如果数据只采样 16 步，就不能训练 32 步动作预测。

### 为什么 `action_horizon` 不是 16？

一次生成 16 步，但只执行 8 步，然后重新根据最新观测规划。这样比单步重规划快，也比一次执行完整 16 步更能适应环境状态变化。

### 为什么 rollout 比普通 BC 慢？

普通 BC 一次 forward 直接出动作。Diffusion Policy 每次重新规划时要做多次 U-Net forward。当前默认 DDIM 10 step，就是为了把这个成本降下来。

### 成功率上不去时先看什么？

先按顺序排查：

1. action 是否在 `[-1, 1]`。
2. rollout 中动作 shape 是否是 `(B, action_dim)`。
3. `eval.use_mp=false` 下单进程 rollout 是否稳定。
4. loss 是否持续下降。
5. DDIM 10 step 是否太少，可以试 20 step。
6. 训练 epoch 是否太少，1 epoch 只适合 smoke test。

## 14. 本实现的边界

当前实现是 2D RGB + low-dimensional proprio 的 LIBERO Diffusion Policy，不包含点云输入，也没有迁移 arcap 参考代码里的 3D pointcloud 分支、Dex 分支或特殊 action weighting。迁移过来的核心思想是：

```text
DDPM training + configurable DDIM inference
```

这使它适合先在 LIBERO HDF5 和仿真 rollout 中验证 DP 主路径，再决定是否继续扩展到更复杂的感知输入。
