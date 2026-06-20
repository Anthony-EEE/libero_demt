# Journal Extension Thought：在 PRV Guidance 之前加入 DEMT 结构发现

## 1. 核心 Gap

会议版实验结果很强，但从 formulation 到 guidance 的链条太直接：

```text
J(A(D)) -> Z = P x R x V -> z* -> corridor / grasp cue / speed bar
```

这会让 PRV 看起来像专家预先选择的假设。Reviewer 很可能会问：

> 为什么 deployment objective 会自然推出 spatial/contact/temporal guidance，而不是其他 human-controllable features？

Journal 版应该在中间加入一个缺失层：

```text
J(A(D))
  -> human-controllable feature intervention
  -> deployment sensitivity analysis
  -> discovered teaching structures G*
  -> human-executable guidance
```

最关键的一句话：

> PRV should not be the assumption of DEMT; PRV should be the discovered manipulation-specific teaching invariant, if the deployment evidence supports it.

这不是删除 PRV，而是降低 PRV 的地位：PRV 不再是 framework 的前提，而是一个被发现的 abstraction，或者说是 manipulation setting 下的一种 instantiation。

## 2. 三个 Contributions

**Contribution 1:** DEMT structure discovery formulation：在 deployment objective 和 human guidance 之间加入 deployment-evaluated structure-discovery layer，用 discovered learner-sensitive dataset structures 替代 expert-assumed PRV。

**Contribution 2:** Controlled successful-demonstration perturbation：在 matched task-state coverage 下，估计 observable、controllable、perturbable、guidanceable human factors 对 deployment performance 的因果敏感性。

**Contribution 3:** Guidance validation：把发现出来的 `G*` 转换成人可以执行的 guidance，并验证 guidance removal 之后，novice-provided demonstrations 在 held-out tasks 上仍然更好。

这比说 “we automate PRV guidance” 更干净。更强的 claim 是：

> 会议版用 expert knowledge 选择 PRV 作为 teaching structure；journal 版从 deployment evidence 中发现 learner-sensitive teaching structures。

## 3. Human-Controllable Feature Space H

不要从下面这个定义开始：

```text
Z = P x R x V
z* = arg max_z J(A(D_z))
```

而应该从更宽的 feature pool 开始：

```text
H = {h_1, h_2, ..., h_m}
```

其中 `H` 是 human-controllable demonstration factors 的集合。

为了避免 `H` 变成另一个专家拍脑袋的黑箱，需要定义 inclusion criteria。只有满足以下条件的 factor `h_i` 才能进入 `H`：

1. **Observable:** 可以从 demonstration trajectory、robot state、action、perception log 或 task event 中测量。
2. **Controllable:** human demonstrator 可以有意识地改变它。
3. **Perturbable:** controller、scripted policy 或 rollout generator 可以生成它的 controlled variants。
4. **Guidanceable:** 如果发现它重要，可以转换成 training cue。

候选 factors：

- approach waypoint distribution
- approach curvature
- final pre-contact position
- final contact position
- final contact orientation
- gripper closing timing
- gripper opening or release timing
- lift timing
- phase duration ratio
- speed profile
- action smoothness or jerk
- camera visibility or occlusion
- object-side choice
- release height and release offset
- redundant wrist posture, if available

重点不是这些因素都重要。重点是让 deployment evidence 决定哪些因素重要。

## 4. Feature vs Dataset-Level Structure

论文里要区分 single-demonstration feature 和 dataset-level teaching structure。

例子：

- `final contact orientation` 是一个 feature。
- `contact-orientation consistency across demonstrations` 是一个 dataset-level structure。
- `PRV consistency` 是多个 dataset-level structures 上的更高层 abstraction。

因为 `J(A(D))` 评价的是从一个 dataset 训练出来的 policy，而不是单条 demonstration，所以 formulation 应该使用 dataset statistics：

```text
phi_i(D) = variation / concentration / multimodality / phase alignment of factor h_i in dataset D
```

deployment relationship 可以写成：

```text
J(A(D)) ~ F(phi_1(D), phi_2(D), ..., phi_m(D))
```

这比简单说某个 `h_i` 好或不好更严谨。真正的问题是：

> 哪些 human-controllable factors 上的 dataset-level structures 会让 fixed learner 成功或失败？

此外，论文应该避免暗示 “low variance is always good”。更好的概念是 conditional regularity：

> We do not assume consistency or low variance is universally beneficial. We seek dataset-level structures that reduce learner-harmful ambiguity under matched task-state coverage.

例如，如果多个 grasp modes 都清楚地 conditioned on object pose 或 approach side，那么多个 grasp modes 可能是合理的。真正有害的不是 variation 本身，而是在相同 task state 下造成 fixed learner 观察到模糊 observation-action mappings 的 variation。

## 5. Deployment Sensitivity Formulation

生成一个 reference dataset：

```text
D_ref
```

`D_ref` 需要谨慎描述。它是用于估计 relative deployment sensitivity 的 matched successful reference condition，不是 globally optimal gold-standard dataset。

通过 perturb 一个 human-controllable factor 生成 controlled successful datasets：

```text
D_{h_i, sigma}
```

其中 `sigma` 控制 perturbation magnitude。

训练 fixed learner：

```text
theta_{h_i, sigma} = A(D_{h_i, sigma})
```

评估 deployment：

```text
J_{h_i, sigma} = J(A(D_{h_i, sigma}))
```

测量 deployment drop：

```text
Delta J_i(sigma) = J(A(D_ref)) - J(A(D_{h_i, sigma}))
```

如果可行，使用多个 perturbation magnitudes 来估计 dose-response relation：

```text
sigma increases -> phi_i(D) changes more -> J(A(D)) decreases
```

这比单点 `Delta J` 更有说服力，因为它显示 learner-harmful structure variation 增强时，会导致 monotonic 或 graded deployment degradation。

然后测试 interactions，因为 teaching structure 可能不是独立 factor，而是 factor coupling：

```text
Delta J_ij = J(A(D_ref)) - J(A(D_{h_i,h_j, sigma}))
Delta J_multi = J(A(D_ref)) - J(A(D_multi))
```

如果 combined perturbation 造成的 deployment drop 不能由 single-factor effects 的和解释，那么这个 interaction 就重要。例如 path variation 在 contact orientation 稳定时可能无害，但如果它改变了 final contact mode，就可能有害；timing variation 可能主要在 grasp 或 release 附近才重要。

选择 discovered learner-sensitive structures：

```text
G* = {h_i in H : Delta J_i > tau and h_i is human-executable}
```

更精确地说，因为 learner 看到的是 datasets：

```text
G* = {phi_i : changing phi_i(D) causes a significant deployment drop}
```

最终方法里，`G*` 应该是 statistically defined，而不只是 threshold-defined。一个 factor 或 interaction 进入 `G*`，至少需要满足：

- effect size: deployment drop 有实际意义，而不只是非零；
- seed stability: effect 在 training seeds 和 rollout seeds 上稳定；
- task stability: effect 在 discovery tasks 或预先指定的 task subset 上出现；
- interaction evidence: 当 `Delta J_ij` 或 `Delta J_multi` 大于 single-factor effects 所能解释的程度时，纳入 coupled effects；
- human-guidanceability: discovered structure 可以被表达成 novice training cue。

如果最大的 `Delta J_i` 对应 path variance、contact-pose/orientation dispersion、phase/timing variance，那么 PRV 就成为一个 discovered manipulation structure。

## 6. Controlled Perturbation Study

controller 或 good-policy 方法在科学上成立，因为它是 causal diagnostic intervention。它不需要真实模拟所有人类行为。它需要回答：

> 当所有 demonstrations 都 task-successful 时，哪些 human-controllable feature variations 会让 fixed learner 在训练后失败？

使用可靠的 successful demonstrations 来源：

- manually developed controller；
- strong learned policy；
- replayed LIBERO demonstrations with controlled perturbation and revalidation。

对于 LIBERO 或 simulation data，perturbations 必须在 environment 中 rollout 并重新验证。只离线修改 actions、poses 或 timing 不够，因为那可能生成 physically invalid trajectories。保留下来的 demonstration 必须 rollout-successful，而不是只在 HDF5 trajectory 里看起来成功。

对每个 task，创建：

- `D_ref`: low-perturbation reference dataset。
- `D_{h_i}`: single-factor perturbation datasets。
- `D_{h_i,h_j}`: selected interaction perturbation datasets。
- `D_multi`: 模拟 novice-like mixed variability 的 multi-factor perturbation datasets。
- `D_matched`: matched-coverage controls。

所有保留的 demonstrations 必须 task-successful。否则实验只是在证明 failed demonstrations 不好，而这不是 DEMT 的核心问题。

## 7. Matched-Coverage Control 是核心

最大的实验风险是 `D_ref` 太窄。如果 reference dataset 覆盖的 initial-state 或 object-pose distribution 与 perturbed datasets 不同，那么 performance drop 可能来自 coverage mismatch，而不是 teaching structure。

因此每个 condition 都应该 match：

- number of demonstrations `|D| = N`；
- task initial-state coverage；
- object pose coverage；
- task success rate before training；
- camera/view distribution where possible；
- learner architecture and hyperparameters；
- training seeds or balanced seed sets；
- rollout seeds and evaluation initial states。

目标设计是：

```text
same coverage + same success + one changed dataset-level structure -> deployment effect
```

这使 perturbation study 成为 causal study，而不仅仅是 data-quality comparison。

## 8. Effect Ranking

不要只比较 “good” 和 “bad” datasets。应该报告 factor-impact table：

| Human-controllable factor | Dataset statistic `phi_i(D)` | Demo success | Deployment `J` | `Delta J` | Stability | Failure phase |
| --- | --- | ---: | ---: | ---: | --- | --- |
| final contact orientation | orientation dispersion near close | 100% | low | high | stable | grasp closure |
| gripper closing timing | closing-time variance | 100% | low | high | stable | close/lift |
| approach waypoint | path variance / approach spread | 100% | medium | medium | task-dependent | reach/contact |
| camera occlusion | visibility variance | 100% | low/medium | high/medium | task-dependent | perception |
| action jerk | jerk distribution | 100% | medium | medium | unstable | unstable motion |
| redundant wrist posture | wrist-posture variance | 100% | high | low | stable low | none |

这会直接填补 formulation gap。你不是从 `J` 手工推出 PRV，而是在测量哪些 human-controllable dataset structures 会导致 `J` 下降。

还要报告 interaction effects：

| Interaction | Question | `Delta J_ij` interpretation |
| --- | --- | --- |
| path x contact orientation | Does path variation matter mainly because it changes contact mode? | coupled spatial-contact structure |
| closing timing x lift timing | Does timing variation matter mainly around phase transition? | coupled temporal-phase structure |
| visibility x contact pose | Does visual ambiguity amplify contact ambiguity? | coupled perception-contact structure |

## 9. Clustering 和 Regression 的作用

Clustering 和 regression 有用，但不能作为 causal proof。

正确分工是：

- **Controlled perturbation:** causal evidence。
- **Deployment sensitivity ranking:** 识别 important factors。
- **Regression / feature modelling:** 从 dataset statistics 预测 deployment。
- **Clustering:** 描述 high-`J` 和 low-`J` datasets 长什么样。
- **Human validation:** 测试 discovered structures 能否训练 novices。

Supervised analysis：

```text
J ~ f(path variance, contact dispersion, timing variance, visibility variation, jerk, ...)
```

Interaction-aware analysis：

```text
J ~ f(phi_i(D), phi_j(D), phi_i(D) * phi_j(D), ...)
```

Unsupervised analysis：

- cluster successful datasets by `phi_i(D)`；
- compare high-`J` and low-`J` clusters；
- describe the shared structures in each cluster。

如果 high-`J` clusters 共享 phase-aligned paths、concentrated contact poses/orientations、conditionally regular closing/lift/release timing，那么论文可以说：

> PRV-style conditional regularity is not assumed by DEMT; it emerges as the dominant structure of deployment-successful teaching data.

## 10. Revised DEMT Story

Journal paper 可以这样呈现 framework：

1. DEMT 通过 deployment 定义 teaching quality：

```text
J(A(D))
```

2. Human teaching 由一组 observable、controllable、perturbable、guidanceable factors 表示：

```text
H = {h_1, ..., h_m}
```

3. 每个 factor 诱导 dataset-level structure：

```text
phi_i(D)
```

4. Controlled successful-demonstration perturbations 估计 deployment sensitivity：

```text
Delta J_i = J(A(D_ref)) - J(A(D_{h_i, sigma}))
Delta J_ij = J(A(D_ref)) - J(A(D_{h_i,h_j, sigma}))
```

5. DEMT 用 effect-size、seed-stability、task-stability、interaction、guidanceability criteria 发现 learner-sensitive teaching structures：

```text
G* = {phi_i or phi_ij : statistically stable Delta J above a practical effect threshold}
```

6. `G*` 被转换成 human-executable guidance。

7. 在 manipulation setting 中，如果 `G*` 聚合成 spatial path、contact strategy、temporal phase structures，那么 PRV 被引入为 discovered abstraction：

```text
G* ≈ {spatial regularity, contact regularity, temporal regularity}
```

于是 guidance 变成：

- spatial regularity -> corridor；
- contact regularity -> grasp/contact cue；
- temporal regularity -> phase/speed cue。

这既保留了会议版贡献，也让 framework 更一般化。

## 11. Experiment Sequence

### Experiment 1: Deployment Sensitivity Over H

使用 controller 或 good-policy rollouts。先 perturb 一个 `h_i`，再测试 selected factor pairs 和 multi-factor perturbations。保持所有 demonstrations successful 且 coverage-matched。训练并部署 fixed learner。

Main result：

> 一些 successful-demonstration structures 和 structure couplings 会造成大幅 deployment drop，而其他不会。

### Experiment 2: Discovered Structure Analysis

按 `Delta J_i`、`Delta J_ij`、`Delta J_multi` 排序 factors 和 interactions。Regression 和 clustering 只在 controlled perturbation 之后使用。

Main result：

> 最大的 learner-sensitive factors 或 factor couplings 集中在 spatial path、contact/orientation、phase/timing structure，或者揭示出不同的 task-specific structure。

### Experiment 3: Guidance From G*

把 discovered `G*` 转换成 guidance。

如果 `G*` aligns with PRV：

- corridor guidance；
- contact cue；
- phase/speed cue。

如果 `G*` 不 align with PRV：

- visibility guidance；
- release-zone guidance；
- smoothness/jerk feedback；
- object-side cue；
- other task-specific cues。

Main result：

> DEMT is not committed to PRV; it converts discovered learner-sensitive structures into guidance.

### Experiment 4: Held-Out Human Validation

区分 discovery 和 validation。

Discovery tasks：

- 3-5 个 LIBERO 或 simulation tasks，用来发现 `G*`。

Validation tasks：

- 2-3 个 held-out LIBERO 或 real tasks，用来测试 discovered guidance。

最强版本的 groups：

- no-guidance control；
- generic AR feedback；
- discovered-structure guidance；
- optional expert-PRV guidance from the conference version。

discovered guidance 不一定需要超过 expert guidance。只要它接近 expert guidance，并且明显优于 no guidance 和 generic AR，就足够有力。

## 12. 先 MVP，后 Full Version

完整故事很大：broad `H`、single-factor perturbations、selected pairwise interactions、multi-factor novice-like perturbations、regression、clustering、LIBERO、held-out human validation、optional expert-PRV comparison。这是很强的 journal 方向，但不适合作为第一步实验全部展开。

MVP 只回答中心 gap：

> Can PRV-like structures be discovered from deployment sensitivity rather than assumed?

最小可执行实验：

1. 选择 3 个 LIBERO 或 PyBullet manipulation tasks。
2. 用四个 inclusion criteria 定义 6-8 个 human-controllable factors。
3. 为每个 factor 生成 task-successful、coverage-matched perturbation datasets。
4. 固定 learner、budget、training seeds、evaluation seeds、task-state coverage。
5. 计算 `Delta J_i` ranking。
6. 加入 2-3 个关键 interactions：path x contact，contact x timing，visibility x contact。
7. 检查 high-impact factors 是否聚合成 PRV-style conditional regularity。

如果这一步成立，journal gap 已经被填上 60-70%。Human validation 可以等 structure-discovery result 建立之后再做。

## 13. Realistic Publication Plan

最强 journal version 是：

```text
structure discovery + controlled causality + new held-out human study
```

但更现实的最低 extension 可以是：

```text
structure discovery
  + controlled causality
  + existing conference human-training evidence
  + one additional held-out validation task
```

只要逻辑清楚，这是可以接受的：

1. 新的 discovery experiment 证明 PRV-like structures 是 discovered，而不是 assumed。
2. 现有 human study 证明基于这些 structures 的 guidance 可以训练 novices。
3. 一个小规模 held-out validation 降低结果只 fit 原任务的风险。

## 14. 如果结果不符合 PRV

如果论文设计正确，这不是失败。

可能发现的 non-PRV factors：

- visibility or occlusion regularity；
- release pose regularity；
- gripper state timing；
- action smoothness；
- object-side choice；
- path-contact coupling rather than path alone。

结论可以变成：

> DEMT is not committed to PRV. PRV was the discovered structure for the original manipulation setting, but the journal framework can discover other task-specific teaching factors.

为了避免 p-hacking：

- 预先定义 discovery tasks 和 validation tasks；
- discovery tasks 只用于 identify `G*`；
- held-out validation tasks 用于 test guidance from `G*`；
- 如果 discovery 得到 non-PRV factors，就 validate those factors，而不是强行解释回 PRV。

## 15. Practical Pilot Using This Repo

当前 repo 支持 first pass：

- `datasets/libero_spatial` 有 10 个 spatial-shift tasks。
- `datasets/libero_object` 有 10 个 object-shift tasks。
- 每个 dataset 有 50 条 demonstrations。
- HDF5 files 暴露 `ee_pos`、`ee_ori`、`gripper_states`、`actions`、RGB observations、rewards、dones。
- LIBERO 提供 training 和 rollout evaluation，可以做 fixed-policy comparisons。

Pilot plan：

1. 选择 3 个 LIBERO 或 PyBullet manipulation tasks。
2. 用四个 inclusion criteria 定义 6-8 个 `H` factors。
3. 从 existing demonstrations 提取 dataset statistics `phi_i(D)` 作为 sanity check。
4. 建立 controller 或 perturbation pipeline，生成 causal single-factor datasets。
5. 在 environment 中 rollout and revalidate 每条 perturbed demonstration。
6. Match coverage across all perturbation conditions。
7. 为每个 condition 训练 fixed-budget policies。
8. 用 `Delta J_i` rank factors。
9. 用 `Delta J_ij` 测试 2-3 个 critical interactions。
10. 分析 high-impact `phi_i(D)` 是否 align with PRV-style conditional regularity，或者 reveal other structures。
11. 只有在 ranking 稳定之后，才把 discovered `G*` 转换成 guidance candidates。

## 16. Final Summary

Journal paper 不应该说：

> DEMT assumes PRV and automates PRV guidance.

应该说：

> DEMT defines quality by deployment. A new structure-discovery layer identifies which human-controllable dataset structures causally affect deployment under a fixed learner and budget. In our manipulation setting, the discovered structures align with PRV, so corridor, contact, and phase/speed guidance are justified by deployment evidence rather than expert assumption.
