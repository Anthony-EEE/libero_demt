# Journal Extension Thought: DEMT Structure Discovery Before PRV Guidance

## 1. Core Gap

The conference version is strong experimentally, but the formulation-to-guidance chain is too direct:

```text
J(A(D)) -> Z = P x R x V -> z* -> corridor / grasp cue / speed bar
```

This makes PRV look like an expert-selected assumption. A reviewer can reasonably ask:

> Why should the deployment objective naturally imply spatial/contact/temporal guidance rather than other human-controllable features?

The journal version should insert a missing layer:

```text
J(A(D))
  -> human-controllable feature intervention
  -> deployment sensitivity analysis
  -> discovered teaching structures G*
  -> human-executable guidance
```

The key sentence:

> PRV should not be the assumption of DEMT; PRV should be the discovered manipulation-specific teaching invariant, if the deployment evidence supports it.

This does not delete PRV. It downgrades PRV from the premise of the framework to one discovered abstraction or manipulation instantiation.

## 2. Three Contributions

**Contribution 1:** DEMT structure discovery formulation: insert a deployment-evaluated structure-discovery layer between deployment objective and human guidance, replacing expert-assumed PRV with discovered learner-sensitive dataset structures.

**Contribution 2:** Controlled successful-demonstration perturbation: estimate causal deployment sensitivity of observable, controllable, perturbable, and guidanceable human factors under matched task-state coverage.

**Contribution 3:** Guidance validation: convert discovered `G*` into human-executable guidance and validate that it improves novice-provided demonstrations on held-out tasks after guidance removal.

This is cleaner than saying "we automate PRV guidance." The stronger claim is:

> The conference version used expert knowledge to choose PRV as the teaching structure. The journal version discovers learner-sensitive teaching structures from deployment evidence.

## 3. Human-Controllable Feature Space H

Instead of starting from:

```text
Z = P x R x V
z* = arg max_z J(A(D_z))
```

start from:

```text
H = {h_1, h_2, ..., h_m}
```

where `H` is a broader pool of human-controllable demonstration factors.

To avoid making `H` another expert-picked black box, define inclusion criteria. A factor `h_i` enters `H` only if it is:

1. **Observable:** measurable from demonstration trajectories, robot state, actions, perception logs, or task events.
2. **Controllable:** a human demonstrator can intentionally change it.
3. **Perturbable:** a controller, scripted policy, or rollout generator can create controlled variants of it.
4. **Guidanceable:** if it is discovered to matter, it can be converted into a training cue.

Candidate factors:

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

The point is not that all these factors matter. The point is to let deployment evidence determine which factors matter.

## 4. Feature vs Dataset-Level Structure

The paper should distinguish a single-demonstration feature from a dataset-level teaching structure.

Example:

- `final contact orientation` is a feature.
- `contact-orientation consistency across demonstrations` is a dataset-level structure.
- `PRV consistency` is a higher-level abstraction over multiple dataset-level structures.

Because `J(A(D))` evaluates the policy trained from a dataset, not a single demonstration, the formulation should use dataset statistics:

```text
phi_i(D) = variation / concentration / multimodality / phase alignment of factor h_i in dataset D
```

Then the deployment relationship is:

```text
J(A(D)) ~ F(phi_1(D), phi_2(D), ..., phi_m(D))
```

This is more rigorous than saying one `h_i` is good or bad. The actual question is:

> Which dataset-level structures over human-controllable factors make the fixed learner succeed or fail?

Also, the paper should avoid implying that "low variance is always good." The better concept is conditional regularity:

> We do not assume consistency or low variance is universally beneficial. We seek dataset-level structures that reduce learner-harmful ambiguity under matched task-state coverage.

For example, multiple grasp modes may be valid if each mode is clearly conditioned on object pose or approach side. The harmful case is not variation itself; it is variation that creates ambiguous observation-action mappings for the fixed learner under the same task state.

## 5. Deployment Sensitivity Formulation

Generate a reference dataset:

```text
D_ref
```

`D_ref` should be described carefully. It is a matched successful reference condition used to estimate relative deployment sensitivity, not a globally optimal gold-standard dataset.

Generate controlled successful datasets by perturbing one human-controllable factor:

```text
D_{h_i, sigma}
```

where `sigma` controls perturbation magnitude.

Train the fixed learner:

```text
theta_{h_i, sigma} = A(D_{h_i, sigma})
```

Evaluate deployment:

```text
J_{h_i, sigma} = J(A(D_{h_i, sigma}))
```

Measure deployment drop:

```text
Delta J_i(sigma) = J(A(D_ref)) - J(A(D_{h_i, sigma}))
```

Where possible, use multiple perturbation magnitudes to estimate a dose-response relation:

```text
sigma increases -> phi_i(D) changes more -> J(A(D)) decreases
```

This is stronger than a single-point deployment drop because it shows that increasing learner-harmful structure variation produces monotonic or graded deployment degradation.

Then test interactions, because teaching structure may be a coupling between factors rather than an independent factor:

```text
Delta J_ij = J(A(D_ref)) - J(A(D_{h_i,h_j, sigma}))
Delta J_multi = J(A(D_ref)) - J(A(D_multi))
```

An interaction is important if the combined perturbation causes a deployment drop that cannot be explained by the sum of single-factor effects. For example, path variation may be harmless when contact orientation is stable, but harmful when it changes the final contact mode; timing variation may matter mainly near grasp or release.

Select discovered learner-sensitive structures:

```text
G* = {h_i in H : Delta J_i > tau and h_i is human-executable}
```

More precisely, because the learner sees datasets:

```text
G* = {phi_i : changing phi_i(D) causes a significant deployment drop}
```

In the final method, `G*` should be statistically defined, not only threshold-defined. A factor or interaction enters `G*` only if it satisfies:

- effect size: deployment drop is practically meaningful, not only nonzero;
- seed stability: the effect is stable across training and rollout seeds;
- task stability: the effect appears across discovery tasks or across a pre-specified task subset;
- interaction evidence: coupled effects are included when `Delta J_ij` or `Delta J_multi` is larger than expected from single-factor effects;
- human-guidanceability: the discovered structure can be communicated as a novice training cue.

If the largest `Delta J_i` values correspond to path variance, contact-pose/orientation dispersion, and phase/timing variance, then PRV becomes a discovered manipulation structure.

## 6. Controlled Perturbation Study

The controller or good-policy method is scientifically valid because it is a causal diagnostic intervention. It does not need to simulate all human behavior. It needs to answer:

> When all demonstrations are task-successful, which human-controllable feature variations make a fixed learner fail after training?

Use a reliable source of successful demonstrations:

- manually developed controller,
- strong learned policy,
- or replayed LIBERO demonstrations with controlled perturbation and revalidation.

For LIBERO or simulation data, perturbations must be rolled out and revalidated in the environment. Offline edits to actions, poses, or timing are not enough, because they can create physically invalid trajectories. A retained demonstration should be rollout-successful, not merely successful-looking in an HDF5 trajectory.

For each task, create:

- `D_ref`: low-perturbation reference dataset.
- `D_{h_i}`: single-factor perturbation datasets.
- `D_{h_i,h_j}`: selected interaction perturbation datasets.
- `D_multi`: multi-factor perturbation datasets that mimic novice-like mixed variability.
- `D_matched`: matched-coverage controls.

All retained demonstrations must be task-successful. Otherwise the experiment only proves that failed demonstrations are bad, which is not the DEMT question.

## 7. Matched-Coverage Control Is Core

The biggest experimental risk is that `D_ref` is too narrow. If the reference dataset covers a different initial-state or object-pose distribution than the perturbed datasets, performance drops may be caused by coverage mismatch rather than teaching structure.

Therefore every condition should match:

- number of demonstrations `|D| = N`;
- task initial-state coverage;
- object pose coverage;
- task success rate before training;
- camera/view distribution where possible;
- learner architecture and hyperparameters;
- training seeds or balanced seed sets;
- rollout seeds and evaluation initial states.

The intended design is:

```text
same coverage + same success + one changed dataset-level structure -> deployment effect
```

This is what makes the perturbation study causal rather than just a data-quality comparison.

## 8. Effect Ranking

Do not only compare "good" and "bad" datasets. Report a factor-impact table:

| Human-controllable factor | Dataset statistic `phi_i(D)` | Demo success | Deployment `J` | `Delta J` | Stability | Failure phase |
| --- | --- | ---: | ---: | ---: | --- | --- |
| final contact orientation | orientation dispersion near close | 100% | low | high | stable | grasp closure |
| gripper closing timing | closing-time variance | 100% | low | high | stable | close/lift |
| approach waypoint | path variance / approach spread | 100% | medium | medium | task-dependent | reach/contact |
| camera occlusion | visibility variance | 100% | low/medium | high/medium | task-dependent | perception |
| action jerk | jerk distribution | 100% | medium | medium | unstable | unstable motion |
| redundant wrist posture | wrist-posture variance | 100% | high | low | stable low | none |

This directly fills the formulation gap. You are not deriving PRV by hand from `J`; you are measuring which human-controllable dataset structures cause `J` to drop.

Also report interaction effects:

| Interaction | Question | `Delta J_ij` interpretation |
| --- | --- | --- |
| path x contact orientation | Does path variation matter mainly because it changes contact mode? | coupled spatial-contact structure |
| closing timing x lift timing | Does timing variation matter mainly around phase transition? | coupled temporal-phase structure |
| visibility x contact pose | Does visual ambiguity amplify contact ambiguity? | coupled perception-contact structure |

## 9. Role of Clustering and Regression

Clustering and regression are useful, but they should not be the causal proof.

Correct division:

- **Controlled perturbation:** causal evidence.
- **Deployment sensitivity ranking:** identifies important factors.
- **Regression / feature modelling:** predicts deployment from dataset statistics.
- **Clustering:** describes what high-`J` and low-`J` datasets look like.
- **Human validation:** tests whether discovered structures can train novices.

Supervised analysis:

```text
J ~ f(path variance, contact dispersion, timing variance, visibility variation, jerk, ...)
```

Interaction-aware analysis:

```text
J ~ f(phi_i(D), phi_j(D), phi_i(D) * phi_j(D), ...)
```

Unsupervised analysis:

- cluster successful datasets by `phi_i(D)`;
- compare high-`J` and low-`J` clusters;
- describe the shared structures in each cluster.

If high-`J` clusters share phase-aligned paths, concentrated contact poses/orientations, and conditionally regular closing/lift/release timing, then the paper can state:

> PRV-style conditional regularity is not assumed by DEMT; it emerges as the dominant structure of deployment-successful teaching data.

## 10. Revised DEMT Story

The journal paper can present the framework as:

1. DEMT defines teaching quality through deployment:

```text
J(A(D))
```

2. Human teaching is represented by a broad set of observable, controllable, perturbable, and guidanceable factors:

```text
H = {h_1, ..., h_m}
```

3. Each factor induces dataset-level structure:

```text
phi_i(D)
```

4. Controlled successful-demonstration perturbations estimate deployment sensitivity:

```text
Delta J_i = J(A(D_ref)) - J(A(D_{h_i, sigma}))
Delta J_ij = J(A(D_ref)) - J(A(D_{h_i,h_j, sigma}))
```

5. DEMT discovers learner-sensitive teaching structures with effect-size, seed-stability, task-stability, interaction, and guidanceability criteria:

```text
G* = {phi_i or phi_ij : statistically stable Delta J above a practical effect threshold}
```

6. `G*` is converted into human-executable guidance.

7. In the manipulation setting, if `G*` groups into spatial path, contact strategy, and temporal phase structures, PRV is introduced as the discovered abstraction:

```text
G* ≈ {spatial regularity, contact regularity, temporal regularity}
```

Then guidance becomes:

- spatial regularity -> corridor;
- contact regularity -> grasp/contact cue;
- temporal regularity -> phase/speed cue.

This preserves the conference contribution while making it more general.

## 11. Experiment Sequence

### Experiment 1: Deployment Sensitivity Over H

Use controller or good-policy rollouts. Perturb one `h_i` at a time, then test selected factor pairs and multi-factor perturbations. Keep all demonstrations successful and coverage-matched. Train and deploy the fixed learner.

Main result:

> Some successful-demonstration structures and structure couplings cause large deployment drops, while others do not.

### Experiment 2: Discovered Structure Analysis

Rank factors and interactions by `Delta J_i`, `Delta J_ij`, and `Delta J_multi`. Use regression and clustering only after controlled perturbation.

Main result:

> The largest learner-sensitive factors or factor couplings concentrate around spatial path, contact/orientation, and phase/timing structure, or reveal a different task-specific structure.

### Experiment 3: Guidance From G*

Convert discovered `G*` into guidance.

If `G*` aligns with PRV:

- corridor guidance;
- contact cue;
- phase/speed cue.

If `G*` does not align with PRV:

- visibility guidance;
- release-zone guidance;
- smoothness/jerk feedback;
- object-side cue;
- other task-specific cues.

Main result:

> DEMT is not committed to PRV; it converts discovered learner-sensitive structures into guidance.

### Experiment 4: Held-Out Human Validation

Separate discovery and validation.

Discovery tasks:

- 3-5 LIBERO or simulation tasks used to discover `G*`.

Validation tasks:

- 2-3 held-out LIBERO or real tasks used to test discovered guidance.

Groups for the strongest version:

- no-guidance control;
- generic AR feedback;
- discovered-structure guidance;
- optional expert-PRV guidance from the conference version.

The discovered guidance does not need to beat expert guidance. It is enough if it approaches expert guidance and clearly outperforms no guidance and generic AR.

## 12. MVP First, Full Version Later

The full story is large: broad `H`, single-factor perturbations, selected pairwise interactions, multi-factor novice-like perturbations, regression, clustering, LIBERO, held-out human validation, and optional expert-PRV comparison. That is a strong journal direction, but it is too much for the first experimental step.

The MVP should answer only the central gap:

> Can PRV-like structures be discovered from deployment sensitivity rather than assumed?

Minimum executable experiment:

1. Choose 3 manipulation tasks in LIBERO or PyBullet.
2. Define 6-8 human-controllable factors using the four inclusion criteria.
3. Generate task-successful, coverage-matched perturbation datasets for each factor.
4. Keep learner, budget, training seeds, evaluation seeds, and task-state coverage fixed.
5. Compute `Delta J_i` ranking.
6. Add 2-3 critical interactions: path x contact, contact x timing, visibility x contact.
7. Check whether high-impact factors cluster into PRV-style conditional regularity.

If this succeeds, the journal gap is already 60-70% filled. Human validation can come after the structure-discovery result is established.

## 13. Realistic Publication Plan

The strongest journal version is:

```text
structure discovery + controlled causality + new held-out human study
```

But a more realistic minimum extension could be:

```text
structure discovery
  + controlled causality
  + existing conference human-training evidence
  + one additional held-out validation task
```

This is acceptable if the logic is clear:

1. The new discovery experiment shows that PRV-like structures are discovered rather than assumed.
2. The existing human study shows that guidance based on those structures trains novices.
3. A small held-out validation reduces the risk that the result is only fitted to the original task.

## 14. If Results Do Not Match PRV

This is not a failure if the paper is designed correctly.

Possible discovered non-PRV factors:

- visibility or occlusion regularity;
- release pose regularity;
- gripper state timing;
- action smoothness;
- object-side choice;
- path-contact coupling rather than path alone.

The conclusion would become:

> DEMT is not committed to PRV. PRV was the discovered structure for the original manipulation setting, but the journal framework can discover other task-specific teaching factors.

To avoid p-hacking:

- predefine discovery tasks and validation tasks;
- use discovery tasks only to identify `G*`;
- use held-out validation tasks to test guidance from `G*`;
- if discovery finds non-PRV factors, validate those factors rather than forcing them back into PRV.

## 15. Practical Pilot Using This Repo

The local repo supports a first pass:

- `datasets/libero_spatial` has 10 spatial-shift tasks.
- `datasets/libero_object` has 10 object-shift tasks.
- Each dataset has 50 demonstrations.
- HDF5 files expose `ee_pos`, `ee_ori`, `gripper_states`, `actions`, RGB observations, rewards, and dones.
- LIBERO provides training and rollout evaluation for fixed-policy comparisons.

Pilot plan:

1. Pick 3 LIBERO or PyBullet manipulation tasks.
2. Define 6-8 `H` factors using the four inclusion criteria.
3. Extract dataset statistics `phi_i(D)` from existing demonstrations as a sanity check.
4. Build a controller or perturbation pipeline for causal single-factor datasets.
5. Roll out and revalidate every perturbed demonstration in the environment.
6. Match coverage across all perturbation conditions.
7. Train fixed-budget policies for each condition.
8. Rank factors by `Delta J_i`.
9. Test 2-3 critical interactions with `Delta J_ij`.
10. Analyse whether high-impact `phi_i(D)` align with PRV-style conditional regularity or reveal other structures.
11. Translate discovered `G*` into guidance candidates only after the ranking is stable.

## 16. Final Summary

The journal paper should not say:

> DEMT assumes PRV and automates PRV guidance.

It should say:

> DEMT defines quality by deployment. A new structure-discovery layer identifies which human-controllable dataset structures causally affect deployment under a fixed learner and budget. In our manipulation setting, the discovered structures align with PRV, so corridor, contact, and phase/speed guidance are justified by deployment evidence rather than expert assumption.
