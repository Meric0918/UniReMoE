# Stage10 Teacher–Student Retention Design

## Summary

Stage10 re-runs the Stage8 four-type training design from the approved Stage3 checkpoint while adding a frozen Stage3 teacher. The trainable student receives the same cached Stage3 encoder features as the teacher and is regularized to retain the teacher's pHLA and pTCR predictions. Stage10 is a new, isolated stage: no Stage8 source, configuration, scripts, checkpoints, or training behavior are modified.

The primary objective is to reduce PH/PT forgetting without giving up the PHT and consistency improvements sought by Stage8. Stage10 keeps all four Stage8 conditions (`baseline`, `hier_only`, `pt_imp_only`, and `full`) so `Stage10 baseline` differs from the existing `Stage8 baseline` only by teacher retention.

## Goals and Boundaries

- Initialize both teacher and student binding modules from the same fold-4, seed-2026 Stage3 `stage3_multitask_model` checkpoint.
- Freeze the teacher permanently and train only the student downstream binding modules.
- Preserve the Stage8 dataset, four-type balanced sampler, supervised losses, directed constraints, optimizer groups, learning rates, schedules, and joint score.
- Add PH/PT teacher-to-student KL retention to every Stage10 condition.
- Prevent a Stage10 best checkpoint from sacrificing the teacher's PH or PT validation task score by more than 0.002.
- Provide audit, cache, training, ablation, resume, validation, external pTCR comparison, and full-checkpoint export workflows.
- Keep external test sets out of checkpoint selection, threshold selection, and hyperparameter tuning.

Stage10 does not initialize from a Stage8 checkpoint, repair a previously trained Stage8 model, replay separate PH/PT datasets, unfreeze any PLM encoder, or change Stage8 behavior.

## Architecture

### Teacher and student

Stage10 instantiates two lightweight binding models from the Stage3 `binding_model` state:

- **Teacher:** `eval()` for its entire lifetime, `requires_grad_(False)`, excluded from the optimizer and saved only by Stage3 path and SHA-256 lineage.
- **Student:** initialized from the same state and structured identically to the Stage8 binding model. Its adapters, pretrained attention/pooling/classifier modules, and triplet modules use the same three optimizer groups and learning rates as Stage8.

The teacher only needs the PH and PT paths during training. The student performs the existing single-pass PH/PT/PHT forward. Evaluation and deployment use the student; the teacher is never embedded in a Stage10 lean or full checkpoint.

### Frozen encoder features and cache

All peptide, HLA, and TCR PLM encoders remain frozen. Teacher and student consume identical residue tensors from the existing immutable Stage8 cache namespace:

```text
embedding_cache/plm_stage8/fold_4/seed_2026/<stage3-sha-prefix>/
```

This avoids duplicating the current 12 GB cache. Stage10 treats it as a shared Stage3 feature cache and opens it only after validating the full Stage3 SHA-256, model dimensions, sequence lengths, cache dtype, and Stage8 four-type dataset fingerprint. The Stage10 cache command verifies the shared cache and builds it through a Stage10-owned entrypoint only when it is absent.

### Isolation

Stage10 owns its config, model/loss/trainer/checkpoint modules, CLI entrypoints, tests, documentation, output root, and SwanLab namespace. It may import the already-tested Stage8 data audit, cached dataset/collator, sampler, metric primitives, and precompute helpers, but no Stage8 file is edited.

## Data Flow and Loss

The input remains `data/data_triple/fold4/{train_merged_fold4.csv,val_merged_fold4.csv}` with the Stage8 schema `peptide,tcr,hla,PH,PT,PHT`. Audit, deduplication, conflict rejection, sequence validation, split leakage cleanup, and dataset fingerprinting remain unchanged. Training batches remain balanced across `111`, `100`, `010`, and `000`; the default batch size 256 therefore contributes 64 rows per type.

For every batch:

1. Move one set of cached hidden tensors and masks to the device.
2. Run the frozen teacher under `no_grad` and autocast to obtain PH/PT logits.
3. Run the student once to obtain PH/PT/PHT logits.
4. Compute the unchanged Stage8 supervised and condition-dependent constraint losses.
5. Compute teacher retention in float32 over all batch rows:

```text
L_keep_PH = KL(softmax(teacher_PH / T) || softmax(student_PH / T)) * T^2
L_keep_PT = KL(softmax(teacher_PT / T) || softmax(student_PT / T)) * T^2
L_keep    = w_PH * L_keep_PH + w_PT * L_keep_PT
L_total   = L_stage8 + lambda_keep * L_keep
```

Defaults are `T=1.0`, `w_PH=1.0`, `w_PT=1.0`, and `lambda_keep=1.0`, which reduces to the direct equal-weight KL stated in the forgetting analysis. These values are configurable and part of the strict resume contract. Teacher probabilities are detached; gradients can flow only through the student logits.

All Stage8 condition semantics are preserved. Retention is always active, while `baseline`, `hier_only`, `pt_imp_only`, and `full` continue to control only the two original directed constraints and their existing ramp.

## Validation and Checkpoint Selection

Stage10 validation evaluates teacher and student on the same ordered validation batches. It reports:

- the student's existing PH, PT, PHT, threshold-selected PHT, by-type, constraint, and joint metrics;
- the teacher's PH and PT metrics;
- PH/PT task-score deltas from teacher to student;
- PH/PT KL divergence, mean absolute positive-probability drift, and threshold-0.5 classification agreement.

The epoch-0 student is identical to the Stage3 teacher in evaluation mode and is always the safe initial best. A trained epoch is eligible for best selection only when both conditions hold:

```text
student_PH_task_score >= teacher_PH_task_score - 0.002
student_PT_task_score >= teacher_PT_task_score - 0.002
```

Among eligible epochs, the existing Stage8 joint score and `early_stopping_min_delta` select the best. An ineligible epoch counts as a non-improving epoch. If no trained epoch is eligible and better, the best checkpoint remains epoch 0.

`primary_success` requires all of the following:

- the best epoch passes the PH/PT retention gate;
- its Stage8 joint score is no more than 0.002 below epoch 0;
- its overall directed-constraint violation rate is below epoch 0.

External datasets never affect these decisions.

## Configuration and Commands

`configs/plm_stage10.yaml` mirrors Stage8 defaults under a `stage10` section and adds:

```yaml
keep_loss_weight: 1.0
phla_keep_weight: 1.0
ptcr_keep_weight: 1.0
distillation_temperature: 1.0
legacy_task_degradation_tolerance: 0.002
```

The output roots are `trained_model/PLM_Stage10` and `swanlog/PLM_Stage10`. The wrapper exposes:

```text
./run_stage10.sh audit
./run_stage10.sh cache
./run_stage10.sh train --condition <condition>
./run_stage10.sh ablations
./run_stage10.sh evaluate --checkpoint <stage10-checkpoint>
./run_stage10.sh evaluate-tcr --checkpoint <stage10-checkpoint> [--stage8-checkpoint <stage8-best>]
./run_stage10.sh export --checkpoint <stage10-lean> --output <stage10-full>
```

Training retains the Stage8 overrides for device, worker count, SwanLab mode, Stage3 checkpoint, run timestamp, resume checkpoint, and debug-only per-type limit.

## Checkpoints and Artifacts

Condition outputs live under:

```text
trained_model/PLM_Stage10/fold_4/seed_2026/<condition>/
```

Lean best/last checkpoints use `stage10_binding_model`, contain only the student binding state, and record the source Stage3 lineage, teacher policy, Stage10 config, data audit, training contract, optimizer/scheduler/scaler state, RNG state, self-contained best student state, validation retention metrics, and artifact role. Only a `last` checkpoint may resume training.

Resume reconstructs the teacher from the configured Stage3 checkpoint and rejects source SHA, condition, data fingerprint, effective sample size, model config, or Stage10 hyperparameter mismatches.

Full export uses `stage10_multitask_model`: it combines the three Stage3 PLMs with the Stage10 student binding state. It retains Stage3/Stage10 lineage and validation metadata but does not carry optimizer state or duplicate teacher weights.

Each completed run also writes validation predictions containing labels, student probabilities, teacher PH/PT probabilities, probability drifts, and source rows, plus a JSON summary with baseline/best metrics and retention eligibility.

## External pTCR Evaluation

The external evaluator loads Stage3 teacher and Stage10 student once, encodes each unique peptide/TCR with the Stage3 encoders, and evaluates both on identical rows at the fixed threshold 0.5. When a Stage8 checkpoint is provided, it must reference the same Stage3 SHA and is evaluated on the same encoded tensors.

Outputs include per-dataset Stage3/Stage8/Stage10 metrics, Stage10-minus-Stage3 and Stage10-minus-Stage8 deltas, aggregate counts, probability summaries, and a row-level CSV with labels and all available model probabilities. AUROC, AUPR, accuracy, F1, MCC, precision, recall, specificity, and loss are reported. External results are descriptive only and never cause the training or evaluation command to fail for performance reasons.

## Failure Handling

Stage10 fails immediately if:

- the Stage3 checkpoint type, fold, seed, model config, upstream lineage, or SHA does not match;
- the shared cache metadata or dataset fingerprint does not match;
- the four-type audit reports unresolved conflicts, unsupported label combinations, or a required type missing after Stage8-equivalent filtering; length and noncanonical rejections remain recorded and filtered exactly as in Stage8;
- any teacher parameter is trainable, appears in an optimizer group, receives a gradient, or the teacher enters training mode;
- a resume checkpoint is not a Stage10 `last` artifact or its training contract differs;
- a Stage8 comparison checkpoint references another Stage3 source.

## Testing and Acceptance

Tests cover:

- Stage10 config defaults, validation, paths, and strict unknown-key handling;
- teacher/student equality at initialization in evaluation mode;
- permanent teacher evaluation mode, frozen parameters, no gradients, and optimizer exclusion;
- zero KL for equal logits, positive KL for different logits, float32 computation, and student-only gradients;
- exact Stage8 loss equivalence when `keep_loss_weight=0` and unchanged behavior of all four conditions;
- retention eligibility, rejection of a higher-joint forgetting epoch, and epoch-0 fallback;
- one-epoch CPU training, artifact contents, teacher-state immutability, strict resume, and training-contract mismatch rejection;
- lean/full checkpoint validation and export without teacher duplication;
- paired Stage3/Stage8/Stage10 metric and row-report generation;
- Stage8 and Stage3 regression suites remaining unchanged and passing.

Implementation acceptance uses unit tests, data audit, shared-cache validation, and a debug limited training/evaluation smoke test. Formal external pTCR results are produced once after training and reported without hard performance thresholds so the test sets do not become development or selection data.
