# Stage10 Teacher–Student Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated Stage10 that re-trains the Stage8 four-type design from Stage3 while a frozen Stage3 teacher constrains student PH/PT drift.

**Architecture:** Stage10 constructs frozen teacher and trainable student binding modules from the same Stage3 state, feeds both the same immutable Stage3 residue cache, and adds equal-weight PH/PT KL retention to the unchanged Stage8 loss. A PH/PT validation gate protects best-checkpoint selection; deployment checkpoints contain only the student plus Stage3 encoder lineage.

**Tech Stack:** Python 3.11, PyTorch, NumPy, PyYAML, HDF5 embedding cache, `unittest`, SwanLab integration, Bash entrypoint.

## Global Constraints

- Do not modify any Stage8 source, config, script, checkpoint, cache contents, or behavior.
- Initialize Teacher and Student from the configured fold-4, seed-2026 Stage3 checkpoint; do not initialize from Stage8.
- Keep all PLM encoders frozen and reuse `embedding_cache/plm_stage8/fold_4/seed_2026/<stage3-sha-prefix>` after strict metadata validation.
- Preserve Stage8 data cleanup, balanced four-type sampling, supervised/constraint loss, stop-gradient directions, optimizer groups, schedules, and joint score.
- Enable teacher retention for all four conditions: `baseline`, `hier_only`, `pt_imp_only`, `full`.
- Default retention parameters are `lambda_keep=1.0`, `w_PH=1.0`, `w_PT=1.0`, `T=1.0`; PH/PT task-score degradation tolerance is `0.002`.
- External pTCR datasets are report-only and cannot affect checkpoint, threshold, or hyperparameter selection.
- Preserve all unrelated dirty-worktree changes and stage/commit only explicit Stage10 files.

---

### Task 1: Stage10 configuration and shared-cache contract

**Files:**
- Create: `source/plm_unified/stage10_config.py`
- Create: `configs/plm_stage10.yaml`
- Create: `tests/test_plm_stage10.py`

**Interfaces:**
- Consumes: shared `ModelConfig`, `CacheConfig`, and `SwanLabConfig` from `source/plm_unified/config.py`.
- Produces: `Stage10Condition`, `Stage10Paths`, `Stage10RuntimeConfig`, `Stage10TrainingConfig`, `Stage10ExperimentConfig`, `load_stage10_config()`, and `apply_stage10_overrides()`.

- [ ] **Step 1: Write failing configuration contract tests**

```python
def make_stage10_config() -> Stage10ExperimentConfig:
    root = Path("/tmp/stage10-contract")
    return Stage10ExperimentConfig(
        paths=Stage10Paths(root, root, root / "cache", root / "models", root / "logs")
    )

class Stage10ConfigTests(unittest.TestCase):
    def test_defaults_match_stage8_and_add_retention(self):
        root = Path("/tmp/stage10-contract")
        config = Stage10ExperimentConfig(
            paths=Stage10Paths(root, root, root, root, root)
        )
        self.assertEqual(tuple(value.value for value in Stage10Condition), (
            "baseline", "hier_only", "pt_imp_only", "full",
        ))
        self.assertEqual(config.stage10.keep_loss_weight, 1.0)
        self.assertEqual(config.stage10.phla_keep_weight, 1.0)
        self.assertEqual(config.stage10.ptcr_keep_weight, 1.0)
        self.assertEqual(config.stage10.distillation_temperature, 1.0)
        self.assertEqual(config.stage10.legacy_task_degradation_tolerance, 0.002)
        self.assertEqual(config.stage10.selection_weights, (0.6, 0.2, 0.2))

    def test_cache_dir_reuses_stage8_namespace(self):
        root = Path("/tmp/stage10-contract")
        config = Stage10ExperimentConfig(
            paths=Stage10Paths(root, root, root / "cache", root, root)
        )
        self.assertEqual(
            config.cache_dir("a" * 64),
            root / "cache/plm_stage8/fold_4/seed_2026/aaaaaaaaaaaa",
        )

    def test_retention_values_must_be_valid(self):
        config = make_stage10_config()
        config.stage10.distillation_temperature = 0.0
        with self.assertRaisesRegex(ValueError, "temperature"):
            config.validate()
```

- [ ] **Step 2: Run the tests and confirm they fail because Stage10 config is absent**

Run:

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_stage10.Stage10ConfigTests -v
```

Expected: import failure for `source.plm_unified.stage10_config`.

- [ ] **Step 3: Implement the Stage10 config types and validation**

Create the Stage8-equivalent fields plus the five retention fields. Keep the config section named `stage10`, reject unknown YAML sections/keys, lock fold/seed and approved data filenames, validate all nonnegative weights, require positive temperature/tolerance, require selection weights to sum to one, and return the Stage8 shared cache path.

```python
class Stage10Condition(str, Enum):
    BASELINE = "baseline"
    HIER_ONLY = "hier_only"
    PT_IMP_ONLY = "pt_imp_only"
    FULL = "full"

@dataclass
class Stage10Paths:
    project_root: Path
    data_root: Path
    cache_root: Path
    output_root: Path
    swanlog_root: Path
    esmc_model: Path = Path("/home/mjp/model/ESMC-300M")
    tcr_bert_model: Path = Path("/home/mjp/model/TCR-Bert")
    stage3_checkpoint: Path = Path(
        "trained_model/PLM_Triplet/fold_4/seed_2026/"
        "triplet_best_20260802_215908_285880.pt"
    )

@dataclass
class Stage10RuntimeConfig:
    fold: int = 4
    seed: int = 2026
    device: str = "auto"
    num_workers: int = 2

@dataclass
class Stage10TrainingConfig:
    data_subdir: str = "data_triple/fold4"
    train_filename: str = "train_merged_fold4.csv"
    val_filename: str = "val_merged_fold4.csv"
    reject_noncanonical_sequences: bool = True
    batch_size: int = 256
    eval_batch_size: int = 256
    max_epochs: int = 15
    min_epochs: int = 5
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    pht_loss_weight: float = 1.0
    phla_loss_weight: float = 0.3
    ptcr_loss_weight: float = 0.2
    pht_score_weight: float = 0.6
    phla_score_weight: float = 0.2
    ptcr_score_weight: float = 0.2
    constraint_margin: float = 0.05
    hierarchy_lambda: float = 0.05
    pt_implication_lambda: float = 0.03
    constraint_ramp_epochs: int = 5
    joint_degradation_tolerance: float = 0.002
    adapter_lr: float = 2.5e-5
    pretrained_attention_lr: float = 5e-5
    triplet_module_lr: float = 1e-4
    weight_decay: float = 1e-2
    scheduler_warmup_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    mixed_precision: str = "bf16"
    log_every_steps: int = 50
    keep_loss_weight: float = 1.0
    phla_keep_weight: float = 1.0
    ptcr_keep_weight: float = 1.0
    distillation_temperature: float = 1.0
    legacy_task_degradation_tolerance: float = 0.002

@dataclass
class Stage10ExperimentConfig:
    paths: Stage10Paths
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    runtime: Stage10RuntimeConfig = field(default_factory=Stage10RuntimeConfig)
    stage10: Stage10TrainingConfig = field(default_factory=Stage10TrainingConfig)
    swanlab: SwanLabConfig = field(default_factory=SwanLabConfig)

    @property
    def training(self) -> Stage10RuntimeConfig:
        return self.runtime

    def cache_dir(self, source_sha256: str) -> Path:
        return (
            self.paths.cache_root / "plm_stage8"
            / f"fold_{self.runtime.fold}"
            / f"seed_{self.runtime.seed}"
            / source_sha256[:12]
        )
```

- [ ] **Step 4: Add repository YAML defaults and verify parsing**

Use Stage8 model/cache/runtime/training values unchanged, set output roots to `PLM_Stage10`, and add SwanLab tags `stage10` and `teacher-student-retention`.

Run:

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_stage10.Stage10ConfigTests -v
```

Expected: all `Stage10ConfigTests` pass.

- [ ] **Step 5: Commit only configuration deliverables**

```bash
git add configs/plm_stage10.yaml source/plm_unified/stage10_config.py tests/test_plm_stage10.py
git commit -m "feat: add stage10 configuration contract"
```

---

### Task 2: Frozen teacher, trainable student, and KL retention loss

**Files:**
- Create: `source/plm_unified/stage10_model.py`
- Create: `source/plm_unified/stage10_loss.py`
- Modify: `tests/test_plm_stage10.py`

**Interfaces:**
- Consumes: `TripletBindingModel`, `ModelConfig`, `Stage10Condition`, and `compute_stage8_loss()`.
- Produces: `Stage10StudentOutput`, `Stage10TeacherOutput`, `Stage10BindingModel`, `FrozenStage10Teacher`, `Stage10TeacherStudentModels`, `build_stage10_teacher_student_models()`, `Stage10Loss`, `distillation_kl()`, and `compute_stage10_loss()`.

- [ ] **Step 1: Write failing model and loss tests**

```python
class Stage10ModelAndLossTests(unittest.TestCase):
    def test_teacher_and_student_start_equal_and_teacher_is_frozen(self):
        source = Stage10BindingModel(self.model_config()).state_dict()
        pair = build_stage10_teacher_student_models(
            self.model_config(), source, device=torch.device("cpu")
        )
        pair.teacher.train(True)
        self.assertFalse(pair.teacher.training)
        self.assertFalse(any(p.requires_grad for p in pair.teacher.parameters()))
        self.assertTrue(all(p.requires_grad for p in pair.student.parameters()))
        self.assert_state_equal(pair.teacher.state_dict(), pair.student.state_dict())

    def test_equal_logits_have_zero_keep_loss_and_only_student_gets_gradient(self):
        teacher = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
        student = teacher.clone().requires_grad_(True)
        value = distillation_kl(student, teacher, temperature=1.0)
        self.assertAlmostEqual(float(value), 0.0, places=7)
        value.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

    def test_zero_keep_weight_is_exact_stage8_loss(self):
        phla = torch.tensor([[0.0, 2.0], [2.0, 0.0]], requires_grad=True)
        ptcr = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
        pht = torch.tensor([[0.0, 3.0], [3.0, 0.0]], requires_grad=True)
        labels = torch.tensor([1, 0])
        label_types = ["111", "000"]
        stage8 = compute_stage8_loss(
            phla, ptcr, pht, labels, labels, labels, label_types,
            condition=Stage8Condition.FULL, epoch=3,
        )
        stage10 = compute_stage10_loss(
            phla, ptcr, pht, phla.detach(), ptcr.detach(),
            labels, labels, labels, label_types,
            condition=Stage10Condition.FULL, epoch=3,
            keep_loss_weight=0.0,
        )
        self.assertTrue(torch.equal(stage10.stage8.total, stage8.total))
        self.assertTrue(torch.equal(stage10.total, stage8.total))
```

- [ ] **Step 2: Run the focused tests and verify missing symbols fail**

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_stage10.Stage10ModelAndLossTests -v
```

Expected: import/name failures for Stage10 model and loss APIs.

- [ ] **Step 3: Implement the shared binding model and frozen teacher guard**

`Stage10BindingModel.forward_all()` must be behaviorally identical to Stage8. `forward_legacy_heads()` computes adapters, PH attention/pooling/classifier, and PT attention/pooling/classifier without running the triplet path.

```python
class FrozenStage10Teacher(Stage10BindingModel):
    def train(self, mode: bool = True):
        return super().train(False)

@dataclass
class Stage10TeacherStudentModels:
    teacher: FrozenStage10Teacher
    student: Stage10BindingModel

    def assert_teacher_frozen(self) -> None:
        if self.teacher.training:
            raise RuntimeError("Stage10 teacher must remain in evaluation mode")
        if any(p.requires_grad or p.grad is not None for p in self.teacher.parameters()):
            raise RuntimeError("Stage10 teacher must remain frozen and gradient-free")
```

The builder loads the same state strictly into two separately allocated models, freezes/evaluates the teacher, enables student gradients, moves both to the device, and calls `assert_teacher_frozen()`.

- [ ] **Step 4: Implement float32 teacher-to-student KL and composite loss**

```python
def distillation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")
    student_log_prob = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits.detach().float() / temperature, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * temperature**2

@dataclass
class Stage10Loss:
    total: torch.Tensor
    stage8: Stage8Loss
    keep: torch.Tensor
    keep_phla: torch.Tensor
    keep_ptcr: torch.Tensor
    keep_loss_weight: float

def compute_stage10_loss(
    phla_logits: torch.Tensor,
    ptcr_logits: torch.Tensor,
    pht_logits: torch.Tensor,
    teacher_phla_logits: torch.Tensor,
    teacher_ptcr_logits: torch.Tensor,
    ph_labels: torch.Tensor,
    pt_labels: torch.Tensor,
    pht_labels: torch.Tensor,
    label_types: Sequence[str],
    *,
    condition: Stage10Condition | str,
    epoch: int,
    margin: float = 0.05,
    hierarchy_lambda: float = 0.05,
    pt_implication_lambda: float = 0.03,
    ramp_epochs: int = 5,
    pht_weight: float = 1.0,
    phla_weight: float = 0.3,
    ptcr_weight: float = 0.2,
    keep_loss_weight: float = 1.0,
    phla_keep_weight: float = 1.0,
    ptcr_keep_weight: float = 1.0,
    distillation_temperature: float = 1.0,
) -> Stage10Loss:
    stage8 = compute_stage8_loss(
        phla_logits, ptcr_logits, pht_logits,
        ph_labels, pt_labels, pht_labels, label_types,
        condition=Stage8Condition(condition.value if isinstance(condition, Stage10Condition) else condition),
        epoch=epoch, margin=margin,
        hierarchy_lambda=hierarchy_lambda,
        pt_implication_lambda=pt_implication_lambda,
        ramp_epochs=ramp_epochs,
        pht_weight=pht_weight, phla_weight=phla_weight, ptcr_weight=ptcr_weight,
    )
    keep_phla = distillation_kl(
        phla_logits, teacher_phla_logits,
        temperature=distillation_temperature,
    )
    keep_ptcr = distillation_kl(
        ptcr_logits, teacher_ptcr_logits,
        temperature=distillation_temperature,
    )
    keep = phla_keep_weight * keep_phla + ptcr_keep_weight * keep_ptcr
    return Stage10Loss(
        total=stage8.total + keep_loss_weight * keep,
        stage8=stage8,
        keep=keep,
        keep_phla=keep_phla,
        keep_ptcr=keep_ptcr,
        keep_loss_weight=keep_loss_weight,
    )
```

- [ ] **Step 5: Run loss/model tests plus the Stage8 loss regression tests**

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_stage10.Stage10ModelAndLossTests \
  tests.test_plm_stage8.Stage8ModelAndLossTests -v
```

Expected: all tests pass and no Stage8 test is changed.

- [ ] **Step 6: Commit the model/loss slice**

```bash
git add source/plm_unified/stage10_model.py source/plm_unified/stage10_loss.py tests/test_plm_stage10.py
git commit -m "feat: add stage10 teacher student retention loss"
```

---

### Task 3: Paired validation metrics and retention eligibility

**Files:**
- Create: `source/plm_unified/stage10_evaluation.py`
- Modify: `tests/test_plm_stage10.py`

**Interfaces:**
- Consumes: teacher/student model APIs, Stage8 joint score, binary metrics, and PHT threshold selection.
- Produces: `Stage10Evaluation`, `retention_is_eligible()`, and `evaluate_stage10_models()`.

- [ ] **Step 1: Write failing paired-evaluation tests**

```python
def score_metrics(value: float) -> dict[str, float]:
    return {"auroc": value, "aupr": value}

class Stage10EvaluationTests(unittest.TestCase):
    def test_equal_models_have_zero_drift_and_are_retention_eligible(self):
        result = evaluate_stage10_models(
            teacher, student, [batch], device=torch.device("cpu"),
            mixed_precision="none", margin=0.05,
            selection_weights=(0.6, 0.2, 0.2),
            degradation_tolerance=0.002,
        )
        self.assertTrue(result.retention_eligible)
        self.assertAlmostEqual(result.retention["phla_kl"], 0.0, places=7)
        self.assertAlmostEqual(result.retention["ptcr_kl"], 0.0, places=7)
        self.assertEqual(result.retention["phla_agreement"], 1.0)
        self.assertEqual(result.retention["ptcr_agreement"], 1.0)

    def test_higher_joint_candidate_is_ineligible_when_pt_score_drops(self):
        teacher_metrics = {"phla": score_metrics(.90), "ptcr": score_metrics(.90)}
        student_metrics = {"phla": score_metrics(.91), "ptcr": score_metrics(.897)}
        self.assertFalse(retention_is_eligible(
            student_metrics, teacher_metrics, tolerance=0.002
        ))
```

- [ ] **Step 2: Run and verify the new evaluation tests fail**

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_stage10.Stage10EvaluationTests -v
```

Expected: missing `stage10_evaluation` APIs.

- [ ] **Step 3: Implement a one-pass paired evaluator**

Use `@torch.inference_mode()`, force both models to eval, and process each batch once. Collect teacher PH/PT probabilities and student PH/PT/PHT probabilities in source-row order. Compute the exact Stage8 student metrics and joint score, teacher PH/PT binary metrics, and these retention fields:

```python
@dataclass
class Stage10Evaluation:
    student_metrics: dict
    teacher_metrics: dict
    retention: dict[str, float]
    joint_score: float
    selected_pht_threshold: float
    retention_eligible: bool
    predictions: dict[str, np.ndarray]

def retention_is_eligible(student_metrics, teacher_metrics, *, tolerance: float) -> bool:
    return all(
        task_score(student_metrics[task])
        >= task_score(teacher_metrics[task]) - tolerance
        for task in ("phla", "ptcr")
    )
```

Compute KL from clipped two-class probabilities, probability MAE from positive-class probabilities, agreement from threshold-0.5 predictions, and task-score deltas as student minus teacher. Preserve Stage8 by-type and directed-violation formulas verbatim in Stage10 code without modifying Stage8.

- [ ] **Step 4: Add an equivalence assertion against Stage8 evaluation**

For the same student and validation loader, assert Stage10 student `phla`, `ptcr`, `pht`, `violations`, and joint metrics match `evaluate_stage8_model()` within `1e-7`.

- [ ] **Step 5: Run paired evaluation and Stage8 evaluation tests**

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_stage10.Stage10EvaluationTests \
  tests.test_plm_stage8.Stage8ModelAndLossTests -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit evaluation support**

```bash
git add source/plm_unified/stage10_evaluation.py tests/test_plm_stage10.py
git commit -m "feat: add stage10 paired retention evaluation"
```

---

### Task 4: Stage3 teacher lineage and Stage10 checkpoint lifecycle

**Files:**
- Create: `source/plm_unified/stage10_checkpoint.py`
- Modify: `tests/test_plm_stage10.py`

**Interfaces:**
- Consumes: `Stage10ExperimentConfig` and generic Torch checkpoint primitives.
- Produces: `Stage3TeacherSource`, `checkpoint_sha256()`, `load_torch_checkpoint()`, `load_stage3_teacher_source()`, `atomic_torch_save()`, `stage10_training_contract()`, lean/full/resume validators, `extract_stage10_binding_state()`, and `export_stage10_full_checkpoint()`.

- [ ] **Step 1: Write failing source/checkpoint contract tests**

Cover Stage3 type/tasks/model/fold/seed/upstream validation, binding-state extraction, Stage3 SHA lineage, lean checkpoint validation, training-contract mismatch, and full export.

```python
def test_full_export_contains_student_not_teacher(self):
    export_stage10_full_checkpoint(source, lean, destination)
    full = load_torch_checkpoint(destination, mmap=False)
    self.assertEqual(full["checkpoint_type"], "stage10_multitask_model")
    self.assertNotIn("teacher", full["model"])
    self.assert_state_equal(
        extract_stage10_binding_state(full), lean["model"]
    )
```

- [ ] **Step 2: Run checkpoint tests and confirm missing APIs fail**

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_stage10.Stage10CheckpointTests -v
```

- [ ] **Step 3: Implement Stage3 teacher source validation**

Validate `format_version=1`, `checkpoint_type=stage3_multitask_model`, task set `{phla,ptcr,triplet}`, exact model config, fold 4, seed 2026, Stage1 upstream lineage, encoder keys, and binding keys. Return:

```python
@dataclass
class Stage3TeacherSource:
    path: Path
    sha256: str
    checkpoint: dict[str, Any]
    binding_state: dict[str, torch.Tensor]
    lineage: dict[str, Any]
```

- [ ] **Step 4: Implement strict lean/resume/full contracts**

Lean type is `stage10_binding_model`; full type is `stage10_multitask_model`. The training-contract descriptor includes model config, every Stage10 field, fold, seed, effective data, and teacher policy. Resume requires matching contract SHA and source SHA. Full export starts from the Stage3 checkpoint, removes its old `binding_model.*`, inserts only `binding_model.<student-key>`, records Stage10 metadata, and drops training-only state.

- [ ] **Step 5: Run Stage10 checkpoint and Stage8 checkpoint regressions**

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_stage10.Stage10CheckpointTests \
  tests.test_plm_stage8.Stage8CacheAndCheckpointTests -v
```

- [ ] **Step 6: Commit checkpoint support**

```bash
git add source/plm_unified/stage10_checkpoint.py tests/test_plm_stage10.py
git commit -m "feat: add stage10 checkpoint lifecycle"
```

---

### Task 5: Stage10 trainer, hard retention gate, and deterministic resume

**Files:**
- Create: `source/plm_unified/stage10_trainer.py`
- Modify: `tests/test_plm_stage10.py`

**Interfaces:**
- Consumes: teacher/student pair, Stage10 config/loss/evaluation/checkpoint APIs, Stage8-compatible cached loaders, scheduler, and tracker.
- Produces: `build_stage10_optimizer()`, `candidate_improves()`, and `Stage10Trainer`.

- [ ] **Step 1: Write failing optimizer and CPU lifecycle tests**

```python
class Stage10TrainerTests(unittest.TestCase):
    def test_optimizer_contains_every_student_parameter_and_no_teacher_parameter(self):
        optimizer = build_stage10_optimizer(pair.teacher, pair.student, config)
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(optimized, {id(p) for p in pair.student.parameters()})
        self.assertTrue(optimized.isdisjoint({id(p) for p in pair.teacher.parameters()}))

    def test_ineligible_high_joint_epoch_does_not_replace_epoch_zero(self):
        candidate = SimpleNamespace(retention_eligible=False, joint_score=.99)
        self.assertFalse(candidate_improves(
            candidate, best_joint=.80, min_delta=.0001
        ))

    def test_eligible_joint_improvement_replaces_epoch_zero(self):
        candidate = SimpleNamespace(retention_eligible=True, joint_score=.81)
        self.assertTrue(candidate_improves(
            candidate, best_joint=.80, min_delta=.0001
        ))

    def test_one_epoch_checkpoint_has_student_only_and_resume_is_self_contained(self):
        summary = trainer.fit()
        checkpoint = load_torch_checkpoint(summary["last_checkpoint"], mmap=False)
        self.assertEqual(checkpoint["checkpoint_type"], "stage10_binding_model")
        self.assertNotIn("teacher", checkpoint)
        resumed.fit(resume=summary["last_checkpoint"])
```

- [ ] **Step 2: Run trainer tests and confirm they fail before implementation**

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_stage10.Stage10TrainerTests -v
```

- [ ] **Step 3: Implement the student-only optimizer and teacher guards**

Mirror the three Stage8 optimizer groups using only `student` modules. Before training, after teacher forward, and after backward, call a teacher invariant check. The optimizer constructor must reject duplicate parameters and any teacher/student parameter ID overlap.

Implement the selection predicate exactly as:

```python
def candidate_improves(
    evaluation: Stage10Evaluation,
    *,
    best_joint: float,
    min_delta: float,
) -> bool:
    return (
        evaluation.retention_eligible
        and evaluation.joint_score > best_joint + min_delta
    )
```

- [ ] **Step 4: Implement one training epoch**

```python
self.student.train()
self.teacher.eval()
with autocast_context:
    with torch.no_grad():
        teacher_output = self.teacher.forward_legacy_heads(
            batch["peptide_hidden"], batch["peptide_mask"],
            batch["hla_hidden"], batch["hla_mask"],
            batch["tcr_hidden"], batch["tcr_mask"],
        )
    student_output = self.student.forward_all(
        batch["peptide_hidden"], batch["peptide_mask"],
        batch["hla_hidden"], batch["hla_mask"],
        batch["tcr_hidden"], batch["tcr_mask"],
    )
    loss = compute_stage10_loss(
        student_output.phla_logits,
        student_output.ptcr_logits,
        student_output.pht_logits,
        teacher_output.phla_logits,
        teacher_output.ptcr_logits,
        batch["ph_labels"], batch["pt_labels"], batch["pht_labels"],
        batch["label_types"],
        condition=self.condition, epoch=epoch,
        margin=values.constraint_margin,
        hierarchy_lambda=values.hierarchy_lambda,
        pt_implication_lambda=values.pt_implication_lambda,
        ramp_epochs=values.constraint_ramp_epochs,
        pht_weight=values.pht_loss_weight,
        phla_weight=values.phla_loss_weight,
        ptcr_weight=values.ptcr_loss_weight,
        keep_loss_weight=values.keep_loss_weight,
        phla_keep_weight=values.phla_keep_weight,
        ptcr_keep_weight=values.ptcr_keep_weight,
        distillation_temperature=values.distillation_temperature,
    )
self.scaler.scale(loss.total).backward()
self.models.assert_teacher_frozen()
```

Track total, Stage8 supervised/head/constraint losses, total/PH/PT keep losses, gradient norm, effective constraint lambdas, and configured keep weight.

- [ ] **Step 5: Implement best selection, artifacts, and resume**

Evaluate epoch 0, capture it as best, then accept trained candidates only when `evaluation.retention_eligible` and joint improves by `early_stopping_min_delta`. Ineligible candidates increment `bad_epochs`. Save student-only best/last states, source lineage, teacher policy, evaluation metrics, retention metrics, self-contained best student, optimizer/scheduler/scaler/RNG, and strict contract. Write row predictions with student PH/PT/PHT, teacher PH/PT, drifts, labels/types/source rows. Compute `primary_success` from the three approved conditions.

- [ ] **Step 6: Verify bytewise teacher immutability and deterministic resume**

Snapshot cloned teacher state before fit and compare after fit/resume. Resume a copied `last` after deleting companion best/prediction files; confirm they are regenerated. Confirm a changed keep weight or tolerance is rejected as a contract mismatch and a `best` artifact cannot resume.

- [ ] **Step 7: Run trainer, loss, and Stage8 regression tests**

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_stage10.Stage10TrainerTests \
  tests.test_plm_stage10.Stage10ModelAndLossTests \
  tests.test_plm_stage8 -v
```

- [ ] **Step 8: Commit trainer support**

```bash
git add source/plm_unified/stage10_trainer.py tests/test_plm_stage10.py
git commit -m "feat: train stage10 with retention gating"
```

---

### Task 6: Audit, shared-cache, training, validation, ablation, and export CLIs

**Files:**
- Create: `source/plm_unified/stage10_precompute.py`
- Create: `source/audit_plm_stage10_data.py`
- Create: `source/precompute_plm_stage10_embeddings.py`
- Create: `source/train_plm_stage10.py`
- Create: `source/run_plm_stage10_ablations.py`
- Create: `source/evaluate_plm_stage10.py`
- Create: `source/export_plm_stage10_full.py`
- Create: `run_stage10.sh`
- Modify: `tests/test_plm_stage10.py`

**Interfaces:**
- Consumes: Stage10 config/model/trainer/checkpoint/evaluation and unmodified Stage8 data/cache APIs.
- Produces: complete `run_stage10.sh audit|cache|train|ablations|evaluate|export` workflow.

- [ ] **Step 1: Write failing CLI parsing and smoke tests**

Patch expensive model/cache functions and assert each entrypoint forwards config, Stage3 override, condition, device, worker, resume, output, and debug-limit arguments correctly. Assert `run_stage10.sh` rejects unknown actions with exit code 2.

- [ ] **Step 2: Run CLI tests and verify entrypoints are absent**

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_stage10.Stage10CliTests -v
```

- [ ] **Step 3: Implement Stage10 audit and shared-cache precompute**

The audit entrypoint invokes `audit_stage8_data()` unchanged but writes under `PLM_Stage10/data_audits/fold_4`. `stage10_precompute.py` mirrors Stage8 encoder loading/precompute behavior with `config.stage10.mixed_precision` and writes only to `config.cache_dir(source.sha256)`. If the cache exists, validate it and return without rewriting unless `--overwrite` is explicit.

- [ ] **Step 4: Implement the training and ablation entrypoints**

`train_plm_stage10.py` must audit first, validate/build source, open the shared cache, build loaders, construct teacher/student before clearing the large Stage3 checkpoint, instantiate `Stage10Trainer`, and log source SHA, teacher frozen state, retention parameters, data sizes, and condition. The ablation runner executes the four conditions in enum order and optionally verifies/precomputes cache first.

- [ ] **Step 5: Implement validation and full-export entrypoints**

The validation command accepts lean or full Stage10 checkpoints, validates source SHA/config, reconstructs teacher/student, runs paired validation, and writes JSON containing student, teacher, retention, eligibility, joint score, threshold, and source metadata. Export accepts a lean completed best checkpoint and emits the default sibling `stage10_full_<timestamp>.pt` unless `--output` is given.

- [ ] **Step 6: Implement the Bash dispatcher**

```bash
case "$ACTION" in
  audit) exec "$PYTHON_BIN" "$PROJECT_ROOT/source/audit_plm_stage10_data.py" --config "$CONFIG" "$@" ;;
  cache) exec "$PYTHON_BIN" "$PROJECT_ROOT/source/precompute_plm_stage10_embeddings.py" --config "$CONFIG" "$@" ;;
  train) exec "$PYTHON_BIN" "$PROJECT_ROOT/source/train_plm_stage10.py" --config "$CONFIG" "$@" ;;
  ablations) exec "$PYTHON_BIN" "$PROJECT_ROOT/source/run_plm_stage10_ablations.py" --config "$CONFIG" "$@" ;;
  evaluate) exec "$PYTHON_BIN" "$PROJECT_ROOT/source/evaluate_plm_stage10.py" --config "$CONFIG" "$@" ;;
  export) exec "$PYTHON_BIN" "$PROJECT_ROOT/source/export_plm_stage10_full.py" --config "$CONFIG" "$@" ;;
  *) echo "Usage: $0 {audit|cache|train|ablations|evaluate|evaluate-tcr|export} [options]" >&2; exit 2 ;;
esac
```

Leave the `evaluate-tcr` dispatch target for Task 7, but make the usage stable now.

- [ ] **Step 7: Run CLI tests and real read-only audit/cache validation**

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_stage10.Stage10CliTests -v
./run_stage10.sh audit
./run_stage10.sh cache
```

Expected: tests pass; audit reports no blocking conflicts/errors; cache validates the existing Stage3 SHA/fingerprint without rewriting shards.

- [ ] **Step 8: Commit the executable pipeline**

```bash
git add configs/plm_stage10.yaml run_stage10.sh \
  source/plm_unified/stage10_precompute.py \
  source/audit_plm_stage10_data.py \
  source/precompute_plm_stage10_embeddings.py \
  source/train_plm_stage10.py \
  source/run_plm_stage10_ablations.py \
  source/evaluate_plm_stage10.py \
  source/export_plm_stage10_full.py \
  tests/test_plm_stage10.py
git commit -m "feat: add stage10 training and evaluation pipeline"
```

---

### Task 7: External Stage3/Stage8/Stage10 pTCR comparison

**Files:**
- Create: `source/evaluate_plm_stage10_tcr_sets.py`
- Create: `source/plm_unified/stage10_external.py`
- Modify: `run_stage10.sh`
- Modify: `tests/test_plm_stage10.py`

**Interfaces:**
- Consumes: Stage3 online encoders, Stage3 teacher binding state, Stage10 checkpoint, optional Stage8 checkpoint, and external `peptide,tcr,label` CSVs.
- Produces: `evaluate_binding_models_on_rows()`, `build_external_comparison_report()`, JSON/Markdown summaries, and row-level probability CSVs.

- [ ] **Step 1: Write failing pure comparison/report tests**

```python
def test_comparison_report_contains_signed_deltas_for_available_models(self):
    report = build_external_comparison_report(
        labels=np.array([0, 1, 1, 0]),
        probabilities={
            "stage3": np.array([.1, .8, .7, .2]),
            "stage8": np.array([.2, .6, .4, .3]),
            "stage10": np.array([.1, .75, .65, .2]),
        },
        threshold=0.5,
    )
    self.assertIn("stage10_minus_stage3", report["deltas"])
    self.assertIn("stage10_minus_stage8", report["deltas"])
    self.assertEqual(report["metrics"]["stage10"]["samples"], 4)
```

Also test the no-Stage8 case, source-SHA mismatch, identical row order, fixed threshold propagation, positive/negative probability summaries, and CSV columns.

- [ ] **Step 2: Run external evaluator tests and confirm missing APIs fail**

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_stage10.Stage10ExternalEvaluationTests -v
```

- [ ] **Step 3: Implement shared-row multi-model inference**

Read/filter external files exactly as Stage8, collect all unique peptide/TCR sequences, load Stage3 encoders once, and encode each unique sequence once. Instantiate Stage3 teacher binding, Stage10 student binding, and optional Stage8 binding. For each pair batch, materialize hidden tensors once and call each model's `ptcr` path, preserving one label/order array.

```python
@dataclass(frozen=True)
class EncodedPairSequences:
    peptide_index: dict[str, int]
    tcr_index: dict[str, int]
    peptide_hidden: np.ndarray
    peptide_mask: np.ndarray
    tcr_hidden: np.ndarray
    tcr_mask: np.ndarray
```

`evaluate_binding_models_on_rows(models, rows, encoded, *, device, pair_batch_size, threshold)` returns `(reports_by_model, probabilities_by_model)`. For each contiguous row batch it resolves peptide/TCR indices, creates each hidden/mask tensor once, evaluates model names in sorted order through the `ptcr` path, appends positive probabilities under the same row order, and finally calls `compute_binary_metrics` separately for each model using one shared label array.

- [ ] **Step 4: Implement report and artifact generation**

For every dataset and model, report AUROC, AUPR, accuracy, F1, MCC, precision, recall/sensitivity, specificity, loss, counts, positive rate, and probability means. Compute numeric Stage10-minus-reference deltas for common metrics. Write one row CSV per run with dataset, source row, peptide, TCR, label, `prob_stage3`, optional `prob_stage8`, and `prob_stage10`. JSON must record all checkpoint paths/SHA values, shared Stage3 lineage, threshold source, device, batch sizes, filters, and unique counts.

- [ ] **Step 5: Wire the `evaluate-tcr` action and validate argument rules**

Require `--checkpoint`; make `--stage8-checkpoint` optional; default data directory to `data/data_TCR/test_data_set`; default threshold to 0.5; reject threshold outside `(0,1)` and any Stage8 source SHA mismatch. Never provide a test-set threshold-selection flag.

- [ ] **Step 6: Run external tests and a small-file smoke test**

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_stage10.Stage10ExternalEvaluationTests -v
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/evaluate_plm_stage10_tcr_sets.py --help
```

Expected: tests pass and CLI documents fixed-threshold comparison inputs.

- [ ] **Step 7: Commit external comparison support**

```bash
git add run_stage10.sh source/evaluate_plm_stage10_tcr_sets.py \
  source/plm_unified/stage10_external.py tests/test_plm_stage10.py
git commit -m "feat: compare stage10 ptcr retention externally"
```

---

### Task 8: Documentation, complete verification, and implementation handoff

**Files:**
- Create: `PLM_STAGE10.md`
- Modify: `tests/test_plm_stage10.py` only if a discovered acceptance gap requires a Stage10 regression.

**Interfaces:**
- Consumes: all completed Stage10 commands and artifact schemas.
- Produces: operator documentation and verified implementation status.

- [ ] **Step 1: Write operator documentation**

Document the teacher/student data flow, exact loss equation/defaults, four conditions, retention gate, cache reuse, command examples, checkpoint contents, resume rules, validation/external report fields, and the rule that external tests do not select models or thresholds. Include commands for audit, cache validation, baseline debug run, all ablations, internal evaluation, tri-model pTCR evaluation, and full export.

- [ ] **Step 2: Run formatting/static syntax checks without rewriting files**

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python -m compileall -q \
  source/plm_unified/stage10_*.py source/*plm_stage10*.py tests/test_plm_stage10.py
bash -n run_stage10.sh
git diff --check
```

Expected: all commands exit 0 with no output from `git diff --check`.

- [ ] **Step 3: Run the complete relevant unit regression suite**

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_stage10 \
  tests.test_plm_stage8 \
  tests.test_plm_triplet -v
```

Expected: zero failures and zero errors.

- [ ] **Step 4: Run real data/cache checks and the limited training smoke path**

```bash
./run_stage10.sh audit
./run_stage10.sh cache
./run_stage10.sh train \
  --condition baseline \
  --limit-per-type 4 \
  --num-workers 0 \
  --swanlab-mode disabled \
  --device cuda
```

Expected: audit is clean, cache validates without shard rewrite, and the debug run emits Stage10 best/last/summary/predictions with Teacher frozen and retention metrics present. If CUDA is unavailable, run the same command with `--device cpu` and `mixed_precision: none` in a temporary config outside the repository.

- [ ] **Step 5: Verify the final diff contains no Stage8 modifications**

```bash
git diff --name-only dc56859..HEAD | sort
git diff --name-only dc56859..HEAD | rg '(^|/)stage8|PLM_STAGE8|plm_stage8' && exit 1 || true
```

Expected: implementation commits contain Stage10 and design/plan files only; no Stage8 path appears.

- [ ] **Step 6: Commit documentation and any final Stage10-only test correction**

```bash
git add PLM_STAGE10.md tests/test_plm_stage10.py
git commit -m "docs: document stage10 retention workflow"
```

- [ ] **Step 7: Record formal experiment commands without running test-driven tuning**

```bash
./run_stage10.sh ablations --device cuda
./run_stage10.sh evaluate-tcr \
  --checkpoint trained_model/PLM_Stage10/fold_4/seed_2026/baseline/stage10_best_<run-timestamp>.pt \
  --stage8-checkpoint trained_model/PLM_Stage8/fold_4/seed_2026/baseline/stage8_best_20260809_011230_337330.pt \
  --threshold 0.5
```

The formal external report is descriptive. Do not alter Stage10 based on its test-set results; any later tuning requires a new validation-only experiment design.
