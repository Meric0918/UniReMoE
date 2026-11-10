# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

UnifyImmun predicts peptide binding to **HLA (PH / pHLA)**, **TCR (PT / pTCR)** and the **peptide-HLA-TCR triple (PHT)** on top of frozen protein-language-model residue embeddings: ESM-C for peptide/HLA, TCR-BERT for the TCR CDR3.

Work is organised as a series of **independent "stage" experiments**. Each stage is its own pipeline — own YAML config, own entry scripts in `source/`, own modules in `source/plm_unified/`, own tests directory, own runbook in the repo root. Stages are added *beside* earlier ones and must not modify, overwrite or re-train them; several runbooks state this explicitly.

The legacy `run_all_phases.py` / `HLA_test.py` / `TCR_test.py` / `*_output_score.py` scripts named in README.md are **not present in this repo** — do not invent them.

## Stage map

Each stage has a root-level runbook (`PLM_STAGE<X>.md`, mostly Chinese). **Read the relevant runbook before touching a stage's code or training flow** — it is authoritative for commands, conditions, checkpoint contracts and acceptance evidence.

| Stage | Summary |
|---|---|
| 1 | Base PH/PT dual-task model: per-residue adapters over a frozen HDF5 cache, one shared peptide adapter, phase 1A warm-up then 1B alternating tasks with module-targeted FGM. Most later stages initialise from its `best_joint`. |
| 1_encoder | Stage 1 variant replacing the MLP adapters with 3-layer/8-head Transformer encoders |
| 2 | Progressive unfreezing of top ESM-C / TCR-BERT blocks with discriminative LRs, starting from a Stage 1 `last` checkpoint |
| 3 | Adds an explicit PHT triplet branch (`triplet_*` modules, `train_plm_triplet.py`) |
| 5 / 6 / 7 | Joint three-task training with frozen PLMs; PH-coverage + task-conflict ablation; read-only evaluation of the HLA-conditioning gain over Stage 1 |
| 8 | Four-type constrained PHT training from Stage 1 `best_joint`, data in `dataset/PHT/classes/PHT_{000,010,100,111}`. The most-reused base for later branches. |
| 9 | Pairwise TCR contrastive ranking within one pHLA, from a Stage 3 `triplet_best` |
| 10 / 12 | Teacher-student anti-forgetting (frozen Stage 3 teacher, KD on PH/PT); Stage8-aligned constrained retraining |
| 11 | Two-level contrastive learning over the sequential PHT path, trained from scratch |
| 13 | GraphSAGE fusion on the Stage 1 dual-task model |
| 14 / 15 | Hyper-PMT heterogeneous-hypergraph fusion on the Stage 8 contract; the same Shared Hyper-PMT dropped into Stage 1's PH/PT contract |
| 16 / 17 / 18 | The three "priority" experiments: independent PT validation splits in model selection with degradation gates; frozen-PLM gradient isolation; stronger PT replay + Stage 1 teacher retention. They share `stage16_{config,data,evaluation,selection,trainer,checkpoint,pipeline}.py` on purpose, so comparison rules cannot drift. See `STAGE16_18_IMPLEMENTATION.md` for how to run them. |
| adapter / PHT | Adapter and random-sequence-encoder variants of the two-level PHT cross-attention path |
| MOE / MOE_PT / MOE_consis | MoE experts replacing the interaction FFNs (v1); residue-level PT→PHT gated fusion (v2); four-class triplet consistency constraints |
| final | Combines `stage_MOE_PT` fusion with `stage_MOE_consis` auxiliary/consistency losses; frozen PLMs, downstream randomly initialised, joint or sequential training |

`train_plm_stage8_pt_retention.py` is an older combination prototype — not a substitute for the Stage 16/17/18 entry points.

## Environment

- Python 3.10, CUDA PyTorch (2.2.1 upstream). Server-local checkpoints, referenced by absolute path in the YAML: ESM-C `/home/mjp/model/ESMC-300M` (hidden 960), TCR-BERT `/home/mjp/model/TCR-Bert` (hidden 768).
- `requirements-plm.txt` is the dependency set for these pipelines (**not** `requirements.txt`). It pins a `transformers` git fork that ESM-C requires and that must match the checkpoint's exported version. Install after a CUDA torch build is already present.
- Server runs use the `unifyimmun` conda env with `OMP_NUM_THREADS=2` and `MKL_THREADING_LAYER=GNU`.
- `data/`, `trained_model/`, `embedding_cache/`, `swanlog/`, `logs/`, `analysis_results/` are gitignored. `.gitignore` is allowlist-style (`*` then `!*.ext`), so a new file type needs an explicit `!` rule to be trackable.

## Common commands

Run entry scripts **from the repo root**. Each resolves its own default config under `configs/`, and paths inside the YAML are relative to `paths.project_root` (`..`), i.e. the repo root.

Stage 1 (the pattern the other stages follow — verify checkpoints, build cache, debug, seeds, resume, evaluate):

```bash
python source/smoke_test_plm_encoders.py --config configs/plm_stage1.yaml --device cuda
python source/precompute_plm_embeddings.py --config configs/plm_stage1.yaml --fold 1 --device cuda
python source/train_plm_stage1.py --config configs/plm_stage1.yaml --fold 1 --seed 9999 \
  --limit 4096 --num-workers 0 --swanlab-mode local
python source/run_plm_stage1_seeds.py --config configs/plm_stage1.yaml --fold 1 \
  --seeds 42 3407 2026 --device cuda --swanlab-mode online
python source/train_plm_stage1.py --config configs/plm_stage1.yaml --fold 1 --seed 42 \
  --resume trained_model/PLM_Unified/fold_1/seed_42/last.pt
python source/evaluate_plm_stage1.py --config configs/plm_stage1.yaml --fold 1 --seed 42
```

Later stages expose the same verbs through `source/{audit,precompute,train,evaluate,analyze,run}_plm_stage<X>*.py`, and several have a `run_stage<X>.sh` wrapper that dispatches subcommands:

```bash
./run_stage8.sh audit      # then: warmup | train | ablations
python source/train_plm_stage16.py --dry-run
```

**Always run the stage's `audit_plm_stage<X>_data.py` (or `./run_stage<X>.sh audit`) before training it** — the audits validate class directories, labels, duplicate/conflicting triples, sequence lengths and amino-acid alphabets, and abort on conflicts.

### Conventions enforced by the runbooks

- `--limit` is smoke-test only; never use it for a real experiment.
- Debug seed is **9999**; experiment seeds are **42 / 3407 / 2026**. They must not collide.
- `--num-workers 0` keeps HDF5 cache reads in the main process, which makes data problems debuggable.
- Mixed precision is `bf16`, autocast only on CUDA.
- To evaluate a new split, add it to the config's `*_splits`, rebuild the cache, then pass `--hla-split` / `--tcr-split`. Fixed-width caches are large (tens of GB for all test sets at once) — build per disk budget, and rebuild explicitly with `--overwrite`.
- Cache dtype, shard size and precompute batch size belong under `cache:` in the YAML, never in code.
- Don't start a formal (multi-round, multi-seed) training run on your own; runbooks are explicit that implementation work stops short of launching them.

## Tests

Tests are synthetic — no real checkpoints, no GPU. Two import styles coexist, and both require the repo root as cwd:

```bash
python -m pytest tests/                               # whole suite
python -m unittest tests.stage1.test_plm_unified      # one module (dotted path)
python -m pytest tests/stage8/test_plm_stage8.py      # one file
python -m pytest tests/stage8/test_plm_stage8.py -k benchmark
```

Older tests import `source.plm_unified.<module>`; the Stage 16-18 tests insert `source/` on `sys.path` and import `plm_unified.<module>` via `tests/stage16_18_helpers.py` (which also provides the `fixture(stage, condition)` factory of tiny deterministic encoders). Match the style of the directory you are adding to. `tests/<stage>/` also holds dated result markdown and benchmark output — those are records, not fixtures.

## Shared architecture

`source/plm_unified/` holds everything; `source/*.py` are thin CLI wrappers that turn YAML into a config dataclass and drive the modules. Within the package, imports are relative (`from .stage8_config import ...`).

Shared foundation, used by nearly every stage:

- **`config.py`** — typed dataclasses (`PathsConfig`, `ModelConfig`, `CacheConfig`, `TrainingConfig`, `SwanLabConfig`, `ExperimentConfig`) plus `load_config` and `apply_overrides`; derives per-fold/per-seed `cache_dir()` / `run_dir()` and validates that checkpoints exist. Each stage adds `stage<X>_config.py` with its own conditions and validation.
- **`encoders.py`** — frozen `ESMCResidueEncoder` / `TCRBertResidueEncoder`, `OnlineEncoderBundle` for `--no-cache`, and `checkpoint_fingerprint`, which hashes checkpoint *structure* (file sizes plus small samples) rather than the multi-GB weights.
- **`cache.py`** — sharded fixed-width HDF5 residue cache with a SQLite sequence index; `EmbeddingCacheWriter` / `EmbeddingCache` + `CacheLocator(shard, row, length)`, and a `MANIFEST_VERSION`.
- **`data.py`** — CSV split discovery (`split_path` maps task/split/fold onto the files under `dataset/`), sequence normalisation, cached and online dataloaders; `Task = Literal["phla", "ptcr"]`.
- **`attention.py` / `model.py`** — `SequenceAdapter`, `CrossAttentionBlock`, `MaskedAttentionPooling`, `BindingClassifier`; `UnifiedBindingModel` = one shared peptide adapter + HLA/TCR adapters + two cross-attention branches + two heads, with a single `forward(task, peptide, p_mask, receptor, r_mask)` returning `BindingOutput(logits, cross_attention, pooling_weights)`.
- **`fgm.py`** — `ModuleFGM`, per-parameter FGM restricted to explicitly selected modules (in 1B: the current task's peptide adapter + receptor adapter), with backup/restore and BN handling.
- **`trainer.py`** — `Stage1Trainer` (1A `warmup` → `warmup_last.pt`, then 1B alternating with FGM), selection on `best_joint_score` = mean of PH AUROC/AUPR and PT AUROC/AUPR, early stopping on `max_rounds` / patience / `min_delta`, checkpoints `warmup_last.pt` / `last.pt` / `best_joint.pt`. Also exports the shared `set_global_seed`, `build_optimizer`, `build_scheduler`, `resolve_device`, `load_training_checkpoint`, `peek_swanlab_run_id`.
- **`metrics.py` / `evaluation.py`** — `BinaryMetrics` (loss, AUROC, AUPR, ACC, MCC, F1, precision/recall/sensitivity/specificity) with `joint_score` / `prefixed_metrics`; `evaluate_model` is the shared inference loop.
- **`tracking.py`** — `SwanLabTracker`, import-free when `mode="disabled"` so CPU runs and tests need no swanlab. Default project `unifyimmun`, per-fold group, experiment named per fold/seed.
- **`artifacts.py`** — timestamped artifact names (`create_artifact_timestamp` / `validate_artifact_timestamp`), so checkpoints and result files carry run identity.

A stage then adds its own `stage<X>_{config,data,model,loss,trainer,checkpoint,evaluation}.py` subset. Follow that naming when adding a stage, and keep the new files isolated from existing ones.

### Data layout

- `dataset/` — the current data root (`paths.data_root`), e.g. `dataset/PHT/classes/PHT_{000,010,100,111}/{train,val}.csv` for Stage 8 and its descendants. `dataset/README.md` is the split-audit report (entity-level split, seed, row counts, unseen-peptide/HLA/TCR/pair/triple splits).
- `data/data_HLA/*.csv`, `data/data_TCR/*.csv` — legacy fold-based splits (`{train,val}_fold_{1..5}.csv` plus `{dataset,independent,external,HPV,neoantigen}_set.csv` and `{covid,independent,triple}_set.csv`); columns `peptide`, `HLA`, `tcr`. Not checked in.
- `embedding_cache/plm_stage<X>/...`, `trained_model/<StageName>/fold_{f}/seed_{s}/`, `swanlog/` — generated, gitignored.

## Conventions

- Code is typed: `from __future__ import annotations`, dataclasses, `Literal`. Match that style.
- Never hardcode absolute paths in modules — resolve through `Path` against `paths.project_root`. The two PLM checkpoints are the only absolute paths, and they live in the config.
- When a change would alter an existing stage's behaviour, add a new stage instead. Checkpoints, configs and recorded results of finished stages are evidence for the paper and are treated as immutable.
