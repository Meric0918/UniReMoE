# UniReMoE manuscript — reverse outline, claim–evidence map, self-review

Companion to `main.tex`. Everything here is derived from the `stage_final`
code analysis and `dataset/README.md`. **No `stage_final` training run exists in
the repository**, so every empirical claim is marked `needs evidence`.

---

## 1. Reverse outline

**Thesis.** Existing unified pHLA/pTCR predictors share one undifferentiated
interaction parameter set, fuse the two binding events only after pooling, and
produce three mutually unconstrained outputs. UniReMoE fixes all three with a
shared–private expert interaction layer, an HLA-conditioned residue-level
pTCR→triple gated transfer, and label-aware directional consistency.

### Introduction

| ¶ | Topic sentence | Evidence / explanation under it |
|---|---|---|
| 1 | Recognition needs two successive events, so neither pair alone answers the question. | Presentation then TCR engagement; triple is what vaccine/TCR-T design needs; screening is costly. |
| 2 | Predictors are built one event at a time, leaving the triple unanswered. | pHLA mature, pTCR data-scarce; both single-relation; heuristic min/product combination assumes conditional independence. |
| 3 | Unified multi-task models supply the triple but create two problems. | (a) one interaction parameter set serves two chemistries → interference, and share-all/share-none is a bad trade; (b) pooled-vector fusion destroys residue evidence. |
| 4 | A third problem: the three outputs may contradict each other, and the obvious fix is too strong. | No term penalises incoherence; a hard hierarchy fails because heads condition differently and are uncalibrated. |
| 5 | We propose UniReMoE with three coupled designs. | C1 routing-based sharing; C2 pre-pooling residue transfer under HLA-conditioned gate; C3 label-aware directional hinges. |
| 6 | *(placeholder)* Experiment summary. | **Empty by design — no run exists.** |
| 7 | Contribution list. | Four numbered items, the fourth empirical and unfilled. |

Mapping check: ¶2→problem statement, ¶3–4→the three technical reasons, ¶5→one
contribution per reason, in the same order. Clean.

### Method

| Subsection | Message | Three elements present? |
|---|---|---|
| Overview | One network, peptide axis preserved until pooling. | map of subsections ✔ |
| Problem formulation | Three binary problems, four observed label classes. | definition + why `110` matters ✔ |
| Frozen representations | PLMs frozen and absent from the graph; three encoders unify the space. | design ✔, motivation ✔ |
| Shared–private experts | Allocation between general and specific chemistry should be learned. | motivation ✔ design ✔ advantage ✔ |
| Gated transfer | Triple needs to know *where* the TCR engages, modulated by allele. | motivation ✔ design ✔ advantage ✔ |
| Directional consistency | Couple outputs softly; every update agrees with its own supervision. | motivation ✔ design ✔ advantage + non-claims ✔ |
| Training objective | Total loss, delayed ramp, two training modes. | design ✔ + algorithm ✔ |
| Data protocol | Entity-level splits; the audit is load-bearing because aux supervision is projected. | ✔ |
| Implementation | From scratch, two LR groups, schedules. | ✔ |

Every Method subsection maps to exactly one Introduction contribution or to the
protocol needed to evaluate them.

---

## 2. Claim–evidence map

### Architectural claims (verifiable in code today)

| Claim | Evidence | Status |
|---|---|---|
| Both PLMs are frozen and absent from the training graph. | `encoders.py:178,269` `requires_grad_(False)`; instantiated only in the precompute action, `stage_final_cli.py:105-142`; training reads fp16 HDF5. | **supported** |
| Interaction blocks have no position-wise FFN; the expert mixture replaces it. | `AttentionResidual`, `stage_MOE_model.py:21-38`, vs. baseline `attention.py:31-86` which has the FFN sublayer. | **supported** |
| Two globally shared + two branch-private experts per relation. | shared pool registered once `stage_MOE_model.py:99`; private `:49`; concatenation order `:60`. | **supported** |
| The mixture is dense; routing is sample-level, expert computation residue-level. | `gates` are `[B,4]`, softmax over all 4, no top-k/capacity, `stage_MOE_model.py:54-62`. | **supported** |
| Routers respect task input boundaries (pHLA never reads TCR; pTCR never reads HLA). | context construction `stage_MOE_model.py:109`, `:133`, `:147`. | **supported** |
| β is a per-residue scalar gated on triple state, pTCR state and pooled HLA summary. | `PTFusion`, `stage_MOE_PT_model.py:20-39`; gate input width 3·d; output `[B,L_P]`. | **supported** |
| β starts near 0.12 and must be opened by learning. | final gate bias init −2.0, `stage_MOE_PT_model.py:27`. | **supported** |
| No gradient truncation on either transfer path. | `tests/stage_final/test_combination.py:46-61`. | **supported** |
| The four consistency terms are directional with a stop-gradient on the label-designated reference. | `stage_MOE_consis_loss.py:49-60`; per-class gradient signs asserted `tests/stage_final/test_stage_final.py:87-105`. | **supported** |
| One forward pass yields all three logits on triple batches. | `forward_all`; `forward_batch` in `stage_MOE_consis_model.py:15-21`. | **supported** |
| Downstream trained from scratch; checkpoints of other architectures are refused. | `stage_final_cli.py:246`; `stage_final_checkpoint.py:53-56`; `test_combination.py:72-78`. | **supported** |
| Three config-only ablations are reachable. | `pt_fusion_enabled` bool-validated; `aux_ph_weight`, `aux_pt_weight`, `hierarchy_lambda`, `pt_implication_lambda` validated finite-and-nonnegative (`stage_final_config.py:105-108`), and absent from the strictly-positive list at `:132-136`; zeroing all four reduces the loss to native BCE exactly, `tests/stage_final/test_stage_final.py:118-121`. | **supported** — but each needs its own YAML (no CLI flag) and must train from scratch, since objective weights are part of the resume contract |

### Empirical claims (all unfilled)

| Claim | Where | Status |
|---|---|---|
| Any accuracy number, for any task or baseline. | Tables 4–9 | **needs evidence** — no run exists |
| Consistency reduces measured violation rate at no cost to ranking/calibration. | §Results consistency | **needs evidence** |
| The shared fraction αᵣ differs meaningfully between relations. | §Results routing | **needs evidence** |
| The transfer gate concentrates at biologically plausible positions. | §Discussion (i)–(ii) | **needs evidence** |
| The triple branch beats s_min / s_prod. | Table main + §classes | **needs evidence** — and this is the load-bearing test, see risk R1 |

### Claims about prior work (verified against the published records)

The bibliography is now complete: 26 entries, each verified against its
Crossref record or official proceedings page. Verifying them forced three
corrections to the draft, because the original criticism of the unified models
did not match what those papers do.

| Claim as originally drafted | What the papers actually do | Fix applied |
|---|---|---|
| Unified models "share a single undifferentiated interaction parameter set", so the two chemistries compete in one set of weights. | **Wrong for both.** UnifyImmun gives each relation its *own* cross-attention branch over a shared peptide encoder. UniPMT routes everything through one heterogeneous graph. | Reframed to the defensible point: both **fix the shared/specific boundary by hand before training**, and neither lets the data decide how much interaction capacity the relations hold in common. |
| Unified models "fuse the two binding events only after pooling", giving the triple head one scalar summary. | **Wrong as stated.** UnifyImmun has **no triple head at all**. UniPMT does predict the triple, but from graph node/edge embeddings. | Reframed per model: UnifyImmun can only answer a triple question by combining two pair scores; UniPMT forms the triple prediction away from the residues where binding occurs. |
| MoE-TCR applies expert layers "at the level of whole task towers". | Close, and now stated precisely: its experts are whole pan-specific predictors chosen by a **sparse top-$k$ gate conditioned on the epitope**. | Sharpened, and used to contrast with our dense, in-block, residue-level mixture. |

Two further consequences:
- A **Related Work subsection on unified models** was added
  (§\ref{sec:related:unified} in the draft). They are the closest prior art and
  previously appeared only in the Introduction. It also states what this work
  inherits rather than improves: the idea that abundant presentation data
  should support the scarce recognition task is theirs.
- **ESM-C is not the 2021 ESM paper.** It is a December 2024 EvolutionaryScale
  release with no peer-reviewed paper or DOI; the 300M variant has 30 layers
  and embedding dimension 960, which matches the hidden size this pipeline
  consumes. The Method now cites the release itself, not the ESM-1b paper.

### Second verification pass: the §7.1 prior-work comparison

Two subagents verified the methods named in `贡献和创新点（对比）.md` §7.1
against the papers' own text. **That document is an AI chat transcript** (it has
`## Prompt:` / `## Response:` structure and `utm_source=chatgpt.com` links), so
its descriptions were treated as hypotheses. Six of them were wrong:

| Hypothesis from the doc | Verified reality | Consequence |
|---|---|---|
| Fusion-pMT has an independent PT branch | **No PT branch.** Single head, pure PH→PHT transfer, one-hot inputs, no PLM, sequential full-parameter finetuning | Our "independently supervised PT branch" increment is real and sharper than assumed |
| "TransTCR, 2025" is a paper | **It is one of two models** in Niu et al., "Attention-aware differential learning…", Brief. Bioinform. 26(1):bbaf038 (the other is TranspMHC) | Cite the paper, never "TransTCR"; model named in prose only |
| StriMap explicitly models TCR–HLA interaction | **Not found in two full-text passes.** pHLA is treated as one joint ligand; HLA conditions recognition via the frozen embedding, which is not a contact model | Removed that claim. Our concession to StriMap is now about α/β chains + structure, which it has and we don't |
| NeoMUST is MMoE-style | **CGC/PLE-style** ("customized gate control"); and **both its tasks are PH-level** — no PT, no PHT | Repositioned as multi-task *machinery* prior, not a three-relation competitor |
| HMCN "uses labels to avoid harmful gradients" | **No label conditioning at all.** Its squared hinge is a pure function of predictions and is **symmetric in gradient** — both operands move | Strengthens contribution 3's distinction |
| C-HMCNN is a distant relative | **It is the closest relative** and *is* label-aware: it multiplies by the binary label inside a max, so the label picks the candidate set and the max picks the recipient | Contribution 3 narrowed to three specific differences (soft vs architectural; stop-gradient vs candidate masking; per-sample label-determined direction vs fixed taxonomy edge) |

**Contribution 1 is framed as task–mechanism fit, by author decision.** The
paper does not claim the expert topology as its own invention — CGC's
attribution is stated in bold in Related Work, in the Introduction, in the
contribution list and in the positioning table. What it claims instead is the
correspondence between the mechanism and *this* biology, the placement that
makes the mechanism usable here, and an empirical test of the correspondence.
This is a legitimate framing for the venue, but it shifts the burden of proof:
a mechanism-novelty claim is settled by the architecture diagram, whereas a
fit claim is settled by evidence, which makes the first two blocks of
Table~\ref{tab:ablation} and the $\alpha_r$ analysis **load-bearing rather
than optional**. See the open items below.

The argument as written rests on four task properties, in Method §3.4:
(i) the two receptors read different parts of one shared peptide — anchors in
the groove versus the exposed central bulge — so part of the chemistry is
common and part is binding-mode specific; (ii) the events are mechanically
coupled, since anchor binding fixes the conformation the receptor reads;
(iii) the degree of coupling is sample dependent, which is what justifies a
learned gate over a fixed partition; (iv) the peptide is the only entity all
three relations share and their input scopes are unequal, so a position-wise
expert on the peptide axis is the only module every relation can reach with
its own inputs. Property (iv) is the strongest and is specific to this problem.

The priority statement is deliberately bounded to application and placement:
expert mixtures already exist in this domain at predictor level (MoE-TCR) and
over two peptide–MHC objectives on one input pair (NeoMUST), and the paper
says so immediately before stating the gap.

**Two things that were overclaimed and remain corrected:**

1. **The shared+private expert topology is CGC's**, not ours. Expert counts are
   a tuned hyper-parameter in that scheme, so "2 shared + 2 private" is a
   setting within CGC, not a departure from it. The draft now says this in
   bold in Related Work, in the Introduction, in the contribution list, in the
   Abstract and in the Key Points.
2. **Per-task gate-weight analysis is already published.** MMoE's Fig. 6
   reports each task's softmax gate distribution and reads off the
   shared/non-shared balance; PLE §5.3/Fig. 8 reports mean ± s.d. of every
   gate's expert weights and concludes its shared experts dominate. So
   $\alpha_r$ is **a convenient summary of a known diagnostic**, not a new
   analysis. The earlier phrasing ("makes the split a learned, reportable
   quantity", "the shared fraction becomes a quantity we can report") would not
   have survived review and has been replaced.

What contribution 1 can still defend, on three axes the two papers do not cover:
(i) the relations read **different input compositions** — PLE handles
heterogeneous *sample spaces*, which is not the same thing; (ii) the experts are
**position-wise and their output keeps the residue axis**, flowing into a later
interaction stage, whereas MMoE/CGC/PLE experts are MLPs on pooled vectors
feeding a tower; (iii) the gate is conditioned on a **relation summary of the
current pair**, not on raw input features (PLE's upper gates already condition
on a learned representation — the task's own previous fusion — so only the
*relation-summary* part is ours).

**Still open, flagged by the verification and not resolved:**
- No novelty search was done for contribution 3 beyond HMCN and C-HMCNN. The
  near-miss literatures are later logic-constraint work from the same group,
  and the asymmetric-stop-gradient family in self-supervised learning (mean
  teacher, BYOL, SimSiam, R-Drop), where "stop-gradient on the operand treated
  as the reliable target" is routine though not label-conditioned per sample.
  **Check at least one from each group before writing "to the best of our
  knowledge" anywhere in the paper.**
- HMCN credits the squared-hinge form to Wehrmann & Barros (CVPR 2018) and
  Wehrmann, Mattjie & Barros (PRL 2018). Those were not opened, so the form's
  true origin is unverified; the draft credits HMCN, which may be one step
  short of correct.
- Which of the candidate baselines are actually runnable: **StriMap** is the
  best bet (code, weights, Zenodo archive, documented finetuning path) but
  needs an ESMFold step and a single-chain restriction that departs from its
  published setting; **Niu et al.** has public code, though training scripts
  were not confirmed; **Fusion-pMT has no public implementation**, so that row
  may be unobtainable.

### Claims deliberately NOT made

| Tempting claim | Why it is excluded |
|---|---|
| The shared+private expert split is our invention. | It is CGC's topology (Tang et al., RecSys 2020). The paper claims the task-fit argument, the placement and the test — not the mechanism. |
| Reporting how much sharing each task chose is a new analysis. | MMoE Fig. 6 and PLE §5.3/Fig. 8 already report per-task gate-weight distributions. α_r is a scalar summary of a known diagnostic, used here as evidence for the fit claim. |
| First use of mixture-of-experts for TCR or peptide–MHC prediction. | MoE-TCR and NeoMUST precede us. The bounded claim is: first to place a shared+private mixture inside the residue-level binding interaction serving all three relations. |
| Per-position FFN experts are novel. | Standard in the sparse-expert transformer line; only the combination with a shared/private split and relation-conditioned gate is particular here. |
| More complete molecular modelling than StriMap. | StriMap uses paired α/β chains and predicted structure; we use one CDR3β and no structure. |
| The squared-hinge consistency form is ours. | HMCN's Eq. 16 — and HMCN itself credits it further back. |
| Better handling of gradient conflict than NeoMUST. | We apply no gradient surgery at all (no CAGrad, no PCGrad). |
| Efficiency gain from the expert layer. | The mixture is dense; all four experts always run. |
| Residue-level routing. | Gates are `[B,4]`, i.e. per sample. |
| Routing uniformity as a mechanism. | `balance_lambda = 0.0` in every reported config. |
| The three tasks mutually promote each other. | No single-task configuration exists; task weights are validated `> 0`, so Δ(joint − single) is not computable. |
| A probability hierarchy ŷ_PHT ≤ min(ŷ_PH, ŷ_PT) holds after training. | The terms are soft squared hinges with margin 0.05; stop-gradient does not remove conflict on shared parameters. |
| Parameter-matched no-fusion control. | `pt_fusion_enabled: false` leaves the 165k fusion parameters registered but unused. |
| Expert-layer ablations (plain FFN, all-shared, private-only, uniform gate). | Not configurable; would need new code. Flagged in Table ablation notes. |

---

## 3. Five-dimension self-review

### Contribution
- **R1 (highest rejection risk). Class `110` is absent, so the triple label is the conjunction of the pair labels on every usable row.** A reviewer will ask why a triple branch is needed. Mitigations in the draft: the limitation is stated first in the Limitations box; the s_min/s_prod rows are built into the main table as the fair control; the Conclusion names acquiring `110` supervision as the key future direction. **This cannot be fully resolved by writing — it needs either `110` data or a decisive s_min/s_prod win.**
- R2: the shared–private expert split is close to progressive-layered-extraction-style multi-task expert separation. The draft distinguishes placement (inside each interaction block, on residue-resolved tensors) and purpose (relation chemistry, not task towers), but the related-work entry must be written carefully against the actual paper.
- R3: contribution 2 is a gated residual injection, which is individually simple. The draft earns it on *where* it acts (pre-pooling, per residue) and on the HLA conditioning, not on mechanism novelty.

### Writing clarity
- Terminology is fixed by macros (`\PH`, `\PT`, `\PHT`, `\method`) so it cannot drift.
- "sample-level routing with residue-level expert computation" is used verbatim everywhere; "residue-level routing" appears nowhere.
- Each Method subsection follows motivation → design → advantage.
- Outstanding: the Introduction's experiment paragraph is a placeholder, so paragraph flow into the contribution list is currently abrupt.

### Experimental strength
- **No results at all.** This is the binding constraint on the whole draft.
- No external predictor is wrapped by the code, so every baseline row needs work outside this stage. The repository holds only pMTnet and ERGO-II benchmark outputs from earlier work.
- Single seed per invocation and no multi-seed runner, so "mean ± s.d. over seeds" requires repeated manual runs; the main-table caption must match whatever is actually run.
- The expert-layer ablations that would isolate contribution 1 are **not implementable by configuration**. Either write them or weaken the contribution-1 claims.

### Evaluation completeness
- Strong: entity-level splits, ten unseen benchmarks, four-class behaviour, violation diagnostics, calibration metrics, routing and gate statistics, cold-entity audit.
- Gaps: no external cohort; unseen-pair/unseen-triple are combination splits, not strictly cold; a few TCRs in the two unseen-TCR benchmarks appear in some task's training pool; train/eval class priors differ (1:3 vs 1:1) and interact with the MCC-selected threshold.
- All threshold-dependent metrics inherit the validation threshold choice; the draft separates them from AUROC/AUPRC explicitly.

### Method design soundness
- The stop-gradient argument is sound as stated and is unit-tested per class, and the draft is careful not to over-read it.
- Mechanisms are not co-timed (auxiliary BCE from epoch 1, consistency ramping from epoch 4 to 8), so attribution needs the ablations rather than training curves. Stated in Limitations.
- Sequential mode puts the triple task last in every round, a recency-bias risk; stated and flagged for the joint-vs-sequential comparison.
- Auxiliary supervision is projected from triple rows, not extra data, and it leaks on the uncleaned corpus. The draft says every reported model is trained on a cleaned copy.

---

## 4. Build notes

**This machine has no TeX engine** (no `pdflatex`/`xelatex`/`lualatex`/`latexmk`/
`bibtex`/`tectonic`, no `/Library/TeX`), so the manuscript has **never been
compiled**. Everything below comes from reading `oup-authoring-template.cls`
v1.5 plus mechanical checks on the source. Build on Overleaf or after
installing MacTeX/TinyTeX, with `pdflatex → bibtex → pdflatex → pdflatex`.

Mechanical checks that pass right now: all 19 environments balanced; all 11
`tabular*` blocks have correct cell counts including `\multicolumn` spans; no
undefined or duplicate `\label`/`\ref`; all 19 `\citep` keys resolve against
`references.bib` with no uncited entries; no `\bottomrule`/`\cmidrule`/
`\subfloat`/`\cite`; no non-ASCII bytes.

Class quirks that are already handled in the source — **do not "simplify" these
away**:

| Quirk | Why it matters |
|---|---|
| `\newcommand{\societylogo}{}` in the preamble | The class uses `\societylogo{}` in the page header but never defines it. |
| `\title[short]{long}` keeps its optional argument | The no-argument branch calls `\authormark`, which the class never defines → undefined control sequence. |
| Both `\vol{}` and `\volume{}` are set | `\vol` feeds the running heads, `\volume` the first-page opener box used by the `modern` style. Omitting `\volume` leaves a stray comma on page 1. |
| Back matter uses `\section*{}` | Dropping `unnumsec` to get numbered body sections would otherwise number the back matter 6.–10., which Bioinformatics does not do. |
| `\PHm`/`\PTm`/`\PHTm` in math, `\PH`/`\PT`/`\PHT` in prose | The prose forms set as italic multi-letter variables in math, and `\PHT`'s en dash becomes two binary minus signs inside a subscript. The prose forms must stay plain text because they appear in `\title[...]` and `\subsection{...}`. |
| No float inside `boxtext` | `boxtext` redefines `\@xfloat` to call `float.sty` macros the class does not load. The Limitations box contains no float. |
| `\begin{enumerate}[1.]`, not `enumitem` keys | `enumitem` is not loaded, so `[label=...]`, `[nosep]`, `[leftmargin=*]` would error. |
| `\botrule`, `\cline` | `booktabs` is not loaded; the class defines its own rules and has no `\bottomrule`/`\cmidrule`. |

Known cosmetic items left open:
- The Limitations box renders its heading as **Box 1** *Limitations*; the class
  appends the optional argument to an auto-numbered "Box N" label and cannot
  produce the bare word.
- `\figph` uses a fixed-height `\parbox`, so placeholder text longer than the
  box produces an Overfull `\vbox` warning. The teaser box is the tight one.
- Every `references.bib` entry is `@misc`. When filled in, each must be
  **retyped** as `@article`/`@inproceedings` — the abbrv styles ignore
  `journal`/`volume`/`pages` on `@misc`. Keep `year` non-empty throughout, or
  author-year natbib raises "Bibliography not compatible with author-year
  citations".

## 5. Ordered next actions

1. ~~Fill the bibliography.~~ **Done.** 32 verified entries; no stubs remain.
   The prior-work claims were checked and the draft corrected (see §2).
   Still worth a human pass: confirm the author lists render as the journal
   expects, and decide whether to cite ESM-C's release page or a later paper if
   one appears.
2. **Implement contribution 1's ablations — no longer optional.** Because
   contribution 1 is now a fit claim rather than a mechanism claim, its
   evidence is the whole argument. Needed, and none reachable by configuration
   today: matched-parameter plain FFN (separates decomposition from capacity),
   all-shared experts, private-only experts, uniform/non-learned gate, gate on
   peptide summary only, experts after pooling (removes residue resolution
   while keeping the topology), and a triple branch reading the pre-expert
   representation (keeps experts but blocks their reach into the triple).
   Expert counts are hard-coded and there is no gate-override switch, so each
   needs code. Also needed for contribution 3: a flag to drop the stop-gradient,
   giving the symmetric-hinge control.
3. **Check the α_r predictions actually hold** before the fit argument goes to
   review. If α_r is flat across the three relations and across samples, the
   gate is not using the decomposition and the honest conclusion is that any
   gain comes from added capacity — Method §3.4 and the Results prose already
   commit to reporting that outcome if it occurs.
4. **Run the model.** Joint and sequential, on the cleaned data copy, then fill
   the tables from `metrics.json` / `metrics.md` / `history.json`.
5. **Decide on baselines.** StriMap is the most plausible external comparator
   (public code, weights, documented finetuning), at the cost of an ESMFold
   step and a single-chain restriction; Niu et al. has public code; Fusion-pMT
   has none. Either run them under our splits or restrict the main table to our
   own variants plus s_min/s_prod, and say which.
6. **Novelty-check contribution 3** against the asymmetric-stop-gradient family
   (mean teacher, BYOL, SimSiam, R-Drop) and later logic-constraint work before
   writing "to the best of our knowledge" anywhere.
7. **Write the placeholder prose blocks last**, in this order: Results prose →
   Discussion → Introduction experiment paragraph → Abstract Results → Key
   Points.
8. Delete `\figph`, `\NA` and `\TODO` and draw the real figures.
