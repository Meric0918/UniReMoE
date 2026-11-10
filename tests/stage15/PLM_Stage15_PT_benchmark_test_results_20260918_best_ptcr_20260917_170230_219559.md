# Stage15 PT benchmark results

- Checkpoint: `/home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Stage15/fold_1/seed_42/stage15_best_ptcr_20260917_170230_219559.pt`
- SHA-256: `f97e0ba14bf212b6e1108cff08e5ae18e931d79c138f73b3d610f505e7d09e5e`
- Fixed threshold: `0.5`
- Samples: **247477**

| Dataset | Head | Samples | AUROC | AUPR | Accuracy | MCC | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `PT/test_unseen_pair` | `ptcr` | 6124 | 0.650999 | 0.400760 | 0.797518 | 0.223841 | 0.313400 |
| `PT/test_unseen_peptide` | `ptcr` | 98393 | 0.692447 | 0.518418 | 0.727887 | 0.290303 | 0.457467 |
| `PT/test_unseen_tcr` | `ptcr` | 142960 | 0.903677 | 0.904240 | 0.826896 | 0.654050 | 0.829297 |
