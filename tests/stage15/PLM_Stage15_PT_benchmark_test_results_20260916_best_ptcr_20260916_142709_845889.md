# Stage15 PH/PT benchmark results

- Checkpoint: `/home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Stage15/fold_1/seed_42/stage15_best_ptcr_20260916_142709_845889.pt`
- SHA-256: `68688fe543eba6187d9f6ee3763a66e84ab884d83e2951a3f16cc7b01736620a`
- Fixed threshold: `0.5`
- Samples: **298034**

| Dataset | Head | Samples | AUROC | AUPR | Accuracy | MCC | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `PT/test_unseen_pair` | `ptcr` | 10274 | 0.907959 | 0.931461 | 0.859743 | 0.721370 | 0.854489 |
| `PT/test_unseen_peptide` | `ptcr` | 144800 | 0.861369 | 0.888465 | 0.795311 | 0.592437 | 0.786980 |
| `PT/test_unseen_tcr` | `ptcr` | 142960 | 0.903676 | 0.904239 | 0.826896 | 0.654050 | 0.829297 |
