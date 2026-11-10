# Stage15 PH/PT benchmark results

- Checkpoint: `/home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Stage15/fold_1/seed_42/stage15_best_ptcr_20260916_200557_137979.pt`
- SHA-256: `0175c14996be81c13e4a9940970282dfae718333de820b2109e9b6496fabca6d`
- Fixed threshold: `0.5`
- Samples: **298034**

| Dataset | Head | Samples | AUROC | AUPR | Accuracy | MCC | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `PT/test_unseen_pair` | `ptcr` | 10274 | 0.767257 | 0.816238 | 0.720167 | 0.455319 | 0.679380 |
| `PT/test_unseen_peptide` | `ptcr` | 144800 | 0.626292 | 0.694697 | 0.628163 | 0.285627 | 0.522923 |
| `PT/test_unseen_tcr` | `ptcr` | 142960 | 0.890894 | 0.894251 | 0.819026 | 0.639457 | 0.824827 |
