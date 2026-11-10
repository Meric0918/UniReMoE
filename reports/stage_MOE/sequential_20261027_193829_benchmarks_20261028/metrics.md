# Stage MOE evaluation

Checkpoint: `trained_model/PLM_Stage_MOE/seed_42/sequential_online_20261027_193829/best.pt`

| Dataset | N | AUROC | AUPRC | MCC | F1 | Threshold | Cold audit |
|---|---:|---:|---:|---:|---:|---:|---|
| ph/test_unseen_hla | 47748 | 0.844921 | 0.825739 | 0.531674 | 0.753717 | 0.098079 | hla: 0 seen in training |
| ph/test_unseen_pair | 64 | 0.659180 | 0.607481 | 0.139010 | 0.440000 | 0.098079 | Mixed / combination split |
| ph/test_unseen_peptide | 62764 | 0.956536 | 0.955051 | 0.783913 | 0.894347 | 0.098079 | peptide: 0 seen in training |
| pt/test_unseen_pair | 10274 | 0.812486 | 0.855856 | 0.568513 | 0.734559 | 0.654011 | Mixed / combination split |
| pt/test_unseen_peptide | 144800 | 0.709756 | 0.770141 | 0.410336 | 0.579867 | 0.654011 | peptide: 0 seen in training |
| pt/test_unseen_tcr | 142960 | 0.866384 | 0.881061 | 0.626652 | 0.795489 | 0.654011 | tcr: 5 seen in training |
| pht/test_unseen_hla | 19670 | 0.779976 | 0.642055 | 0.174989 | 0.461388 | 0.000001 | hla: 0 seen in training |
| pht/test_unseen_peptide | 14118 | 0.981625 | 0.983658 | 0.831629 | 0.915247 | 0.000001 | peptide: 0 seen in training |
| pht/test_unseen_tcr | 7698 | 0.999416 | 0.999456 | 0.955911 | 0.977993 | 0.000001 | tcr: 4 seen in training |
| pht/test_unseen_triple | 660 | 0.808269 | 0.768781 | 0.333953 | 0.616179 | 0.000001 | Mixed / combination split |
