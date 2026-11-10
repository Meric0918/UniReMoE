# Stage MOE evaluation

Checkpoint: `trained_model/PLM_Stage_MOE/seed_42/joint_online_20261027_193829/best.pt`

| Dataset | N | AUROC | AUPRC | MCC | F1 | Threshold | Cold audit |
|---|---:|---:|---:|---:|---:|---:|---|
| ph/test_unseen_hla | 47748 | 0.872767 | 0.858858 | 0.619021 | 0.806266 | 0.057493 | hla: 0 seen in training |
| ph/test_unseen_pair | 64 | 0.563477 | 0.510273 | 0.063628 | 0.482759 | 0.057493 | Mixed / combination split |
| ph/test_unseen_peptide | 62764 | 0.965484 | 0.961858 | 0.816643 | 0.910690 | 0.057493 | peptide: 0 seen in training |
| pt/test_unseen_pair | 10274 | 0.762409 | 0.830802 | 0.560940 | 0.710922 | 0.622000 | Mixed / combination split |
| pt/test_unseen_peptide | 144800 | 0.661672 | 0.739841 | 0.384559 | 0.511343 | 0.622000 | peptide: 0 seen in training |
| pt/test_unseen_tcr | 142960 | 0.892945 | 0.901076 | 0.641386 | 0.802472 | 0.622000 | tcr: 5 seen in training |
| pht/test_unseen_hla | 19670 | 0.864188 | 0.735581 | 0.743150 | 0.876015 | 0.000000 | hla: 0 seen in training |
| pht/test_unseen_peptide | 14118 | 0.979166 | 0.980995 | 0.851019 | 0.922504 | 0.000000 | peptide: 0 seen in training |
| pht/test_unseen_tcr | 7698 | 0.999271 | 0.999333 | 0.967553 | 0.983830 | 0.000000 | tcr: 4 seen in training |
| pht/test_unseen_triple | 660 | 0.800115 | 0.774932 | 0.373831 | 0.631763 | 0.000000 | Mixed / combination split |
